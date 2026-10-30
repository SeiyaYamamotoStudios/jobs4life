"""Outbound mail: the SMTP settings and the one function that sends a message.

**Provider-agnostic on purpose.** The deployment sends through Brevo's SMTP relay
(`smtp-relay.brevo.com:587`, STARTTLS), but nothing here knows that: it is
Python's stdlib `smtplib` and `email.message.EmailMessage`, so changing provider
is changing five environment variables. No vendor SDK, no HTTP API.

**The SMTP password is a server credential.** It is the same class of secret as
the master key, with the same rules: read once, by `MailSettings.from_env`, from
the environment of this process; kept out of the dataclass's `repr`; never logged,
never written to `runs`, never in a trace, never in an exception message, never
rendered. Every error raised from here is built from a literal plus an exception
*class name* and a numeric SMTP code -- never `str(exc)`, which for
`SMTPRecipientsRefused` carries the recipient address and for some servers echoes
the dialogue -- and is raised `from None` so the chained original does not reach a
traceback either. The worker stores `type(exc).__name__: message` in
`tasks.last_error` and logs the traceback; neither can contain the password.

**TLS is not optional.** `ssl.create_default_context()` (certificate and hostname
verified), STARTTLS on every port but 465 and implicit TLS on 465, and a server
that does not offer STARTTLS is refused *before* any credential is sent: login
happens only after the channel is encrypted, so there is no code path that puts the
password on a plaintext connection. A connect/read timeout bounds every call so a
black-holed relay cannot pin a worker thread.

**Retry classification.** A refused login or a 5xx answer will get the same answer
next time (`permanent=True`, which the handler turns into `PermanentTaskError`); a
timeout, a dropped connection or a 4xx is worth the queue's backoff. Delivery is
at-least-once either way -- see `handlers/digest.py` for the duplicate window.
"""

from __future__ import annotations

import contextlib
import smtplib
import ssl
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import parseaddr
from typing import Any
from urllib.parse import urlsplit

from jfl_core.digest_schedule import DigestSchedule

DEFAULT_SMTP_PORT = 587
DEFAULT_SMTP_TIMEOUT = 30.0
_IMPLICIT_TLS_PORT = 465


class MailConfigError(RuntimeError):
    """Mail settings are incomplete or malformed. Names variables, never values."""


class MailDeliveryError(RuntimeError):
    """A message was not accepted by the mail server. The text is a literal built
    from a class name and an SMTP code: it never carries a credential, an address
    or a server response.
    """

    def __init__(self, message: str, *, permanent: bool) -> None:
        super().__init__(message)
        self.permanent = permanent


@dataclass(frozen=True, slots=True)
class MailSettings:
    """Everything the worker needs to send mail. `None` where it is not configured
    (see `WorkerSettings.mail`), never a half-filled instance.
    """

    host: str
    port: int
    username: str | None
    password: str | None = field(repr=False)
    # The From header, bare or `Name <addr>`.
    sender: str
    # Absolute, no trailing slash: every link in an email is built from it.
    public_base_url: str
    # Signs unsubscribe links. Dedicated; see `jfl_core.email_links`.
    link_secret: str = field(repr=False)
    schedule: DigestSchedule
    timeout: float = DEFAULT_SMTP_TIMEOUT

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> MailSettings | None:
        """None when `JFL_SMTP_HOST` is unset -- mail is simply off, and the web
        app says so on the settings page. If a host IS set, everything else
        needed must be too, and a gap fails at boot naming the variables.
        """
        host = (env.get("JFL_SMTP_HOST") or "").strip()
        if not host:
            return None

        def value(name: str) -> str:
            return (env.get(name) or "").strip()

        missing = [
            name
            for name in ("JFL_MAIL_FROM", "JFL_PUBLIC_BASE_URL", "JFL_EMAIL_LINK_SECRET")
            if not value(name)
        ]
        username, password = value("JFL_SMTP_USERNAME"), env.get("JFL_SMTP_PASSWORD") or ""
        if bool(username) != bool(password):
            missing.append("JFL_SMTP_PASSWORD" if username else "JFL_SMTP_USERNAME")
        if missing:
            raise MailConfigError(
                "JFL_SMTP_HOST is set but these are missing: " + ", ".join(sorted(missing))
            )

        raw_port = value("JFL_SMTP_PORT")
        try:
            port = int(raw_port) if raw_port else DEFAULT_SMTP_PORT
        except ValueError:
            raise MailConfigError("JFL_SMTP_PORT must be a port number") from None
        if not 0 < port < 65536:
            raise MailConfigError("JFL_SMTP_PORT must be a port number")

        sender = value("JFL_MAIL_FROM")
        if "@" not in parseaddr(sender)[1] or "\n" in sender or "\r" in sender:
            raise MailConfigError("JFL_MAIL_FROM must be an address, or 'Name <address>'")

        return cls(
            host=host,
            port=port,
            username=username or None,
            password=password or None,
            sender=sender,
            public_base_url=normalise_base_url(value("JFL_PUBLIC_BASE_URL")),
            link_secret=value("JFL_EMAIL_LINK_SECRET"),
            schedule=DigestSchedule.from_env(env),
        )


def normalise_base_url(raw: str) -> str:
    """`https://host[/path]` without a trailing slash. Plain http is accepted for
    localhost only: a link in an email that sends a token over cleartext is not one
    to ship, and a typo here should stop the boot rather than the first unsubscribe.
    """
    parts = urlsplit(raw)
    local = parts.hostname in {"localhost", "127.0.0.1", "::1"}
    if not parts.hostname or parts.scheme not in {"https", "http"}:
        raise MailConfigError("JFL_PUBLIC_BASE_URL must be an absolute https:// URL")
    if parts.scheme == "http" and not local:
        raise MailConfigError("JFL_PUBLIC_BASE_URL must be https:// (http only for localhost)")
    if parts.query or parts.fragment:
        raise MailConfigError("JFL_PUBLIC_BASE_URL must not carry a query or fragment")
    return raw.rstrip("/")


SmtpFactory = Callable[..., Any]


class SmtpSender:
    """Sends one `EmailMessage` per call, on its own connection.

    A digest goes out once a day per user, so a connection per message costs
    nothing and leaves no idle socket to expire between sends. `smtp` and
    `smtp_ssl` are the stdlib classes by default; tests pass fakes with the same
    constructor, so no test opens a socket.
    """

    def __init__(
        self,
        settings: MailSettings,
        *,
        smtp: SmtpFactory = smtplib.SMTP,
        smtp_ssl: SmtpFactory = smtplib.SMTP_SSL,
        context_factory: Callable[[], ssl.SSLContext] = ssl.create_default_context,
    ) -> None:
        self._settings = settings
        self._smtp = smtp
        self._smtp_ssl = smtp_ssl
        self._context_factory = context_factory

    def send(self, message: EmailMessage) -> None:
        settings = self._settings
        context = self._context_factory()
        client: Any = None
        try:
            if settings.port == _IMPLICIT_TLS_PORT:
                client = self._smtp_ssl(
                    settings.host, settings.port, timeout=settings.timeout, context=context
                )
                client.ehlo()
            else:
                client = self._smtp(settings.host, settings.port, timeout=settings.timeout)
                client.ehlo()
                if not client.has_extn("starttls"):
                    # Nothing has been authenticated or sent yet: refuse here.
                    raise MailDeliveryError(
                        "mail server does not offer STARTTLS; refusing to send without TLS",
                        permanent=True,
                    )
                client.starttls(context=context)
                client.ehlo()
            if settings.username is not None and settings.password is not None:
                client.login(settings.username, settings.password)
            client.send_message(message)
        except MailDeliveryError:
            raise
        except smtplib.SMTPAuthenticationError as exc:
            raise MailDeliveryError(
                f"mail server refused the login (SMTP {exc.smtp_code})", permanent=True
            ) from None
        except smtplib.SMTPResponseException as exc:
            code = exc.smtp_code
            raise MailDeliveryError(
                f"mail server answered {type(exc).__name__} (SMTP {code})",
                permanent=isinstance(code, int) and code >= 500,
            ) from None
        except smtplib.SMTPRecipientsRefused:
            raise MailDeliveryError("mail server refused the recipient", permanent=True) from None
        except (smtplib.SMTPException, OSError) as exc:
            # ssl.SSLError, TimeoutError and ConnectionError are OSErrors.
            raise MailDeliveryError(
                f"mail delivery failed ({type(exc).__name__})", permanent=False
            ) from None
        finally:
            if client is not None:
                # Closing a connection that is already dead is best effort.
                with contextlib.suppress(Exception):
                    client.quit()
