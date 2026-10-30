"""SMTP settings and the sender, against fake smtplib classes. No socket is opened
(the root conftest would refuse one), no mail is sent.
"""

from __future__ import annotations

import base64
import logging
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any

import pytest
from jfl_core.crypto.envelope import MasterKey
from jfl_worker.mailer import (
    MailConfigError,
    MailDeliveryError,
    MailSettings,
    SmtpSender,
    normalise_base_url,
)
from jfl_worker.settings import WorkerSettings

PASSWORD = "hunter2-very-secret-smtp-key"
SECRET = "link-secret-" + "x" * 30

ENV = {
    "JFL_DATABASE_URL": "postgresql+psycopg://unused/unused",
    "JFL_SMTP_HOST": "smtp-relay.brevo.com",
    "JFL_SMTP_USERNAME": "user@example.invalid",
    "JFL_SMTP_PASSWORD": PASSWORD,
    "JFL_MAIL_FROM": "jobs4life <digest@example.invalid>",
    "JFL_PUBLIC_BASE_URL": "https://app.example.invalid/",
    "JFL_EMAIL_LINK_SECRET": SECRET,
}


def settings(**overrides: Any) -> MailSettings:
    env = {k: v for k, v in ENV.items() if k != "JFL_DATABASE_URL"}
    env.update(overrides)
    result = MailSettings.from_env(env)
    assert result is not None
    return result


def message() -> EmailMessage:
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = "a@example.invalid", "b@example.invalid", "hi"
    msg.set_content("body")
    return msg


class FakeSmtp:
    """Records the conversation; `script` can make a step raise."""

    instances: list[FakeSmtp] = []
    fail_at: dict[str, Exception] = {}
    offers_starttls = True

    def __init__(self, host: str, port: int, timeout: float | None = None, **kw: Any) -> None:
        self.host, self.port, self.timeout, self.kw = host, port, timeout, kw
        self.calls: list[str] = []
        self.sent: list[EmailMessage] = []
        FakeSmtp.instances.append(self)

    def _step(self, name: str) -> None:
        self.calls.append(name)
        if name in FakeSmtp.fail_at:
            raise FakeSmtp.fail_at[name]

    def ehlo(self) -> None:
        self._step("ehlo")

    def has_extn(self, name: str) -> bool:
        return FakeSmtp.offers_starttls

    def starttls(self, context: ssl.SSLContext | None = None) -> None:
        self._step("starttls")
        assert context is not None and context.verify_mode == ssl.CERT_REQUIRED

    def login(self, user: str, password: str) -> None:
        self._step("login")

    def send_message(self, msg: EmailMessage) -> None:
        self._step("send")
        self.sent.append(msg)

    def quit(self) -> None:
        self.calls.append("quit")


@pytest.fixture(autouse=True)
def _reset() -> None:
    FakeSmtp.instances = []
    FakeSmtp.fail_at = {}
    FakeSmtp.offers_starttls = True


def sender(cfg: MailSettings) -> SmtpSender:
    return SmtpSender(cfg, smtp=FakeSmtp, smtp_ssl=FakeSmtp)


# -- settings ---------------------------------------------------------------------


def test_unset_host_means_mail_is_off() -> None:
    assert MailSettings.from_env({}) is None
    assert MailSettings.from_env({"JFL_SMTP_PASSWORD": "x"}) is None


def test_a_host_demands_the_rest_and_names_them_without_values() -> None:
    with pytest.raises(MailConfigError) as exc:
        MailSettings.from_env({"JFL_SMTP_HOST": "smtp.example.invalid"})
    text = str(exc.value)
    for name in ("JFL_MAIL_FROM", "JFL_PUBLIC_BASE_URL", "JFL_EMAIL_LINK_SECRET"):
        assert name in text


def test_username_and_password_come_as_a_pair() -> None:
    env = {k: v for k, v in ENV.items() if k != "JFL_SMTP_PASSWORD"}
    with pytest.raises(MailConfigError, match="JFL_SMTP_PASSWORD"):
        MailSettings.from_env(env)


def test_defaults_and_normalisation() -> None:
    cfg = settings()
    assert cfg.port == 587
    assert cfg.public_base_url == "https://app.example.invalid"
    assert cfg.schedule.hour == 8 and cfg.schedule.timezone == "Europe/London"


@pytest.mark.parametrize("bad", ["http://app.example.invalid", "app.example.invalid", "ftp://x.y"])
def test_a_base_url_must_be_https(bad: str) -> None:
    with pytest.raises(MailConfigError):
        normalise_base_url(bad)


def test_http_is_allowed_for_localhost_only() -> None:
    assert normalise_base_url("http://localhost:8000/") == "http://localhost:8000"


def test_secrets_are_not_in_any_repr() -> None:
    cfg = settings()
    assert PASSWORD not in repr(cfg) and SECRET not in repr(cfg)
    env = {
        **ENV,
        "JFL_MASTER_KEY": base64.b64encode(MasterKey.generate().material).decode(),
    }
    worker = WorkerSettings.from_env(env)
    assert worker.mail is not None
    assert PASSWORD not in repr(worker) and SECRET not in repr(worker)


# -- the sender -------------------------------------------------------------------


def test_starttls_then_login_then_send_in_that_order() -> None:
    sender(settings()).send(message())
    (smtp,) = FakeSmtp.instances
    assert smtp.calls == ["ehlo", "starttls", "ehlo", "login", "send", "quit"]
    assert smtp.timeout is not None and smtp.port == 587


def test_port_465_uses_implicit_tls_with_a_verifying_context() -> None:
    sender(settings(JFL_SMTP_PORT="465")).send(message())
    (smtp,) = FakeSmtp.instances
    assert "starttls" not in smtp.calls
    assert smtp.kw["context"].verify_mode == ssl.CERT_REQUIRED
    assert smtp.kw["context"].check_hostname is True


def test_a_server_without_starttls_is_refused_before_any_credential_is_sent() -> None:
    FakeSmtp.offers_starttls = False
    with pytest.raises(MailDeliveryError, match="TLS") as exc:
        sender(settings()).send(message())
    (smtp,) = FakeSmtp.instances
    assert "login" not in smtp.calls and "send" not in smtp.calls
    assert exc.value.permanent is True


def test_auth_failure_is_permanent_and_carries_no_secret() -> None:
    FakeSmtp.fail_at["login"] = smtplib.SMTPAuthenticationError(
        535, f"bad credentials {PASSWORD}".encode()
    )
    with pytest.raises(MailDeliveryError) as exc:
        sender(settings()).send(message())
    assert exc.value.permanent is True
    assert PASSWORD not in str(exc.value) and "535" in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


def test_recipient_refusal_leaks_no_address() -> None:
    FakeSmtp.fail_at["send"] = smtplib.SMTPRecipientsRefused(
        {"someone@secret.invalid": (550, b"x")}
    )
    with pytest.raises(MailDeliveryError) as exc:
        sender(settings()).send(message())
    assert "secret.invalid" not in str(exc.value) and exc.value.permanent is True


def test_4xx_and_network_errors_are_retryable() -> None:
    FakeSmtp.fail_at["send"] = smtplib.SMTPDataError(451, b"try later")
    with pytest.raises(MailDeliveryError) as exc:
        sender(settings()).send(message())
    assert exc.value.permanent is False

    FakeSmtp.fail_at = {"starttls": TimeoutError("connect timed out")}
    with pytest.raises(MailDeliveryError) as exc2:
        sender(settings()).send(message())
    assert exc2.value.permanent is False and "TimeoutError" in str(exc2.value)


def test_the_connection_is_closed_even_when_sending_fails() -> None:
    FakeSmtp.fail_at["send"] = smtplib.SMTPDataError(451, b"x")
    with pytest.raises(MailDeliveryError):
        sender(settings()).send(message())
    assert FakeSmtp.instances[0].calls[-1] == "quit"


def test_the_password_never_reaches_the_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    sender(settings()).send(message())
    assert PASSWORD not in caplog.text
