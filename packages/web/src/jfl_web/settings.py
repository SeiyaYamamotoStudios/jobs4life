"""The web app's single environment boundary.

Same rule as `jfl_core.context.RequestContext.from_env`: environment is read once,
here, at construction, and everything below takes what it needs as an argument.
No module in `jfl_web` outside this one touches `os.environ`.

Missing configuration fails at startup and names every variable it is missing at
once -- a server that boots and then 500s on the first login is worse than one
that refuses to boot.

Nothing in this module may render a secret. `WebSettings` keeps the client
secret, the cookie-signing secret and the master key out of its `repr`, because a
settings object is exactly the sort of thing that ends up in a log line or an
error page.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from jfl_core.crypto.envelope import MasterKey, MasterKeyError

# Exactly these three, and never a Gmail scope: restricted scopes require an
# annual CASA security assessment, and email intake is a forwarding address.
GOOGLE_SCOPES = "openid email profile"

_DEFAULT_SESSION_TTL_HOURS = 14 * 24
_TRUE = {"1", "true", "yes", "on"}


class ConfigError(RuntimeError):
    """Configuration is missing or malformed. Never carries a value, only names."""


@dataclass(frozen=True, slots=True)
class WebSettings:
    database_url: str
    google_client_id: str
    google_client_secret: str = field(repr=False)
    google_redirect_uri: str
    # Signs the short-lived cookie holding the OAuth state and PKCE verifier.
    # Not the session: sessions are opaque ids in Postgres.
    oauth_state_secret: str = field(repr=False)
    master_key: MasterKey = field(repr=False)
    session_ttl: dt.timedelta
    # How stale `last_seen_at` may get before a request rolls the expiry forward.
    # Not every request writes: that would make one row per user hot for no gain.
    session_touch_after: dt.timedelta
    # `__Host-` cookies require Secure, which requires HTTPS. Local development
    # over http://localhost cannot use them, so this drops the prefix and the
    # Secure flag -- and says so loudly at startup. Never set in deployment.
    insecure_cookies: bool
    # One free Models-API call when a key is entered, so a typo fails at the form.
    validate_api_keys: bool

    @property
    def session_cookie_name(self) -> str:
        return "jfl_session_dev" if self.insecure_cookies else "__Host-jfl_session"

    @property
    def oauth_cookie_name(self) -> str:
        return "jfl_oauth_dev" if self.insecure_cookies else "__Host-jfl_oauth"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> WebSettings:
        """The ONLY place `jfl_web` reads the environment."""
        source: Mapping[str, str] = os.environ if env is None else env
        missing: list[str] = []

        def required(name: str) -> str:
            value = source.get(name, "").strip()
            if not value:
                missing.append(name)
            return value

        database_url = required("JFL_DATABASE_URL")
        client_id = required("JFL_GOOGLE_CLIENT_ID")
        client_secret = required("JFL_GOOGLE_CLIENT_SECRET")
        redirect_uri = required("JFL_GOOGLE_REDIRECT_URI")
        state_secret = required("JFL_OAUTH_STATE_SECRET")

        master_key: MasterKey | None = None
        try:
            master_key = MasterKey.from_env(source)
        except MasterKeyError as exc:
            # Report it alongside the others rather than first; an operator
            # setting up should see the whole list in one go.
            if "is not set" in str(exc):
                missing.append("JFL_MASTER_KEY")
            else:
                raise ConfigError(str(exc)) from None

        if missing:
            raise ConfigError(
                "missing required environment variables: "
                + ", ".join(sorted(set(missing)))
                + ". See .env.example."
            )
        assert master_key is not None  # unreachable: absence was collected above

        return cls(
            database_url=database_url,
            google_client_id=client_id,
            google_client_secret=client_secret,
            google_redirect_uri=redirect_uri,
            oauth_state_secret=state_secret,
            master_key=master_key,
            session_ttl=dt.timedelta(
                hours=_positive_int(source, "JFL_SESSION_TTL_HOURS", _DEFAULT_SESSION_TTL_HOURS)
            ),
            session_touch_after=dt.timedelta(minutes=5),
            insecure_cookies=_flag(source, "JFL_INSECURE_COOKIES", default=False),
            validate_api_keys=_flag(source, "JFL_VALIDATE_API_KEYS", default=True),
        )


def _flag(source: Mapping[str, str], name: str, *, default: bool) -> bool:
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUE


def _positive_int(source: Mapping[str, str], name: str, default: int) -> int:
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer") from None
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero")
    return value
