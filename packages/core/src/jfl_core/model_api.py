"""What went wrong with a model call, in one vocabulary every package shares.

Every model call site in `jfl_gate` and `jfl_generate` used to turn an SDK
exception into a string (`"bad_request: ..."`) and every worker handler then
matched prefixes of that string. That was enough to tell "retry" from "give up",
and not enough to tell the one failure a user most needs told about: **their
Anthropic account has stopped accepting calls** -- the credit balance ran out,
the key was revoked, the organisation lost access. Anthropic reports running out
of credits as an ordinary HTTP 400 `invalid_request_error`, so under prefix
matching it was a "bad_request", classified as a bug in this code, and the
user's work failed permanently with a generic error.

So there is one classifier, here, and the call sites attach its answer to the
exception they raise (`ModelCallError.api_failure`) rather than encoding it in
a message for someone downstream to parse.

**Why here, and why duck-typed.** `jfl_core` is the one package everything
imports -- the web layer (which renders the categories), the worker (which acts
on them), gate and generate (which produce them) -- and it does not depend on
the `anthropic` SDK. Rather than add that dependency to the base of the stack
for one function, the classifier reads the SDK's stable public surface by
attribute: `status_code` on an `APIStatusError`, `message` and `body` for the
text. Anything without a `status_code` is a connection failure, because the
call sites only ever hand this `APIStatusError` or `APIConnectionError`. The
tests construct real SDK exception objects, so a change in that surface fails
there rather than here in production.

**Nothing in this module stores SDK text.** `ApiFailure` holds a category and a
status code. The categories are what reach the database (`api_key_health`,
`tasks.last_error`); the SDK's message never does, because it is text from a
call authenticated with the user's key and this project does not get to assume
what such text can echo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

# Everything the classifier can say. Three groups, and the group is what callers
# act on -- see `ApiFailure.account_block` and `ApiFailure.transient`.
ApiFailureKind = Literal[
    # Account-level: nothing will succeed until the user does something.
    "credits_exhausted",
    "invalid_key",
    "permission_denied",
    # Transient: the retry ladder is the right answer.
    "rate_limited",
    "overloaded",
    "server_error",
    "connection",
    # Request-level: this request will fail the same way again.
    "bad_request",
    "not_found",
    "other",
]

# The account-level subset. Also the non-`ok` values of `ApiKeyHealthStatus`
# in `jfl_core.models`, which is what the banner reads.
AccountBlock = Literal["credits_exhausted", "invalid_key", "permission_denied"]
ACCOUNT_BLOCKS: tuple[AccountBlock, ...] = get_args(AccountBlock)

_TRANSIENT: frozenset[ApiFailureKind] = frozenset(
    {"rate_limited", "overloaded", "server_error", "connection"}
)

# How Anthropic words an exhausted balance: HTTP 400, `invalid_request_error`,
# "Your credit balance is too low to access the Anthropic API...". Matched
# case-insensitively and on this fragment alone, so a rewording of the rest of
# the sentence still lands.
_CREDIT_BALANCE = "credit balance"

# The prefix of `tasks.last_error` on a task the worker parked because the
# user's account is blocked. It is how the web layer counts and resumes parked
# work (`PostgresTaskRepository.count_parked` / `resume_parked`) without the web
# knowing which task kinds call a model. Followed by the `AccountBlock`.
PARKED_NOTE_PREFIX = "parked: "


def parked_note(block: AccountBlock) -> str:
    """The `tasks.last_error` of a parked task. A literal and a category --
    never an SDK message."""
    return f"{PARKED_NOTE_PREFIX}{block}"


@dataclass(frozen=True, slots=True)
class ApiFailure:
    """One classified model-call failure. No message, on purpose."""

    kind: ApiFailureKind
    status_code: int | None = None

    @property
    def account_block(self) -> AccountBlock | None:
        """The account-level category, or None for anything a retry or a
        code fix could change."""
        if self.kind in ACCOUNT_BLOCKS:
            return self.kind
        return None

    @property
    def transient(self) -> bool:
        return self.kind in _TRANSIENT

    @property
    def label(self) -> str:
        """The prefix a call site puts on its error text (the `runs.error`
        column and the raised exception's message).

        The pre-existing prefixes are kept exactly -- `rate_limited`,
        `authentication_error`, `permission_denied`, `not_found`,
        `bad_request`, `api_status_<code>`, `connection_error` -- because the
        handlers' own non-account classification tables and the `runs` history
        are written in them. `credits_exhausted` is new: it used to be
        indistinguishable from `bad_request`, which was the bug.
        """
        match self.kind:
            case "credits_exhausted":
                return "credits_exhausted"
            case "invalid_key":
                return "authentication_error"
            case "permission_denied":
                return "permission_denied"
            case "rate_limited":
                return "rate_limited"
            case "bad_request":
                return "bad_request"
            case "not_found":
                return "not_found"
            case "connection":
                return "connection_error"
            case _:
                return f"api_status_{self.status_code}"


def _text_of(exc: BaseException) -> str:
    """The SDK's message plus its parsed body's `error.message`, for matching
    only -- never stored."""
    parts = [str(getattr(exc, "message", "") or exc)]
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            parts.append(str(error.get("message", "")))
        parts.append(str(body.get("message", "")))
    return " ".join(parts).lower()


def classify_api_error(exc: BaseException) -> ApiFailure:
    """Classify an `anthropic.APIStatusError` or `anthropic.APIConnectionError`.

    The one place this decision is made. See the module docstring for why it
    reads attributes rather than importing the SDK's classes.
    """
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        # APIConnectionError and its APITimeoutError subclass carry no status.
        return ApiFailure("connection")
    if status == 400:
        if _CREDIT_BALANCE in _text_of(exc):
            return ApiFailure("credits_exhausted", status)
        return ApiFailure("bad_request", status)
    if status == 401:
        return ApiFailure("invalid_key", status)
    if status == 403:
        return ApiFailure("permission_denied", status)
    if status == 404:
        return ApiFailure("not_found", status)
    if status == 429:
        return ApiFailure("rate_limited", status)
    if status == 529:
        return ApiFailure("overloaded", status)
    if status >= 500:
        return ApiFailure("server_error", status)
    return ApiFailure("other", status)


class ModelCallError(RuntimeError):
    """Base of `jfl_gate.gate.GateError` and `jfl_generate.errors.GenerateError`.

    `api_failure` is set when the failure was the model API refusing the call,
    and None for everything else those errors cover (a refusal, a truncated
    answer, a precondition). It is how a worker handler asks "is this the
    user's account?" without parsing the message.
    """

    def __init__(self, message: object, *, api_failure: ApiFailure | None = None) -> None:
        super().__init__(message)
        self.api_failure = api_failure


def account_block_of(exc: BaseException) -> AccountBlock | None:
    """The account-level category carried by `exc`, if any."""
    failure = getattr(exc, "api_failure", None)
    if isinstance(failure, ApiFailure):
        return failure.account_block
    return None
