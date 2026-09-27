"""The one classifier for model-API failures, against real SDK exception objects.

Built with `httpx2.Response` and no network, the way the SDK itself builds them
from a response, so a change in the SDK's exception surface (`status_code`,
`message`, `body`) fails here rather than silently reclassifying a production
failure. The case that motivated all of it: Anthropic reports an exhausted
credit balance as an ordinary HTTP 400 `invalid_request_error`, which prefix
matching filed as "bad_request" -- a bug in this code -- and failed the user's
work permanently.
"""

from __future__ import annotations

from typing import Any, get_args

import anthropic
import httpx2
import pytest
from jfl_core import models
from jfl_core.model_api import (
    ACCOUNT_BLOCKS,
    ApiFailure,
    ModelCallError,
    account_block_of,
    classify_api_error,
    parked_note,
)

_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

CREDIT_BODY = {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": (
            "Your credit balance is too low to access the Anthropic API. "
            "Please go to Plans & Billing to upgrade or purchase credits."
        ),
    },
}


def _status_error(cls: Any, status: int, body: dict[str, Any] | None = None) -> Any:
    message = f"Error code: {status} - {body}" if body else f"Error code: {status}"
    return cls(message, response=httpx2.Response(status, request=_REQUEST, json=body), body=body)


def test_an_exhausted_credit_balance_is_account_level_not_a_bad_request() -> None:
    exc = _status_error(anthropic.BadRequestError, 400, CREDIT_BODY)
    failure = classify_api_error(exc)
    assert failure.kind == "credits_exhausted"
    assert failure.account_block == "credits_exhausted"
    assert not failure.transient
    assert failure.label == "credits_exhausted"


def test_the_credit_match_ignores_case_and_reads_the_body_alone() -> None:
    """Matched on the body's `error.message` even when the exception's own
    message says nothing useful, and case-insensitively."""
    body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "CREDIT BALANCE low"},
    }
    exc = anthropic.BadRequestError(
        "400", response=httpx2.Response(400, request=_REQUEST, json=body), body=body
    )
    assert classify_api_error(exc).kind == "credits_exhausted"


def test_an_ordinary_bad_request_stays_request_level() -> None:
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens"}}
    failure = classify_api_error(_status_error(anthropic.BadRequestError, 400, body))
    assert failure.kind == "bad_request"
    assert failure.account_block is None
    assert not failure.transient
    assert failure.label == "bad_request"


@pytest.mark.parametrize(
    ("cls", "status", "kind", "label"),
    [
        (anthropic.AuthenticationError, 401, "invalid_key", "authentication_error"),
        (anthropic.PermissionDeniedError, 403, "permission_denied", "permission_denied"),
    ],
)
def test_a_rejected_or_unauthorised_key_is_account_level(
    cls: Any, status: int, kind: str, label: str
) -> None:
    failure = classify_api_error(_status_error(cls, status))
    assert failure.kind == kind
    assert failure.account_block == kind
    assert failure.label == label


@pytest.mark.parametrize(
    ("cls", "status", "kind"),
    [
        (anthropic.RateLimitError, 429, "rate_limited"),
        (anthropic.OverloadedError, 529, "overloaded"),
        (anthropic.InternalServerError, 500, "server_error"),
        (anthropic.ServiceUnavailableError, 503, "server_error"),
    ],
)
def test_rate_limits_and_overloads_are_transient(cls: Any, status: int, kind: str) -> None:
    failure = classify_api_error(_status_error(cls, status))
    assert failure.kind == kind
    assert failure.transient
    assert failure.account_block is None


def test_the_legacy_labels_are_kept_for_the_prefixes_handlers_still_read() -> None:
    assert classify_api_error(_status_error(anthropic.RateLimitError, 429)).label == "rate_limited"
    assert classify_api_error(_status_error(anthropic.OverloadedError, 529)).label == (
        "api_status_529"
    )
    assert classify_api_error(_status_error(anthropic.NotFoundError, 404)).label == "not_found"


@pytest.mark.parametrize(
    "exc",
    [anthropic.APIConnectionError(request=_REQUEST), anthropic.APITimeoutError(request=_REQUEST)],
)
def test_connection_failures_and_timeouts_are_transient(exc: Exception) -> None:
    failure = classify_api_error(exc)
    assert failure.kind == "connection"
    assert failure.transient
    assert failure.label == "connection_error"


def test_an_unfamiliar_4xx_is_request_level() -> None:
    failure = classify_api_error(_status_error(anthropic.UnprocessableEntityError, 422))
    assert failure.kind == "other"
    assert not failure.transient
    assert failure.account_block is None


def test_the_failure_carries_no_message() -> None:
    """Nothing the SDK said is kept: SDK text came back from a call made with
    the user's key, and the category is all anyone downstream needs."""
    failure = classify_api_error(_status_error(anthropic.BadRequestError, 400, CREDIT_BODY))
    assert set(ApiFailure.__slots__) == {"kind", "status_code"}
    assert "credit" not in repr(failure).replace("credits_exhausted", "")


def test_the_category_travels_on_the_error_and_is_read_back() -> None:
    failure = ApiFailure("invalid_key", 401)
    assert account_block_of(ModelCallError("authentication_error: x", api_failure=failure)) == (
        "invalid_key"
    )
    assert account_block_of(ModelCallError("model refused to respond")) is None
    assert account_block_of(RuntimeError("anything")) is None


def test_the_blocks_are_exactly_the_non_ok_key_health_statuses() -> None:
    """The banner reads `api_key_health.status`; the worker writes an
    `AccountBlock` into it. One list, two names -- they must not drift."""
    assert set(get_args(models.ApiKeyHealthStatus)) == {"ok", *ACCOUNT_BLOCKS}


def test_the_park_note_is_a_literal_and_a_category() -> None:
    assert parked_note("credits_exhausted") == "parked: credits_exhausted"
