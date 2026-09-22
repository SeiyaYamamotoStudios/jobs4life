"""The `suggest_profile_settings` handler's decisions that need no database.

The handler as a whole runs against real Postgres in
`tests/test_profile_suggestions_worker_integration.py`. What is worth testing
here is the classifier -- which failures are worth retrying and which are not --
for the reason `test_capability_clusters_handler.py` gives: it is coupled to
another package's wording, and that coupling should be visible.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from jfl_generate.prompts import build_profile_suggestions_prompt
from jfl_worker.handlers.profile_suggestions import _classify, _run_id
from jfl_worker.registry import PermanentTaskError


class TestClassify:
    @pytest.mark.parametrize(
        ("message", "code"),
        [
            ("authentication_error: invalid x-api-key", "api_key_rejected"),
            ("permission_denied: no access to this model", "api_key_rejected"),
            ("model refused to respond: reasoning_extraction", "model_refused"),
            ("model output was truncated at max_tokens (4096)", "model_error"),
            ("bad_request: schema is invalid", "model_error"),
        ],
    )
    def test_failures_a_retry_cannot_fix_are_permanent(self, message: str, code: str) -> None:
        assert _classify(message) == (code, True)

    @pytest.mark.parametrize(
        "message",
        [
            "rate_limited: too many requests",
            "api_status_529: overloaded",
            "connection_error: connection reset",
            "could not parse structured output: expecting value",
            "something nobody has seen before",
        ],
    )
    def test_transient_and_unrecognised_failures_are_retried(self, message: str) -> None:
        assert _classify(message) == ("model_error", False)


def test_the_classifier_still_matches_what_the_call_actually_raises() -> None:
    """The coupling, made visible -- see `test_extraction_handler.py`'s sibling
    test for the identical rationale.
    """
    import inspect

    from jfl_generate import profile_suggestions

    source = inspect.getsource(profile_suggestions)
    for prefix in (
        "authentication_error",
        "permission_denied",
        "bad_request",
        "model refused to respond",
        "model output was truncated",
    ):
        assert prefix in source, f"profile_suggestions.py no longer says {prefix!r}"

    # And the prompt this hangs off still builds, which is the cheapest
    # possible smoke test that the generate package is importable from here.
    assert build_profile_suggestions_prompt(max_suggestions=30, now=datetime.now(UTC))


class TestRunId:
    def test_a_well_formed_payload_is_read(self) -> None:
        wanted = uuid.uuid4()
        assert _run_id({"suggestion_run_id": str(wanted)}) == wanted

    def test_a_payload_with_no_id_is_permanent(self) -> None:
        with pytest.raises(PermanentTaskError):
            _run_id({})

    def test_a_payload_whose_id_is_not_a_uuid_is_permanent(self) -> None:
        with pytest.raises(PermanentTaskError) as caught:
            _run_id({"suggestion_run_id": "not-a-uuid"})
        # `last_error` is read back in admin queries and quoted into logs, so
        # the message names neither the value nor the payload.
        assert "not-a-uuid" not in str(caught.value)
