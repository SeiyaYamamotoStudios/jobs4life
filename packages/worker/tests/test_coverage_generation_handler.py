"""The `generate_coverage` handler's decisions that need no database.

The handler as a whole is exercised against real Postgres in
`tests/test_coverage_generation_worker_integration.py`. What is worth testing
here is the classifier -- which failures are worth retrying and which are
not -- because that decision spends the user's money when it is wrong, and
because it is coupled to another package's wording. Same reasoning as
`test_extraction_handler.py`'s sibling test.
"""

from __future__ import annotations

import uuid

import pytest
from jfl_worker.handlers.coverage_generation import _classify, _job_id, _permanent
from jfl_worker.registry import PermanentTaskError


class TestClassify:
    @pytest.mark.parametrize(
        ("message", "code"),
        [
            ("no job 11111111-1111-1111-1111-111111111111 for this user", "no_job"),
            ("job has no requirements to check", "no_requirements"),
            ("authentication_error: invalid x-api-key", "api_key_rejected"),
            ("permission_denied: no access to this model", "api_key_rejected"),
            ("model refused to respond: reasoning_extraction", "model_refused"),
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


def test_the_classifier_still_matches_the_messages_run_coverage_actually_raises() -> None:
    """The coupling, made visible -- see `test_extraction_handler.py`'s
    identical rationale.
    """
    import inspect

    from jfl_generate import coverage, jobs

    source = inspect.getsource(jobs) + inspect.getsource(coverage)
    for prefix in (
        "no job ",
        "job has no requirements to check",
        "authentication_error",
        "permission_denied",
        "bad_request",
        "model refused to respond",
    ):
        assert prefix in source, f"jfl_generate.jobs/coverage no longer says {prefix!r}"


class TestJobId:
    def test_a_well_formed_payload_is_read(self) -> None:
        wanted = uuid.uuid4()
        assert _job_id({"job_id": str(wanted)}) == wanted

    @pytest.mark.parametrize("payload", [{}, {"job_id": None}, {"job_id": 7}, {"other": "thing"}])
    def test_a_payload_without_one_is_permanently_failed(self, payload: dict[str, object]) -> None:
        with pytest.raises(PermanentTaskError):
            _job_id(payload)

    def test_a_malformed_id_never_appears_in_the_error(self) -> None:
        with pytest.raises(PermanentTaskError) as raised:
            _job_id({"job_id": "not-a-uuid-and-maybe-a-secret"})
        assert "not-a-uuid-and-maybe-a-secret" not in str(raised.value)


def test_permanent_builds_the_one_message_shape_the_web_layer_parses() -> None:
    error = _permanent("no_api_key")
    assert str(error) == "coverage generation failed permanently: no_api_key"
