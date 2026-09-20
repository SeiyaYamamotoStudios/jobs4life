"""The `score_application` handler's decisions that need no database.

The handler as a whole is exercised against real Postgres in
`tests/test_scoring_worker_integration.py`. What is worth testing here is the
classifier -- which failures are worth retrying and which are not -- because
that decision spends the user's money when it is wrong, and because it is
coupled to two other modules' wording and that coupling should be visible.
"""

from __future__ import annotations

import uuid

import pytest
from jfl_worker.handlers.scoring import _classify, _score_id
from jfl_worker.registry import PermanentTaskError


class TestClassify:
    @pytest.mark.parametrize(
        ("message", "code"),
        [
            ("authentication_error: invalid x-api-key", "api_key_rejected"),
            ("permission_denied: no access to this model", "api_key_rejected"),
            ("model refused to respond: reasoning_extraction", "model_refused"),
            ("bad_request: schema is invalid", "model_error"),
            ("model output was truncated at max_tokens (4096)", "model_error"),
            ("job has no requirements to score against", "no_requirements"),
            ("job has no requirements to check", "no_requirements"),
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
            "expected 4 coverage results, got 3",
            "something nobody has seen before",
        ],
    )
    def test_transient_and_unrecognised_failures_are_retried(self, message: str) -> None:
        """The default is retryable on purpose, same as extraction's: a wording
        change upstream degrades this to "retried once too often", which costs
        a little money, rather than to "gave up on a 529", which loses the work.
        """
        assert _classify(message) == ("model_error", False)


def test_the_classifier_still_matches_the_messages_the_generate_calls_raise() -> None:
    """The coupling, made visible.

    `_classify` matches on the prefixes `jfl_generate.scoring` and
    `jfl_generate.jobs` build their `GenerateError` messages from. That is a
    real dependency on other modules' wording, and this test is where it is
    stated rather than discovered.
    """
    import inspect

    from jfl_generate import jobs, scoring

    scoring_source = inspect.getsource(scoring)
    for prefix in (
        "authentication_error",
        "permission_denied",
        "bad_request",
        "model refused to respond",
        "model output was truncated",
        "job has no requirements to score against",
    ):
        assert prefix in scoring_source, f"scoring.py no longer says {prefix!r}"

    assert "job has no requirements to check" in inspect.getsource(jobs)


class TestScoreId:
    def test_a_well_formed_payload_is_read(self) -> None:
        wanted = uuid.uuid4()
        assert _score_id({"score_id": str(wanted)}) == wanted

    @pytest.mark.parametrize(
        "payload", [{}, {"score_id": None}, {"score_id": 7}, {"other": "thing"}]
    )
    def test_a_payload_without_one_is_permanently_failed(self, payload: dict[str, object]) -> None:
        with pytest.raises(PermanentTaskError):
            _score_id(payload)

    def test_a_malformed_id_never_appears_in_the_error(self) -> None:
        """`last_error` is read back in admin queries and quoted into logs, so
        nothing from a payload is formatted into it.
        """
        with pytest.raises(PermanentTaskError) as raised:
            _score_id({"score_id": "not-a-uuid-and-maybe-a-secret"})
        assert "not-a-uuid-and-maybe-a-secret" not in str(raised.value)
