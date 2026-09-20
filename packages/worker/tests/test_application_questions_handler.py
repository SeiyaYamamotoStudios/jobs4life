"""The `check_application_answer` / `draft_application_answer` handlers'
decisions that need no database -- NEXT.md's task 4.

The handlers as a whole are exercised against real Postgres in
`tests/test_application_questions_worker_integration.py`. What is worth
testing here is the classifier -- which failures are worth retrying and which
are not -- same reasoning as `test_extraction_handler.py`'s and
`test_title_suggestions_handler.py`'s.
"""

from __future__ import annotations

import uuid

import pytest
from jfl_worker.handlers.application_questions import _answer_id, _classify
from jfl_worker.registry import PermanentTaskError


class TestClassify:
    @pytest.mark.parametrize(
        ("message", "code"),
        [
            ("authentication_error: invalid x-api-key", "api_key_rejected"),
            ("permission_denied: no access to this model", "api_key_rejected"),
            ("model refused to respond: reasoning_extraction", "model_refused"),
            ("bad_request: schema is invalid", "model_error"),
            (
                "model output was truncated at max_tokens (2000)",
                "model_error",
            ),
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
            "no sentences found in the input text",
            "model output misaligned with input sentences: expected [1], got []",
            "something nobody has seen before",
        ],
    )
    def test_transient_and_unrecognised_failures_are_retried(self, message: str) -> None:
        assert _classify(message) == ("model_error", False)


def test_the_classifier_still_matches_the_messages_answers_and_gate_actually_raise() -> None:
    """The coupling, made visible -- see `test_extraction_handler.py`'s sibling
    test for the identical rationale. `jfl_generate.answers` and
    `jfl_gate.gate` build their error messages from these prefixes; if either
    changes its wording, this fails loudly instead of quietly degrading a
    permanent failure into "retried once too often".
    """
    import inspect

    from jfl_gate import gate as gate_module
    from jfl_generate import answers as answers_module

    answers_source = inspect.getsource(answers_module)
    for prefix in ("authentication_error", "permission_denied", "bad_request"):
        assert prefix in answers_source, f"answers.py no longer says {prefix!r}"

    gate_source = inspect.getsource(gate_module)
    for prefix in ("model refused to respond", "model output was truncated"):
        assert prefix in gate_source, f"gate.py no longer says {prefix!r}"


class TestAnswerId:
    def test_a_well_formed_payload_is_read(self) -> None:
        wanted = uuid.uuid4()
        assert _answer_id({"answer_id": str(wanted)}) == wanted

    @pytest.mark.parametrize(
        "payload", [{}, {"answer_id": None}, {"answer_id": 7}, {"other": "thing"}]
    )
    def test_a_payload_without_one_is_permanently_failed(self, payload: dict[str, object]) -> None:
        with pytest.raises(PermanentTaskError):
            _answer_id(payload)

    def test_a_malformed_id_never_appears_in_the_error(self) -> None:
        with pytest.raises(PermanentTaskError) as raised:
            _answer_id({"answer_id": "not-a-uuid-and-maybe-a-secret"})
        assert "not-a-uuid-and-maybe-a-secret" not in str(raised.value)
