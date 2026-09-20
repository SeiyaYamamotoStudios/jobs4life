"""The `extract_cv_facts` handler's decisions that need no database.

The handler as a whole runs against real Postgres in
`tests/test_cv_facts_worker_integration.py`. What is worth testing here is the
classifier -- which failures are worth retrying and which are not -- for the
same reason `test_extraction_handler.py` tests its own: it is coupled to
another package's wording, and that coupling should be visible.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from jfl_generate.prompts import build_cv_facts_prompt
from jfl_worker.handlers.cv_facts import _classify, _sent_document_id
from jfl_worker.registry import PermanentTaskError


class TestClassify:
    @pytest.mark.parametrize(
        ("message", "code"),
        [
            ("no text found in the CV", "no_cv_text"),
            ("authentication_error: invalid x-api-key", "api_key_rejected"),
            ("permission_denied: no access to this model", "api_key_rejected"),
            ("model output was truncated at max_tokens (16000)", "cv_too_long"),
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


def test_the_classifier_still_matches_the_messages_cv_facts_actually_raises() -> None:
    """The coupling, made visible -- see `test_extraction_handler.py`'s sibling
    test for the identical rationale. A wording change in `cv_facts.py` should
    fail here rather than degrade quietly into "retried three times".
    """
    import inspect

    from jfl_generate import cv_facts

    source = inspect.getsource(cv_facts)
    for prefix in (
        "no text found in the CV",
        "authentication_error",
        "permission_denied",
        "bad_request",
        "model output was truncated",
        "model refused to respond",
    ):
        assert prefix in source, f"cv_facts.py no longer says {prefix!r}"

    # And the prompt this hangs off still builds, which is the cheapest smoke
    # test that the generate package is importable from here.
    assert build_cv_facts_prompt(now=datetime.now(UTC))


class TestSentDocumentId:
    def test_a_well_formed_payload_is_read(self) -> None:
        wanted = uuid.uuid4()
        assert _sent_document_id({"sent_document_id": str(wanted)}) == wanted

    @pytest.mark.parametrize(
        "payload", [{}, {"sent_document_id": None}, {"sent_document_id": 7}, {"other": "thing"}]
    )
    def test_a_payload_without_one_is_permanently_failed(self, payload: dict[str, object]) -> None:
        with pytest.raises(PermanentTaskError):
            _sent_document_id(payload)

    def test_a_malformed_id_never_appears_in_the_error(self) -> None:
        """`last_error` is read back in admin queries and quoted into logs."""
        with pytest.raises(PermanentTaskError) as raised:
            _sent_document_id({"sent_document_id": "not-a-uuid-and-maybe-a-secret"})
        assert "not-a-uuid-and-maybe-a-secret" not in str(raised.value)
