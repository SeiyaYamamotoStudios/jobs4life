"""The `suggest_titles` handler's decisions that need no database.

The handler as a whole is exercised against real Postgres in
`tests/test_title_suggestions_integration.py`. What is worth testing here is
the classifier -- which failures are worth retrying and which are not -- same
reasoning as `test_extraction_handler.py`'s: it is coupled to another
package's wording, and that coupling should be visible.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from jfl_generate.prompts import build_title_suggestion_prompt
from jfl_worker.handlers.title_suggestions import _classify, _split_phrases, _suggestion_id
from jfl_worker.registry import PermanentTaskError


class TestClassify:
    @pytest.mark.parametrize(
        ("message", "code"),
        [
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


def test_the_classifier_still_matches_the_messages_suggest_titles_actually_raises() -> None:
    """The coupling, made visible -- see `test_extraction_handler.py`'s sibling
    test for the identical rationale.
    """
    import inspect

    from jfl_generate import titles

    source = inspect.getsource(titles)
    for prefix in (
        "authentication_error",
        "permission_denied",
        "bad_request",
        "model refused to respond",
    ):
        assert prefix in source, f"titles.py no longer says {prefix!r}"

    # And the prompt this all hangs off still builds, which is the cheapest
    # possible smoke test that the generate package is importable from here.
    assert build_title_suggestion_prompt(
        "engineering manager",
        other_includes=[],
        excludes=[],
        application_titles=[],
        now=datetime.now(UTC),
    )


class TestSuggestionId:
    def test_a_well_formed_payload_is_read(self) -> None:
        wanted = uuid.uuid4()
        assert _suggestion_id({"suggestion_id": str(wanted)}) == wanted

    @pytest.mark.parametrize(
        "payload", [{}, {"suggestion_id": None}, {"suggestion_id": 7}, {"other": "thing"}]
    )
    def test_a_payload_without_one_is_permanently_failed(self, payload: dict[str, object]) -> None:
        with pytest.raises(PermanentTaskError):
            _suggestion_id(payload)

    def test_a_malformed_id_never_appears_in_the_error(self) -> None:
        with pytest.raises(PermanentTaskError) as raised:
            _suggestion_id({"suggestion_id": "not-a-uuid-and-maybe-a-secret"})
        assert "not-a-uuid-and-maybe-a-secret" not in str(raised.value)


class TestSplitPhrases:
    def test_splits_trims_and_dedupes(self) -> None:
        assert _split_phrases("engineering manager, , engineering manager, head of eng ") == [
            "engineering manager",
            "head of eng",
        ]

    def test_empty_text_is_no_phrases(self) -> None:
        assert _split_phrases("") == []
