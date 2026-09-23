"""The `generate_cv_draft` handler's decisions that need no database.

The handler as a whole is exercised against real Postgres in
`tests/test_draft_generation_worker_integration.py`. What is worth testing
here is the classifier -- which failures are worth retrying and which are
not -- and the payload parsing, same reasoning as `test_extraction_handler.py`
and `test_coverage_generation_handler.py`.
"""

from __future__ import annotations

import uuid

import pytest
from jfl_core.profile import Profile
from jfl_worker.handlers.draft_generation import (
    _application_id,
    _classify,
    _kind,
    _permanent,
    header_name,
)
from jfl_worker.registry import PermanentTaskError


class TestClassify:
    @pytest.mark.parametrize(
        ("message", "code"),
        [
            ("no job 11111111-1111-1111-1111-111111111111 for this user", "no_job"),
            ("job has no requirements to draft against", "no_requirements"),
            (
                "no coverage recorded for job 11111111-1111-1111-1111-111111111111 -- "
                "run `jfl job coverage ...` first",
                "no_coverage",
            ),
            ("authentication_error: invalid x-api-key", "api_key_rejected"),
            ("permission_denied: no access to this model", "api_key_rejected"),
            (
                "model output was truncated at max_tokens (8000); the document is too "
                "long for one call",
                "ad_too_long",
            ),
            ("model refused to respond: reasoning_extraction", "model_refused"),
            ("bad_request: schema is invalid", "model_error"),
            (
                "claim gate output does not line up with the CV's lines: expected 3, got 2",
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
            "something nobody has seen before",
        ],
    )
    def test_transient_and_unrecognised_failures_are_retried(self, message: str) -> None:
        assert _classify(message) == ("model_error", False)


def test_the_classifier_still_matches_the_messages_generate_draft_actually_raises() -> None:
    """The coupling, made visible -- both `jfl_generate.draft.generate_draft`
    and `jfl_gate.gate.check_text` (the automatic claim-gate pass) build error
    text off the same literal prefixes, which is what lets one classifier
    cover `GenerateError` and `GateError` alike.
    """
    import inspect

    from jfl_gate import gate
    from jfl_generate import draft

    source = inspect.getsource(draft) + inspect.getsource(gate)
    for prefix in (
        "no job ",
        "job has no requirements to draft against",
        "no coverage recorded for job",
        "authentication_error",
        "permission_denied",
        "bad_request",
        "model refused to respond",
        "model output was truncated",
    ):
        assert prefix in source, f"jfl_generate.draft/jfl_gate.gate no longer says {prefix!r}"


class TestApplicationId:
    def test_a_well_formed_payload_is_read(self) -> None:
        wanted = uuid.uuid4()
        assert _application_id({"application_id": str(wanted)}) == wanted

    @pytest.mark.parametrize(
        "payload", [{}, {"application_id": None}, {"application_id": 7}, {"other": "thing"}]
    )
    def test_a_payload_without_one_is_permanently_failed(self, payload: dict[str, object]) -> None:
        with pytest.raises(PermanentTaskError):
            _application_id(payload)

    def test_a_malformed_id_never_appears_in_the_error(self) -> None:
        with pytest.raises(PermanentTaskError) as raised:
            _application_id({"application_id": "not-a-uuid-and-maybe-a-secret"})
        assert "not-a-uuid-and-maybe-a-secret" not in str(raised.value)


class TestKind:
    def test_a_recognised_kind_is_read(self) -> None:
        assert _kind({"kind": "cv_bullets"}) == "cv_bullets"
        assert _kind({"kind": "cover_letter"}) == "cover_letter"
        # The complete CV -- what the web's "Write the CV" queues.
        assert _kind({"kind": "cv_document"}) == "cv_document"

    @pytest.mark.parametrize("payload", [{}, {"kind": None}, {"kind": "resume"}, {"kind": 7}])
    def test_anything_else_is_permanently_failed(self, payload: dict[str, object]) -> None:
        with pytest.raises(PermanentTaskError):
            _kind(payload)


def test_permanent_builds_the_one_message_shape_the_web_layer_parses() -> None:
    error = _permanent("no_coverage")
    assert str(error) == "draft generation failed permanently: no_coverage"


class TestHeaderName:
    def test_the_account_name_when_the_profile_names_no_one(self) -> None:
        assert header_name(Profile(), "Morgan Fictional") == "Morgan Fictional"

    def test_empty_when_neither_names_anyone(self) -> None:
        # jfl_generate.cv_document then falls back to the corpus title.
        assert header_name(Profile(), "") == ""

    def test_a_profile_contact_name_wins(self) -> None:
        class _Contact:
            name = "  Morgan F.  "

        class _WithContact:
            contact = _Contact()

        assert header_name(_WithContact(), "Account Name") == "Morgan F."  # type: ignore[arg-type]
