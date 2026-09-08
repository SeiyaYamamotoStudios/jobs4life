"""The `extract_job_ad` handler's decisions that need no database.

The handler as a whole is exercised against real Postgres in
`tests/test_extraction_integration.py`. What is worth testing here is the
classifier -- which failures are worth retrying and which are not -- because
that decision spends the user's money when it is wrong, and because it is
coupled to another package's wording and that coupling should be visible.
"""

from __future__ import annotations

import uuid

import pytest
from jfl_generate.extract import MAX_TOKENS
from jfl_generate.prompts import build_extract_prompt
from jfl_worker.handlers.extraction import _application_id, _classify
from jfl_worker.registry import PermanentTaskError


class TestClassify:
    @pytest.mark.parametrize(
        ("message", "code"),
        [
            ("no text found in the job ad", "no_job_ad"),
            ("authentication_error: invalid x-api-key", "api_key_rejected"),
            ("permission_denied: no access to this model", "api_key_rejected"),
            (
                f"model output was truncated at max_tokens ({MAX_TOKENS}); "
                "the document is too long for one call",
                "ad_too_long",
            ),
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
        """The default is retryable on purpose.

        A wording change in `jfl_generate.extract` degrades this to "retried
        once too often", which costs a little money; the opposite default would
        degrade to "gave up on a 529", which loses the work.
        """
        assert _classify(message) == ("model_error", False)


def test_the_classifier_still_matches_the_messages_extract_actually_raises() -> None:
    """The coupling, made visible.

    `_classify` matches on the prefixes `jfl_generate.extract` builds its
    `GenerateError` messages from. That is a real dependency on another
    package's wording, and this test is where it is stated rather than
    discovered. If it fails, the fix is either to update `_PERMANENT_FAILURES`
    or -- better -- to give `GenerateError` a typed kind so nothing has to match
    on prose.
    """
    import inspect

    from jfl_generate import extract

    source = inspect.getsource(extract)
    for prefix in (
        "no text found in the job ad",
        "authentication_error",
        "permission_denied",
        "bad_request",
        "model refused to respond",
        "model output was truncated",
    ):
        assert prefix in source, f"extract.py no longer says {prefix!r}"

    # And the prompt this all hangs off still builds, which is the cheapest
    # possible smoke test that the generate package is importable from here.
    assert build_extract_prompt()


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
        """`last_error` is read back in admin queries and quoted into logs, so
        nothing from a payload is formatted into it.
        """
        with pytest.raises(PermanentTaskError) as raised:
            _application_id({"application_id": "not-a-uuid-and-maybe-a-secret"})
        assert "not-a-uuid-and-maybe-a-secret" not in str(raised.value)
