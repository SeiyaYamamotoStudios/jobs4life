"""The one place slice A is allowed to touch the Anthropic API.

`validate_api_key` makes a single `GET /v1/models` call so that a mistyped key
fails at the settings form rather than three screens later. That call costs
nothing -- no tokens in, no tokens out -- but it still constructs a real client,
so it lives behind both halves of the repository guard: the `e2e` marker *and*
`JFL_ALLOW_REAL_API=1`.

    JFL_ALLOW_REAL_API=1 uv run pytest -m e2e

Everything else in slice A makes no model call at all. That is deliberate: the
whole shell can be built, deployed and used for nothing, and an auth bug can
never be mistaken for an engine bug.
"""

from __future__ import annotations

import os

import pytest
from jfl_web.credentials import InvalidApiKeyError, validate_api_key

pytestmark = pytest.mark.e2e


@pytest.fixture
def real_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY is not set")
    return key


def test_a_real_key_validates(real_key: str) -> None:
    validate_api_key(real_key)  # no exception is the assertion


def test_a_bad_key_is_rejected_at_the_form() -> None:
    with pytest.raises(InvalidApiKeyError) as exc:
        validate_api_key("sk-ant-api03-definitely-not-a-real-key-0000000000")
    # The message says the key was rejected and does not quote the key.
    assert "sk-ant" not in str(exc.value)
