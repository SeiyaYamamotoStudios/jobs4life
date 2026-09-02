"""Unit tests for `RequestContext.from_env`'s model selection.

`from_env` is CLAUDE.md's one sanctioned place to read the environment, so the
default and the `JFL_MODEL` override are both pinned here: every call site bills
against `ctx.model`, and a silent divergence between the dataclass default and
the `from_env` default would mean a call billed at the wrong model's rates.
"""

from __future__ import annotations

import uuid

import pytest
from jfl_core.context import RequestContext
from jfl_core.db.tables import LOCAL_USER_ID


@pytest.fixture(autouse=True)
def _required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # from_env requires JFL_DATABASE_URL; keep the other required pieces minimal
    # and let each test set only the JFL_MODEL / JFL_USER_ID variables it cares
    # about, exactly as CLAUDE.md's "from_env is the ONLY place environment is
    # touched" implies -- nothing but os.environ drives this.
    monkeypatch.setenv("JFL_DATABASE_URL", "postgresql://unused")
    monkeypatch.delenv("JFL_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("JFL_USER_ID", raising=False)
    monkeypatch.delenv("JFL_EMBEDDING_DEVICE", raising=False)


def test_model_defaults_to_claude_opus_5_when_jfl_model_is_unset() -> None:
    ctx = RequestContext.from_env()
    assert ctx.model == "claude-opus-5"


def test_model_is_read_from_jfl_model_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JFL_MODEL", "claude-sonnet-5")
    ctx = RequestContext.from_env()
    assert ctx.model == "claude-sonnet-5"


def test_request_context_dataclass_default_model_matches_from_env_default() -> None:
    # RequestContext(...) constructed directly (as every test fixture in this
    # repo does) must land on the same default from_env falls back to, or the
    # two paths would silently diverge on which model a call actually bills.
    ctx = RequestContext(
        user_id=uuid.uuid4(), anthropic_api_key=None, database_url="postgresql://unused"
    )
    assert ctx.model == "claude-opus-5"


def test_other_from_env_fields_are_unaffected_by_the_model_addition() -> None:
    ctx = RequestContext.from_env()
    assert ctx.user_id == LOCAL_USER_ID
    assert ctx.anthropic_api_key is None
    assert ctx.embedding_device == "cuda"
