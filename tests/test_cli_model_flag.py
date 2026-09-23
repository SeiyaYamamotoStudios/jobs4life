"""The CLI's global --model flag.

Lives here rather than under `packages/cli/tests/` only because there is no such
directory yet; `pyproject.toml`'s testpaths covers both `tests` and `packages`.

`_callback` is called directly rather than through typer's CliRunner: every real
command opens Postgres, and the behaviour under test is one assignment at the CLI
boundary, not command dispatch. Calling it directly means the flag's default
(a typer.OptionInfo, not None) is never exercised, so both cases pass explicitly.
"""

from __future__ import annotations

import pytest
from jfl_cli.main import _callback
from jfl_core.context import RequestContext


@pytest.fixture(autouse=True)
def _no_gate_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JFL_GATE_MODEL", raising=False)


def test_model_flag_sets_the_env_var_from_env_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JFL_MODEL", raising=False)
    monkeypatch.setenv("JFL_DATABASE_URL", "postgresql://unused/unused")

    _callback(model="claude-sonnet-5", gate_model=None)

    assert RequestContext.from_env().model == "claude-sonnet-5"


def test_omitting_the_flag_leaves_an_existing_jfl_model_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JFL_MODEL", "claude-opus-5")
    monkeypatch.setenv("JFL_DATABASE_URL", "postgresql://unused/unused")

    _callback(model=None, gate_model=None)

    assert RequestContext.from_env().model == "claude-opus-5"


def test_model_flag_does_not_move_the_claim_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    # --model is the product model; the gate stays where the published number
    # was measured unless --gate-model says otherwise.
    monkeypatch.delenv("JFL_MODEL", raising=False)
    monkeypatch.setenv("JFL_DATABASE_URL", "postgresql://unused/unused")

    _callback(model="claude-opus-5-5", gate_model=None)

    ctx = RequestContext.from_env()
    assert ctx.model == "claude-opus-5-5"
    assert ctx.gate_model == "claude-opus-5"


def test_gate_model_flag_sets_the_gate_model_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JFL_MODEL", raising=False)
    monkeypatch.setenv("JFL_DATABASE_URL", "postgresql://unused/unused")

    _callback(model=None, gate_model="claude-opus-5-5")

    ctx = RequestContext.from_env()
    assert ctx.gate_model == "claude-opus-5-5"
    assert ctx.model == "claude-opus-5-5"  # the product default, untouched by the flag
