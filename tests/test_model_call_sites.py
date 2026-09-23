"""Every Anthropic Messages call in `packages/*/src`, checked structurally.

Written for the move of the product model to Claude Opus 5.5, whose request
surface differs from Opus 5's in ways that fail loudly (a 400) or quietly (a
lower default effort):

- thinking cannot be disabled: `thinking: {type: "disabled"}` and
  `{type: "enabled", budget_tokens}` are both a 400 at every effort level;
- forced `tool_choice` (`any` / `tool`) is a 400;
- effort defaults to `medium`, one level below Opus 5's `high`, so a call that
  leaves it unset reasons less than it did the day before, silently;
- no assistant prefill (as on Opus 5);
- broader safety classifiers, so every call must check for a refusal before it
  reads the content.

This parses the source rather than exercising each function, so a call site
added later is covered without anyone remembering to add it here -- the scan
finds it. The per-module tests with fake clients check behaviour; this checks
the request shape of all of them at once.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

_PACKAGES = Path(__file__).resolve().parents[1] / "packages"
_HAIKU = "claude-haiku-4-5"

# The four deliberately cheap calls. They stay on Haiku whatever the product
# model is, and Haiku 4.5 rejects `effort`, so they must not send it.
_HAIKU_MODULES = {
    "jfl_generate/titles.py",
    "jfl_generate/capabilities.py",
    "jfl_generate/profile_suggestions.py",
    "jfl_generate/pushback.py",
}

# The product-model calls, which must pin effort.
_PRODUCT_MODULES = {
    "jfl_generate/extract.py",
    "jfl_generate/coverage.py",
    "jfl_generate/cv_facts.py",
    "jfl_generate/answers.py",
    "jfl_generate/scoring.py",
    "jfl_generate/draft.py",
    "jfl_generate/cv_document.py",
}

_GATE_MODULE = "jfl_gate/gate.py"


@dataclass(frozen=True)
class CallSite:
    module: str  # e.g. "jfl_generate/extract.py"
    call: ast.Call
    function: ast.FunctionDef
    tree: ast.Module

    @property
    def where(self) -> str:
        return f"{self.module}:{self.call.lineno}"

    def keyword(self, name: str) -> ast.expr | None:
        return next((k.value for k in self.call.keywords if k.arg == name), None)


def _is_messages_call(node: ast.Call) -> bool:
    """`<anything>.messages.create(...)` / `.stream(...)` / `.parse(...)` /
    `.count_tokens(...)` -- the shapes that send a Messages request."""
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in {"create", "stream", "parse", "count_tokens"}
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "messages"
    )


def _call_sites() -> list[CallSite]:
    sites: list[CallSite] = []
    for path in sorted(_PACKAGES.glob("*/src/**/*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        module = str(path.relative_to(path.parents[1]))  # jfl_x/module.py
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and _is_messages_call(node):
                    sites.append(CallSite(module, node, fn, tree))
    # A call inside a nested function is found from both the outer and inner
    # def; keep the innermost (the last one walked for that node).
    by_node: dict[int, CallSite] = {}
    for site in sites:
        by_node[id(site.call)] = site
    return list(by_node.values())


_SITES = _call_sites()


def _module_constant(tree: ast.Module, name: str) -> object:
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if value is not None and any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(value)
    return None


def _dict_keys(node: ast.expr | None) -> set[str]:
    if not isinstance(node, ast.Dict):
        return set()
    return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def test_the_scan_finds_every_known_call_site() -> None:
    # If this shrinks, the scan has stopped seeing calls and every other test
    # here passes vacuously.
    found = {s.module for s in _SITES}
    assert found == _HAIKU_MODULES | _PRODUCT_MODULES | {_GATE_MODULE}, found


@pytest.mark.parametrize("site", _SITES, ids=lambda s: s.where)
def test_no_call_sends_thinking_config(site: CallSite) -> None:
    # Opus 5.5 400s on `{type: "disabled"}` and on `budget_tokens` at every
    # effort level. Omitting `thinking` runs adaptive, which is what every
    # call here wants; nothing needs to name it at all.
    assert site.keyword("thinking") is None, f"{site.where} sends `thinking`"
    source = ast.unparse(site.call)
    assert "budget_tokens" not in source, f"{site.where} sends `budget_tokens`"


@pytest.mark.parametrize("site", _SITES, ids=lambda s: s.where)
def test_no_call_forces_tool_choice(site: CallSite) -> None:
    # Forced `any` / `tool` is a 400 on Opus 5.5. Structured output goes
    # through `output_config.format`, which is unaffected.
    assert site.keyword("tool_choice") is None, f"{site.where} sends `tool_choice`"


@pytest.mark.parametrize("site", _SITES, ids=lambda s: s.where)
def test_no_call_prefills_an_assistant_turn(site: CallSite) -> None:
    messages = site.keyword("messages")
    assert isinstance(messages, ast.List), f"{site.where}: messages is not a literal list"
    for element in messages.elts:
        assert isinstance(element, ast.Dict)
        roles = [
            v.value
            for k, v in zip(element.keys, element.values, strict=True)
            if isinstance(k, ast.Constant) and k.value == "role" and isinstance(v, ast.Constant)
        ]
        assert roles == ["user"], f"{site.where} sends a {roles} turn"


@pytest.mark.parametrize("site", _SITES, ids=lambda s: s.where)
def test_every_call_uses_structured_output(site: CallSite) -> None:
    assert "format" in _dict_keys(site.keyword("output_config")), site.where


@pytest.mark.parametrize(
    "site", [s for s in _SITES if s.module not in _HAIKU_MODULES], ids=lambda s: s.where
)
def test_every_opus_call_pins_effort(site: CallSite) -> None:
    # Opus 5 ran at `high` by default; Opus 5.5 would drop to `medium`. Pinned
    # so the model switch changes the price, not the depth of reasoning.
    assert "effort" in _dict_keys(site.keyword("output_config")), (
        f"{site.where} leaves effort to the model default"
    )


@pytest.mark.parametrize(
    "site", [s for s in _SITES if s.module in _HAIKU_MODULES], ids=lambda s: s.where
)
def test_the_cheap_calls_stay_on_haiku_and_send_no_effort(site: CallSite) -> None:
    model = site.keyword("model")
    assert isinstance(model, ast.Name), site.where
    assert _module_constant(site.tree, model.id) == _HAIKU, site.where
    # Haiku 4.5 rejects `effort`.
    assert "effort" not in _dict_keys(site.keyword("output_config")), site.where


@pytest.mark.parametrize(
    "site", [s for s in _SITES if s.module in _PRODUCT_MODULES], ids=lambda s: s.where
)
def test_product_calls_use_the_configured_product_model(site: CallSite) -> None:
    # `ctx.model`, or a `model` parameter a caller fills from `ctx.model`
    # (jfl_generate.answers) -- never a hard-coded id, never the gate's model.
    model = ast.unparse(site.keyword("model") or ast.Constant(None))
    assert model in {"ctx.model", "model"}, f"{site.where} uses {model}"


def test_the_gate_uses_its_own_model() -> None:
    (site,) = [s for s in _SITES if s.module == _GATE_MODULE]
    assert ast.unparse(site.keyword("model") or ast.Constant(None)) == "ctx.gate_model"


def _first_line(fn: ast.FunctionDef, predicate: object) -> int | None:
    lines = [n.lineno for n in ast.walk(fn) if predicate(n)]  # type: ignore[operator]
    return min(lines) if lines else None


def _is_refusal_check(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Attribute)
        and node.left.attr == "stop_reason"
        and any(isinstance(c, ast.Constant) and c.value == "refusal" for c in node.comparators)
    )


def _reads_content(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "content"


@pytest.mark.parametrize("site", _SITES, ids=lambda s: s.where)
def test_every_call_checks_for_a_refusal_before_reading_content(site: CallSite) -> None:
    refusal = _first_line(site.function, _is_refusal_check)
    content = _first_line(site.function, _reads_content)
    assert refusal is not None, f"{site.where} never checks stop_reason == 'refusal'"
    assert content is not None and refusal < content, (
        f"{site.where} reads response.content before checking for a refusal"
    )


@pytest.mark.parametrize("site", _SITES, ids=lambda s: s.where)
def test_every_refusal_records_its_category(site: CallSite) -> None:
    # The category (`reasoning_extraction`, `cyber`, `bio`, ...) is what tells
    # a classifier false positive from anything else -- CLAUDE.md, 2026-09-02.
    reads_category = any(
        isinstance(n, ast.Attribute) and n.attr == "category" for n in ast.walk(site.function)
    )
    assert reads_category, f"{site.where} does not record the refusal category"
