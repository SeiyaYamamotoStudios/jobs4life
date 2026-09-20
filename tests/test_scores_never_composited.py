"""The two scores are never averaged, combined or ordered against each other,
and a test says so rather than a reviewer.

CLAUDE.md's standing decision, restated in PLAN.md B4: "do I want this" and
"could I get this" diverge constantly, and a role the owner would love and will
not get must never land on the same number as one he would dislike and would
walk into. The failure this file exists to prevent is the small, reasonable-
looking one -- a helper that returns `(a + b) / 2` for a sort order, or an
`overall` field added "just for the list view" -- because that is how the
product's own thesis gets undone by a convenience.

The rule is mechanical, which is what makes it testable:

  1. no module on the scoring path defines an identifier that names a
     composite (overall, combined, composite, average, aggregate, mean);
  2. no arithmetic anywhere on that path has both axes in it;
  3. `ApplicationScore` and the table carry exactly two score columns;
  4. the template renders no expression containing both.

No database and no model call: this reads source.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest
from jfl_core.db import tables
from jfl_core.models import ApplicationScore

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Every module the two numbers pass through, from the call to the page.
SCORE_PATH_MODULES = [
    ROOT / "packages/generate/src/jfl_generate/scoring.py",
    ROOT / "packages/generate/src/jfl_generate/prompts.py",
    ROOT / "packages/generate/src/jfl_generate/schema.py",
    ROOT / "packages/core/src/jfl_core/storage/scores.py",
    ROOT / "packages/worker/src/jfl_worker/handlers/scoring.py",
    ROOT / "packages/web/src/jfl_web/scores.py",
    ROOT / "packages/web/src/jfl_web/routes/applications.py",
]

SCORE_TEMPLATE = ROOT / "packages/web/src/jfl_web/templates/_score.html"

AXES = ("could_get_score", "want_it_score")

# Words a composite would be called. Matched against identifiers only -- prose
# in a docstring saying "never averaged" is the point, not a violation.
_COMPOSITE_WORDS = ("overall", "combined", "composite", "average", "aggregate", "mean")


def _defined_names(tree: ast.AST) -> set[str]:
    """Every name this module defines or assigns to, plus every attribute it
    reads. Enough to catch `overall = ...`, `def overall_score(...)`,
    `score.overall`, and a dict key is caught separately below.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg) or (isinstance(node, ast.keyword) and node.arg):
            names.add(node.arg)
        # Dict keys and template-context keys are strings; a context variable
        # named `overall_score` would arrive this way. Only single-token
        # strings are treated as identifiers, so prose is not caught here.
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and re.fullmatch(r"[a-z][a-z0-9_]*", node.value)
        ):
            names.add(node.value)
    return names


@pytest.mark.parametrize("path", SCORE_PATH_MODULES, ids=lambda p: p.name)
def test_no_module_on_the_score_path_names_a_composite(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text())
    offences = sorted(
        name
        for name in _defined_names(tree)
        if any(word in name.lower() for word in _COMPOSITE_WORDS)
    )
    assert not offences, (
        f"{path.name} names {offences}. The two axes are reported separately and "
        "never averaged, combined or reduced to one number -- see CLAUDE.md."
    )


@pytest.mark.parametrize("path", SCORE_PATH_MODULES, ids=lambda p: p.name)
def test_no_arithmetic_anywhere_has_both_axes_in_it(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp):
            continue
        inside = {
            child.id if isinstance(child, ast.Name) else child.attr
            for child in ast.walk(node)
            if isinstance(child, ast.Name | ast.Attribute)
        }
        assert not set(AXES) <= inside, (
            f"{path.name} does arithmetic on both axes at line {node.lineno}. "
            "Merging them corrupts both -- see CLAUDE.md's decisions log."
        )


def test_the_stored_row_carries_exactly_two_scores() -> None:
    model_scores = {name for name in ApplicationScore.model_fields if name.endswith("_score")}
    assert model_scores == set(AXES)

    column_scores = {c.name for c in tables.application_scores.columns if c.name.endswith("_score")}
    assert column_scores == set(AXES)


def test_the_panel_renders_no_expression_holding_both_axes() -> None:
    """A template is where an average is cheapest to write and hardest to
    notice: `{{ (a + b) / 2 }}` needs no Python at all.
    """
    for expression in re.findall(r"\{\{(.*?)\}\}", SCORE_TEMPLATE.read_text(), re.S):
        assert not all(axis in expression for axis in AXES), expression


def test_the_scheme_would_catch_a_violation(tmp_path: pathlib.Path) -> None:
    """The tests above are only worth having if they can fail. Prove they do."""
    offender = tmp_path / "offender.py"
    offender.write_text("def overall(a, b):\n    return (a + b) / 2\n")
    with pytest.raises(AssertionError, match="names"):
        test_no_module_on_the_score_path_names_a_composite(offender)

    offender.write_text("x = could_get_score + want_it_score\n")
    with pytest.raises(AssertionError, match="arithmetic on both axes"):
        test_no_arithmetic_anywhere_has_both_axes_in_it(offender)
