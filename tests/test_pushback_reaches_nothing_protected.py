"""Pushback cannot touch the protected stores, and this is why -- not a promise.

Six things are never changeable by a disagreement with a score: the claim
gate's verdicts and anything downstream of them, whether a fact is in the
corpus, the per-requirement coverage statuses, the golden set and eval labels,
the "unmeasured" label, and the score an application was already sent under.

`jfl_core.pushback` refuses to name any of them, which
`packages/core/tests/test_pushback_rule.py` covers. This file covers the other
half: the code on the pushback path does not *import* anything that could reach
them, so there is no call to forget to not make.

The one deliberate exception is the web route's use of
`PostgresUserCorpusRepository`, and it is the exception that proves the rule: a
capability claim opens a question, and the user's verbatim answer goes into
their confirmed facts by the **one existing write path**, never a second one.
That is the flywheel working, not pushback reaching into the corpus -- the
score still does not move until the job is scored again. Because the route can
reach that repository, this file checks instead that it reaches nothing else
and that it writes no score.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Modules that hold, or can write, something a pushback may never change.
PROTECTED = (
    "jfl_gate",  # the claim gate's verdicts, and everything downstream
    "jfl_evals",  # the golden set, the eval labels, the harness
    "jfl_core.storage.postgres",  # grounding, spans, requirement coverage
    "jfl_core.corpus_source",  # whether a fact is in the corpus
    "jfl_core.storage.scores",  # the score an application was sent under
)

# The pure rule and the log. Neither may reach any protected module at all.
SEALED = (
    "packages/core/src/jfl_core/pushback.py",
    "packages/core/src/jfl_core/storage/pushbacks.py",
)

ROUTES = "packages/web/src/jfl_web/routes/pushbacks.py"

# The route reads a score to quote the exact number and sentence the user was
# arguing with, which is the stimulus the record is supposed to carry. These
# are the only two methods it may call on that repository: both read.
ALLOWED_SCORE_METHODS = frozenset({"get", "latest"})


def _imports(path: str) -> set[str]:
    tree = ast.parse((ROOT / path).read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


@pytest.mark.parametrize("path", SEALED)
@pytest.mark.parametrize("protected", PROTECTED)
def test_the_rule_and_the_log_import_nothing_protected(path: str, protected: str) -> None:
    offenders = [name for name in _imports(path) if name.startswith(protected)]
    assert not offenders, (
        f"{path} imports {offenders}, which can reach something a pushback may never "
        "change. The list is in this file's docstring."
    )


@pytest.mark.parametrize(
    "protected",
    [p for p in PROTECTED if p not in ("jfl_core.corpus_source", "jfl_core.storage.scores")],
)
def test_the_routes_import_nothing_protected(protected: str) -> None:
    offenders = [name for name in _imports(ROUTES) if name.startswith(protected)]
    assert not offenders, f"{ROUTES} imports {offenders}"


def test_the_routes_only_ever_read_a_score() -> None:
    """The stimulus is read; the score is never written.

    Checked on the syntax rather than at runtime, because the failure it guards
    against is a line of code that looks reasonable -- `scores.mark_done(...)`
    to "apply" a correction -- and which would make the over-claim metric a
    measure of the user's mood.
    """
    tree = ast.parse((ROOT / ROUTES).read_text())
    called: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "scores"
        ):
            called.add(node.func.attr)
    assert called <= ALLOWED_SCORE_METHODS, (
        f"the pushback routes call {sorted(called - ALLOWED_SCORE_METHODS)} on the score "
        "repository; a pushback may read the number it is arguing with and never write it"
    )


def test_no_pushback_module_writes_the_profile() -> None:
    """The profile is not a place corrections accumulate.

    The pushback log *is* the store: a dimension's displacement is a sum over
    it. Nothing here saves a profile, so "the profile drifted towards whatever
    was comfortable" is not a state this feature can produce.
    """
    for path in (*SEALED, ROUTES):
        source = (ROOT / path).read_text()
        assert "save_profile" not in source, f"{path} writes the profile"
        assert "ProfileRepo" not in source, f"{path} reaches the profile store"
