"""Bulk-archive rules -- owner feedback, 2026-09-27: "Archiving needs to allow
for bulk archiving, and maybe even something like - 'archive all scoring < 6/10,
archive all where you have been rejected, etc'".

Pure functions over data the caller already loaded (applications plus their row
scores) -- no SQL, no model call, matching the rest of this package. The
routes in `jfl_web.routes.applications` own fetching the data and writing the
archive; this module owns only "does this application match these rules".

**A score rule never touches an unscored application.** "Not yet scored" and
"scored low" are different facts, and treating the first as the second would
be exactly the kind of claim this whole project exists to catch -- a rule
asserting a low score where nothing was measured. So the score rule only
matches a *finished* run (`RowScore.state == "done"`), and only an axis that
actually came back with a number: an empty profile can finish a run with
`want_it_score` still None (nothing to measure "do I want this" against), and
that silence must not read as a low score either. "Both axes below N" needs
both numbers present -- one axis silent means the claim "both are low" cannot
be made.

**Never a composite.** CLAUDE.md's standing rule: "do I want this" and "could
I get this" are reported separately and never averaged. So the axis is one of
four *explicit* choices -- `want`, `could`, `either`, `both` -- never an
"average below N" option, and there must never be one.

**Rules combine with AND, and an empty rule set matches nothing.** Requiring
at least one active condition means a form submitted with everything
unchecked cannot be read as "archive everything" by omission.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Literal, get_args

from jfl_core.models import Application, ApplicationStatus

from jfl_web.scores import RowScore

ScoreAxis = Literal["want", "could", "either", "both"]

SCORE_AXES: tuple[ScoreAxis, ...] = get_args(ScoreAxis)

AXIS_LABELS: dict[ScoreAxis, str] = {
    "want": '"Do I want this" is below',
    "could": '"Could I get this" is below',
    "either": "either axis is below",
    "both": "both axes are below",
}

# The two statuses that are outcomes rather than pipeline stages -- see
# `jfl_web.routes.applications`'s `_PIPELINE` and `next_status`, which already
# treat `rejected` and `withdrawn` as the ends of the line rather than steps
# along it. Hardcoded rather than derived: the set is small, named in the
# owner's own words ("archive all where you have been rejected"), and
# `tests/test_archive_rules.py` cross-checks it against `_PIPELINE` so the two
# cannot silently drift apart.
TERMINAL_STATUSES: tuple[ApplicationStatus, ...] = ("rejected", "withdrawn")

MIN_SCORE = 1
MAX_SCORE = 10
MIN_DAYS = 1
MAX_DAYS = 3650  # ten years -- long enough that the ceiling is never the point

RULE_ERROR_MESSAGES: dict[str, str] = {
    "no_rule": "Choose at least one rule -- an empty form would archive nothing, on purpose.",
    "bad_score": f"Choose a score axis and a number from {MIN_SCORE} to {MAX_SCORE}.",
    "bad_activity": f"Enter a number of days between {MIN_DAYS} and {MAX_DAYS}.",
}


@dataclass(frozen=True, slots=True)
class ScoreRule:
    axis: ScoreAxis
    threshold: int  # matches when the chosen axis/axes read below this number


@dataclass(frozen=True, slots=True)
class ArchiveRules:
    """One optional condition per kind of rule, ANDed together where more than
    one is set. `is_empty` is what stops a blank form from matching everything.
    """

    score: ScoreRule | None = None
    statuses: frozenset[ApplicationStatus] = field(default_factory=frozenset)
    inactive_days: int | None = None

    @property
    def is_empty(self) -> bool:
        return self.score is None and not self.statuses and self.inactive_days is None


def _score_matches(rule: ScoreRule, row: RowScore | None) -> bool:
    """Only a finished run with a number for the chosen axis can match --
    scoring in flight, a failed run and an application never scored at all
    are all "not yet scored", never "scored low"."""
    if row is None or row.state != "done" or not row.has_numbers:
        return False
    want, could, threshold = row.want_it, row.could_get, rule.threshold
    if rule.axis == "want":
        return want is not None and want < threshold
    if rule.axis == "could":
        return could is not None and could < threshold
    if rule.axis == "either":
        return (want is not None and want < threshold) or (could is not None and could < threshold)
    # "both": either axis silent means "both are low" is not a claim the data
    # supports.
    return want is not None and could is not None and want < threshold and could < threshold


def matches(
    application: Application,
    row_score: RowScore | None,
    rules: ArchiveRules,
    *,
    now: dt.datetime,
) -> bool:
    """Whether this application satisfies every active condition in `rules`.
    An empty rule set matches nothing -- callers should check `is_empty`
    first and not enter this function's loop at all, but this stays safe
    either way."""
    if rules.is_empty:
        return False
    if rules.score is not None and not _score_matches(rules.score, row_score):
        return False
    if rules.statuses and application.status not in rules.statuses:
        return False
    if rules.inactive_days is None:
        return True
    return (now - application.updated_at) >= dt.timedelta(days=rules.inactive_days)


def matching_applications(
    applications: list[Application],
    row_scores: dict[uuid.UUID, RowScore],
    rules: ArchiveRules,
    *,
    now: dt.datetime,
) -> list[Application]:
    """Every application matching every active rule. Order is not guaranteed
    beyond what `applications` was already ordered by -- callers pass in the
    list's own order (most recently updated first)."""
    if rules.is_empty:
        return []
    return [a for a in applications if matches(a, row_scores.get(a.id), rules, now=now)]


def _parse_int(raw: str, *, minimum: int, maximum: int) -> int | None:
    try:
        value = int(raw.strip())
    except (TypeError, ValueError, AttributeError):
        return None
    return value if minimum <= value <= maximum else None


def parse_rule_form(
    *,
    score_enabled: bool,
    score_axis: str,
    score_threshold: str,
    statuses: list[str] | None,
    activity_enabled: bool,
    activity_days: str,
) -> tuple[ArchiveRules | None, str | None]:
    """The rule form's raw fields -> `ArchiveRules`, or `None` plus an error
    code from `RULE_ERROR_MESSAGES`. Shared by the preview and confirm routes
    so the two can never disagree about what a submission means.

    Anything outside the closed sets (`SCORE_AXES`, `TERMINAL_STATUSES`) is
    dropped silently rather than rejected -- the same discipline
    `change_status` applies to `to_status`: a tampered or stale value cannot
    smuggle in behaviour this form never offered.
    """
    score_rule: ScoreRule | None = None
    if score_enabled:
        axis = score_axis if score_axis in SCORE_AXES else None
        threshold = _parse_int(score_threshold, minimum=MIN_SCORE, maximum=MAX_SCORE)
        if axis is None or threshold is None:
            return None, "bad_score"
        score_rule = ScoreRule(axis=axis, threshold=threshold)

    chosen_statuses = frozenset(
        status for status in (statuses or []) if status in TERMINAL_STATUSES
    )

    inactive_days: int | None = None
    if activity_enabled:
        days = _parse_int(activity_days, minimum=MIN_DAYS, maximum=MAX_DAYS)
        if days is None:
            return None, "bad_activity"
        inactive_days = days

    rules = ArchiveRules(score=score_rule, statuses=chosen_statuses, inactive_days=inactive_days)
    if rules.is_empty:
        return None, "no_rule"
    return rules, None
