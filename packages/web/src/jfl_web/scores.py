"""What the scoring panel says -- wording, and nothing else. PLAN.md B4.

The failure messages live here rather than in the worker for the same reason
the database stores a code and not a sentence: the worker writes that column
while holding the user's decrypted API key, and a closed set of codes cannot
carry a secret. Wording is a UI concern, changeable without a migration.

`UNMEASURED` is the one line on this page that is not decoration. There is no
golden set for fit and inventing one would be the synthetic-data prohibition in
a new coat, so the two scores ship labelled unmeasured, in those words, on
screen -- and the measured number this project publishes, the over-claim rate,
is a different number about a different thing and must not be confused with
these.

Nothing here composites the two axes. There is no "overall", no average, and
no ordering of applications by a combined number, because a role the user would
love and will not get and one they would dislike and would walk into must never
land on the same number.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, get_args
from urllib.parse import urlencode

from jfl_core.fit import want_it_basis
from jfl_core.models import (
    Application,
    ApplicationScore,
    ApplicationStatus,
    FitVerdict,
    ScoreErrorCode,
)

UNMEASURED = (
    "These two scores are unmeasured. There is no golden set for fit, so unlike "
    "the claim gate's over-claim rate there is no accuracy figure behind them -- "
    "they are a model's judgement of your own words, shown with its reasoning so "
    "you can disagree with it."
)

COULD_GET_LABEL = "Could I get this"
WANT_IT_LABEL = "Do I want this"

# What the second number actually answers, said once, under it. It is not a
# prediction that you would enjoy the job -- see `docs/profile-schema.md`:
# computed person-job fit predicts satisfaction at rho ~= .28, and people
# forecast their own job satisfaction badly. What it measures is how much of
# what you said matters this ad actually evidences.
WANT_IT_SUBTITLE = (
    "How much of what you said matters this ad actually evidences -- not a prediction "
    "that you would enjoy it."
)

# An empty profile has no number, rather than a 1. Nothing was measured, so
# nothing is claimed.
NO_WANT_IT_SCORE = "No number: you have not recorded anything for this ad to be measured against."

# The four words a constraint or objective comes back as, and what each one
# means to the reader. `silent` is the one worth reading twice: it is not a
# failure of the ad and not a mark against the job -- it is the question to ask
# at interview, which is the most useful thing this panel produces.
VERDICT_WORDING: dict[FitVerdict, str] = {
    "evidenced": "the ad states it",
    "partial": "the ad points that way",
    "silent": "the ad does not say -- ask",
    "contradicted": "the ad states the opposite",
}

SILENCE_NOTE = (
    "Every silence below is a question to ask at interview, not a mark against the "
    "job. What predicts whether a job works out is expectations that turn out to be "
    "met, so these are the ones worth asking about before you accept."
)

STANCE_WORDING: dict[str, str] = {
    "must": "must have",
    "nice": "would like",
    "never": "will not accept",
}


def want_it_summary(score: ApplicationScore) -> str:
    """One sentence saying what the number was derived from.

    Recomputed from the stored verdicts rather than stored beside the number,
    so the tally under a score can never disagree with the verdicts listed
    under it. `jfl_core.fit` owns the arithmetic; this owns the wording.
    """
    basis = want_it_basis(score.constraint_verdicts, score.objective_verdicts)
    if basis.items == 0:
        return NO_WANT_IT_SCORE
    bits = [f"{basis.evidenced} of {basis.items} evidenced"]
    if basis.partial:
        bits.append(f"{basis.partial} partly")
    bits.append(f"{basis.silent} not mentioned")
    if basis.contradicted:
        bits.append(f"{basis.contradicted} contradicted")
    sentence = "Across what you said matters: " + ", ".join(bits) + "."
    if basis.capped:
        sentence += " A must-have or a never is broken, which holds the number down."
    return sentence


# The first score is chained from the read of the ad (see
# `jfl_worker.handlers.extraction._chain_first_score`), so while the ad is being
# read the panel says what is about to happen rather than offering a button.
AWAITS_AD_NOTE = (
    "Scoring starts by itself as soon as the ad has been read -- you chose this "
    "job by adding it, so there is nothing to press."
)

# A pending run that carries an error code: one attempt failed and the queue
# will try again. Not an error yet, and it must not read as one -- the row only
# becomes `failed` when the attempts run out.
RETRYING_NOTE = (
    "Scoring hit a temporary problem and is trying again by itself -- nothing "
    "for you to do. If it keeps failing it stops and says so here."
)

# What adding an application now sets off, said on the add form and beside the
# track button **before** anything is pressed. "Up to" because the coverage
# check is skipped when this exact ad's requirements were already checked.
ADD_COST_NOTE = (
    "Adding it reads the ad and then scores it straight away, on your own "
    "Anthropic API key: up to three model calls -- one to read the ad, one to "
    "check its requirements against your confirmed facts, one to score it. "
    "Nothing else runs unless you ask."
)
TRACK_COST_NOTE = "Reads and scores it on your API key: up to 3 model calls."

# No key stored: nothing is queued, and the reason is said where the user is.
NO_KEY_ADD_NOTE = (
    "You have no Anthropic API key stored, so the ad will be saved but not read "
    "or scored -- both are model calls billed to your own key."
)
NO_KEY_TRACK_NOTE = "No API key stored: it is saved, but not read or scored."


def track_context(has_api_key: bool) -> dict[str, object]:
    """What `_track_button.html` needs to say what pressing it sets off.
    Shared by every page that renders the button, so they cannot disagree.
    """
    return {
        "has_api_key": has_api_key,
        "track_cost_note": TRACK_COST_NOTE,
        "no_key_track_note": NO_KEY_TRACK_NOTE,
    }


# Shown beside the button, because pressing it spends the user's own money and
# may spend it twice.
COST_NOTE = (
    "Scoring calls the model on your own API key. If this job's requirements "
    "have not been checked against your confirmed facts yet, that check runs "
    "first -- two calls rather than one."
)


@dataclass(frozen=True, slots=True)
class ScoreFailure:
    """What to tell the user, and where to send them to fix it."""

    message: str
    fix_url: str | None = None
    fix_label: str | None = None


_FAILURES: dict[ScoreErrorCode, ScoreFailure] = {
    "no_api_key": ScoreFailure(
        "This needs your own Anthropic API key -- scoring is a model call, and it "
        "is billed to you rather than to this app.",
        fix_url="/settings",
        fix_label="Add an API key",
    ),
    "api_key_rejected": ScoreFailure(
        "Anthropic rejected the API key stored here. Replace it and try again.",
        fix_url="/settings",
        fix_label="Replace the key",
    ),
    "model_refused": ScoreFailure(
        "The model declined to score this one. Trying again is worth a go."
    ),
    "model_error": ScoreFailure("Scoring failed. Trying again is worth a go."),
    "credential_unreadable": ScoreFailure(
        "Your stored API key could not be unlocked on the server. Setting it again will fix it.",
        fix_url="/settings",
        fix_label="Set the key again",
    ),
    "no_requirements": ScoreFailure(
        "This job ad has not been read into requirements yet, so there is nothing "
        "to measure your record against. Read the ad first, then score it."
    ),
}

# Anything unrecognised -- a code added to the database before this table caught
# up -- still gets a sentence rather than a blank panel.
_UNKNOWN = ScoreFailure("Scoring failed. Trying again is worth a go.")


def score_failure(code: ScoreErrorCode | None) -> ScoreFailure:
    return _UNKNOWN if code is None else _FAILURES.get(code, _UNKNOWN)


# -- the applications list ----------------------------------------------------
#
# Both numbers on every row, in two cells under two labels. Nothing here adds,
# averages, weights or otherwise combines them, and the list can be sorted by
# either axis alone -- never by a blend of the two. A role the owner would love
# and will not get and one they would dislike and would walk into must never
# land on the same rung of one ranking.

# The pipeline's own order, so sorting by status reads interested -> offer,
# then the two outcomes.
_STATUS_ORDER: dict[str, int] = {s: i for i, s in enumerate(get_args(ApplicationStatus))}

RowScoreState = Literal["unscored", "scoring", "retrying", "failed", "done"]

# The columns the list can be sorted by -- every column header except the
# actions -- and the direction a first click on each one gives. Each key names
# exactly one column; for the two score columns that means exactly one axis.
# There is no "best match" key: ranking by a blend of the two scores would be
# the composite this product refuses to compute, hidden inside a sort.
SortKey = Literal["title", "employer", "status", "could_get", "want_it", "updated"]
SortDirection = Literal["asc", "desc"]

SORT_COLUMNS: dict[SortKey, str] = {
    "title": "Title",
    "employer": "Employer",
    "status": "Status",
    "could_get": COULD_GET_LABEL,
    "want_it": WANT_IT_LABEL,
    "updated": "Updated",
}

# Words read A to Z and the pipeline reads in its own order on a first click;
# scores and recency lead with the highest and the newest.
DEFAULT_DIRECTION: dict[SortKey, SortDirection] = {
    "title": "asc",
    "employer": "asc",
    "status": "asc",
    "could_get": "desc",
    "want_it": "desc",
    "updated": "desc",
}

DEFAULT_SORT: SortKey = "updated"


def parse_sort(raw_key: str | None, raw_direction: str | None) -> tuple[SortKey, SortDirection]:
    """The sort a query string asks for, from a closed set.

    An unknown key falls back to the default sort *and* its default direction
    -- a direction means nothing without the column it was chosen for. An
    unknown direction falls back to the chosen column's default.
    """
    key: SortKey = DEFAULT_SORT
    for candidate in SORT_COLUMNS:
        if raw_key == candidate:
            key = candidate
            break
    else:
        return DEFAULT_SORT, DEFAULT_DIRECTION[DEFAULT_SORT]
    direction: SortDirection = DEFAULT_DIRECTION[key]
    if raw_direction == "asc":
        direction = "asc"
    elif raw_direction == "desc":
        direction = "desc"
    return key, direction


def sort_query(key: SortKey, direction: SortDirection) -> dict[str, str]:
    """The query parameters that name this sort, and nothing redundant: the
    default sort is no parameters at all, a column's default direction is no
    `dir`. So the plain `/applications` URL is the default order, and a
    first-click link reads `?sort=could_get`."""
    if key == DEFAULT_SORT and direction == DEFAULT_DIRECTION[key]:
        return {}
    params: dict[str, str] = {"sort": key}
    if direction != DEFAULT_DIRECTION[key]:
        params["dir"] = direction
    return params


@dataclass(frozen=True, slots=True)
class SortHeader:
    """One sortable column header: where its link goes, and what it says about
    the current order. `aria_sort` is None on every header but the active one
    -- ARIA puts `aria-sort` on one header at a time."""

    key: SortKey
    label: str
    href: str
    active: bool
    direction: SortDirection | None
    aria_sort: Literal["ascending", "descending"] | None
    next_direction: SortDirection


def sort_headers(
    key: SortKey, direction: SortDirection, *, status: str | None
) -> dict[SortKey, SortHeader]:
    """A link per column. Clicking the active column reverses it; clicking any
    other starts that column at its default direction. The status filter rides
    along, so sorting never drops it."""
    headers: dict[SortKey, SortHeader] = {}
    for column, label in SORT_COLUMNS.items():
        active = column == key
        if active:
            next_direction: SortDirection = "asc" if direction == "desc" else "desc"
        else:
            next_direction = DEFAULT_DIRECTION[column]
        params: dict[str, str] = {"status": status} if status else {}
        params.update(sort_query(column, next_direction))
        headers[column] = SortHeader(
            key=column,
            label=label,
            href="/applications" + ("?" + urlencode(params) if params else ""),
            active=active,
            direction=direction if active else None,
            aria_sort=("ascending" if direction == "asc" else "descending") if active else None,
            next_direction=next_direction,
        )
    return headers


@dataclass(frozen=True, slots=True)
class RowScore:
    """What one row of the list says about scoring.

    `could_get` and `want_it` come from the latest *finished* run, so a
    re-score in flight does not blank the numbers the previous run gave --
    `state` says it is running. `want_it` may be None on a finished run: an
    empty profile has nothing for the ad to be measured against.
    """

    state: RowScoreState
    could_get: int | None = None
    want_it: int | None = None
    has_numbers: bool = False


def row_score(latest: ApplicationScore | None, latest_done: ApplicationScore | None) -> RowScore:
    if latest is None:
        return RowScore(state="unscored")
    state: RowScoreState
    if latest.status == "pending":
        state = "retrying" if latest.error_code else "scoring"
    elif latest.status == "failed":
        state = "failed"
    else:
        state = "done"
    if latest_done is None:
        return RowScore(state=state)
    return RowScore(
        state=state,
        could_get=latest_done.could_get_score,
        want_it=latest_done.want_it_score,
        has_numbers=True,
    )


def sort_applications(
    applications: Sequence[Application],
    rows: Mapping[uuid.UUID, RowScore],
    key: str,
    direction: SortDirection | None = None,
) -> list[Application]:
    """Order the list by **one** column.

    Ties -- and every row with nothing to sort on -- keep a neutral order:
    most recently updated first, in either direction. That is the only
    tie-break. For the two score columns it means the other axis is never
    consulted, not even to separate two equal numbers.

    Rows with nothing in the chosen column go last **in both directions**: an
    unscored application, or one with no employer recorded. "Not scored" is not
    a low score, and ranking it as one would be the list asserting something
    nothing measured. A key outside the closed set sorts as the default.
    """
    column: SortKey = parse_sort(key, None)[0]
    if direction is None:
        direction = DEFAULT_DIRECTION[column]

    # The neutral order first; Python's sort is stable, including with
    # `reverse=True`, so every later sort leaves equal rows in this order.
    base = sorted(applications, key=lambda a: a.updated_at, reverse=True)
    if column == "updated":
        return base if direction == "desc" else sorted(base, key=lambda a: a.updated_at)

    def value(application: Application) -> str | int | None:
        if column == "title":
            return application.title.casefold()
        if column == "employer":
            employer = (application.employer or "").strip()
            return employer.casefold() or None
        if column == "status":
            return _STATUS_ORDER.get(application.status, len(_STATUS_ORDER))
        row = rows.get(application.id)
        return None if row is None else getattr(row, column)

    valued = [(value(a), a) for a in base]
    present = [(v, a) for v, a in valued if v is not None]
    missing = [a for v, a in valued if v is None]
    present.sort(key=lambda pair: pair[0], reverse=direction == "desc")
    return [a for _, a in present] + missing
