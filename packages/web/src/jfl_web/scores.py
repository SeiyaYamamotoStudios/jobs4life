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

from dataclasses import dataclass

from jfl_core.fit import want_it_basis
from jfl_core.models import ApplicationScore, FitVerdict, ScoreErrorCode

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
