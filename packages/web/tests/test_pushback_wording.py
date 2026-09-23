"""The pushback card and drift sentence, as pure functions. No database.

What is pinned here: the drift threshold, the before -> after arithmetic the
card quotes, and that nothing the feature puts on screen -- constants, the
readings list, the three templates -- uses the machinery's own words. The
rendered-markup sweep over every card state lives in
`tests/test_pushback_web_integration.py`; this one is the cheap first line.
"""

from __future__ import annotations

import datetime as dt
import inspect
import re
import uuid

import pytest
from jfl_core.models import ApplicationScore, Pushback
from jfl_core.pushback import COULD_GET_OVERALL, WANT_OVERALL, DriftMeter
from jfl_web import pushbacks as wording
from jfl_web.pushbacks import (
    READINGS,
    card,
    drift_line,
    history_line,
    reading_of,
)
from jfl_web.templating import TEMPLATE_DIR

NOW = dt.datetime(2026, 9, 23, 12, 0, tzinfo=dt.UTC)

INTERNAL = (
    "δ",
    "shrinkage",
    "shrunk",
    "dimension",
    "capability",
    "preference",
    "classif",
    "observation",
    "displacement",
    "claim gate",
    "golden set",
    "corpus",
)


def _score(**kw: object) -> ApplicationScore:
    values: dict[str, object] = {
        "id": uuid.uuid4(),
        "application_id": uuid.uuid4(),
        "status": "done",
        "could_get_score": 4,
        "could_get_assessment": "",
        "want_it_score": 5,
        "want_it_assessment": "",
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(kw)
    return ApplicationScore.model_validate(values)


def _applied(
    *,
    kind: str = "preference",
    direction: str = "up",
    delta: float = 1.0,
    target: str = WANT_OVERALL,
    effect: dict[str, object] | None = None,
    disposition: str = "accepted",
    minute: int = 0,
) -> Pushback:
    at = NOW + dt.timedelta(minutes=minute)
    return Pushback(
        id=uuid.uuid4(),
        application_id=uuid.uuid4(),
        score_id=uuid.uuid4(),
        axis="get" if target == COULD_GET_OVERALL else "want",
        dimension=target,
        target_dimension=target,
        user_text="words",
        asserted_direction=direction,
        status="applied",
        classification=kind,
        applied_delta=delta,
        disposition=disposition,
        effect=effect or {"asserted": 1.0, "after_shrinkage": abs(delta)},
        created_at=at,
        updated_at=at,
        applied_at=at,
    )


# -- the drift threshold ---------------------------------------------------------


@pytest.mark.parametrize(
    ("total", "up", "down", "net", "loud"),
    [
        (1, 1, 0, 1.0, False),
        (2, 2, 0, 0.0, False),  # two is not yet a pattern
        (3, 3, 0, 0.0, True),  # three, all upward
        (4, 3, 1, 0.0, True),  # three of four is past two thirds
        (5, 3, 2, 0.0, False),  # three of five is not
        (6, 2, 4, -1.0, False),  # downward never trips it
        (2, 2, 0, 2.0, True),  # the net has reached the most any score can move
    ],
)
def test_the_drift_sentence_escalates_only_past_the_threshold(
    total: int, up: int, down: int, net: float, loud: bool
) -> None:
    line = drift_line(DriftMeter(total=total, upward=up, downward=down, net=net))
    assert line is not None
    assert line.loud is loud


def test_no_corrections_means_no_sentence() -> None:
    assert drift_line(DriftMeter(total=0, upward=0, downward=0, net=0.0)) is None


@pytest.mark.parametrize(
    ("total", "up", "down", "words"),
    [
        (1, 1, 0, "once on this profile, upward."),
        (2, 0, 2, "twice on this profile, all downward."),
        (3, 2, 1, "3 times on this profile, mostly upward."),
        (4, 2, 2, "4 times on this profile, as often up as down."),
    ],
)
def test_the_drift_sentence_reads_as_a_sentence(total: int, up: int, down: int, words: str) -> None:
    line = drift_line(DriftMeter(total=total, upward=up, downward=down, net=0.0))
    assert line is not None
    assert line.text == f"You've pushed back {words}"


# -- the card ----------------------------------------------------------------------


def test_the_first_preference_moves_a_whole_point_and_says_it_counts_everywhere() -> None:
    p = _applied()
    c = card(p, score=_score(), log=[p], sent=False)
    assert c.change is not None
    assert (c.change.label, c.change.before, c.change.after) == ("Do I want this", "5", "6")
    assert wording.EVERY_JOB in c.lines
    assert wording.SMALL_STEPS not in c.lines
    assert c.meaning == "You'd take roles like this more readily than we scored."


def test_a_shrunk_step_says_why_it_moved_only_a_little() -> None:
    first = _applied()
    second = _applied(delta=0.833, effect={"asserted": 1.0, "after_shrinkage": 0.833}, minute=1)
    c = card(second, score=_score(), log=[first, second], sent=False)
    assert c.change is not None
    assert (c.change.before, c.change.after, c.change.shown_as) == ("6", "6.8", 7)
    assert wording.SMALL_STEPS in c.lines


def test_a_stronger_fit_claim_stays_and_links_to_the_fact() -> None:
    p = _applied(
        kind="capability",
        delta=0.0,
        target=COULD_GET_OVERALL,
        disposition="pending_evidence",
        effect={"evidence_required": True},
    )
    c = card(p, score=_score(), log=[p], sent=False)
    assert c.change is not None
    assert (c.change.label, c.change.before, c.change.after) == ("Could I get this", "4", None)
    assert c.link is not None and c.link[2] == f"/pushbacks/{p.id}/evidence"


def test_a_sent_application_says_it_keeps_its_score() -> None:
    p = _applied()
    c = card(p, score=_score(), log=[p], sent=True)
    assert c.change is not None and c.change.after is None
    assert wording.SENT_HOLDS in c.lines


def test_a_broken_must_have_holds_the_number_and_says_so() -> None:
    from jfl_core.models import HardGateBreach

    p = _applied()
    score = _score(
        want_it_score=2,
        hard_gate_breaches=[HardGateBreach(gate="location", breach="On site.")],
    )
    c = card(p, score=score, log=[p], sent=False)
    assert c.change is not None and c.change.after is None
    assert wording.BREACH_HOLDS in c.lines


def test_the_ad_card_never_claims_a_re_read() -> None:
    p = _applied(kind="factual", delta=0.0, disposition="recorded_only")
    c = card(p, score=_score(), log=[p], sent=False)
    assert c.rescore is True
    assert wording.AD_NOT_REREAD in c.lines
    assert c.change is None


def test_a_withdrawn_correction_reads_as_undone() -> None:
    p = _applied().model_copy(update={"withdrawn_at": NOW})
    c = card(p, score=_score(), log=[], sent=False)
    assert c.state == "undone" and not c.can_undo
    assert history_line(p) == "Undone -- it no longer counts."


def test_every_applied_combination_has_a_reading_and_the_list_offers_the_others() -> None:
    for reading in READINGS:
        direction = reading.direction or "up"
        target = COULD_GET_OVERALL if reading.kind == "capability" else WANT_OVERALL
        p = _applied(kind=reading.kind, direction=direction, delta=0.0, target=target)
        assert reading_of(p) == reading
        c = card(p, score=_score(), log=[p], sent=False)
        assert reading not in c.other_readings
        assert len(c.other_readings) == len(READINGS) - 1


# -- no internal words ----------------------------------------------------------------


def _strings_in_module() -> list[str]:
    found: list[str] = []
    for name, value in inspect.getmembers(wording):
        if name.startswith("_") or not name.isupper():
            continue
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, tuple):
            found.extend(v for v in value if isinstance(v, str))
    for reading in READINGS:
        found.extend((reading.choice, reading.meaning))
    return found


def test_no_on_screen_string_uses_an_internal_word() -> None:
    offenders = [
        text for text in _strings_in_module() if any(word in text.lower() for word in INTERNAL)
    ]
    assert not offenders, offenders


_JINJA = re.compile(r"\{#.*?#\}|\{%.*?%\}|\{\{.*?\}\}", re.S)
_TAG = re.compile(r"<[^>]+>")


@pytest.mark.parametrize("name", ["_pushbacks.html", "pushbacks.html", "pushback_evidence.html"])
def test_no_pushback_template_puts_an_internal_word_on_screen(name: str) -> None:
    visible = _TAG.sub(" ", _JINJA.sub(" ", (TEMPLATE_DIR / name).read_text())).lower()
    found = [word for word in INTERNAL if word in visible]
    assert not found, f"{name} shows {found}"
