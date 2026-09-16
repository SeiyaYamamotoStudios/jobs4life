"""Profile setup -- PLAN.md slice B3a: a set of questions, every one optional.

Scoring needs to know what the user wants and what they will not accept, and
conversations cannot remember it across sessions. Every answer is the user's
own words, stored verbatim, and every question can be left blank -- a skipped
question is simply absent, never defaulted or guessed at.

Scope: questions 1-14 and 17 (`jfl_core.profile_questions`). 15/16 (which
become corpus spans) and 18 (CV upload) are out of scope -- see the module
docstring there -- and the page says only that they are coming.

**Q2 reuses the saved job filter instead of duplicating it.** "Which working
arrangements will you consider?" is exactly what `/jobs`'s workplace preset
(`job_filters.workplace_mode`) already answers in a structured, machine-usable
form; a second tickbox set here would be a second place for that choice to go
stale against the first. So this page shows the saved preset read-only, with a
link to change it on `/jobs`, and Q2's own field is free text only -- for
nuance the preset cannot express ("hybrid is fine at one day a fortnight, not
one day a week"). Every other structured value (levels, comp floor, contract
types, disciplines) has no existing home elsewhere, so those live here.

**Per-section save.** Six POST routes, one per group of the page -- hard
gates, discipline, objectives, trajectory, the place, tells -- plus two for
ruled-out decisions (add, reopen). Saving one section never touches another's
answers, and saving an all-blank section is a valid, no-op-if-unchanged save.

Screens:

  GET  /profile                      -- the whole page, grouped by section
  POST /profile/hard-gates           -- questions 1, 3-8 (2 is read-only here)
  POST /profile/discipline           -- question 9
  POST /profile/objectives           -- questions 10/11, up to four slots
  POST /profile/trajectory           -- question 12
  POST /profile/place                -- question 13
  POST /profile/tells                -- question 14
  POST /profile/ruled-out            -- question 17: add an entry
  POST /profile/ruled-out/{id}/reopen -- question 17: mark one reopened
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response
from jfl_core.profile_questions import (
    CONTRACT_TYPE_CHOICES,
    DEFAULT_COMP_CURRENCY,
    DEFAULT_DISCIPLINE_CHOICES,
    DEFERRED_QUESTIONS,
    LEVEL_CHOICES,
    MAX_OBJECTIVES,
    OBJECTIVE_QUESTIONS,
    QUESTIONS_BY_KEY,
    RULED_OUT_QUESTION,
)

from jfl_web.deps import CsrfDep, JobFilterRepoDep, ProfileRepoDep, SessionDep
from jfl_web.jobfilter import WORKPLACE_MODE_NAMES, WORKPLACE_NAMES
from jfl_web.profile import (
    MAX_ANSWER_TEXT,
    MAX_RULED_OUT_TEXT,
    FormTooLongError,
    InvalidCompFloorError,
    checked_text,
    custom_disciplines,
    parse_comp_floor,
    parse_contract_types,
    parse_disciplines,
    parse_levels,
    selected_values,
)
from jfl_web.templating import render

router = APIRouter()

_RULED_OUT_NOT_FOUND = "No ruled-out entry found -- it may belong to another account."


def _objective_slots(profile: ProfileRepoDep) -> list[dict[str, Any]]:
    """`MAX_OBJECTIVES` slots, in ordinal order, blank where the user has not
    filled one in -- so the form always offers exactly four boxes regardless
    of how many are in use.
    """
    by_ordinal = {o.ordinal: o for o in profile.list_objectives()}
    return [
        {
            "ordinal": n,
            "objective_text": (by_ordinal[n].objective_text if n in by_ordinal else ""),
            "evidence_text": (by_ordinal[n].evidence_text if n in by_ordinal else ""),
            # `list_objectives` already excludes a slot whose latest version
            # is blank, so a slot present here always has a real save --
            # `created_at` is that version's timestamp, kept as `updated_at`
            # in the template's terms ("Saved <when>").
            "updated_at": (by_ordinal[n].created_at if n in by_ordinal else None),
        }
        for n in range(1, MAX_OBJECTIVES + 1)
    ]


def _context(
    request: Request,
    session: SessionDep,
    profile: ProfileRepoDep,
    filters: JobFilterRepoDep,
    **extra: Any,
) -> dict[str, Any]:
    answers = profile.get_current_answers()
    disciplines_answer = answers.get("disciplines")
    ctx: dict[str, Any] = {
        "session": session,
        "user": session.user,
        "questions": QUESTIONS_BY_KEY,
        "answers": answers,
        "custom_disciplines": custom_disciplines(
            disciplines_answer.structured if disciplines_answer else None
        ),
        "objectives": _objective_slots(profile),
        "objective_questions": OBJECTIVE_QUESTIONS,
        "ruled_out": profile.list_ruled_out(),
        "ruled_out_question": RULED_OUT_QUESTION,
        "saved_filter": filters.get_filter(),
        "workplace_mode_names": WORKPLACE_MODE_NAMES,
        "workplace_names": WORKPLACE_NAMES,
        "level_choices": LEVEL_CHOICES,
        "contract_type_choices": CONTRACT_TYPE_CHOICES,
        "discipline_choices": DEFAULT_DISCIPLINE_CHOICES,
        "default_currency": DEFAULT_COMP_CURRENCY,
        "deferred_questions": DEFERRED_QUESTIONS,
        "selected_values": selected_values,
        "max_answer_text": MAX_ANSWER_TEXT,
        "max_ruled_out_text": MAX_RULED_OUT_TEXT,
    }
    ctx.update(extra)
    return ctx


def _error(
    request: Request,
    session: SessionDep,
    profile: ProfileRepoDep,
    filters: JobFilterRepoDep,
    message: str,
    status_code: int,
) -> Response:
    return render(
        request,
        "profile.html",
        _context(request, session, profile, filters, error=message),
        status_code=status_code,
    )


@router.get("/profile")
def profile_page(
    request: Request,
    session: SessionDep,
    profile: ProfileRepoDep,
    filters: JobFilterRepoDep,
) -> Response:
    return render(
        request,
        "profile.html",
        _context(request, session, profile, filters, saved="saved" in request.query_params),
    )


@router.post("/profile/hard-gates")
def save_hard_gates(
    request: Request,
    session: SessionDep,
    profile: ProfileRepoDep,
    filters: JobFilterRepoDep,
    _csrf: CsrfDep,
    location_commute: Annotated[str, Form()] = "",
    workplace_arrangements: Annotated[str, Form()] = "",
    levels: Annotated[str, Form()] = "",
    level: Annotated[list[str] | None, Form()] = None,
    comp_floor: Annotated[str, Form()] = "",
    comp_amount: Annotated[str, Form()] = "",
    comp_currency: Annotated[str, Form()] = "",
    contract_types: Annotated[str, Form()] = "",
    contract_type: Annotated[list[str] | None, Form()] = None,
    notice_period: Annotated[str, Form()] = "",
    right_to_work: Annotated[str, Form()] = "",
    categorical_no: Annotated[str, Form()] = "",
) -> Response:
    try:
        comp_structured = parse_comp_floor(comp_amount, comp_currency)
        answers = {
            "location_commute": (checked_text(location_commute), None),
            "workplace_arrangements": (checked_text(workplace_arrangements), None),
            "levels": (checked_text(levels), parse_levels(level or [])),
            "comp_floor": (checked_text(comp_floor), comp_structured),
            "contract_types": (
                checked_text(contract_types),
                parse_contract_types(contract_type or []),
            ),
            "notice_period": (checked_text(notice_period), None),
            "right_to_work": (checked_text(right_to_work), None),
            "categorical_no": (checked_text(categorical_no), None),
        }
    except FormTooLongError as exc:
        return _error(request, session, profile, filters, str(exc), 400)
    except InvalidCompFloorError as exc:
        return _error(request, session, profile, filters, str(exc), 400)

    profile.save_answers(answers)
    return RedirectResponse("/profile?saved=1#hard-gates", status_code=303)


@router.post("/profile/discipline")
def save_discipline(
    request: Request,
    session: SessionDep,
    profile: ProfileRepoDep,
    filters: JobFilterRepoDep,
    _csrf: CsrfDep,
    disciplines: Annotated[str, Form()] = "",
    discipline: Annotated[list[str] | None, Form()] = None,
    discipline_custom: Annotated[str, Form()] = "",
) -> Response:
    try:
        text_value = checked_text(disciplines)
        custom_value = checked_text(discipline_custom, limit=200)
    except FormTooLongError as exc:
        return _error(request, session, profile, filters, str(exc), 400)

    profile.save_answers(
        {
            "disciplines": (
                text_value,
                parse_disciplines(discipline or [], custom_value),
            )
        }
    )
    return RedirectResponse("/profile?saved=1#discipline", status_code=303)


@router.post("/profile/objectives")
def save_objectives(
    request: Request,
    session: SessionDep,
    profile: ProfileRepoDep,
    filters: JobFilterRepoDep,
    _csrf: CsrfDep,
    objective_1: Annotated[str, Form()] = "",
    evidence_1: Annotated[str, Form()] = "",
    objective_2: Annotated[str, Form()] = "",
    evidence_2: Annotated[str, Form()] = "",
    objective_3: Annotated[str, Form()] = "",
    evidence_3: Annotated[str, Form()] = "",
    objective_4: Annotated[str, Form()] = "",
    evidence_4: Annotated[str, Form()] = "",
) -> Response:
    """Four fixed slots (`MAX_OBJECTIVES`), not a dynamic list -- PLAN.md caps
    objectives at four, so four named pairs are simpler than binding a
    numbered field set.
    """
    slots = [
        (1, objective_1, evidence_1),
        (2, objective_2, evidence_2),
        (3, objective_3, evidence_3),
        (4, objective_4, evidence_4),
    ]
    try:
        checked_slots = [
            (n, checked_text(obj_text), checked_text(ev_text)) for n, obj_text, ev_text in slots
        ]
    except FormTooLongError as exc:
        return _error(request, session, profile, filters, str(exc), 400)

    for n, objective_text, evidence_text in checked_slots:
        profile.save_objective(n, objective_text=objective_text, evidence_text=evidence_text)
    return RedirectResponse("/profile?saved=1#objectives", status_code=303)


@router.post("/profile/trajectory")
def save_trajectory(
    request: Request,
    session: SessionDep,
    profile: ProfileRepoDep,
    filters: JobFilterRepoDep,
    _csrf: CsrfDep,
    trajectory: Annotated[str, Form()] = "",
) -> Response:
    try:
        text_value = checked_text(trajectory)
    except FormTooLongError as exc:
        return _error(request, session, profile, filters, str(exc), 400)
    profile.save_answers({"trajectory": (text_value, None)})
    return RedirectResponse("/profile?saved=1#trajectory", status_code=303)


@router.post("/profile/place")
def save_place(
    request: Request,
    session: SessionDep,
    profile: ProfileRepoDep,
    filters: JobFilterRepoDep,
    _csrf: CsrfDep,
    employer_deal_breakers: Annotated[str, Form()] = "",
) -> Response:
    try:
        text_value = checked_text(employer_deal_breakers)
    except FormTooLongError as exc:
        return _error(request, session, profile, filters, str(exc), 400)
    profile.save_answers({"employer_deal_breakers": (text_value, None)})
    return RedirectResponse("/profile?saved=1#place", status_code=303)


@router.post("/profile/tells")
def save_tells(
    request: Request,
    session: SessionDep,
    profile: ProfileRepoDep,
    filters: JobFilterRepoDep,
    _csrf: CsrfDep,
    warning_signs: Annotated[str, Form()] = "",
) -> Response:
    try:
        text_value = checked_text(warning_signs)
    except FormTooLongError as exc:
        return _error(request, session, profile, filters, str(exc), 400)
    profile.save_answers({"warning_signs": (text_value, None)})
    return RedirectResponse("/profile?saved=1#tells", status_code=303)


@router.post("/profile/ruled-out")
def add_ruled_out(
    request: Request,
    session: SessionDep,
    profile: ProfileRepoDep,
    filters: JobFilterRepoDep,
    _csrf: CsrfDep,
    decision_text: Annotated[str, Form()] = "",
) -> Response:
    text_value = decision_text.strip()
    if not text_value:
        return RedirectResponse("/profile#ruled-out", status_code=303)
    try:
        checked = checked_text(text_value, limit=MAX_RULED_OUT_TEXT)
    except FormTooLongError as exc:
        return _error(request, session, profile, filters, str(exc), 400)
    profile.add_ruled_out(checked)
    return RedirectResponse("/profile?saved=1#ruled-out", status_code=303)


@router.post("/profile/ruled-out/{ruled_out_id}/reopen")
def reopen_ruled_out(
    request: Request,
    ruled_out_id: uuid.UUID,
    session: SessionDep,
    profile: ProfileRepoDep,
    filters: JobFilterRepoDep,
    _csrf: CsrfDep,
) -> Response:
    if profile.mark_reopened(ruled_out_id) is None:
        return _error(request, session, profile, filters, _RULED_OUT_NOT_FOUND, 404)
    return RedirectResponse("/profile?saved=1#ruled-out", status_code=303)
