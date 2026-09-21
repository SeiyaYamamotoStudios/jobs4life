"""The profile: five sections, one append-only row -- `docs/profile-schema.md`.

Replaces B3a's eighteen free-text questions, which production held zero rows of.
The shape is the design doc's: constraints, capabilities, disciplines,
objectives, and the self-assessment that is not a preference at all.

Four rules run through every route here, and they are the same rule seen from
four sides.

**Nothing is guessed.** A section left blank stays blank; a field skipped reads
"not stated". A constraint with a value but no must/nice/never stance is
*refused*, not filed under a default -- "I said London" does not say whether
London is a must, and that difference is the only reason to record it.

**Depth is answered, not rated.** A capability's tier comes from two or three
behavioural questions -- did you do it yourself, did you run it in production,
did you review others doing it -- and the page shows which answer produced which
tier. A self-rating would measure confidence, and confidence is not what a CV
claim gets measured against. `interest` is asked separately, because what
someone is good at and what they want to keep doing are different questions and
averaging them answers neither.

**Evidence comes from the corpus, never from the browser.** Capability rows are
pre-proposed from *confirmed* CV facts and carry those facts' span ids. A save
re-derives them server-side; no span id is ever read out of a form. A capability
with no evidence is shown as "claimed, not yet evidenced", which is what it is.

**One section of this page is not a preference.** The self-assessment -- where
your depth is genuine, and the gaps that keep coming up -- is a claim about the
person, so saving it also writes the user's words to the corpus, verbatim,
through the one existing write path (`jfl_core.storage.user_corpus`, and so
`jfl_core.corpus_source`). Deliberately not a second path: two mechanisms for
one kind of fact is how one sentence ends up with two span ids the claim gate
reads as two pieces of evidence. The page says so in plain words, because a tool
that quietly turned "I want more scope" into evidence about what you have done
would be doing the exact thing this project exists to oppose.

No model call anywhere in this module.

Screens:

  GET  /profile                            -- the whole page, five sections
  POST /profile/constraints                -- section 1, all eight kinds at once
  POST /profile/capabilities               -- section 2, add a row by name
  POST /profile/capabilities/{key}         -- section 2, tier one row
  POST /profile/capabilities/{key}/remove  -- section 2, drop a row
  POST /profile/disciplines                -- section 3
  POST /profile/objectives                 -- section 4, four ranked slots
  POST /profile/self-assessment            -- section 5, ALSO to the corpus
  GET  /profile/history                    -- every saved version, newest first
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response
from jfl_core.profile import (
    Capability,
    Profile,
    seed_capabilities_from_facts,
)
from jfl_core.profile_questions import CORPUS_SECTIONS

from jfl_web.deps import (
    CandidateFactRepoDep,
    CsrfDep,
    ProfileStoreDep,
    SessionDep,
    UserCorpusRepoDep,
)
from jfl_web.profile import (
    CAPABILITY_TIERS,
    COMP_COPY,
    CONSTRAINT_FIELDS,
    INTEREST_CHOICES,
    MAX_ITEMS,
    MAX_LABEL,
    MAX_NOTE,
    MAX_OBJECTIVE_TEXT,
    MAX_SELF_ASSESSMENT,
    MAX_TEXT,
    STANCE_CHOICES,
    TIER_DESCRIPTIONS,
    TIER_NAMES,
    TIER_QUESTIONS,
    FormTooLongError,
    InvalidAmountError,
    InvalidChoiceError,
    InvalidYearError,
    MissingStanceError,
    TooManyItemsError,
    answers_for_tier,
    checked_text,
    merge_capabilities,
    parse_capability,
    parse_constraints,
    parse_disciplines,
    parse_objectives,
)
from jfl_web.templating import render

router = APIRouter()

# Every way a submitted form can be wrong. All of them are the user's words not
# fitting, or a form we did not render -- never a failure worth a 500.
_FORM_ERRORS = (
    FormTooLongError,
    InvalidAmountError,
    InvalidChoiceError,
    InvalidYearError,
    MissingStanceError,
    TooManyItemsError,
)

_NO_SUCH_CAPABILITY = "No such capability -- it may belong to another account."


def _visible_capabilities(
    store: ProfileStoreDep, facts: CandidateFactRepoDep
) -> tuple[Profile, list[Capability], dict[str, str]]:
    """The saved profile, the capability rows the page shows, and the text
    behind each row's evidence.

    Rows are everything saved, plus a proposal for every confirmed CV fact not
    already covered. Seeds are re-derived on every request rather than written
    into the profile on sight: a row the user has never looked at is not
    something they have claimed, and writing it in would make the profile say it
    was.

    The evidence map turns span ids into the sentence the user actually
    confirmed, because "evidence: 2 spans" tells nobody whether the evidence
    supports the tier they are about to claim.
    """
    confirmed = facts.list_facts(state="confirmed")
    profile = store.current()
    seeded = seed_capabilities_from_facts(confirmed)
    evidence = {str(f.span_id): f.corpus_text for f in confirmed if f.span_id is not None}
    return profile, merge_capabilities(profile.capabilities, seeded), evidence


def _context(
    session: SessionDep,
    store: ProfileStoreDep,
    facts: CandidateFactRepoDep,
    **extra: Any,
) -> dict[str, Any]:
    profile, capabilities, evidence_texts = _visible_capabilities(store, facts)
    saved_keys = {c.key for c in profile.capabilities}
    version = store.current_version()
    ctx: dict[str, Any] = {
        "session": session,
        "user": session.user,
        "profile": profile,
        "capabilities": capabilities,
        "saved_capability_keys": saved_keys,
        "evidence_texts": evidence_texts,
        "answers_for_tier": answers_for_tier,
        "constraint_fields": CONSTRAINT_FIELDS,
        "stance_choices": STANCE_CHOICES,
        "comp_copy": COMP_COPY,
        "tier_questions": TIER_QUESTIONS,
        "tier_names": TIER_NAMES,
        "tier_descriptions": TIER_DESCRIPTIONS,
        "tiers": CAPABILITY_TIERS,
        "interest_choices": INTEREST_CHOICES,
        "saved_at": version.created_at if version is not None else None,
        "max_note": MAX_NOTE,
        "max_text": MAX_TEXT,
        "max_label": MAX_LABEL,
        "max_items": MAX_ITEMS,
        "max_objective_text": MAX_OBJECTIVE_TEXT,
        "max_self_assessment": MAX_SELF_ASSESSMENT,
    }
    ctx.update(extra)
    return ctx


def _error(
    request: Request,
    session: SessionDep,
    store: ProfileStoreDep,
    facts: CandidateFactRepoDep,
    message: str,
    status_code: int,
) -> Response:
    return render(
        request,
        "profile.html",
        _context(session, store, facts, error=message),
        status_code=status_code,
    )


def _saved(anchor: str) -> RedirectResponse:
    """POST/redirect/GET, and the flag is a bare `saved=1` -- never the message
    itself, which would be text from a query string rendered into a page.
    """
    return RedirectResponse(f"/profile?saved=1#{anchor}", status_code=303)


@router.get("/profile")
def profile_page(
    request: Request,
    session: SessionDep,
    store: ProfileStoreDep,
    facts: CandidateFactRepoDep,
) -> Response:
    return render(
        request,
        "profile.html",
        _context(session, store, facts, saved="saved" in request.query_params),
    )


@router.post("/profile/constraints")
async def save_constraints(
    request: Request,
    session: SessionDep,
    store: ProfileStoreDep,
    facts: CandidateFactRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """All eight constraint kinds in one save.

    The raw form is read rather than declared field by field: eight kinds with a
    stance, a note and a value apiece is twenty-odd parameters, and the shapes
    differ per kind. `require_csrf` has already parsed and cached it.
    """
    form = await request.form()
    submitted = {key: value for key, value in form.items() if isinstance(value, str)}
    try:
        constraints = parse_constraints(submitted)
    except _FORM_ERRORS as exc:
        return _error(request, session, store, facts, str(exc), 400)
    profile = store.current()
    store.save(profile.model_copy(update={"constraints": constraints}))
    return _saved("constraints")


@router.post("/profile/capabilities")
def add_capability(
    request: Request,
    session: SessionDep,
    store: ProfileStoreDep,
    facts: CandidateFactRepoDep,
    _csrf: CsrfDep,
    label: Annotated[str, Form()] = "",
) -> Response:
    """Add a capability the CVs did not propose. It arrives untiered and with no
    evidence, which is exactly what it is: a claim, not a fact.
    """
    try:
        name = checked_text(label, MAX_LABEL)
    except _FORM_ERRORS as exc:
        return _error(request, session, store, facts, str(exc), 400)
    if not name:
        return RedirectResponse("/profile#capabilities", status_code=303)
    profile = store.current()
    if profile.capability(Capability(label=name).key) is None:
        store.save(
            profile.model_copy(
                update={"capabilities": [*profile.capabilities, Capability(label=name)]}
            )
        )
    return _saved("capabilities")


@router.post("/profile/capabilities/{key}")
def save_capability(
    request: Request,
    key: str,
    session: SessionDep,
    store: ProfileStoreDep,
    facts: CandidateFactRepoDep,
    _csrf: CsrfDep,
    hands_on: Annotated[str, Form()] = "",
    production: Annotated[str, Form()] = "",
    oversight: Annotated[str, Form()] = "",
    interest: Annotated[str, Form()] = "",
    last_used: Annotated[str, Form()] = "",
) -> Response:
    """Tier one row, from its behavioural answers.

    The row is looked up among the *visible* capabilities -- saved ones and
    seeds alike -- so tiering a CV-proposed row is the same click as tiering a
    saved one, and the seed's evidence comes from the corpus rather than from
    the form.
    """
    profile, visible, _evidence = _visible_capabilities(store, facts)
    existing = next((c for c in visible if c.key == key), None)
    if existing is None:
        return _error(request, session, store, facts, _NO_SUCH_CAPABILITY, 404)
    try:
        updated = parse_capability(
            existing,
            hands_on=hands_on,
            production=production,
            oversight=oversight,
            interest=interest,
            last_used=last_used,
        )
    except _FORM_ERRORS as exc:
        return _error(request, session, store, facts, str(exc), 400)
    # In place where the row was already saved, appended where it was a seed --
    # a row must not jump down the page because it was tiered.
    saved = list(profile.capabilities)
    position = next((i for i, c in enumerate(saved) if c.key == key), None)
    if position is None:
        saved.append(updated)
    else:
        saved[position] = updated
    store.save(profile.model_copy(update={"capabilities": saved}))
    return _saved("capabilities")


@router.post("/profile/capabilities/{key}/remove")
def remove_capability(
    request: Request,
    key: str,
    session: SessionDep,
    store: ProfileStoreDep,
    facts: CandidateFactRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """Drop a row from the profile. Nothing is really lost: the table is
    append-only, so the version that held it is still readable, and a row seeded
    from a confirmed fact simply comes back as a proposal.
    """
    profile = store.current()
    remaining = [c for c in profile.capabilities if c.key != key]
    if len(remaining) != len(profile.capabilities):
        store.save(profile.model_copy(update={"capabilities": remaining}))
    return _saved("capabilities")


@router.post("/profile/disciplines")
def save_disciplines(
    request: Request,
    session: SessionDep,
    store: ProfileStoreDep,
    facts: CandidateFactRepoDep,
    _csrf: CsrfDep,
    practises: Annotated[str, Form()] = "",
    not_this: Annotated[str, Form()] = "",
) -> Response:
    try:
        disciplines = parse_disciplines(practises, not_this)
    except _FORM_ERRORS as exc:
        return _error(request, session, store, facts, str(exc), 400)
    profile = store.current()
    store.save(profile.model_copy(update={"disciplines": disciplines}))
    return _saved("disciplines")


@router.post("/profile/objectives")
def save_objectives(
    request: Request,
    session: SessionDep,
    store: ProfileStoreDep,
    facts: CandidateFactRepoDep,
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
    """Four fixed ranked slots. Rank is the slot, so clearing the first does not
    shuffle the others up underneath the user.
    """
    slots = [
        (1, objective_1, evidence_1),
        (2, objective_2, evidence_2),
        (3, objective_3, evidence_3),
        (4, objective_4, evidence_4),
    ]
    try:
        objectives = parse_objectives(slots)
    except _FORM_ERRORS as exc:
        return _error(request, session, store, facts, str(exc), 400)
    profile = store.current()
    store.save(profile.model_copy(update={"objectives": objectives}))
    return _saved("objectives")


@router.post("/profile/self-assessment")
def save_self_assessment(
    request: Request,
    session: SessionDep,
    store: ProfileStoreDep,
    facts: CandidateFactRepoDep,
    corpus: UserCorpusRepoDep,
    _csrf: CsrfDep,
    depth_genuine: Annotated[str, Form()] = "",
    recurring_gaps: Annotated[str, Form()] = "",
) -> Response:
    """The one section of this page that also becomes corpus text.

    Two writes, one transaction (`db_conn` owns the boundary): the profile
    version, so what the user said and when stays readable, and the corpus
    statement, so scoring and drafting can actually cite it. `replace_section`
    rather than an append, so re-answering supersedes the earlier statement
    instead of leaving both live, and an emptied box clears the section -- a
    statement the user has withdrawn must stop grounding claims.

    Stored exactly as typed. No model is on this path.
    """
    try:
        depth = checked_text(depth_genuine, MAX_SELF_ASSESSMENT)
        gaps = checked_text(recurring_gaps, MAX_SELF_ASSESSMENT)
    except _FORM_ERRORS as exc:
        return _error(request, session, store, facts, str(exc), 400)

    profile = store.current()
    store.save(
        profile.model_copy(
            update={
                "self_assessment": profile.self_assessment.model_copy(
                    update={"depth_genuine": depth, "recurring_gaps": gaps}
                )
            }
        )
    )
    written = {"depth_genuine": depth, "recurring_gaps": gaps}
    for question_key, section in CORPUS_SECTIONS.items():
        corpus.replace_section(section, [written[question_key]])
    return _saved("self-assessment")


@router.get("/profile/history")
def profile_history(
    request: Request,
    session: SessionDep,
    store: ProfileStoreDep,
) -> Response:
    """Every saved version, newest first. Read-only.

    The table is append-only, so this costs nothing to offer and answers "what
    did I believe about myself in March" -- which is worth having in a tool whose
    subject is how a claim drifts from what was true.
    """
    return render(
        request,
        "profile_history.html",
        {
            "session": session,
            "user": session.user,
            "versions": store.history(),
            "tier_names": TIER_NAMES,
        },
    )
