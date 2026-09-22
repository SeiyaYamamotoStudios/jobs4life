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

**Two model calls are reachable from this page, and neither is made here.**
"Suggest capabilities from my confirmed facts" enqueues a background task
(`jfl_worker.handlers.capability_clusters`) that groups confirmed facts into
capability labels on the user's own key -- because a role is not a capability
and one fact is not one either. What comes back is **proposals**: accept,
rename or reject, never applied silently, and a rename is the user's word from
then on. "Suggest settings from my CVs" enqueues the other
(`jfl_worker.handlers.profile_suggestions`), which reads the CVs in the
sent-document store for the plain settings they state -- disciplines,
where the person has worked, the level the CV describes -- and proposes them
straight into the profile. Nothing in this module calls a model itself; a
two-minute call never runs inside a request.

Screens:

  GET  /profile                            -- the whole page, five sections
  POST /profile/constraints                -- section 1, all eight kinds at once
  POST /profile/capabilities               -- section 2, add a row by name
  POST /profile/capabilities/cluster       -- section 2, ask the model to group
                                              confirmed facts into capabilities
  GET  /profile/capabilities/cluster/{id}  -- section 2, the panel, for polling
  POST .../cluster/{id}/{key}/accept       -- section 2, accept (and rename) one
  POST .../cluster/{id}/{key}/reject       -- section 2, reject one
  POST /profile/capabilities/{key}         -- section 2, tier one row
  POST /profile/capabilities/{key}/remove  -- section 2, drop a row
  POST /profile/suggestions                -- read settings off the user's CVs
  GET  /profile/suggestions/{id}           -- that run's panel, for polling
  POST .../suggestions/{id}/{key}/accept   -- accept one, on the user's terms
  POST .../suggestions/{id}/{key}/reject   -- reject one, remembered across runs
  POST /profile/disciplines                -- section 3
  POST /profile/objectives                 -- section 4, four ranked slots
  POST /profile/self-assessment            -- section 5, ALSO to the corpus
  GET  /profile/history                    -- every saved version, newest first
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response
from jfl_core.profile import (
    Capability,
    Profile,
    capability_key,
    facts_to_cluster,
    propose_capabilities,
)
from jfl_core.storage.credentials import ANTHROPIC_API_KEY
from jfl_core.storage.profile import save_profile

from jfl_web.capabilityclusters import (
    MAX_LABEL as MAX_CLUSTER_LABEL,
)
from jfl_web.capabilityclusters import (
    ClusterView,
    accepted_capabilities,
    cluster_view,
)
from jfl_web.deps import (
    CandidateFactRepoDep,
    CapabilityClusterRepoDep,
    CredentialRepoDep,
    CsrfDep,
    ProfileRepoDep,
    ProfileSuggestionRepoDep,
    RunRepoDep,
    SectionRepoDep,
    SentDocumentRepoDep,
    SessionDep,
    TaskRepoDep,
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
    parse_lines,
    parse_objectives,
    parse_stance,
)
from jfl_web.profilesuggestions import (
    MAX_LEVEL_TEXT,
    MissingLevelTextError,
    NoPlacesGivenError,
    SuggestionConflictError,
    SuggestionsView,
    applied,
    suggestions_view,
)
from jfl_web.sections import profile_sections
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
    store: ProfileRepoDep, facts: CandidateFactRepoDep
) -> tuple[Profile, list[Capability], dict[str, str]]:
    """The saved profile, the capability rows the page shows, and the text
    behind each row's evidence.

    Rows are everything saved, plus a proposal for every confirmed CV fact not
    already covered. Proposals are re-derived on every request rather than
    written into the profile on sight: a row the user has never looked at is
    not something they have claimed, and writing it in would make the profile
    say it was.

    The evidence map turns span ids into the sentence the user actually
    confirmed, because "evidence: 2 spans" tells nobody whether the evidence
    supports the tier they are about to claim.
    """
    confirmed = facts.list_facts(state="confirmed")
    profile = store.current()
    # The pure grouping rule, over facts this function already has --
    # `propose_capabilities_from_facts` is the same thing plus the query, and
    # calling it here would read the facts table twice for one page.
    proposed = propose_capabilities(confirmed, existing=profile.capabilities)
    evidence = {str(f.span_id): f.corpus_text for f in confirmed if f.span_id is not None}
    return profile, merge_capabilities(profile.capabilities, proposed), evidence


def _fact_texts(facts: CandidateFactRepoDep) -> dict[str, str]:
    """Candidate fact id -> the words the user actually confirmed.

    Keyed by the fact rather than by its span, because a clustering proposal
    names facts: the panel has to be able to show what a proposal would carry
    as evidence, and what it left unplaced, in the user's own wording.
    """
    return {str(f.id): f.corpus_text for f in facts.list_facts(state="confirmed")}


def _context(
    session: SessionDep,
    store: ProfileRepoDep,
    facts: CandidateFactRepoDep,
    ui_sections: SectionRepoDep | None = None,
    **extra: Any,
) -> dict[str, Any]:
    profile, capabilities, evidence_texts = _visible_capabilities(store, facts)
    saved_keys = {c.key for c in profile.capabilities}
    states = ui_sections.states() if ui_sections is not None else {}
    # `history(limit=1)` rather than a second repository method: the current
    # version *is* the newest row, and one read of it answers "last saved when".
    latest = store.history(limit=1)
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
        "saved_at": latest[0].created_at if latest else None,
        "max_note": MAX_NOTE,
        "max_text": MAX_TEXT,
        "max_label": MAX_LABEL,
        "max_items": MAX_ITEMS,
        "max_objective_text": MAX_OBJECTIVE_TEXT,
        "max_self_assessment": MAX_SELF_ASSESSMENT,
        "max_cluster_label": MAX_CLUSTER_LABEL,
        "max_suggestion_text": MAX_LEVEL_TEXT,
        # Set by the routes that actually looked one up. None means "not looked
        # up", which the panel renders as the button alone -- never as "no run
        # yet", which would be a claim this context cannot make.
        "cluster": None,
        "suggestions": None,
        # None means "not looked up" -- the empty state is a claim about this
        # account's CVs, and an error re-render has not asked.
        "has_cvs": None,
    }
    ctx.update(extra)
    # Built last, because three of the five summaries count things the context
    # above assembled, and the capabilities section has to know whether a
    # clustering run is in flight. A section a save has just redirected to is
    # forced open whatever is stored -- landing on a folded panel after pressing
    # Save would read as the save having been lost.
    cluster = ctx.get("cluster")
    row = getattr(cluster, "cluster", None)
    ctx["profile_sections"] = profile_sections(
        states,
        profile=profile,
        capabilities=capabilities,
        saved_capability_keys=saved_keys,
        cluster_created_at=getattr(row, "created_at", None),
        cluster_pending=getattr(row, "status", None) == "pending",
        force_open=ctx.get("open_section"),
    )
    return ctx


def _error(
    request: Request,
    session: SessionDep,
    store: ProfileRepoDep,
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

    The anchor is carried twice, as a fragment and as `open=`. A fragment never
    reaches the server, and the section being saved is exactly the one that has
    just stopped being empty and would therefore fold itself away: landing on a
    folded panel after pressing Save reads as the save having been lost. `open=`
    is matched against a fixed set of section names, never rendered.
    """
    return RedirectResponse(f"/profile?saved=1&open={anchor}#{anchor}", status_code=303)


def _cluster_view(
    session: SessionDep,
    clusters: CapabilityClusterRepoDep,
    facts: CandidateFactRepoDep,
    run_repo: RunRepoDep,
    cluster_id: uuid.UUID | None = None,
) -> ClusterView | None:
    """The clustering panel's state, priced.

    Cost comes from `runs` by the run's own trace, the same way the drafting
    screen prices a draft -- this table stores no money, and the one place that
    does is the one built for querying it.
    """
    cluster = clusters.get(cluster_id) if cluster_id is not None else clusters.latest()
    if cluster_id is not None and cluster is None:
        return None
    cost = (
        run_repo.cost_for_trace(session.user.id, cluster.trace_id)
        if cluster is not None and cluster.status != "pending"
        else None
    )
    return cluster_view(cluster, _fact_texts(facts), cost=cost)


@router.get("/profile")
def profile_page(
    request: Request,
    session: SessionDep,
    store: ProfileRepoDep,
    facts: CandidateFactRepoDep,
    clusters: CapabilityClusterRepoDep,
    settings: ProfileSuggestionRepoDep,
    documents: SentDocumentRepoDep,
    run_repo: RunRepoDep,
    ui_sections: SectionRepoDep,
) -> Response:
    return render(
        request,
        "profile.html",
        _context(
            session,
            store,
            facts,
            ui_sections,
            saved="saved" in request.query_params,
            cluster=_cluster_view(session, clusters, facts, run_repo),
            cluster_status=request.query_params.get("cluster"),
            # Never rendered, only matched: anything that is not one of the five
            # section names simply forces nothing open.
            open_section=request.query_params.get("open"),
            suggestions=_suggestions_view(session, store, settings, run_repo),
            suggest_status=request.query_params.get("suggest"),
            has_cvs=bool(documents.list_cvs()),
        ),
    )


@router.post("/profile/constraints")
async def save_constraints(
    request: Request,
    session: SessionDep,
    store: ProfileRepoDep,
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
    store: ProfileRepoDep,
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
    if not any(c.key == capability_key(name) for c in profile.capabilities):
        store.save(
            profile.model_copy(
                update={"capabilities": [*profile.capabilities, Capability(label=name)]}
            )
        )
    return _saved("capabilities")


# -- capability clustering ---------------------------------------------------
#
# Declared BEFORE `/profile/capabilities/{key}`: FastAPI matches in declaration
# order, and `cluster` is a perfectly good capability key as far as the path
# converter is concerned.

CLUSTER_CAPABILITIES_KIND = "cluster_capabilities"

_NO_SUCH_CLUSTER = "No such suggestion -- it may belong to another account."


def _cluster_redirect(flag: str | None = None) -> RedirectResponse:
    """POST/redirect/GET, and the flag is a short literal -- never a message,
    which would be text from a query string rendered into a page.

    Deliberately not `_saved`: enqueueing a run, or being told there is nothing
    to group, saved nothing, and a "saved" banner over either would be the page
    claiming something it did not do.
    """
    suffix = f"?cluster={flag}" if flag else ""
    return RedirectResponse(f"/profile{suffix}#capabilities", status_code=303)


@router.post("/profile/capabilities/cluster")
def start_capability_cluster(
    request: Request,
    session: SessionDep,
    store: ProfileRepoDep,
    facts: CandidateFactRepoDep,
    clusters: CapabilityClusterRepoDep,
    credentials: CredentialRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """Ask the model to group this user's confirmed facts into capabilities.

    One call, on the user's own key, through the queue -- a model call never
    runs inside a request. Three refusals before anything is enqueued, each of
    which says so on the page rather than spending the user's money to find
    out:

      * **no confirmed facts.** There is nothing to group, and the panel points
        at where facts come from;
      * **no API key stored.** Same rule as the title suggestions: nothing is
        enqueued and the panel says to add one;
      * **a run already in flight.** Pressing the button twice buys one call.
    """
    profile = store.current()
    confirmed = facts.list_facts(state="confirmed")
    if not any(f.span_id is not None for f in confirmed):
        return _cluster_redirect("no_facts")
    if credentials.summary(ANTHROPIC_API_KEY) is None:
        return _cluster_redirect("needs_key")
    if clusters.pending() is not None:
        return _cluster_redirect("already_running")
    # Nothing unaccounted for is not a failure and not worth a call -- say so
    # rather than charging for an empty answer.
    sending, _omitted = facts_to_cluster(confirmed, existing=profile.capabilities)
    if not sending:
        return _cluster_redirect("nothing_new")

    # The trace is minted here so the run is priceable whatever happens to it.
    row = clusters.create_pending(trace_id=uuid.uuid4())
    tasks.enqueue(kind=CLUSTER_CAPABILITIES_KIND, payload={"cluster_id": str(row.id)})
    return _cluster_redirect()


@router.get("/profile/capabilities/cluster/{cluster_id}")
def capability_cluster_panel(
    request: Request,
    cluster_id: uuid.UUID,
    session: SessionDep,
    clusters: CapabilityClusterRepoDep,
    facts: CandidateFactRepoDep,
    run_repo: RunRepoDep,
) -> Response:
    """The panel, standalone -- what a pending run polls. Same view-building
    function the inline panel uses, so the two can never render one run's state
    differently.
    """
    view = _cluster_view(session, clusters, facts, run_repo, cluster_id)
    if view is None:
        return render(
            request,
            "error.html",
            {"session": session, "user": session.user, "message": _NO_SUCH_CLUSTER},
            status_code=404,
        )
    return render(
        request,
        "_capability_clusters.html",
        {"session": session, "cluster": view, "max_cluster_label": MAX_CLUSTER_LABEL},
    )


@router.post("/profile/capabilities/cluster/{cluster_id}/{key}/accept")
def accept_capability_proposal(
    request: Request,
    cluster_id: uuid.UUID,
    key: str,
    session: SessionDep,
    store: ProfileRepoDep,
    facts: CandidateFactRepoDep,
    clusters: CapabilityClusterRepoDep,
    _csrf: CsrfDep,
    label: Annotated[str, Form()] = "",
) -> Response:
    """Put one proposal on the profile, under the user's own name for it.

    `label` is the rename box: blank keeps the model's wording, anything else
    is the user's and is stored verbatim on both the proposal and the
    capability. `evidence` comes from the stored proposal's span ids and never
    from the form -- a span id arriving from a browser is a claim about
    grounding the browser does not get to make.

    A capability the user has already tiered or edited is **not** overwritten:
    the merge keeps their row and adds only the evidence it was missing.
    """
    try:
        renamed = checked_text(label, MAX_CLUSTER_LABEL)
    except _FORM_ERRORS as exc:
        return _error(request, session, store, facts, str(exc), 400)
    proposal = clusters.set_proposal_state(cluster_id, key, "accepted", label=renamed or None)
    if proposal is None:
        return _error(request, session, store, facts, _NO_SUCH_CLUSTER, 404)
    profile = store.current()
    store.save(
        profile.model_copy(update={"capabilities": accepted_capabilities(profile, proposal)})
    )
    return _cluster_redirect()


@router.post("/profile/capabilities/cluster/{cluster_id}/{key}/reject")
def reject_capability_proposal(
    request: Request,
    cluster_id: uuid.UUID,
    key: str,
    session: SessionDep,
    store: ProfileRepoDep,
    facts: CandidateFactRepoDep,
    clusters: CapabilityClusterRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """Say no to one grouping. Nothing is written to the profile, and the facts
    it named stay confirmed facts -- a rejected grouping is a rejected *label*,
    never a retracted fact.
    """
    if clusters.set_proposal_state(cluster_id, key, "rejected") is None:
        return _error(request, session, store, facts, _NO_SUCH_CLUSTER, 404)
    return _cluster_redirect()


@router.post("/profile/capabilities/{key}")
def save_capability(
    request: Request,
    key: str,
    session: SessionDep,
    store: ProfileRepoDep,
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
    store: ProfileRepoDep,
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


# -- profile settings read off the user's CVs --------------------------------
#
# The other model call reachable from this page, and like the clustering one it
# is not made here. A CV states claims about the world -- those go through the
# fact-confirmation path and become corpus one at a time. It also states plain
# settings: which disciplines someone practises, where they have worked, the
# level they have been operating at. Those are proposed straight into the
# profile and accepted with a click.
#
# Nothing is applied silently, every proposal shows the CV line behind it, and
# a setting the user has already stated always wins -- the rules live in
# `jfl_web.profilesuggestions`.

SUGGEST_PROFILE_SETTINGS_KIND = "suggest_profile_settings"

_NO_SUCH_SUGGESTION = "No such suggestion -- it may belong to another account."


def _suggest_redirect(flag: str | None = None) -> RedirectResponse:
    """POST/redirect/GET, and the flag is a short literal -- never a message,
    which would be text from a query string rendered into a page.
    """
    suffix = f"?suggest={flag}" if flag else ""
    return RedirectResponse(f"/profile{suffix}#profile-suggestions", status_code=303)


def _suggestions_view(
    session: SessionDep,
    store: ProfileRepoDep,
    suggestions: ProfileSuggestionRepoDep,
    run_repo: RunRepoDep,
    run_id: uuid.UUID | None = None,
) -> SuggestionsView | None:
    """The suggestion panel's state, priced against the profile as it stands.

    Cost comes from `runs` by the run's own trace, the same way the clustering
    panel prices itself -- this table stores no money, and the one place that
    does is the one built for querying it.

    The profile is read here rather than passed in because every proposal is
    shown against what the user has **already** stated: that comparison is the
    difference between "accept this" and "this conflicts with what you said".
    """
    run = suggestions.get(run_id) if run_id is not None else suggestions.latest()
    if run_id is not None and run is None:
        return None
    cost = (
        run_repo.cost_for_trace(session.user.id, run.trace_id)
        if run is not None and run.status != "pending"
        else None
    )
    return suggestions_view(run, store.current(), cost=cost)


@router.post("/profile/suggestions")
def start_profile_suggestions(
    request: Request,
    session: SessionDep,
    store: ProfileRepoDep,
    facts: CandidateFactRepoDep,
    suggestions: ProfileSuggestionRepoDep,
    documents: SentDocumentRepoDep,
    credentials: CredentialRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """Ask the model to read this user's uploaded CVs for profile settings.

    One call, on the user's own key, through the queue -- a model call never
    runs inside a request. Three refusals before anything is enqueued, each of
    which says so on the page rather than spending the user's money to find
    out:

      * **no CVs uploaded.** There is nothing to read, and the panel points at
        where CVs are uploaded;
      * **no API key stored.** Nothing is enqueued and the panel says to add one;
      * **a run already in flight.** Pressing the button twice buys one call.
    """
    if not documents.list_cvs():
        return _suggest_redirect("no_cvs")
    if credentials.summary(ANTHROPIC_API_KEY) is None:
        return _suggest_redirect("needs_key")
    if suggestions.pending() is not None:
        return _suggest_redirect("already_running")

    # The trace is minted here so the run is priceable whatever happens to it.
    row = suggestions.create_pending(trace_id=uuid.uuid4())
    tasks.enqueue(kind=SUGGEST_PROFILE_SETTINGS_KIND, payload={"suggestion_run_id": str(row.id)})
    return _suggest_redirect()


@router.get("/profile/suggestions/{run_id}")
def profile_suggestions_panel(
    request: Request,
    run_id: uuid.UUID,
    session: SessionDep,
    store: ProfileRepoDep,
    suggestions: ProfileSuggestionRepoDep,
    run_repo: RunRepoDep,
) -> Response:
    """The panel, standalone -- what a pending run polls. Same view-building
    function the inline panel uses, so the two can never render one run's state
    differently.
    """
    view = _suggestions_view(session, store, suggestions, run_repo, run_id)
    if view is None:
        return render(
            request,
            "error.html",
            {"session": session, "user": session.user, "message": _NO_SUCH_SUGGESTION},
            status_code=404,
        )
    return render(
        request,
        "_profile_suggestions.html",
        {
            "session": session,
            "suggestions": view,
            "stance_choices": STANCE_CHOICES,
            "max_items": MAX_ITEMS,
            "max_suggestion_text": MAX_LEVEL_TEXT,
        },
    )


@router.post("/profile/suggestions/{run_id}/{key}/accept")
async def accept_profile_suggestion(
    request: Request,
    run_id: uuid.UUID,
    key: str,
    session: SessionDep,
    store: ProfileRepoDep,
    facts: CandidateFactRepoDep,
    suggestions: ProfileSuggestionRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """Put one proposed setting on the profile, on the user's own terms.

    The order matters and is deliberate: the profile is updated **first**, and
    the proposal is marked answered only once that succeeded. A refused accept
    -- a conflict with nothing ticked, a constraint with no stance, a level
    box left empty -- therefore leaves the proposal open and still answerable,
    rather than consuming it to record nothing.

    Nothing is read out of the form that the profile did not already know how
    to hold: a stance from the choices the page offers, the places as the user
    left them, and their own words for a level floor.
    """
    form = await request.form()
    submitted = {k: v for k, v in form.items() if isinstance(v, str)}
    run = suggestions.get(run_id)
    proposal = (
        next((p for p in run.open_proposals if p.key == key), None) if run is not None else None
    )
    if proposal is None:
        return _error(request, session, store, facts, _NO_SUCH_SUGGESTION, 404)

    try:
        stance = parse_stance(submitted.get("stance", ""))
        places = parse_lines(submitted.get("places", "")) if proposal.kind == "location" else None
        level_text = checked_text(submitted.get("level_text", ""), MAX_LEVEL_TEXT)
        updated = applied(
            store.current(),
            proposal,
            stance=stance,
            places=places,
            level_text=level_text,
            replace=submitted.get("replace", "") == "yes",
        )
    except (
        *_FORM_ERRORS,
        SuggestionConflictError,
        MissingLevelTextError,
        NoPlacesGivenError,
    ) as exc:
        return _error(request, session, store, facts, str(exc), 400)

    store.save(updated)
    suggestions.set_proposal_state(run_id, key, "accepted")
    return _suggest_redirect()


@router.post("/profile/suggestions/{run_id}/{key}/reject")
def reject_profile_suggestion(
    request: Request,
    run_id: uuid.UUID,
    key: str,
    session: SessionDep,
    store: ProfileRepoDep,
    facts: CandidateFactRepoDep,
    suggestions: ProfileSuggestionRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """Say no to one proposed setting, permanently.

    Nothing is written to the profile, and the rejection is remembered across
    runs: `answered_keys` is subtracted before a later run's proposals are
    stored, and the key folds the same way, so the same suggestion from the
    same CV -- or from a later one saying the same thing -- is never offered
    again.
    """
    if suggestions.set_proposal_state(run_id, key, "rejected") is None:
        return _error(request, session, store, facts, _NO_SUCH_SUGGESTION, 404)
    return _suggest_redirect()


@router.post("/profile/disciplines")
def save_disciplines(
    request: Request,
    session: SessionDep,
    store: ProfileRepoDep,
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
    store: ProfileRepoDep,
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
    store: ProfileRepoDep,
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
    # `save_profile` rather than `store.save`, because this section has two
    # halves and no caller may perform one of them: the profile row is where
    # the screen reads the words back from, and the corpus span is what the
    # claim gate can cite. It is the one write path, shared with confirming a
    # CV fact, and no model is anywhere on it.
    save_profile(
        store,
        corpus,
        profile.model_copy(
            update={
                "self_assessment": profile.self_assessment.model_copy(
                    update={"depth_genuine": depth, "recurring_gaps": gaps}
                )
            }
        ),
    )
    return _saved("self-assessment")


@router.get("/profile/history")
def profile_history(
    request: Request,
    session: SessionDep,
    store: ProfileRepoDep,
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
