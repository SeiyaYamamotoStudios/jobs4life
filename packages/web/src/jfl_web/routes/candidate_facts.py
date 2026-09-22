"""The confirmation screen: what your CVs claim, and what you say is true.

PLAN.md B6 as redesigned on 2026-09-18. This is the step that protects the
product's thesis. The corpus is what a claim is later measured against, so a
CV's sentences must not become the corpus unexamined -- ground on the CV and
every later CV reads as "supported", and the over-claim number quietly stops
measuring anything. (The 33-CV analysis found exactly that drift.)

Hence the shape of this page, and the shape of what it deliberately lacks:

* **Per fact**: confirm as written, edit, or reject. An edit is stored
  verbatim -- the user's wording, not a tidied version of it, and no model call
  anywhere in this path.
* **Per role**: "all true as written", covering only the facts of that role that
  were on screen when the page was rendered and that have no unanswered probe.
  The rendered fact ids are submitted with the form, so a fact that appeared
  after the page was drawn is never confirmed by a click the user made before it
  existed.
* **Nowhere**: a control that confirms every role at once. That would import a
  CV's drift wholesale under the user's name, which is the single failure this
  screen exists to prevent. There is no such route, and
  `packages/web/tests/test_candidate_facts_routes.py` fails the build if one
  appears.

Rejected facts are never deleted. They collapse into a per-role "not true as
written" list and can be brought back, because "I cannot find or undo the thing
this tool recorded about me" is disqualifying in a truthfulness tool.

The word "corpus" stays in the code and off the screen: to the user this is
**Confirm what's true**, under Background.

Screens:

  GET  /background/facts                          -- every role, its facts, progress
  POST /background/facts/{fact_id}/confirm        -- one fact, the user's words
  POST /background/facts/{fact_id}/reject         -- one fact, kept and visible
  POST /background/facts/{fact_id}/restore        -- bring a rejected fact back
  POST /background/facts/roles/{role_key}/confirm -- one role's on-screen facts
  GET  /corpus/facts                              -- permanent redirect, for bookmarks
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response
from jfl_core.models import CandidateFact
from jfl_core.storage.candidate_facts import ProbeUnansweredError

from jfl_web.deps import CandidateFactRepoDep, CsrfDep, SessionDep
from jfl_web.templating import render

router = APIRouter()

# Where a user with no facts yet is sent: the CV upload screen, which is
# `jfl_web.routes.corpus`'s GET /background. This constant is the only place
# the path is written down.
UPLOAD_PATH = "/background"

# Generous, and a rejection rather than a truncation -- the same rule the
# profile page follows. A shortened statement is not what the user wrote.
MAX_FACT_TEXT = 4000
MAX_PROBE_ANSWER = 1000

_NOT_FOUND = "No such fact -- it may belong to another account."
_PROBE_UNANSWERED = "Answer the question beside that fact before confirming it."
_TOO_LONG = f"That text is longer than {MAX_FACT_TEXT} characters."


def _anchor(role_key: str | None = None) -> str:
    return f"#role-{role_key}" if role_key else ""


class _RoleView:
    """One role as the page renders it: its facts split by state, plus the
    counts the progress line needs. A plain object rather than a dict so the
    template reads as prose.
    """

    def __init__(self, role_key: str, role_label: str) -> None:
        self.role_key = role_key
        self.role_label = role_label
        self.to_check: list[CandidateFact] = []
        self.confirmed: list[CandidateFact] = []
        self.rejected: list[CandidateFact] = []

    @property
    def confirmable(self) -> list[CandidateFact]:
        """The facts a per-role "all true as written" would cover: on screen,
        unchecked, and not waiting on a probe answer.
        """
        return [f for f in self.to_check if not f.needs_probe_answer]

    @property
    def blocked(self) -> int:
        return len(self.to_check) - len(self.confirmable)

    @property
    def total(self) -> int:
        return len(self.to_check) + len(self.confirmed) + len(self.rejected)


def _role_views(facts: CandidateFactRepoDep) -> list[_RoleView]:
    """Roles in CV order, each holding its own facts in CV order."""
    views = {r.role_key: _RoleView(r.role_key, r.role_label) for r in facts.roles()}
    for fact in facts.list_facts():
        view = views.get(fact.role_key)
        if view is None:  # pragma: no cover -- roles() is derived from the same rows
            continue
        if fact.state == "confirmed":
            view.confirmed.append(fact)
        elif fact.state == "rejected":
            view.rejected.append(fact)
        else:
            view.to_check.append(fact)
    return list(views.values())


def _context(session: SessionDep, facts: CandidateFactRepoDep, **extra: Any) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "session": session,
        "user": session.user,
        "roles": _role_views(facts),
        "counts": facts.counts(),
        "upload_path": UPLOAD_PATH,
        "max_fact_text": MAX_FACT_TEXT,
        "max_probe_answer": MAX_PROBE_ANSWER,
    }
    ctx.update(extra)
    return ctx


def _error(
    request: Request,
    session: SessionDep,
    facts: CandidateFactRepoDep,
    message: str,
    status_code: int,
) -> Response:
    return render(
        request,
        "candidate_facts.html",
        _context(session, facts, error=message),
        status_code=status_code,
    )


@router.get("/background/facts")
def facts_page(request: Request, session: SessionDep, facts: CandidateFactRepoDep) -> Response:
    return render(
        request,
        "candidate_facts.html",
        _context(session, facts, saved="saved" in request.query_params),
    )


@router.post("/background/facts/{fact_id}/confirm")
def confirm_fact(
    request: Request,
    fact_id: uuid.UUID,
    session: SessionDep,
    facts: CandidateFactRepoDep,
    _csrf: CsrfDep,
    fact_text: Annotated[str, Form()] = "",
    probe_answer: Annotated[str, Form()] = "",
) -> Response:
    """Confirm one fact. `fact_text` is whatever is in the textarea -- the
    model's proposal if the user left it alone, their own words if they edited
    it. Either way it is stored exactly as submitted.
    """
    if len(fact_text) > MAX_FACT_TEXT or len(probe_answer) > MAX_PROBE_ANSWER:
        return _error(request, session, facts, _TOO_LONG, 400)
    try:
        confirmed = facts.confirm(fact_id, text=fact_text, probe_answer=probe_answer)
    except ProbeUnansweredError:
        return _error(request, session, facts, _PROBE_UNANSWERED, 400)
    if confirmed is None:
        return _error(request, session, facts, _NOT_FOUND, 404)
    return RedirectResponse(
        f"/background/facts?saved=1{_anchor(confirmed.role_key)}", status_code=303
    )


@router.post("/background/facts/{fact_id}/reject")
def reject_fact(
    request: Request,
    fact_id: uuid.UUID,
    session: SessionDep,
    facts: CandidateFactRepoDep,
    _csrf: CsrfDep,
) -> Response:
    rejected = facts.reject(fact_id)
    if rejected is None:
        return _error(request, session, facts, _NOT_FOUND, 404)
    return RedirectResponse(
        f"/background/facts?saved=1{_anchor(rejected.role_key)}", status_code=303
    )


@router.post("/background/facts/{fact_id}/restore")
def restore_fact(
    request: Request,
    fact_id: uuid.UUID,
    session: SessionDep,
    facts: CandidateFactRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """Put a rejected fact back on the unchecked list. Nothing was deleted, so
    there is nothing to reconstruct.
    """
    restored = facts.restore(fact_id)
    if restored is None:
        return _error(request, session, facts, _NOT_FOUND, 404)
    return RedirectResponse(
        f"/background/facts?saved=1{_anchor(restored.role_key)}", status_code=303
    )


@router.post("/background/facts/roles/{role_key}/confirm")
def confirm_role(
    request: Request,
    role_key: str,
    session: SessionDep,
    facts: CandidateFactRepoDep,
    _csrf: CsrfDep,
    fact_id: Annotated[list[uuid.UUID] | None, Form()] = None,
) -> Response:
    """One role's "all true as written".

    Three narrowings, each deliberate. Only facts **submitted with this form**
    are considered, so the click covers what the user was looking at. Of those,
    only ones **belonging to this role**, so a tampered or stale form cannot
    reach across roles. Of those, only ones still **unchecked and not waiting on
    a probe answer** -- a fact that asks "led how many?" is exactly the fact a
    bulk confirm must not sweep up, since the missing number is what
    `scope_inflation` turns on.

    Each survivor goes through the same single-fact `confirm` the individual
    button uses, storing the model's proposal as the user's statement because
    the user has said, of that specific fact, that it is true as written.
    """
    submitted = list(fact_id or [])
    for one in submitted:
        fact = facts.get_fact(one)
        if fact is None or fact.role_key != role_key:
            continue
        if fact.state != "proposed" or fact.needs_probe_answer:
            continue
        facts.confirm(one)
    return RedirectResponse(f"/background/facts?saved=1{_anchor(role_key)}", status_code=303)


@router.get("/corpus/facts", include_in_schema=False)
def legacy_facts_path() -> Response:
    """The old path. A redirect, not a page -- see `jfl_web.routes.corpus`."""
    return RedirectResponse("/background/facts", status_code=301)
