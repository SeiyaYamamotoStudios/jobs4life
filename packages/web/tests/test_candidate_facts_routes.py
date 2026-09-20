"""No route, and no repository method, confirms every candidate fact at once.

CLAUDE.md, 2026-09-18: "the user confirms each, per role, with no global
accept-all". That is not a UI preference -- a one-click import of every fact a
model read out of a stack of CVs would put those CVs' stretches into the corpus
under the user's name, and the corpus is what every later claim is measured
against. The over-claim number would keep being printed and would stop meaning
anything.

So the absence of that control is checked here, mechanically, rather than left
to whoever reviews the template next. Two rules:

  1. every state-changing candidate-fact route is scoped to ONE fact or ONE
     role -- its path carries `{fact_id}` or `{role_key}`;
  2. the repository exposes no method whose name suggests a bulk confirm.

Neither test needs a database: routes are read off every `jfl_web.routes`
module's own `router`, and the repository is inspected, never instantiated.
Discovery walks the package rather than naming one module, so a bulk-confirm
route added somewhere else next year is still caught.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import jfl_web.routes
import pytest
from fastapi.routing import APIRoute
from jfl_core.storage.candidate_facts import PostgresCandidateFactRepository
from jfl_web.app import create_app


def _every_declared_route() -> list[APIRoute]:
    """Every route in every `jfl_web.routes` module, found by walking the
    package -- not by listing the modules, which is how a new one gets missed.
    """
    found: list[APIRoute] = []
    for info in pkgutil.walk_packages(jfl_web.routes.__path__, prefix="jfl_web.routes."):
        module = importlib.import_module(info.name)
        router = getattr(module, "router", None)
        if router is None:
            continue
        found.extend(r for r in router.routes if isinstance(r, APIRoute))
    return found


@pytest.fixture
def fact_routes() -> list[APIRoute]:
    return [r for r in _every_declared_route() if r.path.startswith("/corpus/facts")]


def test_discovery_actually_found_the_routes(fact_routes: list[APIRoute]) -> None:
    """A rule that silently checks nothing passes and proves nothing."""
    assert any("confirm" in route.path for route in fact_routes)


def test_the_screen_is_actually_reachable() -> None:
    """The routes above only matter if the app serves them."""
    source = inspect.getsource(create_app)
    assert "candidate_facts.router" in source


def test_every_confirming_route_is_scoped_to_one_fact_or_one_role(
    fact_routes: list[APIRoute],
) -> None:
    unscoped = [
        route.path
        for route in fact_routes
        if "confirm" in route.path
        and "{fact_id}" not in route.path
        and "{role_key}" not in route.path
    ]
    assert not unscoped, (
        f"these routes confirm candidate facts without naming one fact or one role: "
        f"{unscoped}. A CV's claims may only enter the corpus a role at a time, "
        "deliberately read -- see CLAUDE.md's 2026-09-18 decision."
    )


def test_no_state_changing_fact_route_is_a_bare_collection(
    fact_routes: list[APIRoute],
) -> None:
    """A POST to `/corpus/facts` itself would be a collection-wide write, which
    is the shape a global accept-all would take.
    """
    collection_writes = [
        route.path
        for route in fact_routes
        if route.path.rstrip("/") == "/corpus/facts" and (route.methods or set()) - {"GET", "HEAD"}
    ]
    assert not collection_writes


def test_the_repository_offers_no_bulk_confirm() -> None:
    """The route layer is not the only place such a control could appear. If
    the repository grew `confirm_all`, a future page could call it in one line.
    """
    names = {
        name
        for name, _ in inspect.getmembers(
            PostgresCandidateFactRepository, predicate=inspect.isfunction
        )
        if not name.startswith("_")
    }
    forbidden = {"confirm_all", "confirm_every", "accept_all", "confirm_many", "confirm_facts"}
    assert not (names & forbidden), sorted(names & forbidden)
    assert "confirm" in names  # the single-fact one is what everything goes through


def test_confirm_takes_exactly_one_fact() -> None:
    """Its first parameter is a single id, not a collection -- so "confirm
    these" is not expressible without a caller writing the loop, and a loop is
    where the per-role narrowing lives.
    """
    parameters = list(inspect.signature(PostgresCandidateFactRepository.confirm).parameters)
    assert parameters[:2] == ["self", "fact_id"]
