"""Every kind this worker can run, in one readable list.

Adding a handler is two lines here and nothing anywhere else. Deleting one is a
line here -- and tasks of that kind then sit `pending` rather than failing,
because the claim query only asks for registered kinds.

`calls_model` is not optional and not guessable: it is what
`JFL_DISABLE_MODEL_CALLS` acts on, so a handler that spends a user's key and
says `calls_model=False` would make the incident lever a lie.

`build_registry` takes the settings because a model-calling handler needs two
things from the process environment -- the master key, to unseal the user's
stored API key, and which model to call. They are bound here, at wiring time,
rather than read inside a handler: `WorkerSettings.from_env` is this process's
one environment read, the same rule `RequestContext.from_env` follows at the
CLI boundary.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection

from jfl_worker.handlers.application_questions import CHECK_KIND as CHECK_APPLICATION_ANSWER
from jfl_worker.handlers.application_questions import DRAFT_KIND as DRAFT_APPLICATION_ANSWER
from jfl_worker.handlers.application_questions import (
    build_check_application_answer,
    build_draft_application_answer,
)
from jfl_worker.handlers.boards import KIND as CHECK_BOARD
from jfl_worker.handlers.boards import SCHEDULE_KIND as SCHEDULE_BOARD_CHECKS
from jfl_worker.handlers.boards import (
    TransportFactory,
    build_check_board,
    build_schedule_board_checks,
)
from jfl_worker.handlers.capability_clusters import KIND as CLUSTER_CAPABILITIES
from jfl_worker.handlers.capability_clusters import build_cluster_capabilities
from jfl_worker.handlers.coverage_generation import KIND as GENERATE_COVERAGE
from jfl_worker.handlers.coverage_generation import build_generate_coverage
from jfl_worker.handlers.cv_facts import KIND as EXTRACT_CV_FACTS
from jfl_worker.handlers.cv_facts import build_extract_cv_facts
from jfl_worker.handlers.description import KIND as FETCH_JOB_DESCRIPTION
from jfl_worker.handlers.description import build_fetch_job_description
from jfl_worker.handlers.draft_generation import KIND as GENERATE_CV_DRAFT
from jfl_worker.handlers.draft_generation import build_generate_cv_draft
from jfl_worker.handlers.extraction import KIND as EXTRACT_JOB_AD
from jfl_worker.handlers.extraction import build_extract_job_ad
from jfl_worker.handlers.feed_marks import KIND as PURGE_STALE_FEED_MARKS
from jfl_worker.handlers.feed_marks import purge_stale_feed_marks
from jfl_worker.handlers.profile_suggestions import KIND as SUGGEST_PROFILE_SETTINGS
from jfl_worker.handlers.profile_suggestions import build_suggest_profile_settings
from jfl_worker.handlers.pushback import KIND as CLASSIFY_PUSHBACK
from jfl_worker.handlers.pushback import build_classify_pushback
from jfl_worker.handlers.scoring import KIND as SCORE_APPLICATION
from jfl_worker.handlers.scoring import build_score_application
from jfl_worker.handlers.sessions import KIND as PURGE_EXPIRED_SESSIONS
from jfl_worker.handlers.sessions import purge_expired_sessions
from jfl_worker.handlers.title_suggestions import KIND as SUGGEST_TITLES
from jfl_worker.handlers.title_suggestions import build_suggest_titles
from jfl_worker.registry import HandlerRegistry
from jfl_worker.settings import WorkerSettings

__all__ = [
    "CHECK_APPLICATION_ANSWER",
    "CHECK_BOARD",
    "CLASSIFY_PUSHBACK",
    "CLUSTER_CAPABILITIES",
    "DRAFT_APPLICATION_ANSWER",
    "EXTRACT_CV_FACTS",
    "EXTRACT_JOB_AD",
    "FETCH_JOB_DESCRIPTION",
    "GENERATE_COVERAGE",
    "GENERATE_CV_DRAFT",
    "PURGE_EXPIRED_SESSIONS",
    "PURGE_STALE_FEED_MARKS",
    "SCHEDULE_BOARD_CHECKS",
    "SCORE_APPLICATION",
    "SUGGEST_PROFILE_SETTINGS",
    "SUGGEST_TITLES",
    "build_check_application_answer",
    "build_check_board",
    "build_classify_pushback",
    "build_cluster_capabilities",
    "build_draft_application_answer",
    "build_extract_cv_facts",
    "build_extract_job_ad",
    "build_fetch_job_description",
    "build_generate_coverage",
    "build_generate_cv_draft",
    "build_registry",
    "build_schedule_board_checks",
    "build_score_application",
    "build_suggest_profile_settings",
    "build_suggest_titles",
    "purge_expired_sessions",
    "purge_stale_feed_marks",
]


def build_registry(
    settings: WorkerSettings,
    *,
    board_transport: TransportFactory | None = None,
    description_transport: TransportFactory | None = None,
    board_owners: Collection[uuid.UUID] | None = None,
) -> HandlerRegistry:
    """The worker's handlers. `main()` passes settings and nothing else.

    The keyword arguments exist so a test can run the REAL registry without it
    being able to reach a job board: `board_transport` replaces the httpx
    transport `check_board` would otherwise open, `description_transport` does
    the same for `fetch_job_description`, and `board_owners` confines the
    scheduling pass to the test's own users. All default to production.
    """
    registry = HandlerRegistry()
    registry.register(
        PURGE_EXPIRED_SESSIONS,
        purge_expired_sessions,
        calls_model=False,
    )
    registry.register(
        PURGE_STALE_FEED_MARKS,
        purge_stale_feed_marks,
        # False, same reasoning as the session purge: a DELETE against this
        # deployment's own Postgres, no `anthropic` on the path, nobody's key
        # spent.
        calls_model=False,
    )
    registry.register(
        CHECK_BOARD,
        build_check_board(transport_factory=board_transport),
        # False, and true: public ATS APIs and pure rules, no `anthropic` on the
        # path. The kill switch is for spending a user's key; this spends none.
        calls_model=False,
    )
    registry.register(
        SCHEDULE_BOARD_CHECKS,
        build_schedule_board_checks(only_owners=board_owners),
        calls_model=False,
    )
    registry.register(
        EXTRACT_JOB_AD,
        build_extract_job_ad(master_key=settings.master_key, model=settings.model),
        # True, and this is the line the kill switch acts on. Extraction is one
        # Anthropic call on the user's own key.
        calls_model=True,
    )
    registry.register(
        EXTRACT_CV_FACTS,
        build_extract_cv_facts(master_key=settings.master_key, model=settings.model),
        # True: one Anthropic call per uploaded CV, on the user's own key. The
        # kill switch holds this one too -- a user uploading thirty-three CVs is
        # thirty-three calls, which is exactly the shape of spend the switch is
        # there to stop.
        calls_model=True,
    )
    registry.register(
        SUGGEST_TITLES,
        # No `model=` -- `jfl_generate.titles.suggest_titles` always calls
        # claude-haiku-4-5, never the deployment's configured model.
        build_suggest_titles(master_key=settings.master_key),
        # True: one Anthropic call on the user's own key, same as extraction.
        calls_model=True,
    )
    registry.register(
        CLUSTER_CAPABILITIES,
        # No `model=` -- `jfl_generate.capabilities.cluster_capabilities` always
        # calls claude-haiku-4-5, never the deployment's configured model.
        build_cluster_capabilities(master_key=settings.master_key),
        # True: one Anthropic call on the user's own key. The kill switch holds
        # it, which is also what makes "press it again" safe under an incident
        # -- the run sits `pending` rather than being claimed and charged.
        calls_model=True,
    )
    registry.register(
        SUGGEST_PROFILE_SETTINGS,
        # No `model=` -- `jfl_generate.profile_suggestions.suggest_profile_settings`
        # always calls claude-haiku-4-5, never the deployment's configured model.
        build_suggest_profile_settings(master_key=settings.master_key),
        # True: one Anthropic call on the user's own key, reading their uploaded
        # CVs. The kill switch holds it, which is also what makes "press it
        # again" safe under an incident -- the run sits `pending` rather than
        # being claimed and charged.
        calls_model=True,
    )
    registry.register(
        SCORE_APPLICATION,
        build_score_application(master_key=settings.master_key, model=settings.model),
        # True: one Anthropic call on the user's own key -- two, when the job
        # has no corpus coverage recorded yet and this has to run it first.
        calls_model=True,
    )
    registry.register(
        FETCH_JOB_DESCRIPTION,
        build_fetch_job_description(transport_factory=description_transport),
        # False: an HTTP request to the board's own API, same as CHECK_BOARD,
        # and no Anthropic call. The `extract_job_ad` task it enqueues on
        # success is what the kill switch actually holds.
        calls_model=False,
    )
    registry.register(
        CHECK_APPLICATION_ANSWER,
        build_check_application_answer(master_key=settings.master_key, model=settings.model),
        # True: the assessment call and the claim gate's own automatic pass are
        # both Anthropic calls on the user's own key.
        calls_model=True,
    )
    registry.register(
        DRAFT_APPLICATION_ANSWER,
        build_draft_application_answer(master_key=settings.master_key, model=settings.model),
        # True, same reasoning: the draft call and its automatic gate pass.
        calls_model=True,
    )
    registry.register(
        GENERATE_COVERAGE,
        build_generate_coverage(master_key=settings.master_key, model=settings.model),
        # True: one Anthropic call on the user's own key, same as extraction.
        calls_model=True,
    )
    registry.register(
        GENERATE_CV_DRAFT,
        build_generate_cv_draft(master_key=settings.master_key, model=settings.model),
        # True: two Anthropic calls on the user's own key -- the draft, then
        # the automatic claim-gate pass (see jfl_generate.draft).
        calls_model=True,
    )
    registry.register(
        CLASSIFY_PUSHBACK,
        # No `model=` -- `jfl_generate.pushback.classify_pushback` always
        # calls claude-haiku-4-5, never the deployment's configured model.
        build_classify_pushback(master_key=settings.master_key),
        # True: one Anthropic call on the user's own key, same as title
        # suggestion -- a cheap classification call, still spent on the
        # user's own credential and still held by the kill switch.
        calls_model=True,
    )
    return registry
