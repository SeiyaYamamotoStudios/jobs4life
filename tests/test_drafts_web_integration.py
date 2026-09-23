"""B5's drafting screen, through the real routes against a live Postgres --
`jfl_web.routes.drafts`, and the "CV for this job" panel it shares with the
application page. Marked `integration`; needs `docker compose up -d`
and `alembic upgrade head`. Same stub Google provider, sign-in helper and CSRF
scraping as `test_title_suggestions_web_integration.py`.

No model is ever called here: the enqueue path is checked by inspecting the
`tasks` table, never by letting a task run. Coverage, requirements and drafts
this file needs are written directly through the repositories, the same way
`test_title_suggestions_web_integration.py` writes `title_suggestions` rows
directly rather than running the worker.
"""

from __future__ import annotations

import datetime as dt
import html
import os
import re
import uuid
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import pytest
from fastapi import Request
from fastapi.responses import RedirectResponse, Response
from fastapi.testclient import TestClient
from jfl_core.crypto.envelope import MasterKey
from jfl_core.db.tables import spans as spans_table
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.ids import requirement_id
from jfl_core.models import Draft, JobRequirement, RequirementCoverage, RunRecord
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.postgres import PostgresJobRepository, PostgresRunRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity
from jfl_web.settings import WebSettings
from jfl_worker.chain import queue_next
from jfl_worker.registry import TaskContext
from sqlalchemy import create_engine, delete, insert, select, update
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

GENERATE_COVERAGE_KIND = "generate_coverage"
GENERATE_CV_DRAFT_KIND = "generate_cv_draft"
EXTRACT_JOB_AD_KIND = "extract_job_ad"


class StubGoogle:
    def __init__(self) -> None:
        self.identity: GoogleIdentity | None = None

    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response:
        return RedirectResponse("/auth/google/callback", status_code=303)

    async def fetch_identity(self, request: Request) -> GoogleIdentity:
        assert self.identity is not None
        return self.identity


@pytest.fixture(scope="module")
def database_url() -> str:
    return os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")


@pytest.fixture(scope="module")
def engine(database_url: str) -> Iterator[Engine]:
    created = create_engine(database_url)
    yield created
    created.dispose()


@pytest.fixture
def master_key() -> MasterKey:
    return MasterKey.generate()


@pytest.fixture
def settings(database_url: str, master_key: MasterKey) -> WebSettings:
    return WebSettings(
        database_url=database_url,
        google_client_id="test-client-id",
        google_client_secret="test-client-secret",
        google_redirect_uri="https://testserver/auth/google/callback",
        oauth_state_secret="0" * 43,
        master_key=master_key,
        session_ttl=dt.timedelta(days=14),
        session_touch_after=dt.timedelta(minutes=5),
        insecure_cookies=False,
        validate_api_keys=False,
    )


@pytest.fixture
def google() -> StubGoogle:
    return StubGoogle()


@pytest.fixture
def subs() -> list[str]:
    return []


@pytest.fixture
def client(
    settings: WebSettings, google: StubGoogle, engine: Engine, subs: list[str]
) -> Iterator[TestClient]:
    app = create_app(settings, identity_provider=google)
    with TestClient(app, base_url="https://testserver") as test_client:
        yield test_client
    with engine.begin() as conn:
        conn.execute(delete(users_table).where(users_table.c.google_sub.in_(subs)))


def sign_in(client: TestClient, google: StubGoogle, subs: list[str], engine: Engine) -> uuid.UUID:
    identity = GoogleIdentity(
        sub=f"test-sub-{uuid.uuid4()}", email=f"{uuid.uuid4()}@test.invalid", display_name="T"
    )
    subs.append(identity.sub)
    google.identity = identity
    assert client.get("/auth/google/callback").status_code == 200
    with engine.begin() as conn:
        user_id = conn.execute(
            select(users_table.c.id).where(users_table.c.google_sub == identity.sub)
        ).scalar_one()
    return user_id


def csrf(client: TestClient, path: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None, f"no CSRF token on {path}"
    return match.group(1)


def add_application(
    client: TestClient, *, job_ad: str = "Senior Engineer\n\nAcme. Own the deployment pipeline."
) -> str:
    response = client.post(
        "/applications",
        data={"csrf_token": csrf(client, "/applications/new"), "job_ad": job_ad, "url": ""},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    return response.headers["location"].rsplit("/", 1)[-1]


def get_job_id(engine: Engine, user_id: uuid.UUID, application_id: str) -> uuid.UUID:
    with engine.begin() as conn:
        detail = PostgresApplicationRepository(conn, user_id).get_application(
            uuid.UUID(application_id)
        )
    assert detail is not None and detail.application.job_id is not None
    return detail.application.job_id


def add_requirements(
    engine: Engine, user_id: uuid.UUID, job_id: uuid.UUID, texts: list[str]
) -> list[JobRequirement]:
    with engine.begin() as conn:
        job_repo = PostgresJobRepository(conn)
        requirements = [
            JobRequirement(
                id=requirement_id(job_id, text),
                user_id=user_id,
                job_id=job_id,
                ordinal=i,
                text=text,
                necessity="essential",
            )
            for i, text in enumerate(texts)
        ]
        job_repo.replace_requirements(user_id, job_id, requirements)
    return requirements


def add_coverage(engine: Engine, user_id: uuid.UUID, requirements: list[JobRequirement]) -> None:
    with engine.begin() as conn:
        job_repo = PostgresJobRepository(conn)
        for requirement in requirements:
            job_repo.record_coverage(
                RequirementCoverage(
                    user_id=user_id,
                    requirement_id=requirement.id,
                    trace_id=uuid.uuid4(),
                    status="evidenced",
                    cited_span_ids=[],
                    evidence_note="Traces to the corpus.",
                )
            )


def enqueued_tasks(engine: Engine, user_id: uuid.UUID, kind: str) -> list[object]:
    with engine.begin() as conn:
        return list(
            conn.execute(
                select(tasks_table).where(
                    tasks_table.c.user_id == user_id, tasks_table.c.kind == kind
                )
            ).all()
        )


CITED_SPAN_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
CITED_FACT_TEXT = "Led the platform team at Acme: eight engineers, hiring and on-call."


def add_cited_fact(engine: Engine, user_id: uuid.UUID) -> None:
    """The span the fake gate result cites, so the page can show its words.

    Without it the citation resolves to nothing, which is the *other* case the
    screen has to handle -- a fact the user has since changed.
    """
    with engine.begin() as conn:
        conn.execute(
            insert(spans_table).values(
                id=CITED_SPAN_ID,
                user_id=user_id,
                document_id=None,
                provenance="adjudicated",
                kind="paragraph",
                section_path=None,
                ordinal=0,
                text=CITED_FACT_TEXT,
                content_hash="0" * 64,
                char_start=None,
                char_end=None,
                retired_at=None,
            )
        )


def mark_task_succeeded(engine: Engine, task_id: uuid.UUID) -> None:
    with engine.begin() as conn:
        conn.execute(
            update(tasks_table).where(tasks_table.c.id == task_id).values(status="succeeded")
        )


_GATE_RESULT = {
    "sentences": [
        {
            "index": 1,
            "kind": "title",
            "verdict": None,
            "drift_label": None,
            "cited_span_ids": [],
            "evidence_note": "Document title: not checked against the corpus.",
            "rule_flags": [],
            "text": "CV bullets for Acme",
        },
        {
            "index": 2,
            "kind": "claim",
            "verdict": "supported",
            "drift_label": "supported",
            "cited_span_ids": ["11111111-1111-1111-1111-111111111111"],
            "evidence_note": "Traces cleanly to the corpus.",
            "rule_flags": [],
            "text": "Led the platform team at Acme.",
        },
        {
            "index": 3,
            "kind": "claim",
            "verdict": "review",
            "drift_label": "scope_inflation",
            "cited_span_ids": [],
            "evidence_note": "The corpus does not state the team's size.",
            "rule_flags": [],
            "text": "Managed a large engineering organisation.",
        },
        {
            "index": 4,
            "kind": "claim",
            "verdict": "unsupported",
            "drift_label": "invented_quantity",
            "cited_span_ids": [],
            "evidence_note": "The corpus names no such figure.",
            "rule_flags": [],
            "text": "Cut costs by 40%.",
        },
        {
            "index": 5,
            "kind": "framing",
            "verdict": "supported",
            "drift_label": "framing",
            "cited_span_ids": [],
            "evidence_note": "Framing is never checked.",
            "rule_flags": [],
            "text": "Motivated by a desire to build lasting systems.",
        },
    ]
}


def add_draft(
    engine: Engine,
    user_id: uuid.UUID,
    job_id: uuid.UUID,
    *,
    trace_id: uuid.UUID,
    cost: Decimal | None = Decimal("0.4123"),
) -> Draft:
    draft = Draft(
        user_id=user_id,
        job_id=job_id,
        kind="cv_bullets",
        text="# CV bullets for Acme\n\nLed the platform team at Acme.",
        gate_result=_GATE_RESULT,
        trace_id=trace_id,
    )
    with engine.begin() as conn:
        PostgresJobRepository(conn).record_draft(draft)
        if cost is not None:
            PostgresRunRepository(conn).record(
                RunRecord(
                    user_id=user_id,
                    trace_id=trace_id,
                    component="generate",
                    stage="draft",
                    model="claude-opus-5",
                    cost_usd=cost,
                    outcome="ok",
                    started_at=dt.datetime.now(dt.UTC),
                )
            )
    return draft


def settle_first_read(
    engine: Engine, user_id: uuid.UUID, application_id: str, *, read: bool
) -> None:
    """Finish the read `POST /applications` queued, as the worker would have.

    `read=True` records it as done (the tests add the requirements
    themselves); `read=False` records it as failed, which leaves the ad unread
    and the "Read the ad" step still to do.
    """
    with engine.begin() as conn:
        conn.execute(
            update(tasks_table)
            .where(
                tasks_table.c.user_id == user_id,
                tasks_table.c.kind == EXTRACT_JOB_AD_KIND,
                tasks_table.c.status == "pending",
            )
            .values(status="succeeded" if read else "failed")
        )
        repo = PostgresApplicationRepository(conn, user_id)
        if read:
            repo.finish_extraction(uuid.UUID(application_id), title=None, employer=None)
        else:
            repo.fail_extraction(uuid.UUID(application_id), "model_error")


def ready_to_write(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> tuple[uuid.UUID, str, uuid.UUID]:
    """Signed in, ad read, requirements checked: only "Write the CV" is left."""
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client)
    settle_first_read(engine, user_id, app_id, read=True)
    job_id = get_job_id(engine, user_id, app_id)
    requirements = add_requirements(engine, user_id, job_id, ["5+ years of Python"])
    add_coverage(engine, user_id, requirements)
    return user_id, app_id, job_id


def step(page: str, key: str) -> str:
    """The `<li>` for one of the three steps -- "ad", "check" or "write" -- by
    its position, since the three always render in that order."""
    items = re.findall(r'<li class="step step-[a-z]+".*?</li>', page, re.S)
    assert len(items) == 3, f"expected three steps, found {len(items)}"
    return items[{"ad": 0, "check": 1, "write": 2}[key]]


def text_of(page: str) -> str:
    """What a reader sees: tags gone, entities decoded, whitespace folded."""
    no_scripts = re.sub(r"<(script|style)\b.*?</\1>", " ", page, flags=re.S)
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", no_scripts)).split())


def run_follow_up(engine: Engine, user_id: uuid.UUID, task_id: uuid.UUID) -> None:
    """Mark a task succeeded and queue its next step the way its handler does
    -- the real `jfl_worker.chain.queue_next`, with no model anywhere."""
    mark_task_succeeded(engine, task_id)
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user_id).get_task(task_id)
    assert task is not None
    queue_next(TaskContext(task=task, engine=engine, now=dt.datetime.now(dt.UTC)))


# --------------------------------------------------------------------------
# Where it starts: an action on the application page
# --------------------------------------------------------------------------


def test_a_signed_out_visitor_is_redirected_to_login(client: TestClient) -> None:
    random_id = uuid.uuid4()
    response = client.get(f"/applications/{random_id}/drafts", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_the_application_page_offers_write_the_cv_as_an_action(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """Not a link to another page: the button, and the steps, where the
    person already is."""
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client)
    settle_first_read(engine, user_id, app_id, read=False)

    page = client.get(f"/applications/{app_id}").text
    assert "CV for this job" in page
    assert f'action="/applications/{app_id}/drafts"' in page
    assert ">Write the CV</button>" in page
    assert "Read the ad" in page and "Check it against your facts" in page
    # Ahead of the ad panel, not after the score and the questions.
    assert page.index("CV for this job") < page.index('id="extraction"')


def test_the_application_page_shows_the_latest_cvs_headline(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id, app_id, job_id = ready_to_write(client, google, subs, engine)
    add_draft(engine, user_id, job_id, trace_id=uuid.uuid4())

    page = text_of(client.get(f"/applications/{app_id}").text)
    assert HEADLINE in page
    assert "Read it, copy it or download it" in page


# --------------------------------------------------------------------------
# What it needs first: one sequence, the current step obvious, each step costed
# --------------------------------------------------------------------------


def test_an_unread_ad_makes_reading_it_the_next_step_and_costs_the_whole_chain(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client)
    settle_first_read(engine, user_id, app_id, read=False)

    page = client.get(f"/applications/{app_id}/drafts").text
    assert "step-current" in step(page, "ad") and "about $0.01" in step(page, "ad")
    assert "step-waiting" in step(page, "check") and "about $0.16" in step(page, "check")
    assert "step-waiting" in step(page, "write") and "about $0.35–0.65" in step(page, "write")
    assert (
        "it will read the ad, check it against your confirmed facts and then write the CV "
        "-- about $0.52–0.82 in all, on your own API key"
    ) in text_of(page)
    assert ">Write the CV</button>" in page
    assert "hx-trigger" not in page


def test_an_ad_being_read_shows_that_step_running_and_waits(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """Straight after adding an application its ad is being read. The page
    says so, polls, and offers no button -- a press now would pay to read the
    same ad twice."""
    sign_in(client, google, subs, engine)
    app_id = add_application(client)

    page = client.get(f"/applications/{app_id}/drafts").text
    assert "step-running" in step(page, "ad")
    assert "Reading the ad now" in page
    assert ">Write the CV</button>" not in page
    assert 'hx-trigger="every 3s"' in page


def test_requirements_without_the_check_make_the_check_the_next_step(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client)
    settle_first_read(engine, user_id, app_id, read=True)
    job_id = get_job_id(engine, user_id, app_id)
    add_requirements(engine, user_id, job_id, ["5+ years of Python"])

    page = client.get(f"/applications/{app_id}/drafts").text
    assert "step-done" in step(page, "ad") and "✓" in step(page, "ad")
    assert "step-current" in step(page, "check")
    assert "step-waiting" in step(page, "write")
    assert (
        "it will check it against your confirmed facts and then write the CV "
        "-- about $0.51–0.81 in all"
    ) in text_of(page)
    # The requirements are still there, folded behind their count.
    assert "5+ years of Python" in page
    assert "not checked yet" in page


def test_with_everything_ready_writing_is_the_only_step_left(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    _, app_id, _ = ready_to_write(client, google, subs, engine)

    page = client.get(f"/applications/{app_id}/drafts").text
    assert "step-done" in step(page, "ad") and "step-done" in step(page, "check")
    assert "step-current" in step(page, "write")
    assert "Writing it costs about $0.35–0.65 on your own API key" in text_of(page)
    assert ">Write the CV</button>" in page
    assert ">Write a cover letter</button>" in page


# --------------------------------------------------------------------------
# One button runs the chain -- each missing step queued once
# --------------------------------------------------------------------------


def press(client: TestClient, app_id: str, kind: str = "cv_bullets") -> str:
    response = client.post(
        f"/applications/{app_id}/drafts",
        data={"csrf_token": csrf(client, f"/applications/{app_id}/drafts"), "kind": kind},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    return response.headers["location"]


def unfinished(engine: Engine, user_id: uuid.UUID, kind: str) -> list[Any]:
    return [t for t in enqueued_tasks(engine, user_id, kind) if t.status == "pending"]


def test_one_press_with_the_ad_unread_queues_the_read_with_the_rest_behind_it(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client)
    settle_first_read(engine, user_id, app_id, read=False)
    job_id = get_job_id(engine, user_id, app_id)

    location = press(client, app_id)
    reads = unfinished(engine, user_id, EXTRACT_JOB_AD_KIND)
    assert len(reads) == 1
    assert location == f"/applications/{app_id}/drafts?task={reads[0].id}"
    assert reads[0].payload == {
        "application_id": app_id,
        "then": [
            {"kind": GENERATE_COVERAGE_KIND, "payload": {"job_id": str(job_id)}},
            {
                "kind": GENERATE_CV_DRAFT_KIND,
                "payload": {"application_id": app_id, "kind": "cv_bullets"},
            },
        ],
    }
    # The later steps are queued by the worker as each one finishes, not now.
    assert enqueued_tasks(engine, user_id, GENERATE_COVERAGE_KIND) == []
    assert enqueued_tasks(engine, user_id, GENERATE_CV_DRAFT_KIND) == []

    # Pressing again while it runs queues nothing and shows the same press.
    assert press(client, app_id) == location
    assert len(unfinished(engine, user_id, EXTRACT_JOB_AD_KIND)) == 1


def test_one_press_follows_the_chain_through_each_step_once(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """Coverage missing: one press queues the check with the CV behind it; the
    worker's `queue_next` queues the CV once, however often it is called; and
    the page follows the press from the check to the finished CV."""
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client)
    settle_first_read(engine, user_id, app_id, read=True)
    job_id = get_job_id(engine, user_id, app_id)
    requirements = add_requirements(engine, user_id, job_id, ["5+ years of Python"])

    location = press(client, app_id)
    checks = enqueued_tasks(engine, user_id, GENERATE_COVERAGE_KIND)
    assert len(checks) == 1
    assert checks[0].payload == {
        "job_id": str(job_id),
        "then": [
            {
                "kind": GENERATE_CV_DRAFT_KIND,
                "payload": {"application_id": app_id, "kind": "cv_bullets"},
            }
        ],
    }
    assert enqueued_tasks(engine, user_id, GENERATE_CV_DRAFT_KIND) == []

    running = client.get(location).text
    assert "step-running" in step(running, "check")
    assert "step-waiting" in step(running, "write")
    assert ">Write the CV</button>" not in running
    assert 'hx-trigger="every 3s"' in running

    # The check finishes; its handler queues the CV -- and a redelivery of the
    # same task does not queue a second one.
    add_coverage(engine, user_id, requirements)
    run_follow_up(engine, user_id, checks[0].id)
    run_follow_up(engine, user_id, checks[0].id)
    drafts = enqueued_tasks(engine, user_id, GENERATE_CV_DRAFT_KIND)
    assert len(drafts) == 1
    assert drafts[0].payload == {
        "application_id": app_id,
        "kind": "cv_bullets",
        "after": str(checks[0].id),
    }

    poll_url = f"/applications/{app_id}/drafts/steps?task={checks[0].id}"
    polled = client.get(poll_url)
    assert "step-done" in step(polled.text, "check")
    assert "step-running" in step(polled.text, "write")
    assert "HX-Refresh" not in polled.headers

    # The CV lands: the poll asks for a reload, and the page leads with it.
    mark_task_succeeded(engine, drafts[0].id)
    add_draft(engine, user_id, job_id, trace_id=drafts[0].id)
    finished = client.get(poll_url)
    assert finished.headers.get("HX-Refresh") == "true"
    assert "hx-trigger" not in finished.text

    page = client.get(location).text
    assert HEADLINE in text_of(page)
    assert page.index("draft-headline") < page.index('id="cv-steps"')


def test_a_press_with_everything_ready_queues_only_the_cv(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id, app_id, _ = ready_to_write(client, google, subs, engine)

    location = press(client, app_id)
    assert location.startswith(f"/applications/{app_id}/drafts?task=")
    assert enqueued_tasks(engine, user_id, GENERATE_COVERAGE_KIND) == []
    tasks = enqueued_tasks(engine, user_id, GENERATE_CV_DRAFT_KIND)
    assert len(tasks) == 1
    assert tasks[0].payload == {"application_id": app_id, "kind": "cv_bullets"}

    # Loading the page again enqueues nothing further.
    client.get(location)
    assert len(enqueued_tasks(engine, user_id, GENERATE_CV_DRAFT_KIND)) == 1


def test_check_again_still_queues_one_check(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id, app_id, job_id = ready_to_write(client, google, subs, engine)
    page = client.get(f"/applications/{app_id}/drafts").text
    assert "Check against your facts again" in page

    response = client.post(
        f"/applications/{app_id}/drafts/coverage",
        data={"csrf_token": csrf(client, f"/applications/{app_id}/drafts")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    tasks = enqueued_tasks(engine, user_id, GENERATE_COVERAGE_KIND)
    assert len(tasks) == 1
    assert tasks[0].payload == {"job_id": str(job_id)}


def test_every_post_requires_a_csrf_token(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id, app_id, _ = ready_to_write(client, google, subs, engine)

    assert (
        client.post(
            f"/applications/{app_id}/drafts/coverage", data={"csrf_token": "wrong"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"/applications/{app_id}/drafts", data={"csrf_token": "wrong", "kind": "cv_bullets"}
        ).status_code
        == 403
    )
    assert enqueued_tasks(engine, user_id, GENERATE_COVERAGE_KIND) == []
    assert enqueued_tasks(engine, user_id, GENERATE_CV_DRAFT_KIND) == []


# --------------------------------------------------------------------------
# What they get back: the CV first, the headline, the check beside it
# --------------------------------------------------------------------------

# `_GATE_RESULT`: one supported claim, one to check, one unsupported, and a
# title and a framing sentence that are never checked.
HEADLINE = "1 of 3 claims traces to your confirmed facts; 1 needs checking; 1 isn't supported."


def test_the_result_leads_with_the_cv_and_the_headline(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id, app_id, job_id = ready_to_write(client, google, subs, engine)
    add_cited_fact(engine, user_id)
    add_draft(engine, user_id, job_id, trace_id=uuid.uuid4(), cost=Decimal("0.4123"))

    page = client.get(f"/applications/{app_id}/drafts").text
    assert HEADLINE in text_of(page)
    # Headline, then the text itself, then the check -- and all of it before
    # the steps panel, which now reads as "write another".
    headline_at = page.index("draft-headline")
    text_at = page.index('class="draft-text"')
    check_at = page.index("Needs your attention")
    assert headline_at < text_at < check_at < page.index('id="cv-steps"')
    assert "Led the platform team at Acme." in page
    assert "$0.4123" in page

    # A citation shows the fact's own words, never its id.
    assert CITED_FACT_TEXT in page
    assert str(CITED_SPAN_ID) not in page

    # Each mark is explained where it is shown, in plain words.
    words = text_of(page)
    for meaning in (
        "Supported traces to a fact you confirmed",
        "Check this only partly traces, or says more",
        "Not supported nothing you confirmed backs it",
        "Not checked framing or a heading, never compared",
    ):
        assert meaning in words


def _item_containing(page: str, text: str) -> str:
    items = re.findall(r"<li>\s*<span class=\"status-badge.*?</li>", page, re.S)
    found = [item for item in items if text in item]
    assert found, f"no sentence item containing {text!r}"
    return found[0]


def test_each_flagged_sentence_carries_its_next_action(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id, app_id, job_id = ready_to_write(client, google, subs, engine)
    add_draft(engine, user_id, job_id, trace_id=uuid.uuid4())
    page = client.get(f"/applications/{app_id}/drafts").text

    unsupported = html.unescape(_item_containing(page, "Cut costs by 40%."))
    assert "Not supported" in unsupported
    assert "This number does not appear in anything you have confirmed." in unsupported
    assert "What to do:" in unsupported
    assert "Reword it to what you can show, or cut it" in unsupported
    assert 'href="/background/facts"' in unsupported
    assert 'href="/background"' in unsupported

    review = html.unescape(_item_containing(page, "Managed a large engineering organisation."))
    assert "Check this" in review
    assert "do not state the scope" in review
    assert "What to do:" in review and 'href="/background/facts"' in review

    # Worst first: the unsupported claim is listed before the one to check.
    assert page.index("Cut costs by 40%.") < page.index("Managed a large engineering organisation.")


def test_framing_is_shown_as_not_checked_never_as_supported(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id, app_id, job_id = ready_to_write(client, google, subs, engine)
    add_draft(engine, user_id, job_id, trace_id=uuid.uuid4())
    page = client.get(f"/applications/{app_id}/drafts").text

    for text in ("Motivated by a desire to build lasting systems.", "CV bullets for Acme"):
        item = _item_containing(page, text)
        assert "verdict-not-checked" in item and "Not checked" in item
        assert "verdict-supported" not in item
        assert ">Supported<" not in item


def test_older_drafts_are_folded_and_the_newest_is_open(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id, app_id, job_id = ready_to_write(client, google, subs, engine)
    older = add_draft(engine, user_id, job_id, trace_id=uuid.uuid4())
    newer = add_draft(engine, user_id, job_id, trace_id=uuid.uuid4())
    page = client.get(f"/applications/{app_id}/drafts").text

    def opening_tag(draft_id: uuid.UUID) -> str:
        match = re.search(rf'<details[^>]*data-section="draft\.{draft_id}"[^>]*>', page)
        assert match is not None
        return match.group(0)

    assert re.search(r"\sopen>$", opening_tag(newer.id))
    assert not re.search(r"\sopen>$", opening_tag(older.id))
    assert "Earlier versions" in page
    # Folded, not missing: its whole text is still in the page.
    assert page.count("Cut costs by 40%.") == 2


def test_copy_works_without_javascript_and_the_text_downloads(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id, app_id, job_id = ready_to_write(client, google, subs, engine)
    draft = add_draft(engine, user_id, job_id, trace_id=uuid.uuid4())
    page = client.get(f"/applications/{app_id}/drafts").text

    # The no-JS copy path: the whole text in a readonly box.
    box = re.search(
        r'<textarea id="draft-text-[^"]+" class="draft-text" readonly[^>]*>(.*?)</textarea>',
        page,
        re.S,
    )
    assert box is not None
    assert html.unescape(box.group(1)) == draft.text
    # The button appears only when the script that makes it work is loaded.
    assert re.search(r'<button type="button" class="copy-draft"[^>]*\bhidden\b', page)
    assert "copy.js" in page

    download_url = f"/applications/{app_id}/drafts/{draft.id}/download"
    assert f'href="{download_url}"' in page
    response = client.get(download_url)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.headers["content-disposition"].startswith('attachment; filename="cv-')
    assert response.text == draft.text


_JARGON = re.compile(r"\b(gate|span|spans|grounded|grounding|coverage|corpus|draft kind)\b", re.I)


def test_the_drafting_screens_use_no_internal_words(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The rendered pages, not just the templates: what the Python helpers put
    on screen is swept too. The draft's own text and the checking model's
    verbatim notes are the two things left out -- neither is ours to reword."""
    user_id, app_id, job_id = ready_to_write(client, google, subs, engine)
    add_draft(engine, user_id, job_id, trace_id=uuid.uuid4())

    for path in (f"/applications/{app_id}/drafts", f"/applications/{app_id}"):
        page = client.get(path).text
        page = re.sub(r'<p class="note checker-note">.*?</p>', " ", page, flags=re.S)
        page = re.sub(r"<textarea.*?</textarea>", " ", page, flags=re.S)
        words = text_of(page)
        if path.endswith("/drafts"):
            found = _JARGON.findall(words)
        else:
            # The application page has other panels with their own history;
            # only the CV panel is this feature's.
            panel = words[words.index("CV for this job") : words.index("The ad")]
            found = _JARGON.findall(panel)
        assert not found, f"{path} shows {found}"


# --------------------------------------------------------------------------
# Failures, unknown ids, tenancy
# --------------------------------------------------------------------------


def test_a_failed_step_shows_a_plain_sentence_never_the_raw_error(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client)
    settle_first_read(engine, user_id, app_id, read=True)
    job_id = get_job_id(engine, user_id, app_id)
    add_requirements(engine, user_id, job_id, ["5+ years of Python"])

    location = press(client, app_id)
    task_id = uuid.UUID(location.rsplit("task=", 1)[-1])
    with engine.begin() as conn:
        conn.execute(
            update(tasks_table)
            .where(tasks_table.c.id == task_id)
            .values(
                status="failed",
                last_error="PermanentTaskError: coverage generation failed permanently: no_api_key",
            )
        )

    page = client.get(location).text
    assert "step-failed" in step(page, "check")
    assert "stopped here" in page
    assert "own Anthropic API key" in page
    assert 'href="/settings"' in page
    assert "PermanentTaskError" not in page
    assert "no_api_key" not in page
    # And it can be tried again.
    assert ">Write the CV</button>" in page


def test_an_unknown_or_cross_user_task_id_shows_nothing_special_inline(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """`?task=` naming a task that is not this application's is treated as
    "nothing to show" on the main page -- it must never leak another user's
    task, and it must never 500.
    """
    _, app_id, _ = ready_to_write(client, google, subs, engine)

    page = client.get(f"/applications/{app_id}/drafts?task={uuid.uuid4()}")
    assert page.status_code == 200
    assert "hx-trigger" not in page.text
    assert client.get(f"/applications/{app_id}/drafts?task=not-a-uuid").status_code == 200


def test_the_standalone_fragment_route_404s_for_an_unknown_task(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    app_id = add_application(client)

    response = client.get(f"/applications/{app_id}/drafts/tasks/{uuid.uuid4()}")
    assert response.status_code == 404


def test_another_users_application_404s_on_every_route(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    alice_id, alice_app_id, alice_job_id = ready_to_write(client, google, subs, engine)
    alice_draft = add_draft(engine, alice_id, alice_job_id, trace_id=uuid.uuid4())

    bob_id = sign_in(client, google, subs, engine)
    assert client.get(f"/applications/{alice_app_id}/drafts").status_code == 404
    assert client.get(f"/applications/{alice_app_id}/drafts/steps").status_code == 404
    assert (
        client.get(f"/applications/{alice_app_id}/drafts/{alice_draft.id}/download").status_code
        == 404
    )
    assert (
        client.post(
            f"/applications/{alice_app_id}/drafts/coverage",
            data={"csrf_token": csrf(client, "/applications")},
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/applications/{alice_app_id}/drafts",
            data={"csrf_token": csrf(client, "/applications"), "kind": "cv_bullets"},
        ).status_code
        == 404
    )
    assert (
        client.get(f"/applications/{alice_app_id}/drafts/tasks/{uuid.uuid4()}").status_code == 404
    )
    assert enqueued_tasks(engine, bob_id, GENERATE_CV_DRAFT_KIND) == []

    # Bob's own application cannot reach Alice's draft by its id either.
    bob_app_id = add_application(client, job_ad="Bob's role\n\nAcme.")
    assert (
        client.get(f"/applications/{bob_app_id}/drafts/{alice_draft.id}/download").status_code
        == 404
    )
