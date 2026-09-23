"""Screenshot every main page at three widths, against fictional layout data.

A dev tool, not a test: the layout tests assert on markup because pytest has no
browser, and every overflow bug so far was found by the owner's screenshot
instead. This renders the real templates through the real app and lets a person
(or an agent) *look*.

What it does:

  1. creates the app with the stub identity provider the integration tests use
     and signs in a throwaway user (`google_sub` = `screenshot-harness-<uuid>`);
  2. seeds that user, in the database named by `JFL_DATABASE_URL`, with
     invented data chosen to stress layout -- ~15 applications with long and
     short titles, one titled only by a long URL, both scores, unscored, a
     failed and a pending score, one archived; watched boards with long names
     and many jobs, one shown by its URL; a changes feed with new and closed
     jobs; three CVs (read, unread, failed). Not seeded: drafts, profile
     answers, confirmed facts -- those pages render their empty states;
  3. renders each page through the test client and writes it as a standalone
     HTML file with the stylesheet inlined (so it renders from `file://`);
  4. screenshots each file with headless Firefox at 1440, 1024 and 390 px wide;
  5. deletes the user (every table cascades from `users`), even on failure.

Everything seeded is fiction -- layout test data, never corpus, never a
golden-set item. No model call and no network: scores and extractions are
written straight through the repositories, as the integration tests do.

Run it against a scratch database, never the default one:

    JFL_DATABASE_URL=postgresql+psycopg://jfl:jfl@localhost:5433/jfl_applist \\
        uv run python scripts/screenshot_pages.py --out /path/to/shots

    # just some pages, or just some widths, or HTML only:
    ... --pages applications,jobs --widths 1440,390
    ... --no-screenshots

Output: `<out>/<page>.html` and `<out>/<page>-<width>.png`. Firefox captures
the window, not the whole document, so `--height` (default 1600) decides how
much of a long page is seen. A page *wider* than the window shows its overflow
as a clipped right edge or a sideways scrollbar, which is the thing to look
for.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal

from fastapi import Request
from fastapi.responses import RedirectResponse, Response
from fastapi.testclient import TestClient
from jfl_core.crypto.envelope import MasterKey
from jfl_core.db.tables import applications as applications_table
from jfl_core.db.tables import users as users_table
from jfl_core.ids import requirement_id
from jfl_core.models import JobRequirement, ObservedJob, Workplace
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_core.storage.postgres import PostgresJobRepository
from jfl_core.storage.scores import PostgresScoreRepository
from jfl_core.storage.sent_documents import PostgresSentDocumentRepository
from jfl_intake.adapters.base import FetchResult
from jfl_intake.engine import REPOST_WINDOW, plan_check
from jfl_intake.normalise import fingerprint
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity, OAuthError
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, select, update
from sqlalchemy.engine import Engine

ROOT = pathlib.Path(__file__).resolve().parents[1]
STYLESHEET = ROOT / "packages/web/src/jfl_web/static/style.css"
WIDTHS = (1440, 1024, 390)
HEIGHT = 1600


class StubGoogle:
    """The integration tests' identity provider: no Google, no network."""

    def __init__(self, identity: GoogleIdentity) -> None:
        self.identity = identity

    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response:
        return RedirectResponse("/auth/google/callback", status_code=303)

    async def fetch_identity(self, request: Request) -> GoogleIdentity:
        if self.identity is None:
            raise OAuthError("no identity")
        return self.identity


# -- the fictional data ---------------------------------------------------------

NOW = dt.datetime.now(dt.UTC)

# (title, employer, status, could, want, score state, age)
# score state: "done", "none", "failed", "pending"; could/want None = no number.
APPLICATIONS: list[tuple[str, str | None, str, int | None, int | None, str, dt.timedelta]] = [
    (
        "Senior Engineering Manager, Developer Productivity and Platform Reliability",
        "Northwind Imaginary Logistics",
        "interviewing",
        7,
        8,
        "done",
        dt.timedelta(minutes=12),
    ),
    ("CTO", "Tiny Fictional Co", "interested", 3, 9, "done", dt.timedelta(hours=3)),
    (
        "Head of Engineering",
        "Contoso Pretend Holdings International",
        "applied",
        8,
        1,
        "done",
        dt.timedelta(hours=9),
    ),
    (
        "https://careers.example.invalid/jobs/engineering-manager-payments-infrastructure-"
        "remote-uk-ref-000123456789",
        None,
        "interested",
        None,
        None,
        "none",
        dt.timedelta(days=1),
    ),
    (
        "Engineering Manager",
        "Fabrikam Imaginary",
        "screening",
        6,
        6,
        "done",
        dt.timedelta(days=1, hours=5),
    ),
    (
        "Staff Engineer (Python), Data Platform",
        "Wingtip Toys of Nowhere",
        "rejected",
        5,
        None,
        "done",
        dt.timedelta(days=2),
    ),
    ("Director of Engineering", "Litware", "offer", 9, 7, "done", dt.timedelta(days=3)),
    (
        "Engineering Lead — Machine Learning Infrastructure & Evaluation Tooling",
        "Adventure Works Made-Up Research Laboratory",
        "interested",
        None,
        None,
        "failed",
        dt.timedelta(days=4),
    ),
    (
        "VP Engineering",
        "Proseware",
        "withdrawn",
        4,
        2,
        "done",
        dt.timedelta(days=6),
    ),
    (
        "Principal Engineer",
        "Tailspin Invented",
        "applied",
        None,
        None,
        "pending",
        dt.timedelta(days=8),
    ),
    ("EM", "A. Datum", "interested", 10, 10, "done", dt.timedelta(days=11)),
    (
        "Engineering Manager, Payments",
        "Coho Imaginary Winery and Distributed Systems Consultancy",
        "screening",
        2,
        8,
        "done",
        dt.timedelta(days=15),
    ),
    (
        "Software Development Manager II",
        None,
        "interested",
        None,
        None,
        "none",
        dt.timedelta(days=20),
    ),
    (
        "Group Engineering Manager",
        "Humongous Fictional Insurance",
        "applied",
        7,
        3,
        "done",
        dt.timedelta(days=31),
    ),
    (
        "Technical Program Manager",
        "Lucerne Pretend Publishing",
        "rejected",
        1,
        4,
        "done",
        dt.timedelta(days=45),
    ),
]

ARCHIVED = ("Archived: Engineering Manager, Old Role", "Blue Yonder Invented Airlines")

REQUIREMENTS = [
    ("Eight or more years leading engineering teams of 10+ people", "essential"),
    ("Hands-on Python and distributed systems experience", "essential"),
    ("Experience running an on-call rotation for a customer-facing platform", "essential"),
    ("A track record of hiring and growing senior engineers", "essential"),
    ("Familiarity with Kubernetes, Terraform and observability tooling", "desirable"),
    ("Payments or fintech domain knowledge", "desirable"),
]

AD_TEXT = (
    "Senior Engineering Manager, Developer Productivity and Platform Reliability\n\n"
    "Northwind Imaginary Logistics is a fictional company used only as layout data.\n"
    + "\n".join(text for text, _ in REQUIREMENTS)
)

BOARD_TITLES = [
    "Engineering Manager, Platform",
    "Senior Engineering Manager, Data Infrastructure and Machine Learning Operations",
    "Staff Software Engineer",
    "Engineering Manager, Inference",
    "Head of Developer Experience",
    "Product Designer",
    "Engineering Manager, Payments Risk",
    "Site Reliability Engineer",
    "Account Executive, Enterprise (EMEA)",
    "Engineering Manager, Remote",
]
LOCATIONS: list[tuple[tuple[str, ...], Workplace]] = [
    (("London, UK",), "remote"),
    (("Remote - United Kingdom", "Remote - Ireland", "Remote - Netherlands"), "remote"),
    (("Edinburgh, UK",), "hybrid"),
    (("San Francisco, CA; New York, NY; Seattle, WA",), "onsite"),
    (("Berlin, Germany",), "unknown"),
]


def _job(ext: str, title: str, index: int) -> ObservedJob:
    locations, workplace = LOCATIONS[index % len(LOCATIONS)]
    location = "; ".join(locations)
    return ObservedJob(
        external_id=ext,
        title=title,
        location=location,
        url=f"https://example.invalid/jobs/{ext}",
        fingerprint=fingerprint(title, location),
        workplace=workplace,
        locations=locations,
    )


def _jobs(prefix: str, count: int) -> list[ObservedJob]:
    return [
        _job(f"{prefix}{i}", f"{BOARD_TITLES[i % len(BOARD_TITLES)]} ({i})", i)
        for i in range(count)
    ]


def _apply(
    repo: PostgresBoardRepository, board_id: uuid.UUID, jobs: list[ObservedJob], at: dt.datetime
) -> None:
    state = repo.lock_check_state(
        board_id,
        observed_external_ids=[j.external_id for j in jobs],
        closed_since=at - REPOST_WINDOW,
    )
    assert state is not None
    result = FetchResult(status="complete", jobs=tuple(jobs), expected_total=len(jobs))
    repo.apply_check_plan(plan_check(state, result, observed_at=at), started_at=at, finished_at=at)


def seed(engine: Engine, user_id: uuid.UUID) -> dict[str, uuid.UUID]:
    """Write the fictional data; return the ids the page list needs."""
    ids: dict[str, uuid.UUID] = {}
    with engine.begin() as conn:
        apps = PostgresApplicationRepository(conn, user_id)
        scores = PostgresScoreRepository(conn, user_id)
        for title, employer, status, could, want, state, age in APPLICATIONS:
            is_detail = "detail" not in ids and title.startswith("Senior Engineering Manager")
            app = apps.create_application(
                title=title,
                employer=employer,
                status=status,  # type: ignore[arg-type]
                raw_job_text=AD_TEXT if is_detail else None,
                url="https://example.invalid/jobs/1" if is_detail else None,
                notes="Recruiter said the loop is four rounds." if is_detail else None,
            )
            if is_detail:
                ids["detail"] = app.id
                apps.finish_extraction(app.id, title=None, employer=None)
                assert app.job_id is not None
                PostgresJobRepository(conn).replace_requirements(
                    user_id,
                    app.job_id,
                    [
                        JobRequirement(
                            id=requirement_id(app.job_id, text),
                            user_id=user_id,
                            job_id=app.job_id,
                            ordinal=i,
                            text=text,
                            necessity=necessity,  # type: ignore[arg-type]
                        )
                        for i, (text, necessity) in enumerate(REQUIREMENTS)
                    ],
                )
            if state in ("done", "failed", "pending"):
                row = scores.create_pending(app.id)
                if state == "done":
                    scores.mark_done(
                        row.id,
                        could_get_score=could,
                        could_get_assessment="Fictional layout data: assessed.",
                        want_it_score=want,
                        want_it_assessment="Fictional layout data: assessed.",
                        constraint_verdicts=[],
                        objective_verdicts=[],
                        hard_gate_breaches=[],
                        levers=[],
                        not_stated=[],
                        model="layout-fixture",
                        cost_usd=Decimal("0"),
                        trace_id=uuid.uuid4(),
                    )
                elif state == "failed":
                    scores.mark_failed(row.id, "model_error")
            conn.execute(
                update(applications_table)
                .where(applications_table.c.id == app.id)
                .values(updated_at=NOW - age, created_at=NOW - age - dt.timedelta(days=2))
            )
        archived = apps.create_application(title=ARCHIVED[0], employer=ARCHIVED[1])
        apps.archive(archived.id)

        # CVs for /background's table: read, unread and failed, one with a
        # long unbroken filename.
        cvs = PostgresSentDocumentRepository(conn, user_id)
        for n, (filename, outcome) in enumerate(
            [
                ("cv-2026-engineering-manager-platform-reliability-final-v7.pdf", "read"),
                ("cv.md", "pending"),
                ("Curriculum_Vitae_Imaginary_Person_Director_Of_Engineering_2019.pdf", "failed"),
            ]
        ):
            stored = cvs.add_cv(filename=filename, text=f"Fictional CV {n} for layout only.")
            if outcome == "read":
                cvs.finish_extraction(stored.id, facts_proposed=23)
            elif outcome == "failed":
                cvs.fail_extraction(stored.id, "model_error")

        boards = PostgresBoardRepository(conn, user_id)
        specs: list[tuple[str | None, str, int]] = [
            ("Northwind Imaginary Logistics — Global Engineering Careers", "northwind", 64),
            ("Contoso", "contoso", 12),
            (None, "adventureworksmaderesearchlaboratoryengineering", 23),
            ("Fabrikam Imaginary Holdings (UK & Ireland)", "fabrikam", 7),
        ]
        for label, token, count in specs:
            board = boards.add_board(
                platform="greenhouse",
                board_url=f"https://boards.greenhouse.io/{token}",
                board_key={"token": token},
                label=label,
            )
            jobs = _jobs(token[:4], count)
            # A baseline two days ago, then a check an hour ago that closed
            # the first three and opened four new ones -- the changes feed.
            _apply(boards, board.id, jobs, NOW - dt.timedelta(days=2))
            later = jobs[3:] + [
                _job(f"{token[:4]}n{i}", f"{BOARD_TITLES[i + 1]} — new", i) for i in range(4)
            ]
            _apply(boards, board.id, later, NOW - dt.timedelta(hours=1))
            ids.setdefault("board", board.id)
    return ids


# -- rendering ------------------------------------------------------------------

LINK = re.compile(r'<link rel="stylesheet" href="[^"]*style\.css[^"]*">')
SCRIPT = re.compile(r'<script src="[^"]*"( defer)?></script>')


def standalone(page: str) -> str:
    """The page with its stylesheet inlined and its scripts dropped, so it
    renders from `file://` exactly as the server's CSS lays it out. Scripts
    are enhancement only (htmx, the section toggle), and a file:// page could
    not reach them anyway."""
    css = STYLESHEET.read_text()
    page, found = LINK.subn(lambda _: f"<style>\n{css}\n</style>", page)
    if not found:
        raise SystemExit("no stylesheet <link> found -- has base.html changed?")
    return SCRIPT.sub("", page)


def screenshot(
    html: pathlib.Path, png: pathlib.Path, width: int, height: int, scratch: pathlib.Path
) -> None:
    profile = pathlib.Path(tempfile.mkdtemp(prefix="ffprofile-", dir=scratch))
    try:
        subprocess.run(
            [
                "firefox",
                "--headless",
                "--no-remote",
                "--profile",
                str(profile),
                "--screenshot",
                str(png.resolve()),
                f"--window-size={width},{height}",
                f"file://{html.resolve()}",
            ],
            check=True,
            timeout=90,
            capture_output=True,
        )
    finally:
        shutil.rmtree(profile, ignore_errors=True)


@contextmanager
def signed_in_client(database_url: str) -> Iterator[tuple[TestClient, Engine, uuid.UUID]]:
    engine = create_engine(database_url)
    sub = f"screenshot-harness-{uuid.uuid4()}"
    identity = GoogleIdentity(
        sub=sub, email=f"{uuid.uuid4()}@layout.invalid", display_name="Lay Out"
    )
    settings = WebSettings(
        database_url=database_url,
        google_client_id="layout-client-id",
        google_client_secret="layout-client-secret",
        google_redirect_uri="https://testserver/auth/google/callback",
        oauth_state_secret="0" * 43,
        master_key=MasterKey.generate(),
        session_ttl=dt.timedelta(days=1),
        session_touch_after=dt.timedelta(minutes=5),
        insecure_cookies=False,
        validate_api_keys=False,
    )
    app = create_app(settings, identity_provider=StubGoogle(identity))
    try:
        with TestClient(app, base_url="https://testserver") as client:
            assert client.get("/auth/google/callback").status_code == 200
            with engine.begin() as conn:
                user_id = conn.execute(
                    select(users_table.c.id).where(users_table.c.google_sub == sub)
                ).scalar_one()
            yield client, engine, user_id
    finally:
        with engine.begin() as conn:
            conn.execute(delete(users_table).where(users_table.c.google_sub == sub))
        engine.dispose()


def save_job_filter(client: TestClient) -> None:
    page = client.get("/jobs").text
    token = re.search(r'name="csrf_token" value="([^"]+)"', page)
    assert token is not None
    client.post(
        "/jobs/filter",
        data={
            "csrf_token": token.group(1),
            "workplace": ["remote", "hybrid", "unknown"],
            "title_includes": "engineering manager, head of, engineer",
        },
        follow_redirects=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--pages", default="", help="comma-separated page names; default all")
    parser.add_argument("--widths", default=",".join(map(str, WIDTHS)))
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--no-screenshots", action="store_true")
    args = parser.parse_args()

    database_url = os.environ.get("JFL_DATABASE_URL")
    if not database_url:
        print("set JFL_DATABASE_URL to a scratch database", file=sys.stderr)
        return 2
    out: pathlib.Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    widths = [int(w) for w in args.widths.split(",") if w]
    wanted = {p for p in args.pages.split(",") if p}

    with signed_in_client(database_url) as (client, engine, user_id):
        ids = seed(engine, user_id)
        save_job_filter(client)
        pages = {
            "applications": "/applications",
            "applications-by-could-get": "/applications?sort=could_get",
            "applications-archived": "/applications?archived=1",
            "jobs": "/jobs",
            "changes": "/changes",
            "boards": "/boards",
            "board": f"/boards/{ids['board']}",
            "profile": "/profile",
            "background": "/background",
            "application": f"/applications/{ids['detail']}",
            "application-cv": f"/applications/{ids['detail']}/drafts",
        }
        written = []
        for name, path in pages.items():
            if wanted and name not in wanted:
                continue
            response = client.get(path)
            if response.status_code != 200:
                print(f"{name}: {path} answered {response.status_code}", file=sys.stderr)
                continue
            html = out / f"{name}.html"
            html.write_text(standalone(response.text))
            written.append(html)
            if name == "applications":
                # The same list with the first row's Archive confirm open --
                # the state that was clipped past the table's edge.
                opened = out / "applications-archive-open.html"
                opened.write_text(
                    standalone(response.text).replace(
                        '<details class="confirm-remove row-archive">',
                        '<details class="confirm-remove row-archive" open>',
                        1,
                    )
                )
                written.append(opened)

    if args.no_screenshots:
        for html in written:
            print(html)
        return 0
    scratch = pathlib.Path(tempfile.mkdtemp(prefix=".profiles-", dir=out))
    try:
        for html in written:
            for width in widths:
                png = out / f"{html.stem}-{width}.png"
                screenshot(html, png, width, args.height, scratch)
                print(png)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
