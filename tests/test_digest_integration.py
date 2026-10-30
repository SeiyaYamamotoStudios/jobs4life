"""The emailed digest against real Postgres: the repository, the handlers end to
end, and the web controls. (PLAN.md C9.)

Needs `docker compose up -d` and `alembic upgrade head`. **No mail is ever sent**:
the handlers are given a recording `SmtpSender` subclass, and underneath that the
root `conftest.py` refuses any non-loopback socket. No model is involved.

What these pin, in the order the brief states it:
  * the cursor starts at opt-in and advances only after the (fake) server accepts;
  * a refusal leaves the cursor and the send log untouched, so the queue retries;
  * the email never touches `job_feed_state` or `job_feed_marks`;
  * the saved filter applies, other tenants' boards never appear, non-complete
    checks are named, the heartbeat is weekly;
  * the unsubscribe GET never changes anything and the POST needs no CSRF token.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from email.message import EmailMessage

import pytest
from fastapi import Request
from fastapi.responses import RedirectResponse, Response
from fastapi.testclient import TestClient
from jfl_core.crypto.envelope import MasterKey
from jfl_core.db.tables import digest_settings, email_sends, job_feed_marks, job_feed_state, users
from jfl_core.email_links import UNSUBSCRIBE, sign_link_token
from jfl_core.models import EmailSend, ObservedJob, Workplace
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_core.storage.digest import PostgresDigestRepository, PostgresDigestScheduler
from jfl_core.storage.job_filters import PostgresJobFilterRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_intake.adapters.base import FetchResult
from jfl_intake.digest import SEND_DIGEST_KIND, SEND_TEST_EMAIL_KIND
from jfl_intake.engine import REPOST_WINDOW, plan_check
from jfl_intake.normalise import fingerprint
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity
from jfl_web.settings import WebSettings
from jfl_worker.handlers.digest import (
    build_schedule_digests,
    build_send_digest,
    build_send_test_email,
)
from jfl_worker.mailer import MailDeliveryError, MailSettings, SmtpSender
from jfl_worker.registry import PermanentTaskError, TaskContext
from sqlalchemy import create_engine, delete, func, insert, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")
SECRET = "e" * 43
NOW = dt.datetime.now(dt.UTC)


def hours(n: float) -> dt.timedelta:
    return dt.timedelta(hours=n)


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    created = create_engine(DATABASE_URL)
    yield created
    created.dispose()


@contextmanager
def _throwaway_user(engine: Engine) -> Iterator[uuid.UUID]:
    uid = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(insert(users).values(id=uid, email=f"{uid}@test.invalid"))
    try:
        yield uid
    finally:
        with engine.begin() as conn:
            conn.execute(delete(users).where(users.c.id == uid))


@pytest.fixture
def user(engine: Engine) -> Iterator[uuid.UUID]:
    with _throwaway_user(engine) as uid:
        yield uid


@pytest.fixture
def other_user(engine: Engine) -> Iterator[uuid.UUID]:
    with _throwaway_user(engine) as uid:
        yield uid


def mail_settings() -> MailSettings:
    result = MailSettings.from_env(
        {
            "JFL_SMTP_HOST": "smtp.example.invalid",
            "JFL_MAIL_FROM": "jobs4life <digest@example.invalid>",
            "JFL_PUBLIC_BASE_URL": "https://app.example.invalid",
            "JFL_EMAIL_LINK_SECRET": SECRET,
        }
    )
    assert result is not None
    return result


class RecordingSender(SmtpSender):
    def __init__(self) -> None:
        super().__init__(mail_settings())
        self.sent: list[EmailMessage] = []
        self.error: Exception | None = None

    def send(self, message: EmailMessage) -> None:
        if self.error is not None:
            raise self.error
        self.sent.append(message)


# -- board history, written the way the worker writes it -------------------------------


def observed(ext: str, title: str, workplace: Workplace = "remote") -> ObservedJob:
    return ObservedJob(
        external_id=ext,
        title=title,
        location="London, UK",
        url=f"https://example.invalid/jobs/{ext}",
        fingerprint=fingerprint(title, "London, UK"),
        workplace=workplace,
        locations=("London, UK",),
    )


def apply_check(
    engine: Engine,
    user_id: uuid.UUID,
    board_id: uuid.UUID,
    result: FetchResult,
    at: dt.datetime,
) -> None:
    with engine.begin() as conn:
        repo = PostgresBoardRepository(conn, user_id)
        state = repo.lock_check_state(
            board_id,
            observed_external_ids=[j.external_id for j in result.jobs],
            closed_since=at - REPOST_WINDOW,
        )
        assert state is not None
        repo.apply_check_plan(
            plan_check(state, result, observed_at=at), started_at=at, finished_at=at
        )


def complete(*jobs: ObservedJob) -> FetchResult:
    return FetchResult(status="complete", jobs=jobs, expected_total=len(jobs))


def watch(engine: Engine, user_id: uuid.UUID, label: str = "Acme") -> uuid.UUID:
    with engine.begin() as conn:
        board = PostgresBoardRepository(conn, user_id).add_board(
            platform="greenhouse",
            board_url=f"https://example.invalid/{uuid.uuid4().hex[:8]}",
            board_key={"token": uuid.uuid4().hex[:8]},
            label=label,
        )
    return board.id


def baseline_then_new(
    engine: Engine, user_id: uuid.UUID, *new_jobs: ObservedJob, label: str = "Acme"
) -> uuid.UUID:
    """A baseline two days ago, then a complete check an hour ago adding `new_jobs`."""
    board_id = watch(engine, user_id, label)
    base = observed("a1", "Old Role")
    apply_check(engine, user_id, board_id, complete(base), NOW - hours(48))
    apply_check(engine, user_id, board_id, complete(base, *new_jobs), NOW - hours(1))
    return board_id


def opt_in(engine: Engine, user_id: uuid.UUID, *, at: dt.datetime) -> None:
    with engine.begin() as conn:
        assert PostgresDigestRepository(conn, user_id).enable(now=at)


def context(engine: Engine, user_id: uuid.UUID, kind: str, now: dt.datetime = NOW) -> TaskContext:
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user_id).enqueue(kind=kind)
    return TaskContext(task=task, engine=engine, now=now)


def state_of(engine: Engine, user_id: uuid.UUID):  # type: ignore[no-untyped-def]
    with engine.begin() as conn:
        return PostgresDigestRepository(conn, user_id).get_state()


def sends_of(engine: Engine, user_id: uuid.UUID) -> list[str]:
    with engine.begin() as conn:
        return [
            r.kind
            for r in conn.execute(
                select(email_sends.c.kind)
                .where(email_sends.c.user_id == user_id)
                .order_by(email_sends.c.sent_at)
            )
        ]


def feed_rows(engine: Engine, user_id: uuid.UUID) -> int:
    with engine.begin() as conn:
        return sum(
            conn.execute(
                select(func.count()).select_from(t).where(t.c.user_id == user_id)
            ).scalar_one()
            for t in (job_feed_state, job_feed_marks)
        )


def bodies(message: EmailMessage) -> str:
    plain = message.get_body(preferencelist=("plain",))
    assert plain is not None
    return plain.get_content()


def run_digest(
    engine: Engine, user_id: uuid.UUID, sender: RecordingSender, now: dt.datetime = NOW
) -> object:
    handler = build_send_digest(mail=mail_settings(), sender=sender)
    return handler(context(engine, user_id, SEND_DIGEST_KIND, now))


# -- repository ----------------------------------------------------------------------


def test_opting_in_starts_the_cursor_at_now_and_is_idempotent(
    engine: Engine, user: uuid.UUID
) -> None:
    assert state_of(engine, user) is None
    opt_in(engine, user, at=NOW)
    first = state_of(engine, user)
    assert first is not None and first.enabled and first.notified_up_to == NOW
    with engine.begin() as conn:
        assert PostgresDigestRepository(conn, user).enable(now=NOW + hours(5)) is False
    again = state_of(engine, user)
    assert again is not None and again.notified_up_to == NOW  # a double click moved nothing


def test_re_enabling_after_opting_out_does_not_mail_the_weeks_it_was_off(
    engine: Engine, user: uuid.UUID
) -> None:
    opt_in(engine, user, at=NOW - hours(500))
    with engine.begin() as conn:
        assert PostgresDigestRepository(conn, user).disable() is True
        assert PostgresDigestRepository(conn, user).disable() is False
    opt_in(engine, user, at=NOW)
    state = state_of(engine, user)
    assert state is not None and state.enabled and state.notified_up_to == NOW


def test_the_cursor_only_moves_forward_and_a_disabled_digest_is_left_alone(
    engine: Engine, user: uuid.UUID
) -> None:
    opt_in(engine, user, at=NOW - hours(30))
    with engine.begin() as conn:
        repo = PostgresDigestRepository(conn, user)
        assert repo.complete_run(evaluated_at=NOW, notified_up_to=NOW - hours(1), sent=None)
        repo.complete_run(evaluated_at=NOW, notified_up_to=NOW - hours(10), sent=None)
    state = state_of(engine, user)
    assert state is not None and state.notified_up_to == NOW - hours(1)
    with engine.begin() as conn:
        repo = PostgresDigestRepository(conn, user)
        repo.disable()
        assert repo.complete_run(evaluated_at=NOW, notified_up_to=NOW, sent=None) is False
    state = state_of(engine, user)
    assert state is not None and state.notified_up_to == NOW - hours(1)


def test_one_users_repository_cannot_see_or_move_anothers(
    engine: Engine, user: uuid.UUID, other_user: uuid.UUID
) -> None:
    opt_in(engine, user, at=NOW - hours(30))
    with engine.begin() as conn:
        PostgresDigestRepository(conn, user).record_send(
            EmailSend(id=uuid.uuid4(), kind="test", sent_at=NOW)
        )
        theirs = PostgresDigestRepository(conn, other_user)
        assert theirs.get_state() is None
        assert theirs.last_send() is None
        assert theirs.disable() is False
        assert theirs.recipient_email() == f"{other_user}@test.invalid"
    assert state_of(engine, user) is not None


# -- the send handler ----------------------------------------------------------------


def test_a_digest_lists_new_jobs_advances_the_cursor_and_leaves_the_feed_alone(
    engine: Engine, user: uuid.UUID
) -> None:
    baseline_then_new(engine, user, observed("n1", "Engineering Manager, Inference"))
    opt_in(engine, user, at=NOW - hours(24))
    sender = RecordingSender()

    result = run_digest(engine, user, sender)

    assert result["sent"] == "digest"  # type: ignore[index]
    (message,) = sender.sent
    assert message["To"] == f"{user}@test.invalid"
    body = bodies(message)
    assert "Engineering Manager, Inference" in body
    assert "Old Role" not in body  # the baseline is never news
    assert "1 matching your filter of 1" in body
    assert message["Subject"] == "jobs4life: 1 new job on your watched boards"
    state = state_of(engine, user)
    assert state is not None and state.notified_up_to is not None
    assert state.notified_up_to > NOW - hours(1)
    assert sends_of(engine, user) == ["digest"]
    # Independent of the feed: no last-looked time, no marks.
    assert feed_rows(engine, user) == 0


def test_the_same_events_are_not_mailed_twice(engine: Engine, user: uuid.UUID) -> None:
    baseline_then_new(engine, user, observed("n1", "Engineering Manager, Inference"))
    opt_in(engine, user, at=NOW - hours(24))
    sender = RecordingSender()
    run_digest(engine, user, sender)
    run_digest(engine, user, sender, now=NOW + hours(24))
    assert len(sender.sent) == 1  # the second window was quiet, and not a heartbeat yet


def test_a_refusal_raises_and_moves_nothing_so_the_retry_sends_the_same_window(
    engine: Engine, user: uuid.UUID
) -> None:
    baseline_then_new(engine, user, observed("n1", "Engineering Manager, Inference"))
    opt_in(engine, user, at=NOW - hours(24))
    before = state_of(engine, user)
    sender = RecordingSender()
    sender.error = MailDeliveryError("mail delivery failed (TimeoutError)", permanent=False)

    with pytest.raises(MailDeliveryError):
        run_digest(engine, user, sender)
    assert state_of(engine, user) == before
    assert sends_of(engine, user) == []

    sender.error = None
    run_digest(engine, user, sender)
    assert len(sender.sent) == 1 and sends_of(engine, user) == ["digest"]


def test_a_permanent_refusal_fails_the_task_without_retry(engine: Engine, user: uuid.UUID) -> None:
    baseline_then_new(engine, user, observed("n1", "Engineering Manager, Inference"))
    opt_in(engine, user, at=NOW - hours(24))
    sender = RecordingSender()
    sender.error = MailDeliveryError("mail server refused the login (SMTP 535)", permanent=True)
    with pytest.raises(PermanentTaskError):
        run_digest(engine, user, sender)


def test_the_saved_filter_applies_and_the_email_says_how_many_it_hid(
    engine: Engine, user: uuid.UUID
) -> None:
    baseline_then_new(
        engine,
        user,
        observed("n1", "Remote Role", "remote"),
        observed("n2", "Office Role", "onsite"),
    )
    with engine.begin() as conn:
        PostgresJobFilterRepository(conn, user).save_filter(
            workplace_mode="remote_only",
            workplaces=[],
            title_includes="",
            title_excludes="",
            location="",
        )
    opt_in(engine, user, at=NOW - hours(24))
    sender = RecordingSender()
    run_digest(engine, user, sender)
    body = bodies(sender.sent[0])
    assert "Remote Role" in body and "Office Role" not in body
    assert "1 matching your filter of 2" in body


def test_another_users_boards_never_appear(
    engine: Engine, user: uuid.UUID, other_user: uuid.UUID
) -> None:
    baseline_then_new(engine, user, observed("n1", "My Role"), label="Mine")
    baseline_then_new(engine, other_user, observed("x1", "Secret Rival Role"), label="Theirs")
    opt_in(engine, user, at=NOW - hours(24))
    sender = RecordingSender()
    run_digest(engine, user, sender)
    body = bodies(sender.sent[0])
    assert "My Role" in body
    assert "Secret Rival Role" not in body and "Theirs" not in body


def test_a_board_that_could_not_be_checked_is_named_and_a_quiet_day_is_not_silent(
    engine: Engine, user: uuid.UUID
) -> None:
    board_id = watch(engine, user, "Broken Co")
    apply_check(engine, user, board_id, complete(observed("a1", "Old Role")), NOW - hours(48))
    apply_check(
        engine,
        user,
        board_id,
        FetchResult(status="unreachable", error_code="timeout"),
        NOW - hours(2),
    )
    opt_in(engine, user, at=NOW - hours(24))
    sender = RecordingSender()
    run_digest(engine, user, sender)
    (message,) = sender.sent
    assert message["Subject"] == "jobs4life: 1 board could not be checked"
    assert "Broken Co" in bodies(message) and "the request timed out" in bodies(message)


def test_a_quiet_day_sends_nothing_but_still_completes_the_run(
    engine: Engine, user: uuid.UUID
) -> None:
    baseline_then_new(engine, user)  # nothing new
    opt_in(engine, user, at=NOW - hours(24))
    with engine.begin() as conn:
        PostgresDigestRepository(conn, user).record_send(
            EmailSend(id=uuid.uuid4(), kind="digest", sent_at=NOW - hours(30))
        )
    sender = RecordingSender()
    run_digest(engine, user, sender)
    assert sender.sent == []
    state = state_of(engine, user)
    assert state is not None and state.last_evaluated_at is not None
    assert state.last_evaluated_at > NOW - hours(1)


def test_the_weekly_heartbeat(engine: Engine, user: uuid.UUID) -> None:
    baseline_then_new(engine, user)
    opt_in(engine, user, at=NOW - hours(24))
    with engine.begin() as conn:
        PostgresDigestRepository(conn, user).record_send(
            EmailSend(id=uuid.uuid4(), kind="digest", sent_at=NOW - dt.timedelta(days=8))
        )
    sender = RecordingSender()
    run_digest(engine, user, sender)
    (message,) = sender.sent
    assert message["Subject"] == "jobs4life: still watching 1 board — nothing new"
    assert sends_of(engine, user) == ["digest", "heartbeat"]


def test_no_boards_means_no_email_at_all(engine: Engine, user: uuid.UUID) -> None:
    opt_in(engine, user, at=NOW - hours(24))
    sender = RecordingSender()
    run_digest(engine, user, sender)
    assert sender.sent == []


def test_a_user_who_is_not_opted_in_gets_nothing(engine: Engine, user: uuid.UUID) -> None:
    baseline_then_new(engine, user, observed("n1", "A Role"))
    sender = RecordingSender()
    assert run_digest(engine, user, sender) == {"skipped": "not opted in"}
    assert sender.sent == []


def test_the_test_email_works_without_opt_in_and_leaves_the_cursor_alone(
    engine: Engine, user: uuid.UUID
) -> None:
    sender = RecordingSender()
    handler = build_send_test_email(mail=mail_settings(), sender=sender)
    handler(context(engine, user, SEND_TEST_EMAIL_KIND))
    (message,) = sender.sent
    assert message["Subject"] == "jobs4life: test email"
    assert sends_of(engine, user) == ["test"]
    assert state_of(engine, user) is None


def test_the_test_email_previews_the_last_week_without_touching_the_cursor(
    engine: Engine, user: uuid.UUID
) -> None:
    baseline_then_new(engine, user, observed("n1", "A Role"))
    opt_in(engine, user, at=NOW - hours(2))
    before = state_of(engine, user)
    sender = RecordingSender()
    result = build_send_test_email(mail=mail_settings(), sender=sender)(
        context(engine, user, SEND_TEST_EMAIL_KIND)
    )
    assert result == {"sent": "test", "previewed": "digest", "jobs_listed": 1}
    (message,) = sender.sent
    assert message["Subject"].startswith("jobs4life test: ")
    text = bodies(message)
    assert "TEST EMAIL" in text
    assert "A Role" in text
    assert state_of(engine, user) == before
    # The preview did not consume the window: the real digest still reports the job.
    sender = RecordingSender()
    run_digest(engine, user, sender)
    (digest,) = sender.sent
    assert "A Role" in bodies(digest)
    assert "TEST EMAIL" not in bodies(digest)


def test_a_quiet_week_previews_as_the_heartbeat(engine: Engine, user: uuid.UUID) -> None:
    baseline_then_new(engine, user)
    sender = RecordingSender()
    build_send_test_email(mail=mail_settings(), sender=sender)(
        context(engine, user, SEND_TEST_EMAIL_KIND)
    )
    (message,) = sender.sent
    assert message["Subject"] == "jobs4life test: still watching 1 board — nothing new"
    assert "in the last 7 days completed" in bodies(message)


def test_with_mail_unconfigured_the_send_kinds_fail_permanently_and_scheduling_is_a_no_op(
    engine: Engine, user: uuid.UUID
) -> None:
    with pytest.raises(PermanentTaskError):
        build_send_digest(mail=None)(context(engine, user, SEND_DIGEST_KIND))
    with pytest.raises(PermanentTaskError):
        build_send_test_email(mail=None)(context(engine, user, SEND_TEST_EMAIL_KIND))
    handler = build_schedule_digests(mail=None, only_users=[user])
    assert handler(context(engine, user, "schedule_digests")) == {"skipped": "mail not configured"}


# -- scheduling ----------------------------------------------------------------------


def test_the_scheduling_pass_enqueues_only_users_whose_digest_is_due(
    engine: Engine, user: uuid.UUID, other_user: uuid.UUID
) -> None:
    mail = mail_settings()
    due_at = mail.schedule.latest_due(NOW)
    opt_in(engine, user, at=due_at - hours(30))  # evaluated before the last digest hour
    opt_in(engine, other_user, at=due_at + dt.timedelta(minutes=1))  # opted in after it
    handler = build_schedule_digests(mail=mail, only_users=[user, other_user])

    first = handler(context(engine, user, "schedule_digests"))
    assert first == {"users_due": 1, "digests_enqueued": 1}  # type: ignore[comparison-overlap]
    with engine.begin() as conn:
        queued = PostgresTaskRepository(conn, user).list_tasks(kind=SEND_DIGEST_KIND)
        assert len(queued) == 1
        assert PostgresTaskRepository(conn, other_user).list_tasks(kind=SEND_DIGEST_KIND) == []

    # A second pass does not pile up a second task for the same user.
    again = handler(context(engine, user, "schedule_digests"))
    assert again["digests_enqueued"] == 0  # type: ignore[index]


def test_a_disabled_user_is_never_due(engine: Engine, user: uuid.UUID) -> None:
    opt_in(engine, user, at=NOW - hours(100))
    with engine.begin() as conn:
        PostgresDigestRepository(conn, user).disable()
        due = PostgresDigestScheduler(conn).due_users(
            evaluated_before=NOW,
            retry_not_before=NOW - hours(1),
            task_kind=SEND_DIGEST_KIND,
            limit=10,
            only_users=[user],
        )
    assert due == []


# -- the web controls ----------------------------------------------------------------


class StubGoogle:
    def __init__(self) -> None:
        self.identity: GoogleIdentity | None = None

    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response:
        return RedirectResponse("/auth/google/callback", status_code=303)

    async def fetch_identity(self, request: Request) -> GoogleIdentity:
        assert self.identity is not None
        return self.identity


def web_settings(*, mail: bool) -> WebSettings:
    return WebSettings(
        database_url=DATABASE_URL,
        google_client_id="test-client-id",
        google_client_secret="test-client-secret",
        google_redirect_uri="https://testserver/auth/google/callback",
        oauth_state_secret="0" * 43,
        master_key=MasterKey.generate(),
        session_ttl=dt.timedelta(days=14),
        session_touch_after=dt.timedelta(minutes=5),
        insecure_cookies=False,
        validate_api_keys=False,
        mail_configured=mail,
        email_link_secret=SECRET,
    )


@contextmanager
def signed_in_client(
    engine: Engine, *, mail: bool = True
) -> Iterator[tuple[TestClient, uuid.UUID]]:
    google = StubGoogle()
    identity = GoogleIdentity(
        sub=f"test-sub-{uuid.uuid4()}", email=f"{uuid.uuid4()}@test.invalid", display_name="T"
    )
    google.identity = identity
    app = create_app(web_settings(mail=mail), identity_provider=google)
    with TestClient(app, base_url="https://testserver") as client:
        assert client.get("/auth/google/callback").status_code == 200
        with engine.begin() as conn:
            uid = conn.execute(
                select(users.c.id).where(users.c.google_sub == identity.sub)
            ).scalar_one()
        try:
            yield client, uid
        finally:
            with engine.begin() as conn:
                conn.execute(delete(users).where(users.c.id == uid))


def csrf(client: TestClient, path: str = "/settings") -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None
    return match.group(1)


def enabled_in_db(engine: Engine, uid: uuid.UUID) -> bool:
    state = state_of(engine, uid)
    return bool(state and state.enabled)


def test_the_settings_page_toggles_the_digest_and_shows_the_address(engine: Engine) -> None:
    with signed_in_client(engine) as (client, uid):
        page = client.get("/settings").text
        assert "Email digest" in page
        assert "@test.invalid" in page.split("Email digest")[1]
        assert "Switch the digest on" in page and "never" in page  # last sent: never

        on = client.post("/settings/email/enable", data={"csrf_token": csrf(client)})
        assert on.status_code == 200  # redirected to /settings
        assert enabled_in_db(engine, uid)
        assert "Switch the digest off" in client.get("/settings").text
        assert "08:00 Europe/London" in client.get("/settings").text

        client.post("/settings/email/disable", data={"csrf_token": csrf(client)})
        assert not enabled_in_db(engine, uid)


def test_toggling_needs_a_csrf_token(engine: Engine) -> None:
    with signed_in_client(engine) as (client, uid):
        response = client.post("/settings/email/enable", data={"csrf_token": "wrong"})
        assert response.status_code == 403
        assert not enabled_in_db(engine, uid)


def test_the_test_button_enqueues_one_task_and_sends_nothing_in_the_request(
    engine: Engine,
) -> None:
    with signed_in_client(engine) as (client, uid):
        for _ in range(3):
            client.post("/settings/email/test", data={"csrf_token": csrf(client)})
        with engine.begin() as conn:
            tasks = PostgresTaskRepository(conn, uid).list_tasks(kind=SEND_TEST_EMAIL_KIND)
        assert len(tasks) == 1 and tasks[0].status == "pending"
        assert "queued" in client.get("/settings").text
        assert sends_of(engine, uid) == []


def test_with_mail_unconfigured_the_section_says_so_and_nothing_is_enabled(
    engine: Engine,
) -> None:
    with signed_in_client(engine, mail=False) as (client, uid):
        page = client.get("/settings").text
        assert "Email is not set up on this server" in page
        assert page.count("disabled") >= 2
        client.post("/settings/email/enable", data={"csrf_token": csrf(client)})
        client.post("/settings/email/test", data={"csrf_token": csrf(client)})
        assert not enabled_in_db(engine, uid)
        with engine.begin() as conn:
            assert PostgresTaskRepository(conn, uid).list_tasks(kind=SEND_TEST_EMAIL_KIND) == []


def test_unsubscribe_get_changes_nothing_and_post_unsubscribes_with_no_session_or_csrf(
    engine: Engine,
) -> None:
    with signed_in_client(engine) as (signed_in, uid):
        opt_in(engine, uid, at=NOW)
        token = sign_link_token(SECRET, UNSUBSCRIBE, uid)
        app = signed_in.app
        # A fresh client: no cookies at all, as a mail client or a link scanner.
        with TestClient(app, base_url="https://testserver") as anonymous:
            confirm = anonymous.get(f"/email/unsubscribe/{token}")
            assert confirm.status_code == 200 and "<button" in confirm.text
            assert confirm.headers["cache-control"] == "no-store"
            assert confirm.headers["referrer-policy"] == "no-referrer"
            assert enabled_in_db(engine, uid)  # a prefetching scanner changed nothing

            # RFC 8058: the mail client POSTs this body, with no CSRF token.
            done = anonymous.post(
                f"/email/unsubscribe/{token}", data={"List-Unsubscribe": "One-Click"}
            )
            assert done.status_code == 200 and "Unsubscribed" in done.text
            assert not enabled_in_db(engine, uid)


def test_a_forged_or_other_users_token_does_nothing(engine: Engine) -> None:
    with signed_in_client(engine) as (client, uid):
        opt_in(engine, uid, at=NOW)
        with TestClient(client.app, base_url="https://testserver") as anonymous:
            for token in (
                sign_link_token("a-different-secret-" + "z" * 30, UNSUBSCRIBE, uid),
                f"{uid.hex}.AAAA",
                "garbage",
            ):
                assert anonymous.get(f"/email/unsubscribe/{token}").status_code == 400
                assert anonymous.post(f"/email/unsubscribe/{token}").status_code == 400
        assert enabled_in_db(engine, uid)


def test_digest_tables_hold_nothing_for_a_deleted_user(engine: Engine) -> None:
    with _throwaway_user(engine) as uid:
        opt_in(engine, uid, at=NOW)
    with engine.begin() as conn:
        assert (
            conn.execute(
                select(func.count())
                .select_from(digest_settings)
                .where(digest_settings.c.user_id == uid)
            ).scalar_one()
            == 0
        )
