"""The emailed digest of watched-board changes, in the background (PLAN.md C9).

Three kinds, none of which calls a model (`calls_model=False`, and true: nothing
on these paths imports `anthropic`; the kill switch is for spending a user's key
and this spends none).

`schedule_digests` (no payload)
-------------------------------

Enqueued by the worker's ticker every fifteen minutes through `enqueue_unique`,
exactly as the board pass is. It finds opted-in users whose digest is due --
their last completed run is older than the most recent configured digest hour
(`DigestSchedule.latest_due`), default 08:00 Europe/London -- across tenants via
`PostgresDigestScheduler`, which returns ids and nothing else, and enqueues a
`send_digest` for each with a repository bound to that user. A user who already
has one queued, or had one created in the last hour, is not enqueued again: a
send that keeps failing is retried hourly, not every pass.

`send_digest` (no payload; the user is the task's owner)
--------------------------------------------------------

Reads, decides, sends, records:

  1. in one short transaction, gather the user's opt-in state, address, boards,
     saved filter and exceptions, events since their own cursor, and checks since
     it, then release the connection -- no transaction is held across SMTP;
  2. `jfl_intake.digest.plan_digest` (pure) decides: digest, weekly heartbeat, or
     nothing. The window is `(notified_up_to, now - SETTLE]`;
  3. for a message, build it and hand it to the SMTP sender. A refusal raises, so
     the queue retries (or, for a permanent refusal, fails the task without retry);
  4. **only after the server has accepted it**, in one transaction: log the send
     and move the cursor to the window's end (`complete_run`). A quiet day with
     nothing to send still advances the cursor and marks the run complete, or the
     scheduler would re-enqueue the user for the rest of the day.

**Delivery is at-least-once, and the residual window is stated rather than
hidden.** If the process dies after SMTP accepts the message and before step 4
commits, the cursor has not moved and the task is reclaimed and run again: the
same window is mailed twice. That is the right side to fail on -- a duplicate
digest costs a glance, a lost one is a job missed -- and closing it would need a
two-phase protocol with the mail provider that SMTP does not offer. A user who
unsubscribes while a send is in flight may receive that one email.

**Independent of the feed.** Nothing here reads or writes `job_feed_state` or
`job_feed_marks`: sending an email does not mark anything as looked at, and
opening `/changes` later shows exactly what it would have shown.

Test email
----------

`send_test_email` (payload: none) is enqueued by the "Send a test email" button
and sent by the worker, never in the request. **It is a preview of the real
thing**: the digest `plan_digest` would build for the last `PREVIEW_WINDOW` (7
days), through the same filter and failure rules, rendered with a banner and a
"test" subject so it can never be read as today's digest. Waiting for 08:00 to
find out whether the content is right is not a test. A user with no watched
boards gets the plain delivery check instead, since there is nothing to preview.
Either way it works whether or not the digest is switched on, is logged in
`email_sends` as kind `test`, and **does not touch the cursor**: a preview that
advanced it would swallow the events the next real digest exists to report.

When mail is not configured (`WorkerSettings.mail is None`), `schedule_digests`
does nothing and the two send kinds fail permanently with a literal message; the
web app has already disabled the controls in that case.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable, Collection, Mapping

from jfl_core.email_links import UNSUBSCRIBE, sign_link_token
from jfl_core.models import EmailSend
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_core.storage.digest import PostgresDigestRepository, PostgresDigestScheduler
from jfl_core.storage.job_filters import PostgresJobFilterRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_intake.digest import (
    SCHEDULE_DIGESTS_KIND,
    SEND_DIGEST_KIND,
    SEND_TEST_EMAIL_KIND,
    SETTLE,
    plan_digest,
)

from jfl_worker.digest_email import build_digest_message, build_test_message
from jfl_worker.mailer import MailDeliveryError, MailSettings, SmtpSender
from jfl_worker.registry import Handler, PermanentTaskError, TaskContext

SCHEDULE_KIND = SCHEDULE_DIGESTS_KIND
SEND_KIND = SEND_DIGEST_KIND
TEST_KIND = SEND_TEST_EMAIL_KIND

# Users enqueued per scheduling pass, and the gap between their sends. The pass
# runs every fifteen minutes and the SMTP relay is rate-limited, so a few
# thousand users drain in a morning and one pass's transaction stays short.
SCHEDULE_BATCH = 200
SEND_SPACING = dt.timedelta(seconds=10)
# A user whose send keeps failing is tried hourly, not on every pass.
RETRY_BACKOFF = dt.timedelta(hours=1)
# How far back the test email's preview digest looks.
PREVIEW_WINDOW = dt.timedelta(days=7)

_NOT_CONFIGURED = "mail is not configured on this server"


def _utcnow() -> dt.datetime:
    return dt.datetime.now(tz=dt.UTC)


def _deliver(sender: SmtpSender, message_factory: Callable[[], object]) -> None:
    """Build and send, turning every failure into the queue's vocabulary.

    A message that cannot be built (an address `EmailMessage` rejects) and a
    server that permanently refuses are the same answer next time; anything else
    is worth the retry ladder. Messages are literals: see `jfl_worker.mailer`.
    """
    try:
        message = message_factory()
    except ValueError:
        raise PermanentTaskError("the email could not be built for this address") from None
    try:
        sender.send(message)  # type: ignore[arg-type]
    except MailDeliveryError as exc:
        if exc.permanent:
            raise PermanentTaskError(str(exc)) from None
        raise


def build_schedule_digests(
    *,
    mail: MailSettings | None,
    batch: int = SCHEDULE_BATCH,
    spacing: dt.timedelta = SEND_SPACING,
    only_users: Collection[uuid.UUID] | None = None,
) -> Handler:
    """`only_users=None` -- what production registers -- schedules every opted-in
    user. A collection narrows the pass to those users; tests pass the ones they
    created.
    """
    confined = None if only_users is None else frozenset(only_users)

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        if mail is None:
            return {"skipped": "mail not configured"}
        due_at = mail.schedule.latest_due(ctx.now)
        enqueued = 0
        with ctx.engine.begin() as conn:
            users = PostgresDigestScheduler(conn).due_users(
                evaluated_before=due_at,
                retry_not_before=ctx.now - RETRY_BACKOFF,
                task_kind=SEND_KIND,
                limit=batch,
                only_users=confined,
            )
            for user_id in users:
                task = PostgresTaskRepository(conn, user_id).enqueue_unique(
                    kind=SEND_KIND, scheduled_at=ctx.now + spacing * enqueued
                )
                if task is not None:
                    enqueued += 1
        return {"users_due": len(users), "digests_enqueued": enqueued}

    return handler


def build_send_digest(
    *,
    mail: MailSettings | None,
    sender: SmtpSender | None = None,
    clock: Callable[[], dt.datetime] = _utcnow,
) -> Handler:
    smtp = sender if sender is not None or mail is None else SmtpSender(mail)

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        if mail is None or smtp is None:
            raise PermanentTaskError(_NOT_CONFIGURED)
        return _send_digest(ctx, mail, smtp, clock)

    return handler


def _send_digest(
    ctx: TaskContext, mail: MailSettings, sender: SmtpSender, clock: Callable[[], dt.datetime]
) -> Mapping[str, object]:
    user_id = ctx.user_id
    now = ctx.now
    until = now - SETTLE

    with ctx.engine.begin() as conn:
        digest = PostgresDigestRepository(conn, user_id)
        state = digest.get_state()
        if state is None or not state.enabled:
            return {"skipped": "not opted in"}
        to = digest.recipient_email()
        if to is None:
            return {"skipped": "no address on the account"}
        since = state.notified_up_to or state.opted_in_at or until
        if until <= since:
            return {"skipped": "nothing to cover yet"}
        board_repo = PostgresBoardRepository(conn, user_id)
        filters = PostgresJobFilterRepository(conn, user_id)
        boards = board_repo.list_boards()
        events = board_repo.events_since(since) if boards else []
        checks = board_repo.checks_since(since) if boards else []
        saved_filter = filters.get_filter()
        exceptions = filters.list_exceptions()
        last = digest.last_send()

    plan = plan_digest(
        events=events,
        checks=checks,
        boards=boards,
        saved_filter=saved_filter,
        exceptions=exceptions,
        since=since,
        until=until,
        last_send_at=None if last is None else last.sent_at,
    )

    sent: EmailSend | None = None
    if plan.kind is not None:
        token = sign_link_token(mail.link_secret, UNSUBSCRIBE, user_id)
        kind = plan.kind
        _deliver(
            sender,
            lambda: build_digest_message(
                plan, to=to, settings=mail, unsubscribe_token=token, now=now
            ),
        )
        sent = EmailSend(
            id=uuid.uuid4(),
            kind=kind,
            sent_at=clock(),
            jobs_listed=len(plan.jobs),
            events_found=plan.events_found,
            boards_failed=len(plan.failed),
        )

    with ctx.engine.begin() as conn:
        PostgresDigestRepository(conn, user_id).complete_run(
            evaluated_at=now, notified_up_to=until, sent=sent
        )
    return {
        "sent": plan.kind or "nothing",
        "jobs_listed": len(plan.jobs),
        "jobs_matching": plan.jobs_matching,
        "events_found": plan.events_found,
        "boards_failed": len(plan.failed),
        "boards_watched": plan.boards_watched,
    }


def build_send_test_email(
    *,
    mail: MailSettings | None,
    sender: SmtpSender | None = None,
    clock: Callable[[], dt.datetime] = _utcnow,
) -> Handler:
    smtp = sender if sender is not None or mail is None else SmtpSender(mail)

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        if mail is None or smtp is None:
            raise PermanentTaskError(_NOT_CONFIGURED)
        now = ctx.now
        since = now - PREVIEW_WINDOW
        with ctx.engine.begin() as conn:
            to = PostgresDigestRepository(conn, ctx.user_id).recipient_email()
            board_repo = PostgresBoardRepository(conn, ctx.user_id)
            filters = PostgresJobFilterRepository(conn, ctx.user_id)
            boards = board_repo.list_boards()
            events = board_repo.events_since(since) if boards else []
            checks = board_repo.checks_since(since) if boards else []
            saved_filter = filters.get_filter()
            exceptions = filters.list_exceptions()
        if to is None:
            raise PermanentTaskError("there is no address on this account")
        token = sign_link_token(mail.link_secret, UNSUBSCRIBE, ctx.user_id)
        # `last_send_at=None`: a preview always says something -- a digest, or
        # the heartbeat's "nothing new" -- never the silence of a quiet day.
        plan = plan_digest(
            events=events,
            checks=checks,
            boards=boards,
            saved_filter=saved_filter,
            exceptions=exceptions,
            since=since,
            until=now,
            last_send_at=None,
        )
        if plan.kind is None:
            _deliver(
                smtp,
                lambda: build_test_message(to=to, settings=mail, unsubscribe_token=token, now=now),
            )
        else:
            _deliver(
                smtp,
                lambda: build_digest_message(
                    plan,
                    to=to,
                    settings=mail,
                    unsubscribe_token=token,
                    now=now,
                    preview_days=PREVIEW_WINDOW.days,
                ),
            )
        with ctx.engine.begin() as conn:
            PostgresDigestRepository(conn, ctx.user_id).record_send(
                EmailSend(id=uuid.uuid4(), kind="test", sent_at=clock())
            )
        return {
            "sent": "test",
            "previewed": plan.kind or "no boards",
            "jobs_listed": len(plan.jobs),
        }

    return handler
