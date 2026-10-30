"""The digest message: headers, both parts, escaping, and the words on a job's
workplace. Built from a `DigestPlan` by hand; nothing is sent.
"""

from __future__ import annotations

import datetime as dt
import uuid
from email.message import EmailMessage

from jfl_core.models import BoardFilterException, BoardJob, BoardJobEvent, WatchedBoard
from jfl_intake.digest import DigestJob, DigestPlan, FailedBoard
from jfl_intake.display import workplace_notes
from jfl_intake.filtering import Match
from jfl_worker.digest_email import build_digest_message, build_test_message
from jfl_worker.mailer import MailSettings

NOW = dt.datetime(2026, 10, 30, 7, 0, tzinfo=dt.UTC)
USER = uuid.uuid4()
TOKEN = "TOKEN123.sig"
EVIL = '<script>alert("x")</script> & Co'


def cfg() -> MailSettings:
    result = MailSettings.from_env(
        {
            "JFL_SMTP_HOST": "smtp.example.invalid",
            "JFL_MAIL_FROM": "jobs4life <digest@example.invalid>",
            "JFL_PUBLIC_BASE_URL": "https://app.example.invalid",
            "JFL_EMAIL_LINK_SECRET": "s" * 40,
        }
    )
    assert result is not None
    return result


def board(label: str | None = "Acme") -> WatchedBoard:
    return WatchedBoard(
        id=uuid.uuid4(),
        user_id=USER,
        platform="greenhouse",
        board_url="https://boards.example.invalid/acme",
        board_key={"token": "acme"},
        label=label,
        created_at=NOW,
        next_check_at=NOW,
    )


def job(b: WatchedBoard, title: str, **kw: object) -> DigestJob:
    check = uuid.uuid4()
    j = BoardJob(
        id=uuid.uuid4(),
        user_id=USER,
        board_id=b.id,
        external_id=title,
        title=title,
        fingerprint=title,
        first_seen_check_id=check,
        first_seen_at=NOW,
        last_seen_at=NOW,
        **kw,  # type: ignore[arg-type]
    )
    event = BoardJobEvent(kind="new", board_id=b.id, check_id=check, at=NOW, job=j)
    return DigestJob(event=event, match=Match(job=j, reason="filter"), board=b)


def plan_of(*items: DigestJob, **kw: object) -> DigestPlan:
    base: dict[str, object] = {
        "kind": "digest",
        "jobs": tuple(items),
        "events_found": len(items),
        "jobs_matching": len(items),
        "boards_watched": 2,
    }
    base.update(kw)
    return DigestPlan(**base)  # type: ignore[arg-type]


def texts(msg: EmailMessage) -> tuple[str, str]:
    plain = msg.get_body(preferencelist=("plain",))
    html = msg.get_body(preferencelist=("html",))
    assert plain is not None and html is not None
    return plain.get_content(), html.get_content()


def build(plan: DigestPlan) -> EmailMessage:
    return build_digest_message(
        plan, to="owner@example.invalid", settings=cfg(), unsubscribe_token=TOKEN, now=NOW
    )


def test_headers_for_one_click_unsubscribe_and_machine_mail() -> None:
    msg = build(plan_of(job(board(), "Engineering Manager", workplace="remote")))
    assert msg["From"] == "jobs4life <digest@example.invalid>"
    assert msg["To"] == "owner@example.invalid"
    assert msg["List-Unsubscribe"] == "<https://app.example.invalid/email/unsubscribe/TOKEN123.sig>"
    assert msg["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click"
    assert msg["Auto-Submitted"] == "auto-generated"
    assert msg["Message-ID"].endswith("@example.invalid>")


def test_both_parts_carry_the_job_and_every_link() -> None:
    b = board()
    msg = build(
        plan_of(job(b, "Engineering Manager", workplace="remote", url="https://x.invalid/1"))
    )
    plain, html = texts(msg)
    for body in (plain, html):
        assert "Engineering Manager" in body
        assert "https://x.invalid/1" in body
        assert "https://app.example.invalid/changes" in body
        assert "https://app.example.invalid/settings" in body
        assert "https://app.example.invalid/email/unsubscribe/TOKEN123.sig" in body
    assert "1 matching your filter of 1" in plain


def test_subject_is_counts_only() -> None:
    b = board(EVIL)
    one = build(plan_of(job(b, EVIL)))
    assert one["Subject"] == "jobs4life: 1 new job on your watched boards"
    failed = FailedBoard(
        board=b,
        check=_check(b),
    )
    both = build(plan_of(job(b, "A"), job(b, "B"), failed=(failed,)))
    assert both["Subject"] == (
        "jobs4life: 2 new jobs on your watched boards; 1 board could not be checked"
    )
    only_failed = build(plan_of(failed=(failed,), jobs_matching=0))
    assert only_failed["Subject"] == "jobs4life: 1 board could not be checked"
    beat = build(DigestPlan(kind="heartbeat", boards_watched=12))
    assert beat["Subject"] == "jobs4life: still watching 12 boards — nothing new"


def _check(b: WatchedBoard):  # type: ignore[no-untyped-def]
    from jfl_core.models import BoardCheck

    return BoardCheck(
        id=uuid.uuid4(),
        user_id=USER,
        board_id=b.id,
        started_at=NOW,
        finished_at=NOW,
        status="unreachable",
        jobs_seen=0,
        error_code="timeout",
    )


def test_html_is_escaped_and_text_cannot_forge_lines() -> None:
    b = board(EVIL)
    msg = build(plan_of(job(b, EVIL + "\nINJECTED LINE", url="javascript:alert(1)")))
    plain, html = texts(msg)
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "javascript:" not in html and "javascript:" not in plain
    assert "\nINJECTED LINE" not in plain
    # Nothing from a job or label reached a header.
    assert all("script" not in str(v) for k, v in msg.items() if k != "Message-ID")


def test_failed_boards_are_listed_by_label_with_a_plain_description() -> None:
    b = board("Broken Co")
    msg = build(plan_of(failed=(FailedBoard(board=b, check=_check(b)),), jobs_matching=0))
    plain, html = texts(msg)
    for body in (plain, html):
        assert "Broken Co" in body
        assert "the request timed out" in body
        assert "could not be checked" in body.lower() or "COULD NOT BE CHECKED" in body


def test_heartbeat_says_what_it_can_vouch_for() -> None:
    msg = build(DigestPlan(kind="heartbeat", boards_watched=12))
    plain, _ = texts(msg)
    assert "Still watching 12 boards" in plain
    assert "nothing new matches your filter" in plain


def test_overflow_and_hidden_unstated_are_stated() -> None:
    b = board()
    plan = plan_of(job(b, "A"), jobs_matching=45, events_found=60, hidden_unstated=3)
    plain, html = texts(build(plan))
    assert "45 matching your filter of 60" in plain
    assert "and 44 more" in plain.replace("...", "and") or "44 more" in plain
    assert "3 more jobs would match" in plain and "show_unstated=1" in plain


def test_the_hybrid_caveat_is_the_pages_caveat() -> None:
    b = board()
    item = job(b, "Hybrid role", workplace="hybrid", workplace_label="Hybrid")
    match = Match(job=item.match.job, reason="filter", workplace_note="hybrid_days_not_stated")
    item = DigestJob(event=item.event, match=match, board=b)
    plain, html = texts(build(plan_of(item)))
    assert "Hybrid - days not stated" in plain
    assert "days not stated" in html


def test_workplace_notes_cover_each_case_the_page_has() -> None:
    b = board("Acme")
    base = job(b, "J", workplace="onsite", workplace_label="On-Site").match.job
    exc = BoardFilterException(
        id=uuid.uuid4(), board_id=b.id, note="near home", created_at=NOW, updated_at=NOW
    )
    assert workplace_notes(Match(job=base, reason="filter"), b) == []
    assert workplace_notes(
        Match(job=base, reason="filter", workplace_note="says_remote_friendly"), b
    ) == ["The posting says Remote-Friendly"]
    assert workplace_notes(
        Match(job=base, reason="filter", workplace_note="listed_onsite_says_remote_friendly"), b
    ) == ["Listed On-Site · the posting also says Remote-Friendly"]
    assert workplace_notes(Match(job=base, reason="exception", exception=exc), b) == [
        "included by your Acme exception: near home"
    ]
    assert workplace_notes(Match(job=base, reason="unstated"), b) == [
        "included because this board is set to include jobs whose workplace isn't stated"
    ]


def test_the_test_message_lists_no_jobs_and_still_unsubscribes() -> None:
    msg = build_test_message(
        to="owner@example.invalid", settings=cfg(), unsubscribe_token=TOKEN, now=NOW
    )
    plain, html = texts(msg)
    assert msg["Subject"] == "jobs4life: test email"
    assert "email delivery works" in plain
    assert "unsubscribe" in html.lower()
    assert msg["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click"


def test_no_images_and_no_tracking_in_the_html() -> None:
    _, html = texts(build(plan_of(job(board(), "A", url="https://x.invalid/a"))))
    assert "<img" not in html.lower() and "<script" not in html.lower()
