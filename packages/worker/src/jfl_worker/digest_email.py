"""Building the digest and test emails: headers, a plain-text part, an HTML part.

The decisions about *what* is said are `jfl_intake.digest.plan_digest`'s; this
module only turns a `DigestPlan` into a message. It touches no database and no
network, so it is unit-tested by looking at the result.

**What an email contains, and what it never does.**

  * Each listed job: title, board label and platform, the workplace exactly as
    `/changes` badges it (the employer's own label; "days not stated" on a hybrid
    whose days the platform did not give -- `jfl_intake.display.workplace_notes`),
    locations, whether it is new or returned and when, and a link to the posting.
    Then links to `/changes`, to Settings, and the unsubscribe link.
  * **Nothing from the corpus, a CV, an application or a score ever appears here.**
    The inputs are board events and board checks and nothing else; there is no
    parameter through which such content could arrive.
  * **No tracking.** No images, no tracking pixel, no click-tracking redirects: the
    links are the plain URLs. (Brevo can rewrite links and add an open pixel
    unless told not to -- `docs/hosting.md` says to switch both off.)

**Escaping.** The HTML part is rendered by Jinja with autoescaping on, so a job
title or board label of `<script>` arrives as text. The plain-text part is not an
HTML context, but every value goes through `oneline`, so a title containing a line
break cannot forge a line of the email. Header values are only counts and the
fixed sender, address and URLs, never a title or a label, so there is no header to
inject into; `EmailMessage` itself rejects a CR or LF in one. A posting URL is
linked only if it is `http(s)`.

**Templates live beside this module** (`jfl_worker/templates/`), not in
`jfl_web`: the worker builds the email and must not import the web package, and
email HTML (inline styles, tables, no stylesheet) shares nothing with the site's.

**Headers.** `List-Unsubscribe: <https://…>` and `List-Unsubscribe-Post:
List-Unsubscribe=One-Click` (RFC 8058), so a mail client can offer its own
unsubscribe button; the link goes to a route that accepts that POST with no login
and no CSRF token, authenticated by the signed token alone.
`Auto-Submitted: auto-generated` marks it as machine mail.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import format_datetime, make_msgid, parseaddr
from pathlib import Path
from urllib.parse import urlsplit

from jfl_core.models import BoardJob
from jfl_intake.digest import DigestJob, DigestPlan, FailedBoard
from jfl_intake.display import (
    check_error_message,
    platform_label,
    workplace_headline,
    workplace_notes,
)
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from jfl_worker.mailer import MailSettings

TEMPLATE_DIR = Path(__file__).parent / "templates"

_WHITESPACE = re.compile(r"\s+")


def oneline(value: object) -> str:
    """Collapse every run of whitespace, line breaks included, to one space."""
    return _WHITESPACE.sub(" ", str(value)).strip()


_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(enabled_extensions=("html",), default=False),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)
_env.filters["oneline"] = oneline


@dataclass(frozen=True, slots=True)
class LinkSet:
    """Absolute URLs the email links to."""

    changes: str
    changes_unstated: str
    settings: str
    unsubscribe: str


def link_set(settings: MailSettings, unsubscribe_token: str) -> LinkSet:
    base = settings.public_base_url
    return LinkSet(
        changes=f"{base}/changes",
        changes_unstated=f"{base}/changes?show_unstated=1",
        settings=f"{base}/settings",
        unsubscribe=f"{base}/email/unsubscribe/{unsubscribe_token}",
    )


def _safe_url(url: str | None) -> str | None:
    if not url:
        return None
    return url if urlsplit(url).scheme in {"http", "https"} else None


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def _locations(job: BoardJob) -> str:
    return "; ".join(job.locations or ([job.location] if job.location else []))


def _when(at: dt.datetime, tz: dt.tzinfo) -> str:
    return at.astimezone(tz).strftime("%a %d %b, %H:%M")


def _job_view(item: DigestJob, tz: dt.tzinfo) -> dict[str, object]:
    job = item.event.job
    board = item.board
    return {
        "title": job.title,
        "url": _safe_url(job.url),
        "board": (board.label or board.board_url) if board is not None else "a watched board",
        "platform": platform_label(board.platform) if board is not None else "",
        "kind": item.event.kind,
        "kind_note": (
            "back on the board after being taken down" if item.event.kind == "returned" else ""
        ),
        "when": _when(item.event.at, tz),
        "workplace": workplace_headline(job),
        "workplace_notes": workplace_notes(item.match, board),
        "locations": _locations(job),
    }


def _failed_view(item: FailedBoard, tz: dt.tzinfo) -> dict[str, object]:
    board, check = item.board, item.check
    return {
        "board": board.label or board.board_url,
        "platform": platform_label(board.platform),
        "what": check_error_message(check.error_code),
        "held": check.status == "held",
        "when": _when(check.finished_at, tz),
    }


def digest_subject(plan: DigestPlan) -> str:
    """Counts only -- never a title or a label, so nothing from the boards can
    reach a header."""
    if plan.kind == "heartbeat":
        return f"jobs4life: still watching {_plural(plan.boards_watched, 'board')} — nothing new"
    parts: list[str] = []
    if plan.jobs_matching:
        if plan.returned_matching:
            what = "new or returned job" if plan.jobs_matching == 1 else "new or returned jobs"
            if plan.returned_matching == plan.jobs_matching:
                what = "returned job" if plan.jobs_matching == 1 else "returned jobs"
        else:
            what = "new job" if plan.jobs_matching == 1 else "new jobs"
        parts.append(f"{plan.jobs_matching} {what} on your watched boards")
    if plan.failed:
        n = len(plan.failed)
        parts.append(f"{n} board{' ' if n == 1 else 's '}could not be checked")
    return "jobs4life: " + "; ".join(parts)


def _base_message(
    settings: MailSettings, *, to: str, subject: str, links: LinkSet, now: dt.datetime
) -> EmailMessage:
    message = EmailMessage()
    message["From"] = settings.sender
    message["To"] = to
    message["Subject"] = subject
    message["Date"] = format_datetime(now)
    sender_domain = parseaddr(settings.sender)[1].rpartition("@")[2] or None
    message["Message-ID"] = make_msgid(domain=sender_domain)
    message["List-Unsubscribe"] = f"<{links.unsubscribe}>"
    message["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
    message["Auto-Submitted"] = "auto-generated"
    return message


def _attach_bodies(message: EmailMessage, template: str, context: dict[str, object]) -> None:
    message.set_content(_env.get_template(f"{template}.txt").render(**context))
    message.add_alternative(_env.get_template(f"{template}.html").render(**context), subtype="html")


def build_digest_message(
    plan: DigestPlan,
    *,
    to: str,
    settings: MailSettings,
    unsubscribe_token: str,
    now: dt.datetime,
    preview_days: int | None = None,
) -> EmailMessage:
    """The digest or heartbeat for `plan`. `plan.kind` must not be None.

    `preview_days` makes it the "Send a test email" preview: the same message,
    covering the last `preview_days` days rather than the digest cursor's window,
    with a banner and a subject that say so. A preview must never be mistakable
    for a digest, or the owner reads a test as "this is what changed today".
    """
    if plan.kind is None:
        raise ValueError("a plan that sends nothing has no message")
    tz = settings.schedule.tz
    links = link_set(settings, unsubscribe_token)
    context = {
        "heartbeat": plan.kind == "heartbeat",
        "jobs": [_job_view(j, tz) for j in plan.jobs],
        "matching": plan.jobs_matching,
        "returned": plan.returned_matching,
        "found": plan.events_found,
        "hidden_unstated": plan.hidden_unstated,
        "overflow": plan.overflow,
        "failed": [_failed_view(f, tz) for f in plan.failed],
        "boards_watched": plan.boards_watched,
        "boards_phrase": _plural(plan.boards_watched, "board"),
        "links": links,
        "schedule": settings.schedule.describe(),
        "preview_days": preview_days,
        "sent_at": _when(now, tz),
    }
    subject = digest_subject(plan)
    if preview_days is not None:
        subject = subject.replace("jobs4life: ", "jobs4life test: ", 1)
    message = _base_message(settings, to=to, subject=subject, links=links, now=now)
    _attach_bodies(message, "digest", context)
    return message


def build_test_message(
    *, to: str, settings: MailSettings, unsubscribe_token: str, now: dt.datetime
) -> EmailMessage:
    """The "Send a test email" message for a user with no watched boards: proves
    delivery, says nothing about jobs. With boards, the test is a preview digest
    (`build_digest_message(preview_days=...)`)."""
    links = link_set(settings, unsubscribe_token)
    message = _base_message(settings, to=to, subject="jobs4life: test email", links=links, now=now)
    _attach_bodies(
        message,
        "test",
        {
            "links": links,
            "schedule": settings.schedule.describe(),
            "sent_at": _when(now, settings.schedule.tz),
        },
    )
    return message
