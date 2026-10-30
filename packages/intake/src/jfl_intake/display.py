"""How a board, a job's workplace and a failed check are *said* -- one spelling,
shared by the pages and the digest email.

These lived in `jfl_web` until the emailed digest (PLAN.md C9). The email is built
by the worker, which must not import the web package, and it must never say
something the pages would not: a board called "Greenhouse" on `/boards` and
"greenhouse" in the inbox, or a check failure explained in different words, would
be two accounts of one fact. So the words moved here, below both, and `jfl_web`
re-exports them under the names its templates and tests already use.

Nothing here touches storage, the network or the clock.

**The employer's words are never rewritten.** A job whose workplace came with the
employer's own label (a Greenhouse custom field, e.g. "On-Site") is shown with
that label, verbatim; our canonical word is used only where the platform gave an
enum or a boolean, which are not anyone's prose.
"""

from __future__ import annotations

from jfl_core.models import (
    BoardCheckErrorCode,
    BoardJob,
    BoardPlatform,
    WatchedBoard,
    Workplace,
)

from jfl_intake.filtering import Match

WORKPLACE_NAMES: dict[Workplace, str] = {
    "remote": "Remote",
    "hybrid": "Hybrid",
    "onsite": "On-site",
    "unknown": "Workplace not stated",
}


def workplace_display(job: BoardJob) -> str:
    if job.workplace != "unknown" and job.workplace_label:
        return job.workplace_label
    return WORKPLACE_NAMES[job.workplace]


def workplace_headline(job: BoardJob) -> str:
    """The workplace as the pages badge it: "workplace not stated" for an unknown
    one, else the employer's own label (or our word where they gave none)."""
    return "workplace not stated" if job.workplace == "unknown" else workplace_display(job)


def workplace_notes(match: Match[BoardJob], board: WatchedBoard | None) -> list[str]:
    """What `_workplace_badge.html` says about a matched job beyond the badge, as
    plain strings: the evidence note, a hybrid exception's note, and why the job is
    on the list at all. The emailed digest uses this so that it cannot claim more
    about a job's workplace than the page does -- "Hybrid -- days not stated" is
    the caveat that matters most. Keep the two in step; the worker's tests pin the
    words.
    """
    job = match.job
    name = (board.label or board.board_url) if board is not None else "board"
    notes: list[str] = []
    if match.workplace_note == "hybrid_days_not_stated":
        notes.append("days not stated")
    elif match.workplace_note == "listed_onsite_says_remote_friendly":
        notes.append(f"Listed {workplace_display(job)} · the posting also says Remote-Friendly")
    elif match.workplace_note == "says_remote_friendly":
        notes.append("The posting says Remote-Friendly")
    if match.workplace_exception is not None and match.reason != "exception":
        notes.append(f"hybrid, per your {name} exception: {match.workplace_exception.note}")
    if match.reason == "exception":
        suffix = f": {match.exception.note}" if match.exception and match.exception.note else ""
        notes.append(f"included by your {name} exception{suffix}")
    elif match.reason == "unstated":
        notes.append(
            "included because this board is set to include jobs whose workplace isn't stated"
        )
    return notes


PLATFORM_LABELS: dict[BoardPlatform, str] = {
    "greenhouse": "Greenhouse",
    "ashby": "Ashby",
    "lever": "Lever",
    "workday": "Workday",
    "smartrecruiters": "SmartRecruiters",
    "rippling": "Rippling",
    "breezy": "Breezy",
    "teamtailor": "Teamtailor",
    "personio": "Personio",
    "recruitee": "Recruitee",
    "pinpoint": "Pinpoint",
    "workable": "Workable",
}


def platform_label(platform: BoardPlatform) -> str:
    return PLATFORM_LABELS.get(platform, platform)


# What each BoardCheckErrorCode means, in words that make sense to someone who
# has never seen an ATS API. See `jfl_core.models.BoardCheckErrorCode` and
# `docs/ats-platforms.md` for what actually produces each one.
_ERROR_MESSAGES: dict[BoardCheckErrorCode, str] = {
    "not_found": "the board could not be found at that address",
    "http_client_error": "the site rejected the request",
    "rate_limited": "the site is rate-limiting requests",
    "server_error": "the site's server returned an error",
    "timeout": "the request timed out",
    "connection_error": "a connection could not be made",
    "malformed_response": "the response was not in the shape expected",
    "unidentifiable_job": "a job listing could not be identified",
    "count_mismatch": "the reported job count did not match what was listed",
    "page_cap_reached": "the listing went past the page limit before finishing",
    "duplicate_posting": "the listing repeated a posting without finishing",
    "request_budget_exhausted": "too many requests were needed to finish the check",
    "deadline_exceeded": "the check ran out of time",
    "listing_ceiling": "the board has more jobs than can be listed reliably",
    "unsupported_board": "this board's platform is not supported",
    "drop_guard": "the open job count dropped sharply and was held for review",
}

_UNKNOWN_ERROR = "the check did not finish"


def check_error_message(code: BoardCheckErrorCode | None) -> str:
    return _UNKNOWN_ERROR if code is None else _ERROR_MESSAGES.get(code, _UNKNOWN_ERROR)


__all__ = [
    "PLATFORM_LABELS",
    "WORKPLACE_NAMES",
    "check_error_message",
    "platform_label",
    "workplace_display",
    "workplace_headline",
    "workplace_notes",
]
