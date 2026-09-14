"""Display-only helpers for the boards screen: platform names, a default label
for a newly watched board, and plain-English words for a check's error code.

Nothing here touches storage or the network -- it is purely "how do we say this
on a page", the same split `jfl_web.jobads` draws for the application tracker's
extraction failures.
"""

from __future__ import annotations

from jfl_core.models import BoardCheckErrorCode, BoardPlatform
from jfl_intake.detect import BoardRef

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


# Priority order for picking a human label out of a board_key. `site` is
# special-cased below because for Workday it is a careers-site slug worth
# keeping, but for Teamtailor it is a whole hostname ("acme.teamtailor.com")
# and only the first label of that is worth showing.
_LABEL_KEYS = ("token", "name", "company", "tenant", "slug", "subdomain", "company_id", "site")


def default_label(ref: BoardRef) -> str:
    """A readable label for a board that has none yet -- picked from whatever
    identifying field its `board_key` carries, title-cased for display. Never
    used for lookups: the key itself, not this label, is what `detect_board`
    and equality checks compare.
    """
    for key in _LABEL_KEYS:
        value = ref.board_key.get(key)
        if not value:
            continue
        first = value.split(".")[0] if key == "site" else value
        return first.replace("-", " ").replace("_", " ").title()
    return platform_label(ref.platform)


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
