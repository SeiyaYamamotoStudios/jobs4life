"""Display-only helpers for the boards screen: a default label for a newly watched
board. Platform names and the plain-English words for a check's error code live
in `jfl_intake.display` -- the digest email says them too, from the worker, and
must say exactly what the page says -- and are re-exported here.

Nothing here touches storage or the network -- it is purely "how do we say this
on a page", the same split `jfl_web.jobads` draws for the application tracker's
extraction failures.
"""

from __future__ import annotations

from jfl_intake.detect import BoardRef
from jfl_intake.display import PLATFORM_LABELS, check_error_message, platform_label

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


__all__ = ["PLATFORM_LABELS", "check_error_message", "default_label", "platform_label"]
