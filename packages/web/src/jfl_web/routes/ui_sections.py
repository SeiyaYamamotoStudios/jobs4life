"""One route: "I folded this panel away" / "I opened it again".

  POST /ui/sections -- record a section toggle for the signed-in user

Written by `static/sections.js` over htmx, one small upsert per toggle and never
on render. It answers `204 No Content`: there is nothing to swap, and the page
the user is looking at is already in the state they just put it in.

**A failure here is silent on purpose.** The worst outcome of a lost write is
that a panel comes back open next time. Turning that into an error banner would
put a message about layout preferences in front of someone in the middle of
writing a cover letter.

Nothing in this file can change what a page *says* -- only which parts of it
start folded. That is worth stating, because it is why a key this route has
never heard of is recorded rather than rejected: the screens grow panels, a
draft's section is keyed by the draft's own id, and a route that had to know
every key in advance would be a second place to update every time a screen
changes. The shape is validated instead of the value.
"""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Form
from fastapi.responses import Response

from jfl_web.deps import CsrfDep, SectionRepoDep, SessionDep

router = APIRouter()

# Lower-case words, dots, dashes and digits: enough for `application.ad` and for
# `draft.<uuid>`, and not enough for someone to fill the table with prose. The
# cap is generous against the longest key any screen builds (a `draft.` prefix
# plus a UUID is 42 characters).
_KEY = re.compile(r"^[a-z0-9][a-z0-9._-]{0,119}$")


@router.post("/ui/sections")
def record_section_toggle(
    session: SessionDep,
    sections: SectionRepoDep,
    _csrf: CsrfDep,
    section: Annotated[str, Form()],
    is_open: Annotated[str, Form(alias="open")],
    default_open: Annotated[str, Form()] = "true",
) -> Response:
    """Record the toggle, or ignore it. Either way the browser gets a 204.

    `default_open` comes from the page rather than from here, because "what
    would this screen have shown without your choice" depends on state this
    route does not have -- whether a run was pending, whether the ad had been
    read. It is the user's own telemetry about their own account, so the worst a
    tampered value can do is muddle their own record of their own clicks.
    """
    if _KEY.match(section):
        sections.record_toggle(
            section,
            is_open=is_open == "true",
            default_open=default_open == "true",
        )
    return Response(status_code=204)
