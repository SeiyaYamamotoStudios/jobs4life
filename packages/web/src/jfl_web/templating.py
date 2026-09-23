"""Jinja2 setup. One module so no route has to know where templates live.

Autoescaping is on -- the Jinja2Templates default -- which is what makes it safe
to put a display name or an email straight into a page.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from jfl_web.boards import check_error_message
from jfl_web.drafts import (
    coverage_failure,
    draft_failure,
    generate_label,
    kind_label,
    sentence_label,
    sentence_style,
    usd,
)
from jfl_web.jobfilter import workplace_display
from jfl_web.timeformat import humanize, time_compact

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

_templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
# A6: every event and record shows an absolute date plus a relative one --
# `{{ value | humanize_dt }}` -> "Tue 8 Sep 2026, 3 days ago". See
# jfl_web.timeformat for the thresholds.
_templates.env.filters["humanize_dt"] = humanize
# `{{ value | time_compact }}` -> `<time datetime=… title="Wed 23 Sep 2026,
# 13:04 · 13 minutes ago">13 min ago</time>`. For table cells, where the full
# form repeated per column crowded out the columns that carry the content.
_templates.env.filters["time_compact"] = time_compact
# `{{ check.error_code | board_error_message }}` -> plain English for a
# BoardCheckErrorCode. See jfl_web.boards.
_templates.env.filters["board_error_message"] = check_error_message
# `{{ job | workplace_display }}` -> the employer's own label where one was
# given ("On-Site"), else our word. See jfl_web.jobfilter.
_templates.env.filters["workplace_display"] = workplace_display
# `{{ sentence | sentence_label }}` -> "Supported" / "Check this" / "Not
# supported" / "Not checked". `{{ sentence | sentence_style }}` -> the matching
# CSS class. See jfl_web.drafts -- framing is rendered as not checked.
_templates.env.filters["sentence_label"] = sentence_label
_templates.env.filters["sentence_style"] = sentence_style
# `{{ draft.kind | kind_label }}` -> "CV" / "cover letter".
_templates.env.filters["kind_label"] = kind_label
# `{{ kind | generate_label }}` -> "Write the CV" / "Write a cover letter".
_templates.env.filters["generate_label"] = generate_label
# `{{ cost | usd }}` -> "$0.4123", or "—" when nothing was billed yet.
_templates.env.filters["usd"] = usd
# `{{ task.last_error | coverage_failure }}` / `{{ ... | draft_failure }}` ->
# a GenerationFailure for a failed generate_coverage / generate_cv_draft task.
_templates.env.filters["coverage_failure"] = coverage_failure
_templates.env.filters["draft_failure"] = draft_failure


def _asset_url(name: str) -> str:
    """`/static/style.css` plus a short content hash.

    Cloudflare proxies this app and caches static assets aggressively -- observed
    `cache-control: max-age=14400` with `cf-cache-status: HIT` on a stylesheet
    four hours after it changed. The deploy was correct and the edge served the
    old file, which presents as "I deployed and nothing changed" and invites
    debugging the wrong layer entirely.

    A content hash in the query string makes a changed file a changed URL, so a
    new version can never be a cache hit and an unchanged one still caches for
    the full TTL. That is a property of the build rather than of remembering to
    purge, which is the point: purging is a step someone has to take, every time,
    and eventually will not.
    """
    path = STATIC_DIR / name
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    except OSError:
        # A missing asset is a template bug, not a reason to fail the request.
        return f"/static/{name}"
    return f"/static/{name}?v={digest}"


# Computed once at import: the files do not change while the process runs, and
# the container is rebuilt for every deploy.
_templates.env.globals["asset_url"] = _asset_url


def render(
    request: Request, name: str, context: dict[str, Any] | None = None, status_code: int = 200
) -> Response:
    return _templates.TemplateResponse(
        request=request, name=name, context=context or {}, status_code=status_code
    )
