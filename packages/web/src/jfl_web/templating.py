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

from jfl_web.timeformat import humanize

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

_templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
# A6: every event and record shows an absolute date plus a relative one --
# `{{ value | humanize_dt }}` -> "Tue 8 Sep 2026, 3 days ago". See
# jfl_web.timeformat for the thresholds.
_templates.env.filters["humanize_dt"] = humanize


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
