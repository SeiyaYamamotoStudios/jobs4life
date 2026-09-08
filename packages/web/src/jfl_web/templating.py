"""Jinja2 setup. One module so no route has to know where templates live.

Autoescaping is on -- the Jinja2Templates default -- which is what makes it safe
to put a display name or an email straight into a page.
"""

from __future__ import annotations

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


def render(
    request: Request, name: str, context: dict[str, Any] | None = None, status_code: int = 200
) -> Response:
    return _templates.TemplateResponse(
        request=request, name=name, context=context or {}, status_code=status_code
    )
