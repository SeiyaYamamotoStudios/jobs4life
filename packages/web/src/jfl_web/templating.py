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

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

_templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def render(
    request: Request, name: str, context: dict[str, Any] | None = None, status_code: int = 200
) -> Response:
    return _templates.TemplateResponse(
        request=request, name=name, context=context or {}, status_code=status_code
    )
