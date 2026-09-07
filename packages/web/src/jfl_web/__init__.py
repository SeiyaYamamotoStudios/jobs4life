"""jobs4life web app -- slice A: sign-in, sessions, tenancy, credential custody.

`create_app()` is the entry point:

    uv run uvicorn --factory jfl_web.app:create_app --port 8000

The application tracker is slice A5 and is deliberately not here.
"""

from jfl_web.app import create_app

__all__ = ["create_app"]
