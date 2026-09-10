# The jobs4life web app AND the background worker: one image, two commands.
#
# The worker (slice B1) is a separate *container* so it can be stopped without
# taking the site down -- that is the incident response if a user's key starts
# burning money -- but it is not a separate image. It shares the app's code and
# its dependency set, so a second image would be the same bytes built twice and
# one more thing to keep in step.
#
# Two stages so the runtime image carries no build tooling. Only `jfl-web`,
# `jfl-worker` and their workspace dependencies are installed, which keeps
# `sentence-transformers` and `torch` out: they are an optional extra of
# `jfl-core` (~3GB) and nothing in the web or worker path touches them.
# Retrieval is unused in v1 by decision, so this is not a shortcut, it is the
# actual dependency set.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Built at /app, the same path it runs from. A venv's console scripts carry an
# absolute shebang, so building at /build and copying to /app leaves every
# entry point pointing at an interpreter that does not exist -- which surfaces
# as `exec ...: no such file or directory` naming the script rather than the
# missing Python.
WORKDIR /app

# Dependency resolution is cached separately from source: the manifests change
# rarely, the source changes every deploy.
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/
COPY packages/gate/pyproject.toml packages/gate/
COPY packages/generate/pyproject.toml packages/generate/
COPY packages/cli/pyproject.toml packages/cli/
COPY packages/evals/pyproject.toml packages/evals/
COPY packages/web/pyproject.toml packages/web/
COPY packages/worker/pyproject.toml packages/worker/
COPY packages/intake/pyproject.toml packages/intake/

# Sources must exist for the workspace members to build; stub them so the
# dependency layer can resolve before real source is copied.
RUN mkdir -p packages/core/src/jfl_core packages/gate/src/jfl_gate \
      packages/generate/src/jfl_generate packages/cli/src/jfl_cli \
      packages/evals/src/jfl_evals packages/web/src/jfl_web \
      packages/worker/src/jfl_worker packages/intake/src/jfl_intake \
 && touch packages/core/src/jfl_core/__init__.py packages/gate/src/jfl_gate/__init__.py \
      packages/generate/src/jfl_generate/__init__.py packages/cli/src/jfl_cli/__init__.py \
      packages/evals/src/jfl_evals/__init__.py packages/web/src/jfl_web/__init__.py \
      packages/worker/src/jfl_worker/__init__.py packages/intake/src/jfl_intake/__init__.py

RUN uv sync --frozen --no-dev --package jfl-web --package jfl-worker --no-install-workspace

COPY packages/ packages/
COPY migrations/ migrations/
COPY alembic.ini ./
RUN uv sync --frozen --no-dev --package jfl-web --package jfl-worker


FROM python:3.12-slim-bookworm AS runtime

# Runs as a non-root user. The container is reachable only from the host's
# loopback (Cloudflare Tunnel dials out to it), but a container escape should
# not land on uid 0.
RUN groupadd --system --gid 1001 app \
 && useradd --system --uid 1001 --gid app --create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/packages /app/packages
COPY --from=builder --chown=app:app /app/migrations /app/migrations
COPY --from=builder --chown=app:app /app/alembic.ini /app/alembic.ini

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status==200 else 1)"

# The DEFAULT command is the web app; the worker service in
# deploy/docker-compose.yml overrides it with `jfl-worker` (and disables the
# healthcheck below, which probes an HTTP port the worker does not serve).
#
# `--factory`: jfl_web.app exposes create_app(), not a module-level `app`, so
# settings are read at startup rather than at import.
#
# Log level is deliberately NOT debug. Authlib logs the PKCE `code_verifier` at
# DEBUG (authlib/integrations/base_client/sync_app.py), and a verifier in the
# logs undermines the protection PKCE exists to give.
CMD ["uvicorn", "jfl_web.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--log-level", "info", "--proxy-headers", "--forwarded-allow-ips", "*"]
