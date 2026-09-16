# jobs4life

A career system that finds roles worth considering, says honestly how well you fit them,
and stops you overstating that fit when you apply.

Most tools in this space help candidates look better than they are. This one measures
the distance between what is true and what is being claimed, and shows you the number.
Every generated sentence passes a **claim gate** that checks it against a corpus of real
experience and reports whether it is supported, needs review, or is contradicted. The
gate informs; it never blocks.

## Status

- **App** — <https://app-jobs4life.hiltonlabs.org>. Google sign-in, bring your own
  Anthropic API key (envelope-encrypted), application tracker, watched job boards across
  12 ATS platforms, a changes feed, and "track as application".
- **Demo** — <https://jobs4life.hiltonlabs.org>. Static, pre-computed from the real
  pipeline over fictional candidates and job ads.
- **Measured** — over-claim **0.7%** (1/140) and over-flag **2.9%** (2/69) on 210 tier-1
  FEVER items, Opus 5. Framing was 0/209, so tier 1 never exercised the gate's one
  unguarded path (a claim misfiled as framing passes silently); quote the number with
  that caveat.

Design decisions: [`CLAUDE.md`](CLAUDE.md). Sequencing: [`PLAN.md`](PLAN.md). Current
state and known defects: [`NEXT.md`](NEXT.md). Server runbook:
[`docs/hosting.md`](docs/hosting.md).

## Local setup

Verified from a clean machine on Fedora-based Linux (Nobara 44), 2026-09-16.

### 1. Tools

[`uv`](https://docs.astral.sh/uv/) manages Python itself, so no system Python 3.12 is
needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # installs to ~/.local/bin
```

Docker with the compose plugin, for Postgres. On Fedora/Nobara the distro packages work;
Docker's own repo is not required:

```bash
sudo dnf install -y moby-engine docker-compose
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
getent group docker    # confirm your user is listed, then log out and back in
```

Until you log in again, prefix Docker commands with `sg docker -c '...'`.

### 2. Environment

```bash
cp .env.example .env      # then fill it in; every variable is documented inline
set -a; source .env; set +a
```

Load `.env` per shell rather than exporting `ANTHROPIC_API_KEY` globally — a global key
silently shadows any `ant auth login` profile.

`corpus/` and `analysis/` are gitignored and hold real career detail. Copy them from a
private backup; the database is a rebuildable index over `corpus/`.

### 3. Install, database, corpus

```bash
uv sync
docker compose up -d --wait      # Postgres + pgvector on port 5433
uv run alembic upgrade head
uv run jfl ingest                # rebuild the span index from corpus/**/*.md
```

### 4. Check it works

None of these call a model or touch the internet — the test suite refuses both.

```bash
uv run pytest                   # unit
uv run pytest -m integration    # needs the database
uv run ruff check . && uv run ruff format --check .
uv run mypy packages
```

### 5. Run the app

```bash
uv run uvicorn jfl_web.app:create_app --factory --port 8000
uv run jfl-worker                # background queue, separate process
```

`/healthz` answers without sign-in, and `/` redirects to `/login`. Signing in locally
should need two changes to `.env` — `JFL_INSECURE_COOKIES=1` (the `__Host-` session
cookie requires HTTPS) and a `JFL_GOOGLE_REDIRECT_URI` of
`http://localhost:8000/auth/google/callback` registered on the Google OAuth client — but
a local sign-in has not yet been verified. Never set `JFL_INSECURE_COOKIES` in
deployment.

`JFL_DISABLE_MODEL_CALLS=1` makes the worker leave model-calling tasks pending instead
of running them.

## Commands that cost money

These use a real Anthropic API key. API credit is billed separately from a Claude
subscription.

```bash
uv run jfl check "a sentence"          # claim gate; also --file cv.pdf
uv run jfl job add / coverage           # requirements from a pasted ad, corpus coverage
uv run jfl draft JOB_ID                 # generate, then gate automatically
JFL_ALLOW_REAL_API=1 uv run pytest -m e2e
uv run inspect eval packages/evals/src/jfl_evals/tasks.py --model none/none -T limit=210
```

Measured costs vary about 2x with output length: a whole-CV `jfl check` has cost
$0.13–0.64, and the full 210-item eval $2.07 on Opus 5. The eval defaults to 5 items, so
a full run has to be asked for with `-T limit=210`. See
[`packages/evals/README.md`](packages/evals/README.md).

## Deploying

```bash
./deploy/deploy.sh
```

Ships `git archive HEAD` — committed, tracked files only — so `corpus/` and uncommitted
changes can never reach the server. Builds on the VPS, migrates, restarts, and checks
`/healthz` locally and publicly.

## Licence

MIT © 2026 Seiya
