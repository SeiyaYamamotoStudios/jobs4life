# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**job4life** — a career system, eventually at `hiltonlabs.org/job4life`. It finds roles
worth considering, says honestly how well the author fits them, plans the work to close
the gap, and stops them overstating that fit when they apply.

Most tools in this space help candidates look better than they are. This one measures
the distance between what is true and what is being claimed, and shows the number.
**That thesis is the product and it is also the eval metric — do not let a design
decision quietly undermine it.**

Headline metric is **over-claim rate**: claims passed as grounded that were not. Not
accuracy. Over-claiming is the error with a real cost. The Inspect harness is the
headline deliverable and survives every scope change; given a choice between more
features and a working harness, choose the harness.

Long-running, ~10 hrs/week. One deadline: something demonstrable and honestly
described by **late September 2026**. After that, product work, no deadline.

v1 is **single-user with the seams left in**. Do not build auth, sessions, or tenancy
enforcement yet. Hosting is deferred — assume a container on a small VPS behind
Cloudflare; nothing should depend on a specific host.

Owner: engineering manager, ~20 years, JVM/Python/Clojure. Do not explain general
engineering concepts. Do explain anything specific to the Python AI ecosystem. RTX 5090
available for local embeddings and inference.

## What is agentic, and what is not

Agentic means **the model chooses the next action**. Most of this system is a pipeline
with model calls in it, which is a different and smaller claim. Keep making the smaller
one.

Genuinely agentic: fit analysis, education planning, corpus interrogation when the gate
flags something the author says is true, company research.

**Not** agentic, though several use models: intake, scoring, generation, the claim gate,
dedup, extraction. Fixed control flow throughout. Do not build these as agent loops, and
**push back if asked to**.

When a tool-use loop is needed, write it against the Anthropic SDK directly.

## Domain map

Twelve domains, listed so nothing gets architecturally excluded. Only 1 and 2 are built.

1. **Corpus and claim gate** — grounding for everything generated. Slice one.
2. **Generation** — CV bullets, cover letters, free-text, interview prep. Every output
   passes the gate. Slice two.
3. **Intake** — pluggable sources, wide and noisy by design; deterministic filter from a
   natural-language prompt.
4. **Scoring** — two separate axes, **never one composite**: do I want this, and could I
   get this. They diverge constantly and merging them corrupts both.
5. **Fit analysis** — per-requirement verdicts with citations, plus a verifier that
   checks each citation supports the claim attached to it.
6. **Education planning** — every step names the requirement it closes and the evidence
   it would produce. **A step that creates no citable evidence is not a step.**
7. **Application tracking.**
8. **Feedback ingestion** — rejection reasons paired with what was sent. The only
   external ground truth in the system, and scarce. Capture it carefully.
9. **Offer evaluation.**  10. **Accounts and credentials.**  11. **Artifact import.**
12. **Observability and cost attribution** — the `runs` table, built for querying.

## Decisions log

**2026-08-24 — No golden set, no Inspect harness for now.** Owner's call, made twice
after the case for an eval was put. v1 judges claims with a single model call over the
stuffed corpus and ships on that. Consequences, recorded so they are not rediscovered:

- There is no over-claim number, so prompt changes cannot be verified. A change that
  fixes one failure and causes two others looks identical to a change that works.
- The "shows you the number" thesis is unmeasured in v1.
- Retrieval is unused: `span_embeddings`, the HNSW index and the embedding interface
  stay in place but nothing writes to them.

If this reverses, the cheapest path back is labelling real verdicts already captured in
`runs` rather than authoring a golden set from scratch.

## Build order

- **Now** — schema, corpus ingestion, then the end-to-end path: whole corpus in
  context, single call, verdict out. Runnable by hand. CLI only. No Inspect harness.
- **Next** — generation behind the gate, review queue with write-back, sent-document
  consistency checks.
- **Then** — intake with pluggable sources, deterministic prompt filter, unmeasured fit
  scoring.
- **Later** — FastAPI and browser UI, MCP server, accounts, education planning, the rest.

## Stack — decided, do not relitigate

Python 3.12 · `uv` workspace · `pydantic` · Postgres + `pgvector` via `docker compose`
(not SQLite) · `alembic` · `sentence-transformers` with a CPU fallback behind the same
interface · `Inspect` (AISI) for evals · Anthropic SDK called directly.

**No LangChain, LangGraph, CrewAI, or AutoGen.** If a tool-use loop is needed, write it.
CLI only in this iteration — no web UI, no MCP server, no API.

## Architectural constraints

These are cheap now and expensive to retrofit. They exist because this gets deployed later.

- Every table has a `user_id` column, FK to `users`. v1 seeds one local user:
  `0425d123-ed29-5a6a-a06d-d00267574046`, inserted by migration 0002.
- The API key is per-request context passed from the entry point — never a
  module-level environment read. `RequestContext.from_env()` is the only place
  environment is touched.
- Storage sits behind a repository interface. No SQL above that layer.
- Embeddings sit behind an interface with GPU and CPU implementations.
- `core` contains no HTTP or framework types.
- The sent-document store must never be reachable from a grounding query. Enforced
  structurally (separate tables, separate repository interface), not by a WHERE clause
  or a prompt instruction. If a mild stretch gets through once and then becomes a
  grounding reference, drift ratchets.
- **No scraping of LinkedIn or Indeed** — and not only for legal reasons: a hiring
  manager who sees a scraper in the architecture reads it as poor judgment. Use ATS APIs
  open by design (Greenhouse, Lever, Ashby, Workable) plus RSS. LinkedIn stays a manual
  paste surface.
- **Gmail restricted scopes are not worth the annual security assessment.** v1 is a
  dedicated forwarding address, modelled as one job source among several.
- **Credential custody**: envelope encryption, never logged, never in the `runs` table,
  never in a trace. Non-negotiable.
- Corpus markdown in `corpus/` is the source of truth; the database is a rebuildable
  index over it. Span IDs derive from content and section, never insertion order.
- Inspect eval logs stay as Inspect's own files on disk, never in Postgres.

## Repo shape

`packages/core` (storage, schemas, ingestion, embeddings) · `packages/gate` (the gate
itself) · `packages/evals` (Inspect tasks, golden set). Migrations in `migrations/`.

## Commands

| Task              | Command |
|-------------------|---------|
| Install           | `uv sync` |
| Database up       | `docker compose up -d` |
| Migrate           | `uv run alembic upgrade head` |
| New migration     | `uv run alembic revision --autogenerate -m "..."` |
| Unit tests        | `uv run pytest` |
| Integration tests | `uv run pytest -m integration` (needs the database) |
| Lint              | `uv run ruff check . && uv run ruff format --check .` |
| Typecheck         | `uv run mypy packages` |

`JFL_DATABASE_URL` must be set for anything touching the database; see `.env.example`.
Postgres is on port **5433** to avoid colliding with a local install.

**Autogenerated migrations need two hand edits every time:** alembic omits the
`import pgvector.sqlalchemy` line, and `CREATE EXTENSION IF NOT EXISTS vector` must be
added to the first migration that runs on a fresh database.

## Testing

Testing triangle: many component/unit tests, some integration tests, a handful of
critical end-to-end tests. Markers: `integration` (live Postgres), `e2e` (real model
API). Both are excluded from the default `pytest` run.

No golden set or eval harness in v1 — see the decisions log. If one is ever added: it
is a test fixture, never part of a user flow, and **do not generate synthetic items or
adjust labels to improve a score.**

## Conventions

- `main` is the default branch. Branch for changes; don't commit or push unless asked.
- Keep the working tree clean — no stray scratch files committed.
- Match the surrounding code's style once there is surrounding code.

## Delegation

Orchestrate with the session model; delegate implementation to subagents one tier
down. If the session is Opus, spawn coding subagents with `model: "sonnet"`. Keep
planning, architecture decisions, review of returned work, and anything touching
this file in the orchestrating session.

Override the tier when the work warrants it — gnarly debugging, subtle concurrency,
a design the smaller model has already fumbled once. Pass `model: "opus"` for those
rather than re-reviewing a bad diff twice. The tier is a default, not a rule.

Notes:
- Give each subagent enough context to work cold; it does not inherit the session's.
- `subagent_type: "fork"` inherits full context but always runs the parent's model —
  a `model` override on a fork is ignored.
- This standing preference is the request to delegate; no need to ask each time.
  It does not mean every task needs a subagent — small, local edits stay inline.

## Updating this file

Update CLAUDE.md when a decision here goes out of date: stack chosen, commands
changed, architecture shifted. Record decisions and non-obvious constraints, not a
directory listing.
