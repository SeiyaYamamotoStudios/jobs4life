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

**2026-09-01 — Repository is private.** Flipping private to public later preserves the
whole commit history, so nothing is lost by waiting; public to private does not retract
what has been cloned or indexed. Revisit when there is something worth showing.

**2026-09-01 — The owner's corpus is a test fixture, not a backlog.** Filling its gaps by
hand produces no code, and does by hand exactly what domain 2 automates: a gap becomes a
question, the answer becomes an `adjudicated` span. Verdict is a function of the corpus,
so a thin corpus produces *more flags*, not different behaviour — the mechanism is
identical whatever it holds. Run against real material to find defects; never schedule
work to improve one person's coverage. One finding from that gap work is mechanism and
survives: the record is a present-tense snapshot, so a historically accurate claim about
an earlier team composition reads as drift. Spans need validity periods.

**2026-09-01 — The deterministic tier buys guarantees, not savings.** Refines the same
day's earlier "deterministic tier first, model tier for the residue". The cost argument
in that entry was wrong: the corpus is already cached, a marginal check reads it for
about $0.02, and the tier cannot skip the call anyway — deciding claim-versus-framing
needs the model on every sentence, so the call happens regardless. What the tier actually
buys is a *guarantee no prompt can give*: the model catches `invented_quantity` and
stated-boundary crossings most of the time, and rules catch them every time. That is
worth more than the saving was.

Scope is narrowed to what reads the corpus at runtime and works for anybody: unsourced
numbers, and claims touching the corpus's stated-negative facts. Dropped: "over-claim
phrases already seen to recur" — a phrase list derived from 33 of the owner's CVs is that
archive in disguise, and bakes one use case into the mechanism. The five evidence-dependent
labels need judgement and stay with the model.

**A deterministic rule can prove contact, never crossing — so every rule resolves to
`review`, never `unsupported`.** Term overlap with "Rust: no experience" shows a claim
touches a boundary; only the model can say whether it crosses one. And the analysis found
`invented_quantity` never fired across 2,347 units — every number traced — so the base
rate of true positives is near zero and a literal-absence rule would mostly catch numbers
legitimately *derived* from the corpus (a tenure summed from dates). Hard-failing those is
precisely the over-flagging that gets the tool switched off. Rules therefore run **after**
the model and may only upgrade `supported` to `review` — the direction that catches misses
— never downgrade a flag to a pass. A rule-set verdict names the rule in its reason.
**Build the rules from observed patterns, never from invented ones** — same discipline as
the taxonomy.

**2026-09-01 — Coverage is measured against the corpus, never against the candidate.** The
requirement statuses are `evidenced`, `partial`, `absent`, `contradicted` — deliberately
not met/unmet. The tool reports what it can evidence; it does not rate the person.
`absent` means the corpus is silent, which is the gap-question trigger, and is the claim
gate's corpus-silence problem seen from the other side.

**2026-09-01 — A gap answer is stored verbatim.** The user's own words become the
`adjudicated` span text, with no model anywhere in that path. Having a model tidy an
answer into a neater corpus fact is the ratchet in miniature: the user is then held to
wording they did not choose, by a tool whose whole claim is that it measures distance
from what they actually said.

**2026-09-01 — The claim gate runs automatically on generated text, on demand for pasted
text.** Generation without the gate is the tool this project exists to oppose, and at
application volume an optional check does not get pressed. Pasted text — a recruiter's
question, an imported draft — is the user asking, so it waits to be asked. This does not
weaken "it informs, it never blocks": a flagged draft is still emitted.

**2026-09-01 — Whole documents, never sections.** Asking someone to paste the prose parts
of a CV is the friction that gets a tool abandoned. The splitter handles document
structure instead.

**2026-09-01 — Generated documents influence form, never truth.** Previous CVs are
legitimate for structure, voice, and which topics earned attention, and for consistency
checking. They are never grounding. A mild stretch that becomes a grounding reference
makes the next draft stretch further, and by the fifth the tool measures distance against
its own prior output and reports everything as supported. The corpus grows only through
the user, via answered gap questions and adjudications.

**2026-08-24 — Golden set reinstated, sourced from public data.** Supersedes an earlier
decision to ship without one. The blocker was never the concept but the assumption that
items had to come from the owner's corpus; tier 1 comes from public sources instead, so
the eval costs the owner only the taxonomy. See the drift taxonomy and golden set
sections.

**2026-08-24 — Retrieval unused in v1.** The gate stuffs the whole corpus into context
(30–50k tokens, viable with prompt caching). `span_embeddings`, the HNSW index and the
embedding interface stay in place but nothing writes to them. Revisit if the corpus
outgrows the context window.

**2026-09-01 — Demo and product split; the corpus never leaves the owner's machine in
v1.** The public page at `hiltonlabs.org/job4life` is a pre-computed demo over fictional
material: a small matrix of fictional candidates × job ads, results produced by the real
pipeline and committed as fixtures. The live tool stays local — CLI now, a localhost web
UI later. Three reasons. The verification record is a liability document by design (its
most valuable section is a list of stated boundaries), and its protection is that it
never leaves this machine. A public model-calling page is an unmetered $0.27 per click.
And the standing "no auth in v1" decision stays true instead of being overridden. When
remote access is ever actually needed, gate at the edge with Cloudflare Access — no
in-app OAuth until real multi-user. Demo fixtures are fiction and therefore committable;
they are not golden-set items, so the synthetic-data prohibition does not apply to them —
but every result the demo shows **must be produced by the real pipeline, never
hand-written**: the page claims "this is what the tool outputs," and that claim must be
true.

**2026-09-01 — A gap answer lands in corpus markdown, not the database.** `jfl answer`
appends the user's words to `corpus/answered-questions.md` and re-ingests; the resulting
span is `provenance='document'` like any other corpus fact, recorded in
`gap_questions.resulting_span_id`. This restores "markdown is the source of truth; the
database is a rebuildable index" — which the direct adjudicated-span write had silently
broken — and makes every answer a plain-text line the author can read, edit, and delete.
In a truthfulness tool, "I can't find or fix the fact you recorded about me" is
disqualifying. Never both paths for one fact: two span ids for the same text means the
gate sees it twice. `add_adjudicated_span` remains only for the unbuilt review-items
flow; revisit when that is built.

**2026-09-01 — Slice 2b is split; September needs only the core.** 2b-core: job → draft
(CV bullets, cover letter) → automatic claim-gate pass → per-sentence verdicts,
autonomous mode only — the gate output *is* the question list, per the generation
design. 2b-full (later): interactive gaps-first mode, sent-document-store voice and
form influence, free-text answers. 2b-core uses the corpus alone as input, so the
form-never-truth ratchet holds trivially.

**2026-09-01 — The September deliverable is the demo page plus the measured number.**
A hosted pre-computed demo showing the distance, and the 210-item eval's over-claim and
over-flag rates, honestly caveated. Execution plan, sequencing, budgets and fences live
in `PLAN.md`.

## Vocabulary — two different gates, never say "gate" alone

This collision has cost real time. Always use the qualified name.

- **Claim gate** — generated text + corpus in, per-sentence grounding verdict out. Stops
  over-claiming. Domains 1 and 2.
- **Job filter** — a stream of jobs + the user's stated preferences in, roles worth
  attention out. Starts as one natural-language prompt ("fully remote engineering
  manager jobs in the UK") and sharpens as the user rejects roles and says why.
  Domains 3 and 4.

The package is `jfl_gate` and the CLI is `jfl check`; both mean the **claim gate**.

## How the claim gate behaves

**It informs, it never blocks.** A user may choose to over-claim, and that is their
call — the product's job is to show the distance between what the corpus supports and
what is being claimed, not to enforce honesty. This follows directly from the thesis:
*measures the distance and shows the number*. Nothing in the pipeline may refuse to
emit a draft because the claim gate flagged it.

## How to develop the model-facing parts

Start from near-default model judgement, watch what it actually does on real text, and
tailor from observed behaviour. **Do not design elaborate prompt scaffolding up front** —
the owner has run a simple version of this in a Claude Code chat with no predefined
structure at all and got useful results by iterating on what he saw. Same rule as the
drift taxonomy: no category without evidence it occurs.

## Drift taxonomy

Confirmed 2026-08-24, drawn from failures the owner has actually seen. These become the
`drift_label` values. **Do not add categories without evidence they occur** — speculative
types make the eval look better than it is, since examples get built for failures that
never happen.

The key property: **verdict is a function of the corpus, not of the type.** "Owned the FX
pricing platform" passes or goes to review depending entirely on whether the corpus says
what ownership entailed. Same sentence, different answer, different day.

**Hard fails** — contradicted, or no corpus addition could rescue them:

| Label | Why it cannot be rescued |
|---|---|
| `invented_quantity` | The number appears nowhere in the source |
| `adjacency_substitution` | Corpus says *reviewed*, claim says *built*. Contradicted, not unsupported |

**Evidence-dependent** — the claim shape is legitimate; grounding depends on what the
corpus holds:

| Label | Evidence it requires |
|---|---|
| `scope_inflation` | The boundary stated: squad, team, department |
| `ownership_inflation` | What ownership entailed: decisions, budget, on-call, headcount |
| `outcome_attribution` | A metric or justification linking the work to the outcome directly |
| `strategy_scope` | The scope named: for the team, the org, the function |
| `causality` | Hard evidence of the author's role in the causal chain |

**Not drift**: `framing` (ungroundable by nature — sequence, motivation, what was being
weighed; **never flag it**, over-flagging is what gets the tool switched off) and
`supported` (traces cleanly).

Temporal compression was proposed and **rejected** — not a failure the owner has seen.

This is the standard supported / refuted / not-enough-evidence split arrived at
independently, which means public FEVER-style data maps onto it directly.

## Golden set

Two tiers, since over-claiming is only detectable against ground truth and ground truth
for a CV is private to its author:

- **Tier 1** — built from public sources (FEVER-style data, open-source contribution
  histories, documented public career records). Tests claim-vs-framing and the core
  mechanic. Needs nothing from the owner. Cannot test defensible compression.
- **Tier 2** — 10–15 items from the owner's own corpus, covering only the
  defensible-compression cases tier 1 structurally cannot reach. Optional; without it the
  number is still real, with a stated limitation.

**Do not source items from CVs paired with application outcomes.** Outcome is dominated
by market conditions and competition; it is near-uncorrelated with whether a claim was
grounded, and training on it would build the tool this project exists to oppose.

## Generation (domain 2) — 2a being built, 2b agreed

Anchored on a **job**, created from a pasted job ad — the antithesis of an ATS, same
anchor entity, opposite direction. Fixed control flow throughout: extract requirements,
check corpus coverage, generate, gate. Not an agent loop.

Two interaction modes, one pipeline:

- **Interactive** — gaps become questions before generating; answers become `adjudicated`
  spans, so the corpus improves with every application. This is the flywheel.
- **Autonomous** — "just generate it". Assumptions get made and the gate marks them, so
  the gate output *is* the question list, delivered after instead of before.

Also in scope: free-text answers for application questions and recruiter conversations,
against corpus plus job context.

**A flagged claim may be a corpus gap rather than drift.** The gate cannot tell
"contradicted" from "not recorded" — a true fact the corpus is silent on looks like an
over-claim. Report the two separately; the fix for a gap is to add the fact, which is the
same write-back path.

## Build order

- **Done** — schema, corpus ingestion, the baseline claim gate end to end. Whole corpus
  in context, single call, verdict out, `runs` row written. CLI only.
- **Done** — slice 2a: job entity from a pasted ad, requirement extraction,
  per-requirement corpus coverage, gap questions, verbatim answer write-back. The rule
  tier, the Inspect harness and the FEVER tier-1 golden set.
- **Now** — `PLAN.md` end to end: gap answers to markdown, slice 2b-core (drafting
  behind the claim gate, autonomous mode), the demo fixtures and hosted demo page, the
  full eval run.
- **Then** — 2b-full (interactive mode, sent-document store), local web UI with the job
  queue the ~2-minute gate latency demands.
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

`packages/core` (storage, schemas, ingestion, embeddings) · `packages/gate` (the claim
gate) · `packages/generate` (domain 2: jobs, requirements, coverage, gap questions, and
later drafting) · `packages/cli` (the `jfl` entry point — a delivery mechanism, not a
domain, so it may depend on every package and none may depend on it) · `packages/evals`
(Inspect tasks, golden set). Migrations in `migrations/`.

## Commands

| Task              | Command |
|-------------------|---------|
| Install           | `uv sync` |
| Database up       | `docker compose up -d` |
| Migrate           | `uv run alembic upgrade head` |
| New migration     | `uv run alembic revision --autogenerate -m "..."` |
| Unit tests        | `uv run pytest` |
| Integration tests | `uv run pytest -m integration` (needs the database) |
| End-to-end tests  | `JFL_ALLOW_REAL_API=1 uv run pytest -m e2e` (**costs money**) |
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

**No test spends API credits by accident, and the marker exclusion is not what stops
it.** `addopts` is a default, and an explicit `-m e2e`, an `--override-ini`, or a CI
job with its own `addopts` walks past it -- and none of that protects against the
likeliest mistake, a new test that constructs a client and was never marked. So the
guard sits at the client: an autouse fixture in the root `conftest.py` replaces
`anthropic.Anthropic` and `AsyncAnthropic` with something that raises, and a real
client requires **both** the `e2e` marker **and** `JFL_ALLOW_REAL_API=1`. Either alone
fails. Tests that install their own fake client are unaffected -- their monkeypatch
runs after the fixture. Do not weaken this to one condition.

The golden set and the Inspect harness live in `packages/evals`. This paragraph used to
say there were none in v1; that was superseded by the 2026-08-24 decision above and the
stale wording is corrected here. Standing rules, unchanged: the golden set is a **test
fixture, never part of a user flow**, and **do not generate synthetic items or adjust
labels to improve a score.** Tier-1 items are drawn from public data, never invented.

The harness measures **over-claim rate** — claims passed as grounded that were not — not
accuracy. A gate that flags everything scores perfectly on over-claim rate and is
worthless, so report the over-flagging rate beside it and never collapse the two into one
number.

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
