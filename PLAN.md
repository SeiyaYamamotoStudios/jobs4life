# PLAN — to a working demo at job4life.hiltonlabs.org

Written 2026-09-01 (Fable planning session). Executed by an Opus orchestrating session
delegating to Sonnet implementation agents, per CLAUDE.md's delegation convention.
Decisions behind this plan are in CLAUDE.md's decisions log (2026-09-01 entries);
this file is only sequencing, fences, budgets, and acceptance criteria.

**Deadline:** something demonstrable and honestly described by late September 2026.
~4 weeks × ~10 h/week ≈ 40 h. Committed scope is ~21 h; everything else is stretch.

**The deliverable is two things:** the hosted pre-computed demo (fictional material,
real pipeline output), and the measured number (the 210-item eval's over-claim and
over-flag rates, caveated). The live tool stays on the owner's machine — see the
decisions log for why the corpus never leaves it in v1.

## Standing answers assumed (owner has approved the recommendations)

- Public demo + local private tool, not hosted multi-user. No in-app auth.
- Gap answers append to `corpus/answered-questions.md` and re-ingest.
- 2b-core only for September; 2b-full deferred.

## Open questions — resolved 2026-09-04

1. **Where does `hiltonlabs.org` serve from?** Answered: `www` is Ghost Pro (via Fastly),
   the apex A record is Ghost's own shared redirect server, and the zone is on Cloudflare.
2. **VPS or Pages?** Answered: Cloudflare Pages, on the subdomain `job4life.hiltonlabs.org`,
   connected to the private GitHub repo. See CLAUDE.md's 2026-09-04 decision for why the
   path-based URL is not available and why GitHub Pages is not an option.
3. **Credit top-up:** still outstanding. ~$0.59 remains. Revised ask is ~$15, not the $10
   estimated earlier — see the cost-variance finding in NEXT.md.

---

## W0 — gap answers to markdown (~2 h) — first, it unblocks nothing else but touches shared files

`jfl answer` appends the verbatim answer as a bullet under `corpus/answered-questions.md`
(single h1, so `section_path` is stable), runs ingestion, computes the resulting span id
with `ids.span_id(...)`, stores it in `gap_questions.resulting_span_id`, marks the
question answered. Remove the direct `add_adjudicated_span` call from this path only.

- Fence: `packages/cli/src/jfl_cli/main.py` (answer command), `packages/core` ingest/
  storage only as strictly needed, plus tests. Corpus root must be injectable so tests
  use a tmp dir — check how `run_ingestion` locates `corpus/` before assuming.
- Acceptance: answering a question creates a line in the markdown file and a
  `provenance='document'` span; re-running `jfl ingest` is idempotent; the old DB-only
  path is gone; unit tests cover append format, id computation, and idempotency.

## W1 — slice 2b-core: drafting behind the claim gate (~10 h)

New `drafts` table (id, user_id FK, job_id FK, kind CHECK `cv_bullets|cover_letter`,
text, gate_result JSONB, trace_id, created_at) + migration. `jfl_generate/draft.py`:
one model call — corpus in a cached system block exactly as coverage does it, job +
requirements + latest coverage in the user message — then an **automatic** claim-gate
pass on the output via `jfl_gate.check_text` (the decisions log requires this: the gate
runs automatically on generated text). CLI: `jfl draft JOB_ID --kind cv|cover_letter`,
printing the draft with per-sentence verdicts (reuse the existing printers, framing as
NOT CHECKED) followed by the open gap questions. A flagged draft is still emitted — the
gate informs, it never blocks.

- Grounding input is the corpus **only**. 2b-core must not read `sent_documents` at all.
- `packages/generate` now depends on `packages/gate`; the CLI package split was done
  precisely so this cycle-free dependency works — no workspace surgery expected.
- Exactly one `runs` row per model call, as everywhere. Expected cost ≈ $0.33/draft
  (draft call + gate pass, warm cache).
- Acceptance: end-to-end on the owner's real corpus and one real job ad — draft out,
  verdicts attached, run rows written; unit tests with fake clients; one integration
  test for the drafts table. Verified by the orchestrator on real material before the
  workstream closes (evidence before code).

## W2 — demo fixtures: fiction authoring (~4 h, no API calls) — parallel with W1

`demo/fixtures/`: 3 fictional candidates (each a verification record in the corpus
format, 40–60 spans, including a stated-boundaries section) × 3 job ads. Each candidate
built to exhibit specific labeled drift when their fictional CV claims are checked —
over-claiming in ways the taxonomy names, plus supported claims and framing, plus at
least one under-claim. Fictional names, invented companies, no resemblance to real
people. Committable — that is the point of fiction.

- Acceptance: each record parses through `parse_document` cleanly; boundary sections
  match the corpus-format markers; a README in `demo/fixtures/` states these are demo
  fixtures, not golden-set items, and why that distinction matters.

## W3 — demo results generation (~2 h + credit) — needs W1 + W2 + top-up

A script (committable, `demo/generate_results.py`) that, per candidate, ingests the
fictional corpus under a scratch user id and runs the real pipeline for all 9
combinations: extract → coverage → draft → gate. Saves full JSON per combination to
`demo/fixtures/results/`. These are committable (fiction).

Budget, from W1's **measured** per-call costs rather than estimates: 3 extractions
(~$0.012 each, one per job ad, reused across candidates) + 9 coverage (~$0.157) + 9
drafts (~$0.43, which is a draft call *and* its gate pass) ≈ **$5.30**. Hard-stop the
script if projected spend exceeds $8. Note the draft figure is ~29% above the original
$0.33 estimate because the draft call and the gate call sit behind different instruction
prefixes and so cannot share a cache entry — each pays a full corpus cache write, and
neither ever gets a cache read.

- Acceptance: 9 result files, each carrying the real `runs`-style token/cost numbers so
  the demo can honestly show what a check costs; a regeneration is one command.

## W4 — the demo page (~5 h) — needs W3

Static site: a build script renders `demo/site/` from the fixtures (results embedded
as JSON, small vanilla JS for the candidate × job picker — no framework, no server).

**Amended 2026-09-05: no template engine.** This said "Jinja at build time", which does
not survive contact with the rest of the sentence: if every dynamic element is rendered
client-side from the embedded JSON, the server side has exactly one substitution to make
— the JSON blob itself. Jinja would be a dependency with no work to do, and the build
container then needs it too. `demo/build_site.py` is stdlib-only and swaps a placeholder
in `demo/template/index.html`, which stays editable as real HTML. The page's centrepiece is the thesis: what was claimed, what the
corpus supports, the distance, per-sentence. Verdict colours as in the CLI; framing
rendered as NOT CHECKED — the page must never assert a verification that didn't happen.
A visible note that the material is fictional and the results are unedited real
pipeline output, with the per-run cost and latency shown.

- Deploy per the answers to open questions 1–2 (default: Cloudflare Pages + a route).
- Acceptance: page works from `file://` (truly static), passes a squint test on mobile
  width, and deploys to the real URL.

## W5 — the full eval run (~1 h + $4.44) — needs top-up; independent of W2–W4

`inspect eval ... -T limit=210` with `JFL_ALLOW_REAL_API` semantics respected (the eval
is outside pytest; its own default limit of 5 is the guard). Record over-claim rate,
over-flag rate, and the framing × over-claim counter in `packages/evals/README.md`,
with the CV-domain caveat already written there. This also measures the framing prompt
change from 2026-09-01, which is currently unverified.

## W6 — stretch only: local web UI + job queue (~12 h)

`tasks` table + `SELECT … FOR UPDATE SKIP LOCKED` worker (attempt counter, max
attempts, global `JFL_DISABLE_MODEL_CALLS` kill-switch), FastAPI (`root_path`-aware
from day one) + Jinja/htmx on localhost. Do not start unless W0–W5 are done and the
weeks allowed slack. This is October work wearing September clothes.

---

## Orchestration rules (all learned this session; do not relax)

1. Every agent brief names an explicit **file fence**, including root `tests/`
   ownership. Overlaps caused every integration failure this session.
2. **Subagents never run git commands.** A `git stash` by one agent reverted all four
   agents' work; recovery was luck. The orchestrator commits and pushes at green
   checkpoints (owner has approved push-as-you-go on `scaffold-and-schema`).
3. Agents get an explicit **API budget or "no API calls"** in the brief. The test suite
   cannot spend (conftest guard); scripts still can.
4. Evidence before code: measure on real material before and after; an agent reporting
   an untested fix is asked to run it, not thanked.
5. Sonnet implements; Opus orchestrates, reviews, and keeps CLAUDE.md/PLAN.md edits to
   itself. Escalate a brief to Opus-tier only on demonstrated fumble or genuinely
   subtle work.
6. Verify agent self-reports independently when they touch shared state (the corpus
   near-miss and the stash misattribution both happened this session).

## Weekly map

- **Week 1:** W0, W1 started; W2 in parallel.
- **Week 2:** W1 finished and verified on real material; top-up; W3.
- **Week 3:** W4 built and deployed (needs open questions 1–2 answered); W5 run.
- **Week 4:** buffer for what slipped; September writeup with the eval numbers; W6 only
  if genuinely clear.
