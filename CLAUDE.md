# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**jobs4life** — a career system, eventually at `jobs4life.hiltonlabs.org`. It finds roles
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

Long-running, ~10 hrs/week. There was one deadline — something demonstrable and honestly
described by late September 2026 — and it was **met on 2026-09-07**: the demo is live at
`jobs4life.hiltonlabs.org` and the measured numbers are over-claim 0.7% (1/140) and
over-flag 2.9% (2/69) across 210 tier-1 items. Everything after this is product work with
no deadline. **Do not let the absence of a deadline become an argument for scope**: the
owner already uses this daily across scattered Claude conversations, so the bar is
"replaces that", not "demonstrates that".

v1 was single-user with the seams left in. **That phase ended on 2026-09-07** — see the
decisions log. It is now a hosted, logged-in, multi-user web app on a UK VPS behind
Cloudflare. The seams were left in well: every table already carries `user_id`, and
`RequestContext.anthropic_api_key` was already per-request, so multi-tenancy and
bring-your-own-key are enforcement and plumbing rather than migration.

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

**2026-09-15 — Scoring is spent only on jobs the user chooses; "remote friendly" needs
evidence.** Two owner decisions, detailed in `PLAN.md` C7. **A job is scored only when the
user turns it into an application**, never on arrival and never for jobs they merely
browse — users pay for model calls with their own keys, so the model runs when a person has
decided a job is worth it. And workplace filtering has two presets: **remote only**
(strict) and **remote friendly** (remote plus low-commitment hybrid, up to about one day a
week). Platforms label hybrid without the number of days, so **hybrid jobs are included in
remote friendly but badged "days not stated"** — shown for the owner to judge, never
presented as confirmed low commitment. The badge is refined over time by evidence: the
employer's own words ("Remote-Friendly"), a per-board exception in the owner's words, or
later the days read from a posting. Claiming low commitment the label does not state would
be the tool asserting something it cannot see; showing it flagged is not.

**2026-09-10 — Watched job boards: the history is ours, and a check is authoritative or it
changes nothing.** The owner named this the feature he would use most, since LinkedIn has
become largely useless: a personal catalogue of employers' ATS boards, checked daily,
showing what appeared, vanished, came back and was reposted. Design in `PLAN.md` slice C.

**Workday is included despite an undocumented endpoint.** This is a judgement against the
no-scraping rule, made deliberately: it calls the same unauthenticated JSON a public
careers page requests for itself, at a polite rate, and parses structured data rather than
markup — no authentication bypassed, no human-facing page scraped. It is where most large
employers post, so leaving it out would gut the feature. It is labelled fragile, and its
adapter must fail loudly rather than return an empty board when the shape changes.

**History comes from our own observations, never from source dates.** They are not
consistent across platforms: Greenhouse gives timestamps, Workday gives `"Posted Today"`.

**Only a complete check may close a presence interval.** Verified 2026-09-10 against NVIDIA
and Adobe, Workday answers an over-size page with HTTP 200 and zero jobs on one tenant and
HTTP 400 on another; caps an unfiltered listing at 2,000 and then wraps to page 0; wraps
past the end even under that ceiling; and reports `total: 0` mid-pagination. Handled
naively, every one of those records a mass false disappearance — false signal written into
exactly the history this feature exists to show. So unreachable, partial and truncated
checks are recorded and change no job's state. This is the thesis applied to intake: the
tool measures what is true, and a failed fetch is not evidence that jobs vanished.

**A newly watched board's first check is a baseline, not news**, or adding one large board
announces hundreds of "new" jobs and the feed is noise from its first day.

**2026-09-07 — jobs4life becomes a hosted, logged-in, multi-user web app.** Supersedes
"v1 is single-user with the seams left in", "CLI only in this iteration", and the demo /
product split's claim that the live tool stays local. The reasoning is not that the
prototype succeeded — it is that **there was never a prototype to graduate**. The owner
already uses this daily, spread across Claude conversations and other sources, and finds
it useful; the evals and the demo page were evidence of the mechanism, not a trial of the
idea. What conversations structurally cannot do is *remember*: his own words for the gap
are "a clear list of all the applications I have going". That absence is the product.

Consequences, each of which supersedes something above:

- **Google OAuth, sessions, and enforced tenancy**, now, not later. Google login alone —
  no password auth, no email verification, no other providers.
- **Users bring their own Anthropic API key.** The owner does not fund other people's
  model calls; a draft-and-gate cycle is $0.35-0.64 measured, which is unbounded across
  accounts. This makes the project a **custodian of other people's credentials**, so the
  standing envelope-encryption rule stops being theoretical: per-user data key wrapped by
  a master key held only in the host environment, ciphertext in Postgres, never logged,
  never in `runs`, never in a trace, never rendered back to the browser after entry.
- **Hosting is no longer deferred.** OVHcloud VPS-1 (2 vCore / 4 GB / 40 GB), **London**,
  ~£4/month, Ubuntu 24.04, docker compose. UK residency is deliberate: storing UK users'
  career histories and API keys under UK GDPR avoids an international-transfer question
  that would otherwise need answering to every user. Reachable only through a **Cloudflare
  Tunnel** — no public inbound HTTP ports. Runbook in `docs/hosting.md`.
- **The demo page stays exactly as it is**, pre-computed and static on Cloudflare Pages.
  It is evidence, it costs nothing to serve, and it is now the honest front door to a tool
  that lives behind a login. It does not become the app.

**The pre-existing architecture paid for this.** `user_id` on every table and a
per-request `RequestContext.anthropic_api_key` were both decided months earlier for
deployability. Neither needed changing. Keep making that kind of decision.

**2026-09-07 — Every model call is told what time it is.** The owner's stated reason:
Claude "always gets a weird perception of how much time has passed", which breaks
concretely when preparing for an interview tomorrow. So the current timestamp and the
user's live application timeline are injected into model context on every call, and every
prompt and artefact is timestamped in the UI. This is cheap and it is not cosmetic — an
education plan or interview prep that cannot locate itself in time is guessing.

**2026-09-07 — The rating is model-judged, still two axes, still never composited.**
"Non-deterministically build a rating" means a model judgement, not a formula. The
standing rule holds unchanged: **do I want this** and **could I get this** are reported
separately and never averaged, because they diverge constantly and merging them corrupts
both. It ships **unmeasured and labelled as such** — there is no golden set for fit, and
inventing one would be the synthetic-data prohibition in a new coat.

**2026-09-07 — Email intake is a forwarding address, not an inbox integration.**
Reaffirmed, not changed, because the request "point it at my email inbox" reads as
otherwise. Reading a Gmail inbox needs restricted scopes and therefore an annual
third-party security assessment (CASA), which is real recurring money and work. The
capability is kept and the mechanism changed: a dedicated forwarding address, modelled as
one job source among several. Revisit only if the friction proves real in use.

**2026-09-07 — Renamed to jobs4life, except inside the prompts.** The thesis is that
people should be able to move from job to job across a working life, not hold one job for
life — so the singular name argued the opposite of the product. Renamed everywhere it is
user-facing or documentary.

**Deliberately not renamed: the four prompt instruction blocks** (`jfl_gate/prompt.py`'s
`_INSTRUCTIONS`, and `_EXTRACT_INSTRUCTIONS` / `_COVERAGE_INSTRUCTIONS` /
`_DRAFT_INSTRUCTIONS` in `jfl_generate/prompts.py`), which still say "job4life". Prompt
text is what the 2026-09-05 over-claim and over-flag numbers were measured against, and
this project does not get to assert that a change is too small to matter — that is exactly
the reasoning the deleted heuristic rules were built on. The cost of being sure is $2.07,
and a re-run is already required for the two known prompt defects (document titles read as
assertions; the model re-splitting its own input). The prompt rename rides along with that
batch, measured in the same run. Until then the prompts name a product that has been
renamed, which is invisible to users and costs nothing.

**The Python identifiers stay `jfl_*` / `JFL_*`.** The abbreviation expands to "jobs for
life" as readily as it did to "job for life", so renaming ~100 identifiers, the CLI entry
point, and every environment variable would be churn with no reader-facing gain. The
GitHub repo stays `job-for-life` for the same reason.

**2026-09-05 — The product model is Opus 5, and the reason is not accuracy.** Both
models were run over the whole 210-item tier-1 set and compared **paired**, item by
item (`packages/evals/scripts/compare_eval_runs.py`). On the headline numbers they are
indistinguishable: over-claim 0.7% (1/140) and over-flag 2.9% (2/69) on both, and the
paired test shows **zero discordant items** on either — the same single item over-claims
under both models, the same two over-flag. Compared as independent proportions this
would have been an underpowered draw; paired, it is a clean "no difference".

They differ sharply on a third number, which is why that number now exists. **Corpus
silence read as contradiction** — ground truth `review`, gate said `unsupported` — is
22.9% (16/70) on Opus and **71.4% (50/70)** on Sonnet, discordant 34–0 in one direction,
McNemar exact p < 0.001. Neither headline rate sees this, because it is neither an
over-claim nor an over-flag. It matters more here than the raw figure suggests: a gap in
the corpus is supposed to become a gap question and then a new span — that is the
flywheel — and a model that calls silence "contradicted" converts a fixable gap into an
accusation. For a tool whose whole claim is measuring distance honestly, telling someone
the corpus contradicts them when it is merely silent is the worst-tempered error
available.

**And Sonnet is not meaningfully cheaper.** The price sheet says 40% of Opus; the
measured full run was $1.8451 against $2.0715, **11% less**. Sonnet emitted **157,162
output tokens against Opus's 68,420** — 2.3x — and output is ~92% of a check's cost, so
the verbosity almost exactly cancels the price advantage. Do not infer cost from the rate
card for this workload; measure it. Sonnet stays selectable (`--model`, `$JFL_MODEL`) and
its numbers get published beside Opus's, because model choice is a product option, not a
hidden default.

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

**A deterministic rule can prove contact, never crossing — so every rule resolves to
`review`, never `unsupported`.** Rules run **after** the model, may only move `supported`
to `review`, never touch `drift_label`, and never touch framing. A rule names itself in
the sentence's `reason` and in `rule_flags`.

**2026-09-01, later the same day — the two heuristic rules were measured and deleted.**
`unsourced-number` and `boundary-contact` are gone; see `rules.py`'s docstring for the
full evidence. The short version: across three real runs the boundary rule escalated 16,
9 and 14 claims from `supported` to `review` while `drift_label` stayed `supported` —
about 22% of all claims, overriding a model that had traced them cleanly. It matched the
owner's **own name** (boundaries are written "<name> is not …", so the name sits in that
section and scores as rare-hence-distinctive), the word `hold` from the heading
"boundaries to hold", and ordinary CV vocabulary like `systems`, `team`, `code`. The
number rule fired only on years, version strings and phone numbers. This was not a
threshold that needed tuning: unigram overlap cannot separate "he does not have X" from
"his depth *is* in X", because a boundary span states both halves. Reading the negation
is judgement, which is the model's job — and the model already has that section in
context.

What replaced them is **definitional, not statistical**: a cited span id either exists in
the corpus or it does not, and a `supported` claim either carries a citation or it does
not. Neither can produce a false positive because neither estimates anything. They fire
zero times on the owner's real material, which is the correct behaviour for a guarantee.
**Build the rules from observed patterns, never from invented ones** — and delete them
when the observation says to.

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
v1.** The public page at `jobs4life.hiltonlabs.org` is a pre-computed demo over fictional
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

**2026-09-04 — The demo is a subdomain on Cloudflare Pages, not a path.** Supersedes
`hiltonlabs.org/jobs4life` throughout. `www.hiltonlabs.org` is Ghost Pro, and Ghost Pro
breaks behind a Cloudflare proxy — a custom domain cannot even be activated with the
orange cloud on, and proxied sites fail at certificate renewal — so no Worker route can
intercept a path on that host. Ghost's own subdirectory feature is the inverse (Ghost at
`/blog`, static at the root) and is a paid Business add-on. The apex `178.128.137.126` is
**Ghost's shared apex-redirect server**, not ours; leave that A record alone. GitHub Pages
is out because it serves private repos only on paid plans. So: `jobs4life.hiltonlabs.org`
on Cloudflare Pages (free, DNS already in this account), publishing `demo/site` — **only
that directory is published, never the repo** — with `corpus/` and `analysis/` gitignored
so real career data is not in the repo at all. Results are embedded in the HTML at build
time, so the page stays self-contained and works from `file://`.

**Amended 2026-09-07 — publish by direct upload, not a GitHub connection.** The entry
above assumed Cloudflare would clone the repo and build it, which is why it named a
*build output directory*. It then became true that `demo/site/` is committed and there is
no build command, and at that point the git connection buys only auto-deploy while paying
for it with third-party read access to a private repo holding a career system. A demo
page that changes monthly does not need to republish on every push. So: `npx wrangler
pages deploy demo/site --project-name=jobs4life`, authenticated by a `CLOUDFLARE_API_TOKEN`
scoped to **Pages:Edit only**, never an account-wide token. That token is a deploy
credential, not a `RequestContext` one — nothing under `packages/` reads it, and it must
never reach the `runs` table or a trace. Reconnect GitHub only if auto-deploy ever matters
more than the access does.

**2026-09-02 — Never name a structured-output property `reason`.** Every gate call began
returning `stop_reason: "refusal"`, category `reasoning_extraction`, blocked as apparent
reverse engineering or duplication of model outputs — on prompts that had worked the day
before. Bisected against the live API: the system prompt alone is fine, the schema alone
is fine, together they refuse; dropping or renaming the `reason` property clears it.
Independent of `effort` and `max_tokens`, and not scale-sensitive. Renamed to
`evidence_note` in the gate (f87bf78) and coverage (f0fd4ec). The name was always wrong —
the field holds a note about what the corpus supports, not the model's reasoning — but the
lesson generalises: **a long labelling prompt plus a schema demanding a label and a reason
per item reads as a distillation harvest.** Check any new schema against this. The failure
is quiet: HTTP 200, empty content, and a downstream JSON parse error naming the wrong
cause, while the cache write still bills.

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
  in context, single call, verdict out, `runs` row written.
- **Done** — slice 2a: job entity from a pasted ad, requirement extraction,
  per-requirement corpus coverage, gap questions, verbatim answer write-back. The rule
  tier, the Inspect harness and the FEVER tier-1 golden set.
- **Done** — slice 2b-core (drafting behind the claim gate), the nine demo results, the
  live demo page, and the measured number on 210 items across two models.
- **Now — the hosted app, sliced so each one is usable the day it lands.** Detail in
  `PLAN.md`; the ordering principle is that **the tracker comes before the cleverness**,
  because the tracker is the part conversations cannot do.
  - **A** — Google login, enforced tenancy, per-user API key custody, and the application
    tracker with timestamps. Usable alone.
  - **B** — the existing engine behind a UI: coverage, gap questions, drafting, the claim
    gate. Needs the job queue first — a ~2-minute gate call never runs in a request.
  - **C** — intake: Ashby / Greenhouse / Lever / Workable for named employers, plus the
    forwarding address, plus the two-axis model-judged rating.
  - **D** — interview stage tracking, and the rejection-feedback loop that turns an
    outcome into a skill-up plan whose every step names citable evidence.
- **Later** — 2b-full (interactive gaps-first mode, sent-document store), span validity
  periods, offer evaluation, artifact import, MCP server.

## Stack — decided, do not relitigate

Python 3.12 · `uv` workspace · `pydantic` · Postgres + `pgvector` via `docker compose`
(not SQLite) · `alembic` · `sentence-transformers` with a CPU fallback behind the same
interface · `Inspect` (AISI) for evals · Anthropic SDK called directly.

Plus **FastAPI + Jinja + htmx**, server-rendered, for the web app — no SPA, no JS build
chain. Google OAuth via **Authlib**. Envelope encryption via **cryptography** (AES-GCM).

**No LangChain, LangGraph, CrewAI, or AutoGen.** If a tool-use loop is needed, write it.
The CLI stays and stays supported — it is the fastest path to exercising the engine
against real material, which is how every defect that mattered has been found. Still no
MCP server.

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
  never in a trace. Non-negotiable — and as of 2026-09-07 no longer hypothetical, because
  users supply their own Anthropic API keys. A key is write-only from the browser's point
  of view: it can be replaced, never read back. The master key lives in the host
  environment and never in Postgres, so a database compromise alone yields no usable key.
- **Tenancy is enforced structurally, not by convention.** A repository is constructed
  with the `user_id` it may act on and has no per-call override; there is no code path
  where forgetting a `WHERE user_id = ...` is possible, because callers cannot express it.
  A `WHERE` clause that a reviewer has to notice is not enforcement.
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

**A second guard sits at the socket, for the same reason** — added 2026-09-10 with the
board-watching engine, whose checks make real HTTP requests. An autouse fixture beside the
Anthropic one refuses any Python-level TCP connection to a non-local address and refuses to
resolve any name other than `localhost`, so a test that reaches for the internet fails
loudly instead of being quietly recorded as an `unreachable` board. It lifts under exactly
the API guard's two conditions — the `e2e` marker **and** `JFL_ALLOW_REAL_API=1` — and the
same rule holds: do not weaken it to one. Loopback, the unspecified address and Unix sockets
stay reachable; `httpx.MockTransport` and fake transports never open a socket at all.

**Know its limits rather than overstate them.** It patches Python's `socket` module, so it
covers httpx, urllib and anything built on Python sockets — but **not C libraries that do
their own networking** (libpq, libcurl/pycurl), and not raw UDP. The database keeps working
under it precisely because libpq bypasses it. An adapter that ever adopts a C-level HTTP
client walks straight past this guard, so that change must bring its own isolation.
`getaddrinfo` is only a pre-filter; enforcement is in `connect`, which checks the address it
is actually given — so a name that resolves somewhere unexpected is still refused.

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
