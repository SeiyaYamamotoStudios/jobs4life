# PLAN — jobs4life as a tool the owner actually uses

Written 2026-09-07, replacing the September demo plan (archived at
`docs/PLAN-september-demo.md`, delivered in full). Decisions behind this plan are in
CLAUDE.md's decisions log, 2026-09-07 entries; this file is sequencing, designs, fences
and acceptance criteria only.

**There is no deadline and that is a risk, not a licence.** The September deliverable is
shipped. The bar from here is not "demonstrates the idea" — the owner already uses this
daily across scattered Claude conversations and finds it useful. The bar is **"replaces
that"**, and the thing conversations structurally cannot do is remember. In his words:
*"a clear list of all the applications I have going."*

So the ordering principle throughout: **the tracker comes before the cleverness.** Every
slice below must be usable on its own the day it lands. If a slice would only make sense
once the next one exists, it is sliced wrong.

## The four slices

| | What lands | Usable alone as |
|---|---|---|
| **A** | Google login, enforced tenancy, per-user API key custody, application tracker | The list of live applications, with real timestamps |
| **B** | Queue, quick-actions, paste-an-ad, two scores, the engine behind a UI | Paste an ad and get it scored, drafted and gated without a terminal |
| **C** | Watched job boards: adapters, our own history, what changed | A personal catalogue of employers showing what appeared, vanished and came back |
| **D** | Interview prep, and rejection feedback into a plan | A closed loop from rejection to citable evidence |

---

## Slice A — the shell that remembers

Nothing model-facing. No Anthropic call anywhere in this slice except one optional
validation ping. That is deliberate: it means A can be built, deployed and used while
costing nothing, and it isolates auth bugs from engine bugs.

### A1 — Google login

Authorization Code flow with PKCE via **Authlib**. Scopes `openid email profile`, nothing
else — no Gmail scope, ever (see the forwarding-address decision).

**Identify users by Google's `sub`, never by email.** Email addresses change hands;
`sub` is stable and unique forever. Store email and name for display only, refreshed on
each login.

### A2 — Sessions

Opaque session id in a cookie, session row in Postgres. **Not a JWT** — this app holds
other people's API keys, so instant server-side revocation matters more than statelessness
at one-VPS scale.

Cookie: `__Host-` prefix, `HttpOnly`, `Secure`, `SameSite=Lax`, no `Domain` attribute.
Row carries `user_id`, `created_at`, `expires_at`, `last_seen_at`, and a rolling
expiry. Logging out deletes the row, not just the cookie.

### A3 — Tenancy, enforced structurally

Per CLAUDE.md: a repository is **constructed with** the `user_id` it may act on and has
no per-call override. There must be no code path where forgetting a `WHERE user_id = ...`
is expressible.

**Acceptance is a test, not a review:** a test that reflects over every public repository
method and asserts none of them accepts a `user_id` argument. A `WHERE` clause a reviewer
has to spot is not enforcement, and this is the failure that leaks one user's career
history to another.

### A4 — Credential custody

Envelope encryption, per the non-negotiable standing rule.

- **KEK** — 32 bytes, base64, from `JFL_MASTER_KEY` in the host environment. Never in
  Postgres. A database compromise alone therefore yields nothing usable.
- **DEK** — 32 random bytes per user, AES-GCM-encrypted under the KEK.
- **The API key** — AES-GCM-encrypted under that user's DEK.
- Table `user_credentials`: `user_id`, `provider`, `wrapped_dek`, `dek_nonce`,
  `ciphertext`, `nonce`, `key_hint` (last 4 characters, for display), `created_at`,
  `last_used_at`.

**Write-only from the browser's side.** A key can be set or replaced, never read back —
the UI shows `sk-ant-…4f2a` and a Replace button. Never logged, never in `runs`, never in
a trace, never in an error message.

Validate on entry with one cheap Anthropic call so a typo fails at the form rather than
three screens later.

### A5 — The application tracker

The wedge. Two tables:

- `applications` — `user_id`, `job_id`, `status`, `source`, `notes`, timestamps
- `application_events` — every status transition, with a timestamp and optional note

Statuses: `interested` → `applied` → `screening` → `interviewing` → `offer`, plus
terminal `rejected` and `withdrawn`. Transitions are recorded, never overwritten: the
event log *is* the timeline, and it is what later gets injected into model context.

Job entry already works — `jfl job add` parses a pasted ad today. Slice A puts a textarea
in front of it.

### A6 — Timestamps everywhere

Per the 2026-09-07 decision. Current time plus the user's live application timeline go
into model context on every call from slice B onwards; in slice A it is UI only — every
event shows an absolute date *and* a relative one ("Tue 8 Sep, 3 days ago"), because
"tomorrow" is exactly what the owner said gets lost.

### Slice A acceptance

1. Two Google accounts, side by side: neither can see or reach the other's applications
   by any URL, including guessed ids.
2. The repository-method reflection test passes.
3. An API key can be set and replaced, never read back, and appears nowhere in logs,
   `runs`, or a traceback.
4. Deployed on the VPS behind the Cloudflare Tunnel, reachable over HTTPS, with the
   demo page at `jobs4life.hiltonlabs.org` still serving and untouched.
5. The owner can add a real job ad and move it through states — and prefers doing that
   to keeping the list in a conversation. **If he doesn't, slice A has failed** and no
   amount of slice B fixes it.

---

## Slice B — the engine, behind a UI

Revised 2026-09-08 after the owner used slice A. His feedback drives the ordering
below; everything except B2 needs the queue, which is why the queue is first.

### B1 — the job queue, first

A gate call takes ~2 minutes and an extraction is not much quicker. Nothing that slow
runs inside a request. His words set the requirement exactly: **fast input, slow
processing is acceptable** — so the form returns immediately and the work happens behind
it.

`tasks` table with `SELECT … FOR UPDATE SKIP LOCKED`, an attempt counter, a max-attempts
cap, and a global `JFL_DISABLE_MODEL_CALLS` kill switch. The worker is a separate
container so it can be stopped without taking the site down — which is also the incident
response if a user's key starts burning money. It also gets the jobs nothing else owns:
purging expired sessions, and later the demo regeneration.

### B2 — status quick-actions (no queue, do it early)

A dropdown plus submit is wrong for the 90% case. A primary button for the **likely next
state** — "Mark as applied", "Mark as screening" — with the full picker still present for
jumps and reversals. Cheap, and the most-used control in the app.

### B3 — paste an ad, get an application

Replaces the current long form. A paste box, an optional URL, and nothing else required.
The extraction engine already exists (`jfl_generate.extract`, exercised today by
`jfl job add`) so this is a form, a task, and a results view.

**Paste is primary and the URL is best-effort.** Most ATS pages render client-side and
many refuse non-browser fetches, and LinkedIn and Indeed are excluded by standing
decision — not only legally, but because a scraper in the architecture reads as poor
judgement. So: try the URL, fall back quietly, never pretend it is reliable.

This is also the slice that proves the queue end to end, which is why it comes first
among the model-facing work.

### B4 — two scores on arrival, never one

On add, the application is scored automatically, with a paragraph for each score.

**Two axes, never composited** — the standing decision, and the owner's three requests map
onto it cleanly: *chances* is **could I get this**, *alignment to interest* is **do I want
this**, and *appropriateness* is a blend of the two, which is exactly what must not be
built. A role he would love and will not get, and one he would hate and would walk into,
must never land on the same number; averaging them hides the disagreement precisely when
it is the useful signal.

Ships **labelled unmeasured**. There is no golden set for fit and inventing one would be
the synthetic-data prohibition in a new coat. The over-claim rate remains the measured
number and must not be confused with these.

### B5 — the rest of the engine's screens

Coverage, gap questions, drafting ("Generate a CV" from an application), and the claim
gate with per-sentence verdicts. All built; all needing a UI and the queue.

**Framing renders as `NOT CHECKED`, never as supported** — the same rule the demo page
obeys. Per-run cost is shown, because the user is paying for it with their own key.

### B6 — corpus upload

A user with no corpus has nothing to measure against, so this gates B's usefulness for
anyone but the owner. Markdown upload through the existing ingestion, spans stored
per-user. "Markdown is the source of truth" holds: uploads are stored and re-ingestible,
never only indexed.

## Slice C — watching job boards

Revised 2026-09-10. The owner's words: *"perhaps the one I would find most useful
myself"* — LinkedIn has become largely useless, so this builds a personal catalogue of
employers he is interested in and shows what changed. **The input is one line; the
aggregation, history and comparison are the product.**

**Sequencing: C1–C6 come before B4.** They need no model calls and nothing from B4, and
this is the feature the owner has said he would use most.

### C1 — sources

One adapter per ATS, each normalising to a thin record — platform, external id, title,
location, URL. Full descriptions are fetched lazily, only for jobs the user opens.

The user pastes the board's URL and the adapter is chosen by URL pattern — deterministic.
Nothing tries to discover a company's ATS from its name; web search may *suggest* boards
later (C8).

Verified live 2026-09-10 — public, unauthenticated, real jobs returned from a real
employer's board: **Greenhouse, Ashby, Lever, Rippling, SmartRecruiters, Breezy,
Teamtailor (RSS), Personio (XML), Workday, Workable, Recruitee, Pinpoint** — twelve.
Unconfirmed: **BambooHR** — search surfaced only BambooHR's own careers page and
aggregators, never a customer's board. Everything else is an AI-parsed careers page (C8)
or a paste.

Finding a real tenant mattered more than finding the endpoint. Guessed slugs failed for
reasons that had nothing to do with the platform — Workable's own board is not
`workable`, Pinpoint customers use their own subdomains — which is the same reason the
input is a pasted board URL and not a company name.

**Workday is included** despite an undocumented endpoint — see CLAUDE.md, 2026-09-10.

### C2 — the history is ours, never the source's

Every job's timeline comes from our own checks: first seen, last seen, and **presence
intervals**. Source dates are not trusted, because they are not consistent — Greenhouse
gives real timestamps, Workday gives prose (`"Posted Today"` on all 20 of a sample).

Presence is stored as intervals, not per-check sightings: a job continuously present is
one row, and vanishing then returning opens a second. That represents exactly the events
that matter — appeared, disappeared, came back — and stays compact, where per-check
sightings for one 2,600-job board checked daily would be ~950k rows a year.

**Store everything, filter the view.** A company's hiring velocity, or whether its
engineering-manager roles keep churning, is signal even when the jobs are not for you —
and history that was not kept cannot be recovered, whereas a filter is cheap to change.

### C3 — a check is authoritative, or it changes nothing

The rule the whole feature rests on: **only a complete, successful check may close a
presence interval.** A job absent from a check is gone only if that check provably saw the
whole board. Unreachable, errored, partial and truncated checks are recorded as such and
change no job's state.

This is not theoretical. Workday, verified 2026-09-10 against NVIDIA and Adobe:

- **A page size over 20 fails, inconsistently.** NVIDIA returns HTTP 200 with zero jobs
  and no `total` key — silent. Adobe returns HTTP 400. Same mistake, one loud and one
  silent; both are failures and neither may ever read as "empty board".
- **An unfiltered listing exposes at most 2,000 jobs, then wraps.** Offset 2000 returns
  page 0 exactly, and offsets 2500–8000 keep returning full pages. NVIDIA really has
  2,630 (its facet counts sum to that), so 630 are invisible to paging.
- **It wraps past the end even under the ceiling.** Adobe (730) returns a short page of 10
  at offset 720, then page 0 again at 740.
- **`total` reads 0 on intermediate pages**, so it cannot be used to stop.

So Workday pagination stops on the first of: a short page, a job id already seen in this
check (wrap), or the offset reaching page 0's `total`. A board at the 2,000 ceiling is
split by a facet whose values partition it with every slice under the ceiling — NVIDIA's
`jobFamilyGroup`, 15 values, largest 1,725 — and the union is **verified against the
partition's summed count**. No such facet: the check is marked truncated. Unique ids short
of the expected total: the check is incomplete.

**A drop guard**, from an observed pattern rather than an invented one: a "successful"
check returning far fewer jobs than the last complete one holds intervals open and flags
the board. The silent 200-with-zero-jobs above is exactly what a broken adapter looks
like, and it is far likelier than a company closing every role overnight.

### C4 — a baseline is not news

The first check of a newly watched board is a **baseline**. Its jobs are "open when you
started watching", never "new" — otherwise adding Anthropic's board announces 595 new jobs
and the feed is noise from its first day. "New" means first seen by a check *after* the
baseline.

Likewise the UI says **"seen since 24 Jul"**, never "posted 24 Jul". We know when we saw a
job, not when it was posted.

### C5 — returned versus reposted

Two different events, kept distinct:

- **Returned** — the same external id vanishes and comes back. Read straight off the
  presence intervals.
- **Reposted** — a *different* external id with the same normalised title and location, at
  the same board, appearing after an equivalent vanished. A deterministic fingerprint.

Greenhouse's `internal_job_id` is **not** a repost signal — verified: one requisition is
listed under two public ids with different titles ("Account Executive, AI Native" and
"…Startups"). That is one requisition's variants, not a repost.

**Jobs are keyed at the finest grain a platform exposes — the posting, not the
requisition.** Adobe's Workday page 0 on 2026-09-10: 20 postings, 20 distinct
requisitions, 5 carrying a posting suffix (`R171808` → posting `R171808-1`). Keying by
requisition would throw that suffix away, so a role re-listed as `-2` would read as a job
that never went down — hiding exactly the repost pattern this slice exists to show, in
history that could never afterwards be re-keyed. `requisition_id` is stored alongside
where a platform provides one (Workday, Greenhouse), but **no repost rule uses it yet**: a
`-2` re-listing has not been observed, and rules come from observed patterns. Storing it is
what lets that pattern be learned from real history later.

Assumed until the owner says otherwise: a repost falls within 60 days of the
disappearance. Configurable.

### C6 — cadence

Daily scheduled checks through the worker, staggered and rate-limited per platform, plus
"check now". The deterministic path makes no model calls and costs nothing — but a
2,630-job Workday board is 130+ requests a check, so politeness is a requirement, not a
courtesy. Watches are per user in v1: simple, and tenant-safe by construction; sharing a
board's fetch across users is an optimisation for later.

### C7 — what you see, and what changed

Settled with the owner on 2026-09-15. Built and deployed already: the Boards screen, and
`/jobs` — one aggregated list of every open job across watched boards, through **one saved
filter used everywhere**, with per-board exceptions in the owner's own words.

**Workplace modes — two named presets, replacing loose checkboxes as the main control.**
- **Remote only** — strict. Jobs the employer states are remote. Nothing labelled on-site
  or hybrid, whatever the location text says. The owner's example: Primer, remote-first.
- **Remote friendly** — remote only, **plus low-commitment hybrid: from a couple of days a
  month up to one day a week.**

**Hybrid is included in remote friendly, flagged rather than excluded** (owner,
2026-09-15). Platforms say "Hybrid" without saying how many days, so a hybrid job appears
under remote friendly **badged "Hybrid — days not stated"**, never presented as confirmed
low commitment. The owner eyeballs these, and the badge is refined over time as evidence
arrives:
- the employer's own words — `remote-friendly`, `remote first` and similar, in a platform
  field, Greenhouse custom metadata or location text — mark a job as stated low-commitment.
  This resolves Anthropic's 38 jobs labelled On-Site in metadata whose location reads
  "Remote-Friendly (Travel-Required)": excluded under remote only, included under remote
  friendly, with the conflict visible on the row rather than silently resolved;
- a per-board exception in the owner's words (e.g. "Anthropic: ~1 day a week") marks that
  board's matching jobs as known low-commitment;
- the owner can mark a board's hybrid as too heavy, taking it out of remote friendly;
- later, the number of days read from a posting's description when it is fetched.

A job labelled **on-site** is still excluded from remote friendly unless the employer's
words or an exception say otherwise.

Jobs whose workplace is unstated keep the existing per-board rule and hidden-count line.

**The "what changed" feed.**
- **Since the user last looked** — not since the service last checked. Each user has a
  last-looked time, advanced when they view the feed.
- An event (new, gone, returned, reposted) appears when it happens and **stays visible for
  24 hours after the user first sees it, or until they dismiss it** — whichever comes first.
  Dismissal is per user and per event.
- The feed goes through the same saved filter and exceptions as `/jobs`.
- A board's baseline check is never news (C4), and a failed or partial check produces no
  events (C3).

**"Track as application"** — a button on a job that turns it into an application. It uses
the model exactly as a manual paste does: fetch the posting's description from the platform
(lazily, only for this job), then run the existing B3 extraction through the queue. Nothing
is extracted or scored for jobs the user only looks at.

**Scoring happens only on an explicit action.** A job is scored when the user turns it into
an application (B4), never on arrival and never for jobs they merely browse. This spends
the user's own key only when they have decided a job is worth it, and supersedes C8's
"scoring on arrival".

### C7a — suggested title expansions, informed by what we know about the user

Agreed with the owner 2026-09-15. When the user adds a title phrase to the filter, a cheap
model call (Claude Haiku 4.5, ~$0.001 per phrase; confirm on first real uses) **suggests
adjacent titles** — `engineering manager` → EM, SEM, senior engineering manager, engineering
lead, software development manager. **Suggestions are offered with a tickbox each, never
added silently;** only accepted ones become match terms, so matching stays exact and free.
Runs once per phrase, through the queue, on the user's own key, under the model kill switch,
and is cached.

**Context makes the suggestions right, not generic.** Abbreviations are ambiguous — "SEM" is
also Search Engine Marketing — so the call is given what the user has already told us:
- their other title phrases, **including excludes** (excluding "marketing" settles SEM);
- the titles of applications they are tracking (archived ones excluded);
- later, once corpus upload exists (B6), a short excerpt of their current role and seniority
  — **an excerpt, not the corpus**: minimum necessary data, and the call stays a fraction of
  a penny.

Context biases suggestions toward the user's current track, which is mostly what they want;
the prompt should still allow a few adjacent-but-different titles, and the tickbox is the
safeguard either way.

Schema note, per CLAUDE.md 2026-09-02: return a list of titles, optionally with a very short
gloss per title — **never a property named `reason`**, and no label-plus-reason-per-item
shape, which is what tripped the `reasoning_extraction` classifier before.

### C8 — later in slice C

- **AI-parsed careers pages** — manual trigger, costs the user per check, and guarded
  against SSRF, since it means fetching user-supplied URLs server-side.
- **Suggested boards and companies** — labelled as model suggestions, with the skew named:
  a model knows famous companies and least about precisely the small ones worth finding.
  **No sponsorship and no adverts, ever.**
- **The email forwarding address** as a source.
- ~~Scoring on arrival~~ — **superseded 2026-09-15**: scoring happens only when the user
  turns a job into an application (see C7).

## Slice D — the loops that close

### D1 — interview preparation

Its own section, showing every application in `interviewing`. Paste the recruiter's prep
notes; get a plan anchored to the actual interview date — which is what the timestamps in
A6 exist for.

**It must say what is not worth preparing.** The owner raised this and it is not a
footnote: this tool exists to stop someone presenting a distorted picture of themselves,
and a prep plan that rehearses them into a person they are not is that same failure in
different clothes. Naming the two or three things that matter, and explicitly saying the
rest is over-preparation, is the product's thesis applied to interviews.

### D2 — rejection, feedback, and the plan

A rejection captures its reason, and that reason **is stored verbatim**. No model tidies
it. It is the only external ground truth in the system and it is scarce — the same
reasoning that keeps gap answers in the user's own words.

That feeds a gap-analysis and action-plan section, **separate from applications** because
it is about the person over time rather than about one role.

**Every step names the requirement it closes and the evidence it would produce. A step
that creates no citable evidence is not a step.** That is what stops a plan degenerating
into "get better at Kafka" and makes it "build X, which produces Y". And when that
evidence exists it becomes a corpus span — which is the flywheel closing: a rejection
eventually improves the corpus the claim gate measures against.

## Cross-cutting, before the model-facing work

- **Error references.** Every unhandled exception gets a short reference shown to the
  user and logged with the traceback. Not a browser traceback: the failure that motivated
  this was on `/auth/google/callback`, which is not behind authentication, and a stack
  through `envelope.py` names the crypto layer to whoever hit it.
- **Drop `users.email`'s UNIQUE NOT NULL.** Email is presentation-only now that identity
  is Google's `sub`; the constraint makes a reassigned address a hard login failure for
  its new owner.
- **The prompt batch.** Two known defects — document titles read as assertions, the model
  re-splitting its own input at initials — plus the deferred prompt rename, in one change
  and one $2.07 eval re-run. **Before drafting goes in front of anyone but the owner**, or
  the published over-claim rate describes a prompt that is not running.


## Risks worth naming now

**Slice A is boring and that is the point.** The temptation will be to jump to B because
the engine is the interesting part. The engine already works and is measured; the reason
the tool isn't used daily is that there is nowhere to put an application. Resist.

**Holding other people's API keys raises the stakes on every mistake.** A logging bug is
now a credential disclosure. This is why custody is in slice A rather than bolted on when
the first non-owner user appears.

**Two prompt defects are known and unfixed** — document titles read as assertions, and
the model re-splitting its own input at initials. Both need a prompt change, which
invalidates the 2026-09-05 eval baseline, so they are batched with the deferred prompt
rename into one $2.07 re-run. Do that before slice B puts drafting in front of a second
user.

**The demo page must not rot.** It is the honest front door and it is pre-computed. If
the pipeline changes materially, regenerate it — nine combinations, ~$2.82 — or take it
down. A page claiming "this is what the tool outputs" must stay true.
