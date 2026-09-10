# ATS platforms — verified behaviour

Evidence behind slice C's board adapters (`packages/intake`). Every row was probed live on
**2026-09-10** against a real employer's board, unauthenticated. These surfaces are
undocumented or loosely documented and they change — re-verify anything older than a few
months before trusting it.

Why this file exists: every trap below, missed, writes false appearances or disappearances
into the history slice C exists to show. See CLAUDE.md, 2026-09-10, and `PLAN.md` C3.

## Summary

| Platform | Endpoint | `external_id` (posting) | `requisition_id` | Completeness | Verified on |
|---|---|---|---|---|---|
| Greenhouse | `GET boards-api.greenhouse.io/v1/boards/{token}/jobs` | `id` | `requisition_id` | single response | Anthropic — 595 |
| Ashby | `GET api.ashbyhq.com/posting-api/job-board/{name}` | `id` | — | single response; honour `isListed` | Ashby — 68 |
| Lever | `GET api.lever.co/v0/postings/{company}?mode=json` | `id` (title is `text`) | — | single response | `leverdemo` — 13 |
| **Workday** | `POST {t}.{wdN}.myworkdayjobs.com/wday/cxs/{t}/{site}/jobs` | `externalPath` suffix, e.g. `R171808-1` | `bulletFields[0]` | **paged, with traps** — below | NVIDIA — 2,630; Adobe — 730 |
| SmartRecruiters | `GET api.smartrecruiters.com/v1/companies/{id}/postings` | `id` | `refNumber` | paged by the **echoed** `limit` — below | Bosch — 4,835 |
| Rippling | `GET api.rippling.com/platform/api/ats/v1/board/{slug}/jobs` | `uuid` (title is `name`) | — | single response, **one row per job × location** — below (v2 pages at 20 — use v1) | Rippling — 648 rows, 347 jobs |
| Breezy | `GET {company}.breezy.hr/json` | `id` | — | single response | Breezy — 3 |
| Teamtailor | `GET {careers site}/jobs.rss` | `guid` | — | single feed | career.teamtailor.com — 12 |
| Personio | `GET {company}.jobs.personio.de/xml` | `id` | — | single feed | Personio — 1 |
| Recruitee | `GET {company}.recruitee.com/api/offers/` | `id` | — | single response | Make — 3 |
| Pinpoint | `GET {company}.pinpointhq.com/postings.json` | `id` | `job.id` | single response, no paging keys | Sun King — 206 |
| **Workable** | `POST apply.workable.com/api/v3/accounts/{sub}/jobs` | `id` (URLs use `shortcode`) | — | **token paging** — below | Devsinc — 33 |

**Unconfirmed: BambooHR.** Search surfaced only BambooHR's own careers page and job
aggregators, never a customer's board to test against.

"Single response" on a small board proves little about caps on a large one. Only Greenhouse
(595), Rippling (648 rows, 347 jobs) and Pinpoint (206) were observed returning a large list in one response.

## Keys: the posting, never the requisition

Every adapter keys a job by the **finest-grained id the platform exposes — the posting**. A
requisition can be listed as several postings, and a re-listing typically gets a new posting
id under the same requisition; keying by requisition would make that look like a job that
never went down, hiding exactly the repost pattern slice C exists to show. `requisition_id` is
stored where a platform provides one, and no repost rule uses it yet. See `PLAN.md` C5.

## Workday — four traps, all verified

Against NVIDIA (`nvidia.wd5` / `NVIDIAExternalCareerSite`) and Adobe (`adobe.wd5` /
`external_experienced`). The request body is
`{"appliedFacets":{},"limit":20,"offset":N,"searchText":""}`.

1. **A page size over 20 fails — inconsistently.** NVIDIA: HTTP 200, zero `jobPostings`, and
   **no `total` key**. Adobe: HTTP 400. Always send `limit: 20`. A 4xx, or a 200 missing
   `total`/`jobPostings`, is a **failed** check — never an empty board. A genuinely empty
   board is `total: 0` *present* with an empty list at offset 0.
2. **An unfiltered listing exposes at most 2,000 jobs, then wraps.** Offset 2000 returned page
   0 exactly (20 of 20 ids identical); offsets 2500–8000 kept returning full pages. NVIDIA's
   facet counts sum to **2,630**, so 630 jobs are invisible to plain paging.
3. **It wraps past the end even below the ceiling.** Adobe (730): offset 700 → 20, offset 720 →
   **10 (the real end)**, offset 740 → page 0 again.
4. **`total` reads 0 on intermediate pages** (NVIDIA 1980, Adobe 700). Trust it only at offset 0.

**Paging stops** on the first of: a short page; an id already collected this check; the offset
reaching offset-0's `total` below 2,000; or a hard page cap (which makes the check `incomplete`).

**A board at the 2,000 ceiling is split by facet.** Offset 0 returns `facets`. Use a group whose
value counts form a partition and whose largest value is under 2,000 — NVIDIA's
`jobFamilyGroup`: 15 values, largest ~1,725. `workerSubType` (max 2,290) and `timeType`
(max 2,628) are too coarse; `locationMainGroup` over-counts because jobs have several
locations. Fetch each slice via `appliedFacets`, union, and **verify the union against the
partition's summed count**. No usable facet → `truncated`.

`postedOn` is prose — `"Posted Today"` — and must never be parsed into a date. The stable id
is also in `bulletFields[0]`; on Adobe, 5 of 20 postings carry a suffix the requisition lacks
(`R171808` vs `R171808-1`).

## Workable — token paging, clean

- Pages carry a `nextPage` token; send it back as `"token"` in the next request body.
- Devsinc: pages of 10, 10, 10, 3; `total` read **33 on every page**; no repeats; the last page
  omits `nextPage`. Unlike Workday, no wrap and no misreported total.
- **Complete** = no token remaining **and** unique ids equal page 1's `total`.
- Workable's own board slug is not `workable`; guessed slugs are unreliable — take the URL.

## SmartRecruiters — a silent page-size cap

Against Bosch (`BoschGroup`, 4,835 postings) — five targeted requests, not an exhaustive
page-through of a large board:

- **A page size over 100 is silently capped.** `limit=200` returned HTTP 200 with 100 postings
  and echoed `"limit": 100`. No error, and not empty — so an adapter that advances `offset` by
  the size it *requested* skips 100 jobs every page and never notices. **Advance by the echoed
  `limit`.**
- `totalFound` read 4,835 at offsets 0, 2400 and 4800 — consistent across pages, unlike
  Workday's.
- The last page (offset 4800) returned exactly 35; offset 4835 returned 0 with no overlap with
  page 0. **Terminates cleanly, no wrap.**
- **Complete** = a short or empty page reached **and** unique ids equal `totalFound`.

Three paged platforms, three different answers to an oversized page: Workday errors on one
tenant and goes silently empty on another; SmartRecruiters silently caps and tells you so;
Workable pages by token and never lets you choose. No adapter may assume it got the page size
it asked for.

## Pinpoint

- `/postings.json` and `/jobs.json` are **different resources**: 206 each on Sun King, id sets
  disjoint apart from one coincidental collision. **Postings are the grain.**
- Each posting's `job` object links to its parent (`job.id`) and carries the employer's own
  `job.requisition_id`, which is often empty. Store **`job.id`** as `requisition_id`: always
  present, and literally the posting's parent. On Sun King every job had exactly one posting.
- Fields are top-level — not JSON:API `attributes`/`relationships`.
- The public posting URL uses a UUID path; the API `id` is numeric.
- `/api/v1/*` is the authenticated API (HTTP 401). Not used.

## Rippling — one row per job, per location

Verified 2026-09-11 against Rippling's own board (v1): **648 rows but only 347 distinct
`uuid`s.** 129 jobs appear more than once — up to 20 copies of a single `uuid` — and across
those copies **only `workLocation` differs** (128 of the 129; one job repeats identically).
"Account Executive, Broker Channel (Pittsburgh or Cleveland)" is two rows: one for Cleveland,
one for Pittsburgh.

- **The job is the `uuid`; a row is a job × location.** Count jobs by distinct `uuid`, or a
  347-job board reads as 648 and the drop guard compares the wrong number.
- **Merge a job's locations deterministically — distinct, sorted — never keep the first row.**
  First-occurrence makes the stored location depend on the order Rippling happens to return
  rows, and location is half the repost fingerprint, so a reordering alone could manufacture a
  false repost.

## Personio — no verifiable posting URL

The XML feed carries no URL field. The obvious pattern, `{company}.jobs.personio.de/job/{id}`,
returned **HTTP 429 on both of two separate attempts**, and the redirect lands on personio.com's
marketing homepage. It is unverified, so the adapter leaves the job URL empty rather than ship a
guess. Do not probe it repeatedly — Personio rate-limits hard, and politeness is a requirement
here, not a courtesy.

## Greenhouse

- `internal_job_id` is **not** a repost signal: one requisition was listed under two public ids
  with different titles ("Account Executive, AI Native" and "…Startups") — variants of one
  requisition, not a repost.
- Real `first_published`/`updated_at` timestamps — but history still comes from our own
  checks, for consistency with platforms that give prose.

## Client identity

The engine identifies itself honestly: `jobs4life-board-check/0.1 (+https://jobs4life.hiltonlabs.org)`.
All twelve platforms accept an honest identifying User-Agent. Workable refuses `Python-urllib`
(HTTP 403) while accepting `python-httpx`, `curl` and honest clients — so a 403 is a failed
check, never an empty board.

**Never disguise the client as a browser.** A platform that refuses honest clients has declined
to be read that way, and the board falls back to copy-and-paste.
