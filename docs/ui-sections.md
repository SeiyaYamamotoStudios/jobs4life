# Collapsible sections, and the spacing they impose

Agreed 2026-09-22, from the owner's complaint: *"We have some overflow and
alignment problems. Sections should be collapsible, with a mechanism to know
that something has changed in a collapsed section, and be intelligent about
what's collapsed by default. Currently there's a lot of scrolling — for example
an application with all the essentials and desirables listed at the top."*

This is the contract. The next screen uses it rather than inventing a third
style.

## The component

One Jinja macro, `templates/_sections.html`, wrapping `<details>`/`<summary>`
and nothing else.

```jinja
{% import "_sections.html" as sections %}
{% call sections.section(timeline_section) %}
  ...
{% endcall %}
```

It takes one argument: a `jfl_web.sections.Section`, built in Python by the
route. The template holds no rules at all — not which sections open, not what
the summary says, not whether there is a marker. That is deliberate: the rules
are the interesting part, and they are unit-testable only if they are not in a
template (`packages/web/tests/test_sections.py`).

| Field | What it is |
|---|---|
| `key` | Stable id, e.g. `application.ad`, `score.levers`, `draft.<uuid>` |
| `title` | The heading, rendered as `<h2>` (or `<h3>` at `level=3`) inside the `<summary>` |
| `summary` | The **count summary**: what is inside, said while it is folded |
| `marker` | The **change marker**: `1 new`, or `updated`, or empty |
| `open` | Whether it renders open — the answer after all three rules below |
| `default_open` | What this page would have shown with no stored choice |
| `level` | 2, or 3 for a section nested inside another |

### Why `<details>` and not a script

It folds with JavaScript switched off. The keyboard reaches the `<summary>` for
free. The browser's own in-page find opens it. The heading keeps its level
inside the `<summary>`, so folding a panel never costs the page its outline. A
collapsed section's content is **still in the HTML** — nothing is fetched on
open, so nothing can fail on open.

`static/sections.js` is the only script involved, and it is enhancement: block
it and every section still works, the choice is simply not remembered. It exists
because the value to post on toggle is the element's `open` *property*, which is
not an attribute htmx can read; the listener reads that one boolean and hands
the request straight back to `htmx.ajax`. It is registered in the **capture
phase**, because `toggle` does not bubble — one listener on `document` then
covers sections that arrive later inside an htmx swap, with nothing to re-bind.

## The three rules, in order

1. **Forced.** Something is pending or needs attention — a running extraction, a
   failed run, an unanswered question, a prerequisite not met. A forced section
   is open **whatever the user has chosen before**, because the choice was made
   about a different situation and this one is transient. This is the one place
   the user's own choice does not win, and it is a deliberate exception: leaving
   someone staring at a folded panel while the thing they paid for runs behind
   it is worse than overriding a preference for as long as the run lasts.
2. **The user's own choice**, if they have ever toggled this section.
3. **The agreed default**, otherwise.

## The defaults

Open by default: **status and timeline**, and **anything pending**. Folded:
**requirements**, **older drafts**, **older scores**. The newest draft and the
current score stay open. Agreed with the owner; the table is the whole of it.

| Section | Open by default | Forced open when | Count summary |
|---|---|---|---|
| `application.status` | yes | — | — |
| `application.timeline` | yes | — | `9 events` |
| `application.notes` | only when something is written | — | `empty` |
| `application.cv` | yes | a step of the CV sequence is running or stopped | `12 of 18 claims trace to your facts` |
| `application.ad` | no | the ad is unread, being fetched, failed, or absent | `12 requirements · 8 essential` |
| `application.score` | yes | not scored yet, pending, or failed | `1 must-have broken · 7 constraints · 2 objectives` |
| `application.questions` | no | any question unanswered, running or failed | `4 questions · 3 answered` |
| `score.constraints` | yes | — | `7 constraints` |
| `score.objectives` | yes | — | `2 objectives` |
| `score.levers` | no | — | `3 claims` |
| `score.not_stated` | no | — | `4 questions` |
| `score.pushbacks` | no | a correction is still being read | `3 corrections` — the box itself always shows; the fold holds past corrections, and the drift sentence under the score is never folded |
| `drafts.requirements` | no | coverage has not been checked | `12 requirements · 8 evidenced` |
| `drafts.generate` | yes | — | — |
| `drafts.history` | yes | — | `3 drafts` |
| `draft.<id>` | only the newest | — | `Tue 8 Sep 2026, 3 days ago · 12 sentences` |
| `profile.*` (five) | only while nothing is stated in it | a clustering run is pending; a save has just redirected here | `2 practised · 1 ruled out` |

Two of these are worth their reasoning.

**The score's constraint and objective verdicts stay open.** The silences are
the product — what the ad says nothing about is what to ask at interview — and
folding them by default would hide what the panel is for. The levers and the
not-stated list fold, because both are prompts to go and do something elsewhere
rather than findings about this job.

**The profile folds the sections you have already answered.** A section you have
stated nothing in is the one that still wants you, so it stays open; an answered
one folds behind a count. A new user's profile therefore looks exactly as it did
before this change, and a filled-in one is short. A save redirects to
`?saved=1&open=<section>#<section>` — a URL fragment never reaches the server,
and landing on a folded panel after pressing Save reads as the save having been
lost.

## The change marker

A dot and a short count — `Drafts ● 1 new` — following the convention the
changes feed already uses: *what changed since you last looked*, counted, with
the moment you last looked recorded rather than assumed.

It is derived from timestamps the data already carries. Nothing new is written
to track it.

| Section | Measured from |
|---|---|
| `application.ad` | `extracted_at` — the ad was re-read → `updated` |
| `application.score` | `updated_at` — a new run landed → `updated` |
| `application.questions` | each answer's `updated_at` → `N new` |
| `application.timeline` | each event's `occurred_at` → `N new` |
| `score.pushbacks` | each correction's `created_at` → `N new` |
| `drafts.history` | each draft's `created_at` → `N new` |
| `profile.capabilities` | the clustering run's `created_at` → `updated` |

Three rules hold everywhere:

- **It clears when the section is opened.** `last_opened_at` is written on every
  toggle, in both directions: opening means you are looking at it now, closing
  means you have just finished. Either way, everything currently inside has been
  seen.
- **An open section never carries one.** Its contents are already on screen; a
  badge counting them is noise.
- **No watermark means no marker.** A section nobody has ever opened or closed
  has no `last_opened_at`, and the honest answer to "what is new since you last
  looked" is then "we have never seen you look". Inventing a baseline would
  announce a year of old drafts as news on a first visit — the same failure a
  newly watched job board's first check avoids by being a baseline rather than
  news (CLAUDE.md, 2026-09-10). The cost is that the marker only starts working
  for a section once it has been touched once. That was accepted.

## Where the state lives, and why there

`ui_section_states(user_id, section_key, is_open, default_open, toggles,
against_default, last_opened_at, …)`, one row per (user, section), upserted on
toggle by `POST /ui/sections` and **never written on render**. A page render is
one `SELECT` of this user's rows.

**Not in `profiles`.** That row is append-only and is read back as what the user
believed about themselves on a given day; a new version per folded panel would
bury real decisions under UI noise. **Not in `localStorage`** either, and the
reason is not sync: the owner asked for *"we will have to track if people go
against this."* So every toggle also records what the screen would have shown
without the stored choice, and `against_default` counts the disagreements. A
default that everybody immediately undoes is then a query rather than something
somebody eventually notices:

```sql
select section_key, count(*) filter (where is_open <> default_open) as against,
       count(*) as users
from ui_section_states group by 1 order by against desc;
```

`default_open` is reported by the page rather than recomputed server-side,
because "what would this screen have shown without your choice" depends on state
the toggle route does not have. It is the user's own telemetry about their own
account, so a tampered value muddles nothing but their own record of their own
clicks.

`section_key` carries no CHECK constraint: the set is open by construction — a
per-draft section is keyed by the draft's own id — so a closed list would need
migrating every time a screen grows a panel. The route validates the key's
**shape** (`^[a-z0-9][a-z0-9._-]{0,119}$`) and silently ignores anything else.
The route answers `204` either way: the worst outcome of a lost write is that a
panel comes back open next time, and an error banner about layout preferences in
front of someone writing a cover letter would be worse than the thing it
reports.

## Spacing and overflow conventions

Fixed once, in `static/style.css`, rather than per page. A new screen inherits
all of this by using the component and the existing classes.

**One spacing scale.** `--space-1` (0.35rem) through `--space-5` (2.4rem). Three
screens each had their own idea of the gap between a heading and what it
introduces, and the seams showed on any page that mixed them. Sections own their
own vertical rhythm: `--space-5` above a top-level section, `--space-4` between
siblings and for a nested one, and `.section-body > :first-child` has its top
margin removed so the first thing in every panel sits in the same place.

**A `<section>` that exists only to carry an htmx polling target adds no margin
of its own** (`section:has(> details.section) { margin: 0 }`). Those wrappers
are what produced the uneven gaps between the ad, the score and the questions
panels.

**Long unbroken strings break rather than push — and which property goes
where is the whole rule.** *Corrected 2026-09-23; the first version of this
paragraph was wrong and cost a desktop layout.*

- `overflow-wrap: break-word` for **all text**, set once on `body` and
  inherited. Ordinary words stay whole; only a run that cannot fit its line at
  all is broken. It does **not** lower an element's min-content width, which is
  why it is safe in a table cell.
- `overflow-wrap: anywhere` **only on runs with no break opportunity**:
  `code`, `.url-text` (a URL shown as its own text — e.g. a board with no
  label), `.id-text`, a draft's `.draft-citations li`, and a `.drift-dimension`
  name. Scope it by class; never on `td`, `th`, `a`, `li`, `dd`, `.note` or any
  other general text selector.

Why it matters: `anywhere` also counts every character as a soft wrap
opportunity *when the browser measures min-content width*. The first version
of this convention put it on `td, th, dd, dt, li, blockquote, .note, a`
wholesale, so every cell's minimum collapsed to about one character, and the
automatic table layout gave /changes' job titles ~40px ("En / gin / eer / ing")
and /boards' employer names "Anthr / opic", while the `nowrap` timestamp and
action cells kept their full width and ran off the container edge.
`tests/test_table_layout.py` fails if it comes back.

**Wide tables: the primary column has a floor, the actions never clip.**

- The column a row is *about* carries `col-primary` on its `<td>` — the job
  title on /jobs and /changes, the board on /boards, the title on
  /applications — and gets `min-width: 14rem` (12rem for the board name). When
  a table is short of room, the secondary columns (employer, locations,
  workplace) give way first; with `break-word` none of them goes below its
  longest word.
- An action's label never wraps (`white-space: nowrap` on the button, summary
  or badge), so the actions column's minimum is its widest control. The cell
  itself wraps, so two actions stack rather than demanding the sum of their
  widths.
- Timestamps in a table cell use `| time_compact`: `<time datetime="…"
  title="Wed 23 Sep 2026, 13:04 · 13 minutes ago">13 min ago</time>`. The long
  `| humanize_dt` form ("Wed 23 Sep 2026, 13 minutes ago") stays where there is
  room and the date matters — detail pages, fact lists, timelines — but
  repeated per column it was most of /boards' width.
- Pages with a `table.boards` (jobs, changes, boards) get `main` up to 80rem;
  reading pages keep the 56rem measure, and those pages' `.lede` keeps it too.
  /changes needs roughly 66rem of column minimums, so it fits without
  scrolling from a ~1100px window upward.

**Every table scrolls inside `.table-wrap`** — at narrow widths only. It
keeps `min-width: 34rem` so a narrow window scrolls it rather than crushing
six columns into two words each; between that and the column minimums above, a
table scrolls inside its wrapper on a small laptop or split screen and never at
a normal desktop width. Under 40rem the floor is reset to 0 and the tables
become cards (see `tests/test_mobile_layout.py`), except `table.cvs`, which
keeps a 30rem floor and scrolls. `table.cvs` had never been given the shared
table rules at all, which is why its rows sat unaligned next to every other
list in the app; it has them now.

**Flex items that hold text get `min-width: 0`.** A flex item's default
`min-width: auto` refuses to shrink below its longest word, which is how one
long employer name pushed a two-column panel wider than the page. Named
selectors rather than a blanket rule, so a deliberate `white-space: nowrap`
elsewhere is not quietly overridden.

**A card's actions belong to the card.** A panel of cards — title
suggestions, profile proposals — puts each card's secondary action (Dismiss,
Reject) in the same action row as its primary one, as a quiet link, not in a
second box underneath. Where both act on the same record, one form with a
`formaction` on the secondary button does it. Watch descendant selectors on a
wrapping section: `.job-filter form` once boxed every form inside the
suggestions panel, which is exactly how Dismiss ended up in a box of its own;
it is `.job-filter > form` now.

**Rows of controls wrap as rows.** `.quick-actions` wraps; `.status-badge` never
does. Text inputs, selects and textareas are `max-width: 100%; box-sizing:
border-box`, because `rows`/`cols` set a width the box model otherwise refuses
to give back.

## What the tests prove, and what they do not

`tests/test_ui_sections_web_integration.py` and
`packages/web/tests/test_sections.py` cover the defaults, forcing, persistence,
the `against_default` count, tenancy, CSRF, and the marker appearing and
clearing.

They assert on **markup**, because there is no browser in the test environment.
What is proved is that the server emits `<details>` with the right `open`
attribute, summary and marker, that a folded section still carries its whole
content, and that the element carries no `hidden` and no `hx-` trigger. What is
**not** proved here: that a browser paints the disclosure triangle, that the
capture-phase `toggle` listener fires, or that the htmx post goes out. Those
need a browser and have not been verified.
