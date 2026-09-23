"""What the paste box does before the worker gets to it.

Slice B3 replaced a six-field form with a textarea, because the owner's words
were "the user has to be realistic that a link may not always work, copy and
pasting in is a good place" and "this needs to be fast input (slow processing is
acceptable)". Fast input means the POST cannot call a model, which means the
application needs a title before anything has read the ad. That is what
`provisional_title` is for: a placeholder, honestly labelled as one, that the
background extraction is allowed to replace and a typed title is not.

The failure messages live here rather than in the worker for the same reason the
database stores a code and not a sentence: the worker writes that column while
holding the user's decrypted API key, and a closed set of codes cannot carry a
secret. Wording is a UI concern, changeable without a migration.

**No URL fetching.** The form takes a URL and stores it as a link, and nothing
in this app requests it. Three reasons, in order of weight: most ATS pages
render client-side, so a fetch returns an empty shell that is strictly worse
than the paste it would be competing with; fetching a URL a user supplies makes
this app an SSRF cannon that would need an allowlist, private-range blocking, a
redirect cap and a timeout budget to be safe; and LinkedIn and Indeed are
excluded by standing decision anyway. The paste box is the feature.
"""

from __future__ import annotations

from dataclasses import dataclass

from jfl_core.models import ExtractionErrorCode

# Comfortably longer than any real job ad (~25k tokens) and short enough that a
# stray paste of a whole website does not become a paid model call.
MAX_AD_CHARS = 100_000

# Longer titles get elided. Long enough for "Senior Engineering Manager,
# Payments Platform (Remote, UK)"; short enough to sit on one line in the list.
MAX_TITLE_CHARS = 80

FALLBACK_TITLE = "Untitled role"

# Leading decoration a pasted ad's first line tends to carry: markdown heading
# marks, bullets, and the box-drawing a PDF copy-paste leaves behind.
_LEADING_NOISE = "#*-•–—=_ \t|>"


def provisional_title(ad_text: str) -> str:
    """A placeholder title taken from the ad's first line of real text.

    Deliberately dumb: this is not extraction, it is a label for a row that
    exists before extraction has run. Getting it slightly wrong is fine, because
    it is marked provisional and gets replaced. Getting it *expensive* would not
    be -- no model call may happen in the request path.
    """
    for line in ad_text.splitlines():
        candidate = " ".join(line.strip(_LEADING_NOISE).split())
        if not any(ch.isalnum() for ch in candidate):
            continue
        if len(candidate) <= MAX_TITLE_CHARS:
            return candidate
        cut = candidate[: MAX_TITLE_CHARS - 1]
        # Trim back to a word boundary if there is one worth trimming to.
        if " " in cut[MAX_TITLE_CHARS // 2 :]:
            cut = cut.rsplit(" ", 1)[0]
        return f"{cut.rstrip()}…"
    return FALLBACK_TITLE


def normalise_url(raw: str) -> str | None:
    """The typed URL, or None. Raises `ValueError` on something that is not one.

    Only a scheme check. The URL is stored and rendered as a link and is never
    requested by this app, so there is nothing here to defend beyond "an
    `href` a browser will treat as a link to elsewhere" -- which rules out
    `javascript:` and friends, and that is the whole job.
    """
    url = raw.strip()
    if not url:
        return None
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("A link needs to start with http:// or https://")
    return url


@dataclass(frozen=True, slots=True)
class ExtractionFailure:
    """What to tell the user, and where to send them to fix it."""

    message: str
    fix_url: str | None = None
    fix_label: str | None = None


_FAILURES: dict[ExtractionErrorCode, ExtractionFailure] = {
    "no_api_key": ExtractionFailure(
        "This needs your own Anthropic API key -- reading an ad is a model call, "
        "and it is billed to you rather than to this app.",
        fix_url="/settings",
        fix_label="Add an API key",
    ),
    "api_key_rejected": ExtractionFailure(
        "Anthropic rejected the API key stored here. Replace it and try again.",
        fix_url="/settings",
        fix_label="Replace the key",
    ),
    "no_job_ad": ExtractionFailure("There is no ad text stored against this application."),
    "ad_too_long": ExtractionFailure(
        "That ad is too long to read in one pass. Paste the role and requirements "
        "on their own, without the company's boilerplate."
    ),
    "model_refused": ExtractionFailure(
        "The model declined to read that text. If it is an ordinary job ad, "
        "trying again with just the role and requirements usually works."
    ),
    "model_error": ExtractionFailure("Reading the ad failed. Trying again is worth a go."),
    "credential_unreadable": ExtractionFailure(
        "Your stored API key could not be unlocked on the server. Setting it again will fix it.",
        fix_url="/settings",
        fix_label="Set the key again",
    ),
}

# Anything unrecognised -- a code added to the database before this table caught
# up -- still gets a sentence rather than a blank panel.
_UNKNOWN = ExtractionFailure("Reading the ad failed. Trying again is worth a go.")


# A pending read or fetch that carries an error code: one attempt failed and
# the queue will try again. Not an error yet -- see `note_extraction_retry`.
EXTRACTION_RETRYING_NOTE = (
    "Reading the ad hit a temporary problem and is trying again by itself -- "
    "nothing for you to do. If it keeps failing it stops and says so here."
)
FETCH_RETRYING_NOTE = (
    "Couldn't reach the board just now -- trying again by itself. If it keeps "
    "failing you will be asked to paste the ad."
)


def extraction_failure(code: ExtractionErrorCode | None) -> ExtractionFailure:
    return _UNKNOWN if code is None else _FAILURES.get(code, _UNKNOWN)
