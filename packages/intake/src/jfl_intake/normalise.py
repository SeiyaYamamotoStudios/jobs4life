"""Normalisation, and the fingerprint the repost signal compares.

A *repost* is a job with a new external id that is really an old role put back
up -- the same title, in the same place, on the same board. External ids cannot
see that (they are exactly what changed), and Greenhouse's `internal_job_id` is
not a substitute: one requisition was verified listed under two public ids with
different titles. So the comparison is on what a person reading the board would
compare: the title and the location, normalised.

**Exactly what is normalised**, in this order:

  1. Unicode NFKC, so a full-width or ligature character compares equal to its
     plain form;
  2. `casefold()` -- lowercase, including the non-ASCII cases `lower()` misses;
  3. every character that is not alphanumeric (`str.isalnum`) becomes a space.
     Punctuation, symbols and separators all go, so `Sr. Engineer (Remote)` and
     `sr engineer remote` compare equal. *Replaced by a space, not deleted*,
     so `Engineer,Platform` does not fuse into one word;
  4. runs of whitespace collapse to one space, and the ends are stripped.

Nothing else. No stemming, no synonym table ("Sr" is not "Senior"), no
reordering. Every one of those would be a guess about what counts as the same
role, and a guess here writes a false "reposted" into the history the owner
relies on. A fingerprint that is too strict misses a repost, which leaves an
honest "new"; one that is too loose invents a relationship.

The fingerprint is `"<title>|<location>"`. The separator cannot occur inside
either half, because step 3 removes it, so the join is unambiguous. A missing
location normalises to the empty string.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalise(text: str | None) -> str:
    """Apply the four steps above. None and whitespace-only become ``""``."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text).casefold()
    spaced = "".join(ch if ch.isalnum() else " " for ch in folded)
    return _WHITESPACE.sub(" ", spaced).strip()


def fingerprint(title: str, location: str | None) -> str:
    return f"{normalise(title)}|{normalise(location)}"


def clean_text(value: object) -> str | None:
    """A display string from an API field: stripped, None if absent or blank.

    Deliberately NOT `normalise` -- this is what the owner reads, so case and
    punctuation are kept. Non-strings are treated as absent rather than
    stringified: an adapter that gets a dict where a title should be has met a
    shape it does not understand, and `str(dict)` is not a title.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
