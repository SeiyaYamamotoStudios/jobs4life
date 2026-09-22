"""How much of a CV is stored, and how much of it is read.

These were one number, `jfl_web.corpus.MAX_CV_CHARS`, which conflated two
different things it was protecting: the row a CV is stored in, and the model
call that reads it. Raising the first meant raising the second, which meant
paying more per CV for nothing -- so the limit could not be raised and the
owner's real PDFs, with their headers, footers and repeated contact blocks,
sat against it.

They are separate here because **the honest answer to a long CV is not the same
on both sides**. Storage can be generous: bytes are cheap and the document is
the author's own record of what they claimed. The model call cannot, because a
user pays for it with their own key. So a long CV is stored whole and read in
part, and the screen says exactly that -- "stored, but only the first N
characters are read" -- rather than truncating quietly, which would make the
facts proposed from it look like the whole of the document.

It lives in `jfl_core` because two packages must agree on the read ceiling and
neither may depend on the other: `jfl_web` writes the sentence the user reads,
and `jfl_generate` makes the call. A constant in one of them and a sentence in
the other is exactly how the two drift apart.
"""

from __future__ import annotations

# What the model is given, per CV, in `jfl_generate.cv_facts.extract_cv_facts`.
#
# ~100k characters is ~25k input tokens: a 40-page CV, several times any real
# one, and already generous for a call the user is billed for. It is
# deliberately NOT raised alongside the storage ceiling -- there is nothing on
# page 41 of a CV that changes which facts a person should confirm, and reading
# it would just cost them more. `jfl_generate` also caps its own output at 16k
# tokens, so a document far past this would truncate the response and fail the
# whole read with `cv_too_long`; stopping at a known point instead turns that
# failure into a sentence.
MAX_CV_READ_CHARS = 100_000


def for_reading(text: str) -> tuple[str, bool]:
    """(what to send the model, whether anything was left behind).

    Cut on a line boundary where there is one within the last 2% of the
    allowance, so the model is not handed half a sentence and asked to quote
    it back as somebody's own words.
    """
    if len(text) <= MAX_CV_READ_CHARS:
        return text, False
    head = text[:MAX_CV_READ_CHARS]
    break_at = head.rfind("\n")
    if break_at >= MAX_CV_READ_CHARS - (MAX_CV_READ_CHARS // 50):
        head = head[:break_at]
    return head, True
