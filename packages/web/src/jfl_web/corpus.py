"""What the CV upload form accepts, and what it says when it does not.

Slice B6, extended for PDF. The rules live here rather than in the route for
the same reason `jfl_web.jobads` holds the job-ad ones: the worker writes a
**code** from a closed set into `cv_extractions.error_code` while holding the
user's decrypted API key, and the sentence a person reads is a UI concern that
must be changeable without a migration.

**PDF is read, and then shown back before it is believed.** This step used to
take `.md` and `.txt` only, and the refusal was deliberate: PDF extraction gets
the column order wrong on exactly the two-column CVs people use, and a mangled
line quoted back as "your own words" is the one thing CV onboarding must never
do. Real CVs are PDFs, so the refusal could not stand -- but the reason behind
it still does. What replaces it is not trust in the extractor; it is a review
screen. The text comes out, the author sees it, edits whatever came out wrong,
and only then is anything stored or read. The class of failure the refusal was
protecting against is closed by the author's eyes, not by a better library.

**`.docx` is still refused.** There is no maintained pure-Python reader for it
(`python-docx` needs lxml, a C extension), and hand-rolling a zip-plus-XML
reader would add a parser for untrusted uploads to earn a format the paste box
already covers. The message names what does work instead of saying "no".

**Plain text is untouched.** A `.md` or `.txt` upload is the bytes the author
wrote; there is nothing to review, so it stores and queues in one step exactly
as before. A batch containing a PDF goes to review as a whole, because one
upload with two different outcomes is worse than one extra screen.

**No model call in the request path.** Uploading stores bytes and queues work;
reading the CV happens in the worker, on the user's key.
"""

from __future__ import annotations

from dataclasses import dataclass

from jfl_core.cv_limits import MAX_CV_READ_CHARS
from jfl_core.ingest.pdf import PdfUnreadable, read_pdf
from jfl_core.models import CvExtractionErrorCode

TEXT_SUFFIXES = (".md", ".txt", ".markdown", ".text")
PDF_SUFFIXES = (".pdf",)
ALLOWED_SUFFIXES = TEXT_SUFFIXES + PDF_SUFFIXES

# Formats worth naming in a refusal rather than lumping into "not supported":
# these are the ones people actually try, and a message that names the format
# is a message that sounds like it was expected.
_WORD_SUFFIXES = (".docx", ".doc", ".odt", ".rtf", ".pages")

# -- what each ceiling protects -----------------------------------------------
#
# These were one number. It protected two different things at once -- the row a
# CV is stored in and the model call that reads it -- so it could not be raised
# for the first without paying for the second. They are separate now.

# The stored document. A `text` column and one `sent_spans` row per non-blank
# line, so what this really bounds is row count: 500k characters is roughly ten
# thousand lines, which is a 200-page document and several times the longest
# real CV. PDFs push far harder against this than markdown did -- repeated
# headers, footers, contact blocks and a two-column layout read twice all land
# in the extracted text -- which is why the old 100k was the limit people met.
MAX_CV_CHARS = 500_000

# What the model is given lives in `jfl_core.cv_limits.MAX_CV_READ_CHARS`,
# imported above rather than redeclared: the sentence the user reads and the cut
# the call actually makes have to be one number, not two that agree today.

# One request. The owner's case is thirty-three CVs in one go; 200 is several
# times that, and it bounds the work one POST does (a store and an enqueue
# each) rather than anything about the account.
MAX_FILES_PER_UPLOAD = 200

# One file, before decoding. A text CV is a few tens of kilobytes; a PDF with
# an embedded photograph and subsetted fonts runs to a few megabytes. 20 MiB is
# past any CV and short of anything that would be worth holding in memory to
# find out.
MAX_FILE_BYTES = 20 * 1024 * 1024

# One request, before decoding. Deliberately below Cloudflare's 100 MB body
# ceiling, so a user who sends too much gets this sentence rather than
# Cloudflare's error page -- a limit the app cannot explain is a limit the user
# cannot act on. Note what this does NOT do: Starlette has already spooled the
# body to disk by the time a route sees an `UploadFile`, so this bounds what is
# decoded, held as `str` and written, not what crossed the wire.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

# A filename is a label, and after the review screen it is a label the browser
# posted back, so it is treated as user input on the way in.
MAX_FILENAME_CHARS = 200


class UploadRejected(Exception):
    """A file the form will not take. The message is shown to the user."""


@dataclass(frozen=True, slots=True)
class UploadedCv:
    """One CV on its way in, plus what to tell the author about how it was read.

    `note` is the caption on the review screen -- "Read from 3 pages of
    cv.pdf" -- and is None for a file that needed no interpreting.
    """

    filename: str
    text: str
    note: str | None = None
    extracted: bool = False


def clean_filename(filename: str) -> str:
    name = " ".join(filename.replace("/", "-").split()).strip()
    return name[:MAX_FILENAME_CHARS] or "CV"


def suffix_of(filename: str) -> str:
    lowered = filename.lower()
    for suffix in ALLOWED_SUFFIXES + _WORD_SUFFIXES:
        if lowered.endswith(suffix):
            return suffix
    return ""


def check_suffix(filename: str) -> None:
    if suffix_of(filename) in ALLOWED_SUFFIXES:
        return
    raise UploadRejected(
        f"{filename} is not a format this step reads. PDF, .md and .txt work; "
        "Word and OpenDocument files do not. Save it as a PDF, or copy the text "
        "out and paste it in below."
    )


def check_upload_size(filename: str, raw: bytes, running_total: int) -> None:
    """Both byte ceilings, checked before anything is decoded or parsed."""
    if len(raw) > MAX_FILE_BYTES:
        raise UploadRejected(
            f"{filename} is {_mib(len(raw))}, past the {_mib(MAX_FILE_BYTES)} one file "
            "may be. If it has photographs or scanned pages in it, save a text-only "
            "copy, or paste the text in instead."
        )
    if running_total > MAX_UPLOAD_BYTES:
        raise UploadRejected(
            f"That is more than {_mib(MAX_UPLOAD_BYTES)} of files in one go. "
            "Upload them in a few batches."
        )


def _mib(count: int) -> str:
    return f"{count / (1024 * 1024):.0f} MB"


def read_upload(filename: str, raw: bytes) -> UploadedCv:
    """One uploaded file as text, however it arrived."""
    if suffix_of(filename) in PDF_SUFFIXES:
        return _from_pdf(filename, raw)
    return UploadedCv(filename=filename, text=decode(filename, raw))


def _from_pdf(filename: str, raw: bytes) -> UploadedCv:
    try:
        pdf = read_pdf(raw)
    except PdfUnreadable as exc:
        # The exception's message is written as the tail of a sentence about a
        # named file, so that one sentence names the file the user must fix.
        raise UploadRejected(f"{filename} {exc}") from None
    pages = "1 page" if pdf.pages == 1 else f"{pdf.pages} pages"
    return UploadedCv(
        filename=filename,
        text=check_length(filename, pdf.text),
        note=f"Read from {pages} of {filename}.",
        extracted=True,
    )


def decode(filename: str, raw: bytes) -> str:
    """UTF-8, strictly. A CV decoded with replacement characters would be
    quoted back to its author with black diamonds in it, which is worse than
    telling them the file did not open.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise UploadRejected(
            f"{filename} is not UTF-8 text. If it came out of a word processor, "
            "save it as a PDF, or copy the text and paste it in instead."
        ) from None
    return check_length(filename, text)


def check_length(label: str, text: str) -> str:
    if not text.strip():
        raise UploadRejected(f"{label} is empty.")
    if len(text) > MAX_CV_CHARS:
        raise UploadRejected(
            f"{label} is {len(text):,} characters, past the {MAX_CV_CHARS:,} this step takes. "
            "If it is a whole portfolio rather than a CV, upload the CV on its own."
        )
    return text


def read_note(length: int) -> str | None:
    """What to say about a CV longer than one model call reads.

    Said plainly, because the alternative -- cutting it and saying nothing --
    would make the facts proposed from the first hundred thousand characters
    look like the facts in the whole document.
    """
    if length <= MAX_CV_READ_CHARS:
        return None
    return (
        f"Stored in full ({length:,} characters), but only the first "
        f"{MAX_CV_READ_CHARS:,} are read."
    )


@dataclass(frozen=True, slots=True)
class ExtractionFailure:
    """What to tell the user, and where to send them to fix it."""

    message: str
    fix_url: str | None = None
    fix_label: str | None = None


_FAILURES: dict[CvExtractionErrorCode, ExtractionFailure] = {
    "no_api_key": ExtractionFailure(
        "This needs your own Anthropic API key -- reading a CV is a model call, "
        "and it is billed to you rather than to this app.",
        fix_url="/settings",
        fix_label="Add an API key",
    ),
    "api_key_rejected": ExtractionFailure(
        "Anthropic rejected the API key stored here. Replace it and try again.",
        fix_url="/settings",
        fix_label="Replace the key",
    ),
    "credential_unreadable": ExtractionFailure(
        "Your stored API key could not be unlocked on the server. Setting it again will fix it.",
        fix_url="/settings",
        fix_label="Set the key again",
    ),
    "no_cv_text": ExtractionFailure("There is no text stored against this CV."),
    "cv_too_long": ExtractionFailure(
        "That CV is too long to read in one pass. Upload it in parts, or trim the "
        "sections that are not about your own work."
    ),
    "model_refused": ExtractionFailure(
        "The model declined to read that text. If it is an ordinary CV, trying again usually works."
    ),
    "model_error": ExtractionFailure("Reading the CV failed. Trying again is worth a go."),
}

_UNKNOWN = ExtractionFailure("Reading the CV failed. Trying again is worth a go.")


def extraction_failure(code: CvExtractionErrorCode | None) -> ExtractionFailure:
    return _UNKNOWN if code is None else _FAILURES.get(code, _UNKNOWN)
