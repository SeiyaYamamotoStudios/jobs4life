"""What the CV upload form accepts, and what it says when it does not.

Slice B6. The rules live here rather than in the route for the same reason
`jfl_web.jobads` holds the job-ad ones: the worker writes a **code** from a
closed set into `cv_extractions.error_code` while holding the user's decrypted
API key, and the sentence a person reads is a UI concern that must be
changeable without a migration.

**Plain text only, and the page says so.** A CV is usually a PDF or a .docx,
and pretending otherwise would be worse than refusing: a PDF text-extraction
layer gets the column order wrong on exactly the two-column CVs people use, and
a mangled line quoted back as "your own words" is the one thing this slice must
never do. So this slice takes `.md` and `.txt` and a paste box, and says plainly
that the others are not supported yet.

**No model call in the request path.** Uploading stores bytes and queues work;
reading the CV happens in the worker, on the user's key.
"""

from __future__ import annotations

from dataclasses import dataclass

from jfl_core.models import CvExtractionErrorCode

ALLOWED_SUFFIXES = (".md", ".txt", ".markdown", ".text")

# Comfortably longer than any CV (a wordy five-page one is ~20k characters) and
# short enough that a stray paste of a whole website does not become a paid
# model call. Measured in characters after decoding, so it means the same thing
# for the upload and the paste box.
MAX_CV_CHARS = 100_000

# The owner has thirty-three generated CVs plus the hand-written ones, and the
# point of this slice is that he uploads all of them. This is a ceiling on one
# request, not on the account.
MAX_FILES_PER_UPLOAD = 50


class UploadRejected(Exception):
    """A file the form will not take. The message is shown to the user."""


@dataclass(frozen=True, slots=True)
class UploadedCv:
    filename: str
    text: str


def check_suffix(filename: str) -> None:
    lowered = filename.lower()
    if not any(lowered.endswith(suffix) for suffix in ALLOWED_SUFFIXES):
        raise UploadRejected(
            f"{filename} is not a plain-text file. This step takes .md and .txt only -- "
            "PDF and Word are not read yet. Copy the text out and paste it in instead."
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
            "copy the text and paste it in instead."
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
