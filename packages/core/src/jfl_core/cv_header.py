"""The top of a CV and its interests, taken from the profile's settings.

Shared by the two places a `CvDocument` gets its header: the worker, when it
generates a CV (`jfl_worker.handlers.draft_generation`), and the CV page's
"update from my profile" button (`jfl_web.routes.cv_documents`). One function,
so a generated CV and a refreshed one cannot read the profile differently.

**Settings, not claims** (`jfl_core.profile.CvHeaderSettings`). Name, tagline,
phone, email, location, links and interests are applied to the document
*after* the claim gate has run and are never part of what it is sent, and none
of them is ever written to the corpus. No model anywhere in this module.
"""

from __future__ import annotations

from jfl_core.cv_document import CvDocument, CvHeader, CvLink
from jfl_core.profile import Profile


def header_name(profile: Profile, display_name: str) -> str:
    """The name the CV should carry: the profile's, if it states one, else the
    account's display name, else "" -- in which case generation falls back to
    the corpus document's own title. Never inferred."""
    return profile.cv_header.name.strip() or display_name.strip()


def header_from_profile(profile: Profile, current: CvDocument) -> CvDocument:
    """`current` with its header and interests replaced by the profile's.

    A name left blank on the profile keeps the one the CV already has, since a
    CV cannot have no name. Every other blank field is left off, never guessed.
    """
    settings = profile.cv_header
    header = CvHeader(
        name=settings.name or current.header.name,
        tagline=settings.tagline,
        contact=settings.contact,
        links=[CvLink(label=link.label, url=link.url) for link in settings.links],
    )
    return current.model_copy(update={"header": header, "interests": list(profile.interests)})


__all__ = ["header_from_profile", "header_name"]
