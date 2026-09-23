"""PLACEHOLDER (cvedit branch) -- `render_cv_html` and `render_cv_pdf`.

The rendering branch owns the real implementations; at merge they replace this
module (or the editing screens are re-pointed at wherever they live). Only the
two signatures matter here:

    render_cv_html(doc: CvDocument) -> str
    render_cv_pdf(doc: CvDocument) -> bytes

What both must honour, and what the screens rely on: they render the document
exactly as stored -- the text of every line, never a gate verdict, a note or
any other annotation. The same document goes to both, so the on-screen preview
is the document the PDF is made from.
"""

from __future__ import annotations

from html import escape

from jfl_core.cv_document import CvDocument


def _paragraphs(doc: CvDocument) -> list[tuple[str, str]]:
    """(kind, text) in reading order -- shared by both placeholder renderers."""
    out: list[tuple[str, str]] = [("name", doc.header.name)]
    if doc.header.tagline:
        out.append(("tagline", doc.header.tagline))
    contact = [*doc.header.contact, *(link.label for link in doc.header.links)]
    if contact:
        out.append(("contact", " | ".join(contact)))
    out.extend(("p", line.text) for line in doc.summary)
    if doc.skills:
        out.append(("h2", doc.skills_heading))
        out.extend(("p", f"{s.label}: {s.text.text}") for s in doc.skills)
    if doc.roles:
        out.append(("h2", doc.experience_heading))
        for role in doc.roles:
            where = ", ".join(x for x in (role.employer, role.location) if x)
            out.append(("h3", f"{role.title} -- {where} ({role.dates})"))
            if role.descriptor:
                out.append(("em", role.descriptor))
            out.extend(("li", b.text) for b in role.bullets)
    if doc.education:
        out.append(("h2", doc.education_heading))
        out.extend(("li", line.text) for line in doc.education)
    if doc.interests:
        out.append(("h2", doc.interests_heading))
        out.append(("p", ", ".join(doc.interests)))
    return out


def render_cv_html(doc: CvDocument) -> str:
    body: list[str] = []
    for kind, text in _paragraphs(doc):
        tag = {"name": "h1", "tagline": "p", "contact": "p"}.get(kind, kind)
        if kind == "li":
            body.append(f"<ul><li>{escape(text)}</li></ul>")
        else:
            body.append(f'<{tag} class="{kind}">{escape(text)}</{tag}>')
    font = "Georgia, serif" if doc.template == "classic" else "Helvetica, Arial, sans-serif"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<style>body{{font-family:{font};margin:2rem;color:#111}}"
        "h1{margin:0}h2{border-bottom:1px solid #999;margin-top:1.4rem}</style>"
        f'</head><body class="cv-{doc.template}">{"".join(body)}</body></html>'
    )


def _pdf_escape(text: str) -> str:
    safe = text.encode("latin-1", "replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def render_cv_pdf(doc: CvDocument) -> bytes:
    """A one-page, text-only PDF. Enough to be a real file; not a layout."""
    lines = [text for _, text in _paragraphs(doc)]
    stream_lines = ["BT", "/F1 10 Tf", "50 800 Td", "14 TL"]
    for text in lines[:55]:
        stream_lines.append(f"({_pdf_escape(text[:110])}) '")
    stream_lines.append("ET")
    stream = "\n".join(stream_lines).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    size = len(objects) + 1
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


__all__ = ["render_cv_html", "render_cv_pdf"]
