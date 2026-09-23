"""`check_cv_edits`: "Check my edits" on a generated CV, in the background, on
the user's own key.

The CV page queues this with the lines the user rewrote and has not had checked
(`jfl_core.cv_lines.unchecked_edits`), named by path. One claim-gate call over
those lines only -- joined one per paragraph, so no sentence crosses two lines
(`jfl_gate.gate.split_units` never splits across blocks) -- and each line gets
the worst verdict among its sentences. Framing-only lines get `framing`, which
the page renders as Not checked, never as supported.

**The result is a new version**, never an edit of an old one: the store is
append-only. The verdicts are written onto the *latest* version, and only onto
lines whose text still matches what was checked (`jfl_core.cv_lines.with_verdicts`)
-- if the user edited again while this ran, a verdict on different words would
be a lie, so that line stays unchecked.

**Not twice for one task.** The new version carries the task's id as its
`trace_id`, which is also every `runs` row's trace; a redelivered task that
finds the latest version already carries it returns without calling anything.

Credential discipline is `jfl_worker.handlers.application_questions`': the key
is fetched, unsealed, used once and dropped; never in the payload, a log line,
`last_error` or `runs`.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from jfl_core.context import GATE_MODEL as DEFAULT_GATE_MODEL
from jfl_core.context import RequestContext
from jfl_core.crypto.envelope import MasterKey, MasterKeyError, SecretUnsealError
from jfl_core.cv_lines import line_at, with_verdicts, worst_verdict
from jfl_core.models import RunRecord
from jfl_core.storage.credentials import PostgresCredentialRepository
from jfl_core.storage.cv_documents import PostgresCvDocumentRepository
from jfl_core.storage.postgres import PostgresGroundingRepository, PostgresRunRepository
from jfl_gate.gate import GateError, check_text
from sqlalchemy.engine import Engine

from jfl_worker.credentials import load_api_key
from jfl_worker.registry import Handler, PermanentTaskError, TaskContext

KIND = "check_cv_edits"

# The marker `jfl_web.cvdocs.check_failure` reads a permanent failure's code
# after. A closed set of codes follows it, never text from an exception.
FAILURE_MARKER = "cv edit check failed permanently: "

_PERMANENT_FAILURES: tuple[tuple[str, str], ...] = (
    ("authentication_error", "api_key_rejected"),
    ("permission_denied", "api_key_rejected"),
    ("model refused to respond", "model_refused"),
    ("bad_request", "model_error"),
    ("model output was truncated", "model_error"),
)


class _RunRecorder:
    """Commits each `runs` row on its own connection -- see
    `jfl_worker.handlers.extraction._RunRecorder`; duplicated for the same
    reason every handler duplicates it."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def record(self, run: RunRecord) -> None:
        with self._engine.begin() as conn:
            PostgresRunRepository(conn).record(run)


def _classify(message: str) -> tuple[str, bool]:
    for prefix, code in _PERMANENT_FAILURES:
        if message.startswith(prefix):
            return code, True
    return "model_error", False


def _uuid(payload: Mapping[str, object], key: str) -> uuid.UUID:
    raw = payload.get(key)
    if not isinstance(raw, str):
        raise PermanentTaskError(f"payload has no {key}")
    try:
        return uuid.UUID(raw)
    except ValueError:
        raise PermanentTaskError(f"payload {key} is not a uuid") from None


def _paths(payload: Mapping[str, object]) -> list[str]:
    raw = payload.get("paths")
    if not isinstance(raw, list) or not all(isinstance(p, str) for p in raw):
        raise PermanentTaskError("payload paths is not a list of strings")
    return list(raw)


def verdicts_for_lines(
    lines: Sequence[tuple[str, str]], sentences: Sequence[Any]
) -> dict[str, tuple[str, str, str]]:
    """path -> (text checked, verdict, note), from the gate's sentences.

    The gate was given the lines joined one per paragraph, in this order, and
    returns its sentences in order with their text. Each sentence is placed in
    the first line (from the current one on) that contains it. A line's verdict
    is its worst sentence's; a line whose sentences were all framing is
    `framing`. A line no sentence landed in gets no verdict and stays
    unchecked -- never a guessed one.
    """
    found: dict[str, list[tuple[str, str]]] = {path: [] for path, _ in lines}
    normalised = [" ".join(text.split()) for _, text in lines]
    # Where the next sentence may start: a line, and a position within it, so
    # two identical sentences in two lines land one in each.
    line_index, offset = 0, 0
    for sentence in sentences:
        text = " ".join(str(getattr(sentence, "text", "")).split())
        if not text or getattr(sentence, "kind", None) == "title":
            continue
        for index in range(line_index, len(lines)):
            start = offset if index == line_index else 0
            position = normalised[index].find(text, start)
            if position < 0:
                continue
            line_index, offset = index, position + len(text)
            kind = getattr(sentence, "kind", None)
            verdict = "framing" if kind == "framing" else str(getattr(sentence, "verdict", ""))
            note = str(getattr(sentence, "evidence_note", ""))
            found[lines[index][0]].append((verdict, note))
            break
    out: dict[str, tuple[str, str, str]] = {}
    for path, line_text in lines:
        results = found[path]
        worst = worst_verdict([verdict for verdict, _ in results])
        if worst is None:
            continue
        note = next(note for verdict, note in results if verdict == worst)
        out[path] = (line_text, worst, note)
    return out


def build_check_cv_edits(
    *, master_key: MasterKey | None, model: str, gate_model: str = DEFAULT_GATE_MODEL
) -> Handler:
    def handler(ctx: TaskContext) -> Mapping[str, object]:
        return _check_cv_edits(ctx, master_key=master_key, model=model, gate_model=gate_model)

    return handler


def _check_cv_edits(
    ctx: TaskContext, *, master_key: MasterKey | None, model: str, gate_model: str
) -> Mapping[str, object]:
    payload = ctx.task.payload
    application_id = _uuid(payload, "application_id")
    version_id = _uuid(payload, "version_id")
    paths = _paths(payload)

    with ctx.engine.begin() as conn:
        repo = PostgresCvDocumentRepository(conn, ctx.user_id)
        already = repo.version_for_trace(ctx.task.id)
        version = repo.get_version(version_id)
    if already is not None:
        return {"application_id": str(application_id), "skipped": "already checked"}
    if version is None or version.application_id != application_id:
        return {"application_id": str(application_id), "skipped": "no such version"}
    lines: list[tuple[str, str]] = []
    for path in paths:
        line = line_at(version.document, path)
        if line is not None and line.origin == "user":
            lines.append((path, line.text))
    if not lines:
        return {"application_id": str(application_id), "skipped": "nothing to check"}

    if master_key is None:
        raise PermanentTaskError(FAILURE_MARKER + "credential_unreadable")
    try:
        with ctx.engine.begin() as conn:
            api_key = load_api_key(PostgresCredentialRepository(conn, ctx.user_id), master_key)
    except (MasterKeyError, SecretUnsealError):
        raise PermanentTaskError(FAILURE_MARKER + "credential_unreadable") from None
    if api_key is None:
        raise PermanentTaskError(FAILURE_MARKER + "no_api_key")

    request = RequestContext(
        user_id=ctx.user_id,
        anthropic_api_key=api_key,
        database_url=ctx.engine.url.render_as_string(hide_password=False),
        model=model,
        gate_model=gate_model,
        trace_id=ctx.task.id,
    )
    text = "\n\n".join(line_text for _, line_text in lines)
    try:
        with ctx.engine.begin() as conn:
            output = check_text(
                request, PostgresGroundingRepository(conn), _RunRecorder(ctx.engine), text
            )
    except GateError as exc:
        code, permanent = _classify(str(exc))
        if permanent:
            raise PermanentTaskError(FAILURE_MARKER + code) from None
        raise

    verdicts = verdicts_for_lines(lines, output.sentences)
    with ctx.engine.begin() as conn:
        repo = PostgresCvDocumentRepository(conn, ctx.user_id)
        latest = repo.latest(application_id)
        if latest is None:
            return {"application_id": str(application_id), "skipped": "no version"}
        checked = with_verdicts(latest.document, verdicts)
        if checked == latest.document:
            return {"application_id": str(application_id), "skipped": "edited again since"}
        saved = repo.add_version(
            application_id,
            checked,
            status="checked",
            gate_result=output.model_dump(mode="json"),
            trace_id=ctx.task.id,
        )
    if saved is None:
        return {"application_id": str(application_id), "skipped": "no such application"}
    return {"application_id": str(application_id), "version_id": str(saved.id)}
