"""The `classify_pushback` handler's decisions.

`_classify`, `_pushback_id` and `dimension_label` are pure and tested the same
way `test_title_suggestions_handler.py` tests its siblings -- no database.

The handler's control flow (skip a redelivered/already-classified task, fail
loudly with no master key or no stored key, fail the row and retry on a
transient model error, fail the row and stop on a permanent one, apply a
successful classification) is exercised here too, against fakes rather than
real Postgres: `PostgresPushbackRepository` and `load_api_key` are
monkeypatched at the names `jfl_worker.handlers.pushback` imports them under,
and `ctx.engine` is a throwaway in-memory sqlite engine that the fakes never
actually query -- it exists only so `ctx.engine.begin()` and
`ctx.engine.url.render_as_string(...)` have something real to call. This
differs from the sibling handler test files, which leave the full flow to a
`*_integration.py` file against real Postgres; that file is out of scope
here, so the flow is covered this way instead of not at all.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from jfl_core.crypto.envelope import MasterKey, MasterKeyError, SecretUnsealError
from jfl_core.models import Pushback
from jfl_generate.errors import GenerateError
from jfl_generate.prompts import build_pushback_classification_prompt
from jfl_generate.pushback import PushbackClassification
from jfl_worker.handlers import pushback as pushback_handler
from jfl_worker.handlers.pushback import (
    _classify,
    _earlier_texts,
    _pushback_id,
    build_classify_pushback,
    dimension_label,
)
from jfl_worker.registry import PermanentTaskError, TaskContext
from sqlalchemy import create_engine

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")
NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


# --- classify / pushback_id / dimension_label (no database) -----------------


class TestClassify:
    @pytest.mark.parametrize(
        ("message", "code"),
        [
            ("authentication_error: invalid x-api-key", "api_key_rejected"),
            ("permission_denied: no access to this model", "api_key_rejected"),
            ("model refused to respond: reasoning_extraction", "model_refused"),
            ("bad_request: schema is invalid", "model_error"),
        ],
    )
    def test_failures_a_retry_cannot_fix_are_permanent(self, message: str, code: str) -> None:
        assert _classify(message) == (code, True)

    @pytest.mark.parametrize(
        "message",
        [
            "rate_limited: too many requests",
            "api_status_529: overloaded",
            "connection_error: connection reset",
            "could not parse structured output: expecting value",
            "something nobody has seen before",
        ],
    )
    def test_transient_and_unrecognised_failures_are_retried(self, message: str) -> None:
        assert _classify(message) == ("model_error", False)


def test_the_classifier_still_matches_the_messages_classify_pushback_actually_raises() -> None:
    """The coupling, made visible -- see `test_title_suggestions_handler.py`'s
    sibling test for the identical rationale.
    """
    import inspect

    from jfl_generate import pushback as pushback_module

    source = inspect.getsource(pushback_module)
    for prefix in (
        "authentication_error",
        "permission_denied",
        "bad_request",
        "model refused to respond",
    ):
        assert prefix in source, f"jfl_generate.pushback no longer says {prefix!r}"

    # And the prompt this all hangs off still builds, which is the cheapest
    # possible smoke test that the generate package is importable from here.
    assert build_pushback_classification_prompt(
        user_text="I actually led that team",
        axis="get",
        direction="up",
        shown_score=4,
        shown_explanation="",
        dimension_label="a capability",
        earlier_texts=[],
        now=NOW,
    )


class TestPushbackId:
    def test_a_well_formed_payload_is_read(self) -> None:
        wanted = uuid.uuid4()
        assert _pushback_id({"pushback_id": str(wanted)}) == wanted

    @pytest.mark.parametrize(
        "payload", [{}, {"pushback_id": None}, {"pushback_id": 7}, {"other": "thing"}]
    )
    def test_a_payload_without_one_is_permanently_failed(self, payload: dict[str, object]) -> None:
        with pytest.raises(PermanentTaskError):
            _pushback_id(payload)

    def test_a_malformed_id_never_appears_in_the_error(self) -> None:
        with pytest.raises(PermanentTaskError) as raised:
            _pushback_id({"pushback_id": "not-a-uuid-and-maybe-a-secret"})
        assert "not-a-uuid-and-maybe-a-secret" not in str(raised.value)


class TestDimensionLabel:
    def test_constraint(self) -> None:
        assert dimension_label("constraint:workplace") == "workplace"

    def test_objective(self) -> None:
        assert dimension_label("objective:1") == "objective 1"

    def test_capability_collapses_to_a_generic_phrase(self) -> None:
        """The key after the colon is an internal id (`jfl_core.profile.Capability`),
        not guaranteed to be readable prose, so it is never interpolated.
        """
        assert dimension_label("capability:fx-pricing-platforms") == "a capability"

    def test_want_overall(self) -> None:
        assert dimension_label("want_overall") == 'the whole "do I want this" number'

    def test_could_get_overall(self) -> None:
        assert dimension_label("could_get_overall") == 'the whole "could I get this" number'

    def test_unknown_dimension_passes_through(self) -> None:
        assert dimension_label("something_else") == "something_else"


# --- fakes for the handler's control flow ------------------------------------


def _pushback(
    *,
    pushback_id: uuid.UUID | None = None,
    dimension: str = "capability:fx",
    status: str = "awaiting_classification",
    user_text: str = "I actually led that team",
    created_at: datetime = NOW,
) -> Pushback:
    return Pushback(
        id=pushback_id or uuid.uuid4(),
        application_id=uuid.uuid4(),
        score_id=uuid.uuid4(),
        axis="get",
        dimension=dimension,
        shown_score=4,
        shown_explanation="Limited evidence of team leadership.",
        user_text=user_text,
        asserted_direction="up",
        status=status,  # type: ignore[arg-type]
        created_at=created_at,
        updated_at=created_at,
    )


class _FakePushbackRepo:
    """Stands in for `PostgresPushbackRepository`. Backed by a dict shared
    across every instance built from the same `store`, since the handler opens
    a fresh connection -- and so, in production, a fresh repository -- for
    each step.
    """

    def __init__(self, store: dict[uuid.UUID, Pushback], user_id: uuid.UUID) -> None:
        self._store = store
        self._user_id = user_id

    def get(self, pushback_id: uuid.UUID) -> Pushback | None:
        return self._store.get(pushback_id)

    def set_classification(
        self,
        pushback_id: uuid.UUID,
        *,
        classification: str,
        new_information: bool,
        note: str = "",
        source: str = "model",
        trace_id: uuid.UUID | None = None,
    ) -> Pushback | None:
        row = self._store[pushback_id]
        updated = row.model_copy(
            update={
                "status": "classified",
                "classification": classification,
                "classification_source": source,
                "classification_note": note,
                "new_information": new_information,
                "error_code": None,
                "trace_id": trace_id,
            }
        )
        self._store[pushback_id] = updated
        return updated

    def mark_classification_failed(self, pushback_id: uuid.UUID, code: str) -> Pushback | None:
        row = self._store[pushback_id]
        updated = row.model_copy(update={"error_code": code})
        self._store[pushback_id] = updated
        return updated

    def recent(self, limit: int = 100) -> list[Pushback]:
        rows = sorted(self._store.values(), key=lambda p: p.created_at, reverse=True)
        return rows[:limit]


def _install_fake_repo(monkeypatch: pytest.MonkeyPatch, store: dict[uuid.UUID, Pushback]) -> None:
    def factory(conn: object, user_id: uuid.UUID) -> _FakePushbackRepo:
        del conn
        return _FakePushbackRepo(store, user_id)

    monkeypatch.setattr(pushback_handler, "PostgresPushbackRepository", factory)


def _task_context(pushback_id: uuid.UUID) -> TaskContext:
    from jfl_core.models import Task

    task = Task(
        id=uuid.uuid4(),
        user_id=USER,
        kind="classify_pushback",
        payload={"pushback_id": str(pushback_id)},
        status="running",
        attempts=1,
        max_attempts=5,
        scheduled_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )
    # A throwaway in-memory engine: the fakes above never issue real SQL
    # against it, it exists only because `ctx.engine.begin()` and
    # `ctx.engine.url.render_as_string(...)` need something real to call.
    engine = create_engine("sqlite:///:memory:")
    return TaskContext(task=task, engine=engine, now=NOW)


# --- the handler's control flow -----------------------------------------------


def test_a_redelivered_task_whose_row_is_already_classified_calls_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = uuid.uuid4()
    store = {pid: _pushback(pushback_id=pid, status="classified")}
    _install_fake_repo(monkeypatch, store)

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("the model must not be called for a task with nothing to do")

    monkeypatch.setattr(pushback_handler, "call_classify_pushback", _boom)
    monkeypatch.setattr(pushback_handler, "load_api_key", _boom)

    handler = build_classify_pushback(master_key=MasterKey.generate())
    result = handler(_task_context(pid))
    assert result is not None and "skipped" in result


def test_a_redelivered_task_whose_row_is_already_applied_calls_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = uuid.uuid4()
    store = {pid: _pushback(pushback_id=pid, status="applied")}
    _install_fake_repo(monkeypatch, store)
    monkeypatch.setattr(
        pushback_handler,
        "call_classify_pushback",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call the model")),
    )

    handler = build_classify_pushback(master_key=MasterKey.generate())
    result = handler(_task_context(pid))
    assert result is not None and "skipped" in result


def test_a_missing_row_is_skipped_not_errored(monkeypatch: pytest.MonkeyPatch) -> None:
    pid = uuid.uuid4()
    _install_fake_repo(monkeypatch, {})

    handler = build_classify_pushback(master_key=MasterKey.generate())
    result = handler(_task_context(pid))
    assert result is not None and "skipped" in result


def test_no_master_key_fails_the_row_and_raises_permanently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = uuid.uuid4()
    store = {pid: _pushback(pushback_id=pid)}
    _install_fake_repo(monkeypatch, store)

    handler = build_classify_pushback(master_key=None)
    with pytest.raises(PermanentTaskError):
        handler(_task_context(pid))

    assert store[pid].error_code == "credential_unreadable"
    assert store[pid].status == "awaiting_classification"


def test_credential_that_cannot_be_unsealed_fails_the_row_and_raises_permanently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = uuid.uuid4()
    store = {pid: _pushback(pushback_id=pid)}
    _install_fake_repo(monkeypatch, store)

    def _raise_unseal(*args: object, **kwargs: object) -> str | None:
        raise SecretUnsealError("nope")

    monkeypatch.setattr(pushback_handler, "load_api_key", _raise_unseal)

    handler = build_classify_pushback(master_key=MasterKey.generate())
    with pytest.raises(PermanentTaskError):
        handler(_task_context(pid))

    assert store[pid].error_code == "credential_unreadable"


def test_master_key_error_fails_the_row_and_raises_permanently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = uuid.uuid4()
    store = {pid: _pushback(pushback_id=pid)}
    _install_fake_repo(monkeypatch, store)

    def _raise_master_key(*args: object, **kwargs: object) -> str | None:
        raise MasterKeyError("nope")

    monkeypatch.setattr(pushback_handler, "load_api_key", _raise_master_key)

    handler = build_classify_pushback(master_key=MasterKey.generate())
    with pytest.raises(PermanentTaskError):
        handler(_task_context(pid))

    assert store[pid].error_code == "credential_unreadable"


def test_no_stored_api_key_fails_the_row_with_no_api_key_and_raises_permanently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = uuid.uuid4()
    store = {pid: _pushback(pushback_id=pid)}
    _install_fake_repo(monkeypatch, store)
    monkeypatch.setattr(pushback_handler, "load_api_key", lambda *a, **k: None)

    handler = build_classify_pushback(master_key=MasterKey.generate())
    with pytest.raises(PermanentTaskError):
        handler(_task_context(pid))

    assert store[pid].error_code == "no_api_key"
    assert store[pid].status == "awaiting_classification"


def test_a_transient_model_failure_fails_the_row_and_re_raises_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = uuid.uuid4()
    store = {pid: _pushback(pushback_id=pid)}
    _install_fake_repo(monkeypatch, store)
    monkeypatch.setattr(pushback_handler, "load_api_key", lambda *a, **k: "sk-ant-test")

    def _raise_transient(*args: object, **kwargs: object) -> PushbackClassification:
        raise GenerateError("rate_limited: too many requests")

    monkeypatch.setattr(pushback_handler, "call_classify_pushback", _raise_transient)

    handler = build_classify_pushback(master_key=MasterKey.generate())
    with pytest.raises(GenerateError):
        handler(_task_context(pid))

    assert store[pid].error_code == "model_error"
    # Not a PermanentTaskError: the row is left `awaiting_classification` and
    # the runner retries the task.
    assert store[pid].status == "awaiting_classification"


def test_a_permanent_model_failure_fails_the_row_and_raises_permanently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = uuid.uuid4()
    store = {pid: _pushback(pushback_id=pid)}
    _install_fake_repo(monkeypatch, store)
    monkeypatch.setattr(pushback_handler, "load_api_key", lambda *a, **k: "sk-ant-test")

    def _raise_permanent(*args: object, **kwargs: object) -> PushbackClassification:
        raise GenerateError("authentication_error: invalid x-api-key")

    monkeypatch.setattr(pushback_handler, "call_classify_pushback", _raise_permanent)

    handler = build_classify_pushback(master_key=MasterKey.generate())
    with pytest.raises(PermanentTaskError):
        handler(_task_context(pid))

    assert store[pid].error_code == "api_key_rejected"


def test_a_successful_call_applies_the_classification_to_the_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = uuid.uuid4()
    store = {pid: _pushback(pushback_id=pid)}
    _install_fake_repo(monkeypatch, store)
    monkeypatch.setattr(pushback_handler, "load_api_key", lambda *a, **k: "sk-ant-test")

    calls: list[dict[str, object]] = []

    def _fake_call(request: object, recorder: object, **kwargs: object) -> PushbackClassification:
        calls.append(kwargs)
        return PushbackClassification(kind="capability", new_information=True, note="a note")

    monkeypatch.setattr(pushback_handler, "call_classify_pushback", _fake_call)

    handler = build_classify_pushback(master_key=MasterKey.generate())
    result = handler(_task_context(pid))

    assert result is not None and result["classification"] == "capability"
    updated = store[pid]
    assert updated.status == "classified"
    assert updated.classification == "capability"
    assert updated.classification_source == "model"
    assert updated.classification_note == "a note"
    assert updated.new_information is True
    assert updated.error_code is None

    assert len(calls) == 1
    assert calls[0]["user_text"] == "I actually led that team"
    assert calls[0]["dimension_label"] == "a capability"


def test_earlier_texts_are_this_users_own_words_on_the_same_dimension_only() -> None:
    pid = uuid.uuid4()
    target = _pushback(pushback_id=pid, dimension="capability:fx", user_text="current")
    same_dim_earlier = _pushback(
        dimension="capability:fx",
        user_text="earlier same dimension",
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    other_dim = _pushback(
        dimension="constraint:workplace",
        user_text="different dimension entirely",
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    later_same_dim = _pushback(
        dimension="capability:fx",
        user_text="not yet said at the time of this pushback",
        created_at=datetime(2026, 9, 30, tzinfo=UTC),
    )
    store = {
        target.id: target,
        same_dim_earlier.id: same_dim_earlier,
        other_dim.id: other_dim,
        later_same_dim.id: later_same_dim,
    }
    repo = _FakePushbackRepo(store, USER)

    assert _earlier_texts(repo, target) == ["earlier same dimension"]


def test_earlier_texts_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    pid = uuid.uuid4()
    target = _pushback(pushback_id=pid, dimension="capability:fx")
    store = {target.id: target}
    for i in range(pushback_handler.MAX_EARLIER_TEXTS + 3):
        older = _pushback(
            dimension="capability:fx",
            user_text=f"earlier {i}",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        store[older.id] = older
    repo = _FakePushbackRepo(store, USER)

    assert len(_earlier_texts(repo, target)) == pushback_handler.MAX_EARLIER_TEXTS
