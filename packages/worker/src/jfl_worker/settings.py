"""Worker configuration, and the kill switch.

`WorkerSettings.from_env()` is this process's equivalent of
`RequestContext.from_env()`: the ONE place the worker reads its environment at
startup. Everything below it is handed what it needs.

`model_calls_disabled()` is the deliberate exception, and the reason is in its
docstring: it is an incident lever, not configuration.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

from jfl_core.crypto.envelope import MasterKey
from jfl_core.db.tables import LOCAL_USER_ID

DISABLE_MODEL_CALLS_ENV = "JFL_DISABLE_MODEL_CALLS"

# Kept in sync with `jfl_gate.pricing.MODEL` and `RequestContext.model` the same
# way those two are kept in sync with each other: by hand, because the worker's
# settings module may not reach across into the gate to read a constant.
DEFAULT_MODEL = "claude-opus-5"

# Explicitly-off values. Anything else non-empty counts as ON, because the
# person typing this is doing it at speed while a user's key burns money, and
# `JFL_DISABLE_MODEL_CALLS=yes` must not quietly keep spending.
_FALSEY = frozenset({"", "0", "false", "no", "off"})


def model_calls_disabled(env: Mapping[str, str] | None = None) -> bool:
    """Is the model-call kill switch on, right now?

    Read from the environment at every call rather than captured into
    `WorkerSettings`, because the brief this implements is explicit that the
    switch is checked **at dispatch time, not at startup**. That matters even
    though a container's environment does not usually change under it: the
    check happens between claiming a task and running it, so the answer used is
    the answer at the moment of spending, and a `docker compose up -d worker`
    with the variable set takes effect on the next task rather than on the next
    thing anyone remembers to restart.

    `env` is injectable so tests can drive it without mutating the process.
    """
    source = os.environ if env is None else env
    return source.get(DISABLE_MODEL_CALLS_ENV, "").strip().lower() not in _FALSEY


# Defaults live here rather than being read back off the class: `slots=True`
# replaces class attributes with slot descriptors, so `cls.poll_interval` inside
# `from_env` would be a descriptor object, not 2.0. It fails silently into a
# nonsense setting, so the values are named once and used twice.
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_BATCH_SIZE = 1
DEFAULT_VISIBILITY_TIMEOUT = 15 * 60.0
DEFAULT_RECLAIM_INTERVAL = 60.0
DEFAULT_RETRY_BASE = 30.0
DEFAULT_RETRY_FACTOR = 2.0
DEFAULT_RETRY_CAP = 10 * 60.0
DEFAULT_KILL_SWITCH_RETRY_DELAY = 60.0
DEFAULT_PURGE_INTERVAL = 60 * 60.0
DEFAULT_BOARD_SCHEDULE_INTERVAL = 15 * 60.0
DEFAULT_ERROR_BACKOFF = 10.0


def _seconds(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    """Every number the loop needs, with the reasoning for each default.

    They are settings rather than constants because the right visibility timeout
    depends on the slowest handler deployed, which changes per slice.
    """

    database_url: str

    # How long to wait after finding nothing to do. Two seconds is the delay a
    # user feels between submitting a form and the work starting; a poll is one
    # indexed query against a partial index, so ~43k of them a day costs
    # Postgres nothing worth measuring. Busy loops drain the queue without
    # sleeping at all, so this is the idle cost only.
    poll_interval: float = DEFAULT_POLL_INTERVAL

    # How many tasks to claim at once. ONE, and not for want of ambition: this
    # worker runs tasks serially, so claiming five would start the visibility
    # clock on four rows nobody is touching yet -- they would look stale and be
    # reclaimed while queued behind a gate call. `claim(limit=...)` takes a
    # batch size because a concurrent worker would want one; this loop does not.
    batch_size: int = DEFAULT_BATCH_SIZE

    # How long a `running` row may sit before another worker may assume its
    # owner is dead. Fifteen minutes against a slowest-known task of ~2 minutes
    # (a claim-gate call): generous on purpose, because reclaiming too early
    # runs the work twice and, when the handler eventually calls a model, twice
    # means paying twice.
    visibility_timeout: float = DEFAULT_VISIBILITY_TIMEOUT

    # How often to sweep for those rows. A minute; the sweep is one indexed
    # query and usually returns nothing.
    reclaim_interval: float = DEFAULT_RECLAIM_INTERVAL

    # Backoff between attempts: 30s, then 60s, then 120s ... capped at 10
    # minutes. The first retry is far enough out to clear a transient API 529 or
    # a Postgres restart, and the cap keeps a task that will eventually succeed
    # from being parked for hours. Deterministic, with no jitter: jitter exists
    # to desynchronise a herd, and this deployment has one worker.
    retry_base: float = DEFAULT_RETRY_BASE
    retry_factor: float = DEFAULT_RETRY_FACTOR
    retry_cap: float = DEFAULT_RETRY_CAP

    # How long a task refused by the kill switch waits before it is looked at
    # again. It will not be claimed at all while the switch is on -- the claim
    # filters by kind -- so this only matters for the switch flipping mid-flight.
    kill_switch_retry_delay: float = DEFAULT_KILL_SWITCH_RETRY_DELAY

    # How often the recurring session purge is enqueued. Hourly: expired
    # sessions are already refused at lookup (`expires_at <= now`), so purging
    # is housekeeping rather than a security boundary.
    purge_interval: float = DEFAULT_PURGE_INTERVAL

    # How often the watched-board scheduling pass is enqueued. Fifteen minutes:
    # each board has its own daily slot (`jfl_intake.scheduling`), so this is
    # the most a check runs late, and a newly added board gets its baseline
    # within this long. The pass is one indexed query when nothing is due.
    board_schedule_interval: float = DEFAULT_BOARD_SCHEDULE_INTERVAL

    # After an unexpected loop-level error (Postgres down, say). Longer than the
    # poll interval so a database outage does not become a log flood.
    error_backoff: float = DEFAULT_ERROR_BACKOFF

    # Maintenance work belongs to somebody, because `tasks.user_id` is NOT NULL
    # and there is no unowned path into that table. The seeded local user is the
    # one row guaranteed to exist in every database (migration 309cf277970b),
    # which makes it the honest owner of work that is nobody's in particular.
    system_user_id: uuid.UUID = LOCAL_USER_ID

    # The KEK, for unsealing a user's stored Anthropic key at the moment a
    # handler needs it. Optional on the dataclass so a test can build settings
    # for the model-free handlers without minting one; `from_env` always
    # supplies it, so a deployed worker with no `JFL_MASTER_KEY` fails at boot
    # rather than one task into someone's first extraction.
    master_key: MasterKey | None = None

    # Which model the handlers call. A product option, not a hidden default --
    # see CLAUDE.md's 2026-09-05 decision. The web app and the CLI read the same
    # `JFL_MODEL`, so a deployment sets it once.
    model: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> WorkerSettings:
        """The only place this process reads its environment.

        A missing `JFL_DATABASE_URL` or `JFL_MASTER_KEY` raises here, at boot,
        rather than at the first task.
        """
        source = os.environ if env is None else env
        return cls(
            database_url=source["JFL_DATABASE_URL"],
            master_key=MasterKey.from_env(source),
            model=source.get("JFL_MODEL") or DEFAULT_MODEL,
            poll_interval=_seconds(source, "JFL_WORKER_POLL_INTERVAL", DEFAULT_POLL_INTERVAL),
            visibility_timeout=_seconds(
                source, "JFL_WORKER_VISIBILITY_TIMEOUT", DEFAULT_VISIBILITY_TIMEOUT
            ),
            reclaim_interval=_seconds(
                source, "JFL_WORKER_RECLAIM_INTERVAL", DEFAULT_RECLAIM_INTERVAL
            ),
            retry_base=_seconds(source, "JFL_WORKER_RETRY_BASE", DEFAULT_RETRY_BASE),
            retry_cap=_seconds(source, "JFL_WORKER_RETRY_CAP", DEFAULT_RETRY_CAP),
            purge_interval=_seconds(source, "JFL_WORKER_PURGE_INTERVAL", DEFAULT_PURGE_INTERVAL),
            board_schedule_interval=_seconds(
                source, "JFL_WORKER_BOARD_SCHEDULE_INTERVAL", DEFAULT_BOARD_SCHEDULE_INTERVAL
            ),
        )

    def retry_delay(self, attempts: int) -> dt.timedelta:
        """Backoff before the next attempt, given how many have been made.

        `attempts` is the count AFTER the failed attempt (it is incremented at
        claim time), so the first failure asks for `retry_delay(1)` and gets
        `retry_base`.
        """
        exponent = max(attempts - 1, 0)
        seconds = min(self.retry_base * (self.retry_factor**exponent), self.retry_cap)
        return dt.timedelta(seconds=seconds)
