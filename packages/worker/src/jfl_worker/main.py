"""Entry point: `jfl-worker`.

Reads the environment once (`WorkerSettings.from_env`), builds the engine and
the registry, installs signal handlers, and runs the loop until told to stop.

SIGTERM matters here specifically: `docker compose stop` sends it and then
SIGKILLs `stop_grace_period` later. Handling it is what turns "the container was
killed and a task is stuck in `running` until the visibility timeout" into "the
worker finished its task and exited". The stuck case still exists for tasks
longer than the grace period -- see `runner.py` -- which is why reclaim exists
too.
"""

from __future__ import annotations

import logging
import signal
from types import FrameType

from sqlalchemy import create_engine

from jfl_worker.handlers import build_registry
from jfl_worker.log import configure_logging, log_event
from jfl_worker.queue import postgres_enqueuer_scope, postgres_queue_scope
from jfl_worker.runner import Worker
from jfl_worker.settings import WorkerSettings


def main() -> int:
    logger = configure_logging()
    settings = WorkerSettings.from_env()

    # `pool_pre_ping`: this process outlives Postgres restarts and idle-timeout
    # reaps, and a stale pooled connection would otherwise surface as one failed
    # task per restart. `pool_size=2` because the loop uses one connection at a
    # time and a handler may open a second.
    engine = create_engine(settings.database_url, pool_pre_ping=True, pool_size=2, max_overflow=2)

    worker = Worker(
        registry=build_registry(settings),
        settings=settings,
        queue_scope=postgres_queue_scope(engine),
        enqueuer_scope=postgres_enqueuer_scope(engine, settings.system_user_id),
        engine=engine,
        logger=logger,
    )

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        log_event(logger, logging.INFO, "worker.signal", signal=signal.Signals(signum).name)
        worker.request_stop()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        worker.run_forever()
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":  # pragma: no cover - container entry point
    raise SystemExit(main())
