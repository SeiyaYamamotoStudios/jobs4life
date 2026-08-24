# job-for-life

A grounding gate for AI-generated job application text. Given a generated sentence and
a corpus of real experience, it decides whether that sentence traces to something true,
and flags it for human review when it does not.

Most tools in this space help candidates look better than they are. This one measures
the distance between what is true and what is being claimed, and shows you the number.

## Status

Iteration 1, in progress. Storage layer and schema only.

## Running it

```bash
uv sync
docker compose up -d
export JFL_DATABASE_URL=postgresql+psycopg://jfl:jfl@localhost:5433/jfl
uv run alembic upgrade head

uv run pytest                  # unit
uv run pytest -m integration   # needs the database
```

Postgres listens on 5433 to avoid colliding with a local install.

## Licence

MIT © 2026 Seiya
