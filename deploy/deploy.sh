#!/usr/bin/env bash
# Ship the current commit to the VPS, build it there, migrate, restart.
#
#   ./deploy/deploy.sh
#
# Source is transferred with `git archive HEAD`, deliberately, rather than rsync
# or scp of the working tree. git archive emits **tracked files only**, so
# `corpus/` and `analysis/` -- real personal career data, gitignored -- cannot
# reach the server even by mistake. An rsync with an --exclude list is one typo
# away from shipping them; this is safe by construction rather than by care.
#
# It also means the box always runs a committed state. Uncommitted local changes
# are simply absent, which is the correct behaviour: "it works on the server"
# should always name a commit.

set -euo pipefail

HOST="${JFL_DEPLOY_HOST:-ubuntu@57.128.186.206}"
REMOTE_DIR="${JFL_DEPLOY_DIR:-/opt/jobs4life}"

cd "$(dirname "$0")/.."

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "WARNING: uncommitted changes will NOT be deployed (git archive ships HEAD)." >&2
  echo "         HEAD is $(git rev-parse --short HEAD): $(git log -1 --pretty=%s)" >&2
  echo >&2
fi

REV=$(git rev-parse --short HEAD)
echo "==> Deploying $REV to $HOST:$REMOTE_DIR"

echo "==> Transferring tracked source"
ssh "$HOST" "rm -rf $REMOTE_DIR/src && mkdir -p $REMOTE_DIR/src"
git archive HEAD | ssh "$HOST" "tar -x -C $REMOTE_DIR/src"

echo "==> Installing compose file"
scp -q deploy/docker-compose.yml "$HOST:$REMOTE_DIR/docker-compose.yml"

echo "==> Building"
ssh "$HOST" "cd $REMOTE_DIR && docker compose build app"

echo "==> Migrating"
# Migrations run in a one-off container against the same image, before the app
# restarts, so a failed migration leaves the previous app running rather than
# starting a new one against a half-migrated schema.
ssh "$HOST" "cd $REMOTE_DIR && docker compose run --rm --no-deps \
  -e JFL_DATABASE_URL=\"postgresql+psycopg://\$(grep '^POSTGRES_USER=' .env | cut -d= -f2):\$(grep '^POSTGRES_PASSWORD=' .env | cut -d= -f2)@db:5432/\$(grep '^POSTGRES_DB=' .env | cut -d= -f2)\" \
  --entrypoint alembic app upgrade head"

echo "==> Restarting"
ssh "$HOST" "cd $REMOTE_DIR && docker compose up -d"

echo "==> Health"
sleep 8
ssh "$HOST" "curl -fsS -o /dev/null -w 'local healthz: %{http_code}\n' http://localhost:8000/healthz" || echo "local healthz FAILED"
curl -fsS -o /dev/null -w "public: %{http_code}\n" https://app-jobs4life.hiltonlabs.org/healthz || echo "public FAILED"

echo "==> Deployed $REV"
