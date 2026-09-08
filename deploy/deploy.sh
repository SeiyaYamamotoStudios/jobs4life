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

# Every step below would otherwise open its own TCP connection, and the box runs
# `ufw limit OpenSSH` -- which blocks a source making 6 or more connections in
# 30 seconds. A deploy trips that and locks itself out mid-run, leaving the app
# built but not restarted. Multiplexing puts every command down ONE connection,
# so the rate limit never sees a burst. The alternative -- relaxing ufw to
# `allow` -- would weaken the box to suit a script, which is the wrong way round.
# %C is an ssh token (a hash of host/port/user), expanded by ssh itself --
# not a mktemp template. Kept under ~/.ssh because a unix socket path is
# capped near 104 characters.
mkdir -p ~/.ssh/cm
CTL="$HOME/.ssh/cm/%C"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$CTL" -o ControlPersist=120)
ssh_run() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }
cleanup() { ssh "${SSH_OPTS[@]}" -O exit "$HOST" 2>/dev/null || true; }
trap cleanup EXIT

cd "$(dirname "$0")/.."

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "WARNING: uncommitted changes will NOT be deployed (git archive ships HEAD)." >&2
  echo "         HEAD is $(git rev-parse --short HEAD): $(git log -1 --pretty=%s)" >&2
  echo >&2
fi

REV=$(git rev-parse --short HEAD)
echo "==> Deploying $REV to $HOST:$REMOTE_DIR"

echo "==> Transferring tracked source"
ssh_run "rm -rf $REMOTE_DIR/src && mkdir -p $REMOTE_DIR/src"
git archive HEAD | ssh_run "tar -x -C $REMOTE_DIR/src"

echo "==> Installing compose file"
scp -q "${SSH_OPTS[@]}" deploy/docker-compose.yml "$HOST:$REMOTE_DIR/docker-compose.yml"

echo "==> Building"
ssh_run "cd $REMOTE_DIR && docker compose build app"

echo "==> Migrating"
# Migrations run in a one-off container against the same image, before the app
# restarts, so a failed migration leaves the previous app running rather than
# starting a new one against a half-migrated schema.
ssh_run "cd $REMOTE_DIR && docker compose run --rm --no-deps \
  -e JFL_DATABASE_URL=\"postgresql+psycopg://\$(grep '^POSTGRES_USER=' .env | cut -d= -f2):\$(grep '^POSTGRES_PASSWORD=' .env | cut -d= -f2)@db:5432/\$(grep '^POSTGRES_DB=' .env | cut -d= -f2)\" \
  --entrypoint alembic app upgrade head"

echo "==> Restarting"
ssh_run "cd $REMOTE_DIR && docker compose up -d"

echo "==> Health"
sleep 8
ssh_run "curl -fsS -o /dev/null -w 'local healthz: %{http_code}\n' http://localhost:8000/healthz" || echo "local healthz FAILED"
curl -fsS -o /dev/null -w "public: %{http_code}\n" https://app-jobs4life.hiltonlabs.org/healthz || echo "public FAILED"

echo "==> Deployed $REV"
