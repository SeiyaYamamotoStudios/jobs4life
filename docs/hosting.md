# Provisioning runbook — jobs4life VPS

Target: a fresh OVHcloud VPS-1 (2 vCore, 4 GB RAM, 40 GB NVMe, Ubuntu 24.04 LTS, London)
taken from "just created" to "Docker is running, Postgres is up, a Cloudflare Tunnel
reaches a placeholder service, SSH is locked down." No application code goes on the box
in this runbook — that's separate work. Every layer below has its own verification step;
don't take the next one on faith.

This assumes you already know Linux. It only explains the parts that are specific to
this setup: why the tunnel replaces open ports, why swap matters at 4 GB, why a logical
dump is not the same as the provider snapshot, and the one Docker/ufw interaction that
will quietly undo the whole "no public ports" design if you get it wrong.

## Placeholders — pick your own values, these are just examples

| Placeholder | Meaning | Used as |
|---|---|---|
| `VPS_IP` | the server's public IPv4, from the OVH panel | example only |
| `deploy` | the non-root account you create | example only |
| `jobs4life-vps01` | the box's own hostname (not a DNS record) | example only |
| `app.jobs4life.hiltonlabs.org` | public hostname for the app | **given in the brief, yours to change** |
| `jobs4life-app` | the Cloudflare Tunnel's name | example only |
| `/opt/jobs4life` | where compose files and dumps live on the box | example only |

Everything below uses these literally — swap in your own where you deviate. Each
command block says who runs it: **root@vps**, **deploy@vps**, **laptop**, or **browser**.

Do not touch the `hiltonlabs.org` apex A record or the `www` CNAME — those are Ghost's
shared redirect server and Ghost Pro respectively, and are unrelated to this box. Do not
touch the existing `jobs4life.hiltonlabs.org` CNAME to Cloudflare Pages — that's the
static demo; this runbook only ever creates a *new* record, `app.jobs4life.hiltonlabs.org`.

---

## 1. First contact

```bash
# laptop
ssh root@VPS_IP
```

Your key was added at creation, so this should drop you straight in with no password
prompt. If it doesn't, stop and sort out the key before doing anything else.

```bash
# root@vps
apt update && apt full-upgrade -y
[ -f /var/run/reboot-required ] && echo "reboot needed" || echo "no reboot needed"
```

If a reboot is flagged (usually a kernel update on a fresh image), do it now while
nothing depends on the box yet:

```bash
# root@vps
reboot
```

```bash
# laptop — reconnect after ~30s
ssh root@VPS_IP
```

```bash
# root@vps
hostnamectl set-hostname jobs4life-vps01
timedatectl set-timezone Europe/London
hostnamectl status   # confirm hostname
timedatectl           # confirm "Time zone: Europe/London"
```

---

## 2. Deploy user, then lock SSH down

**Keep this root session open for the entire rest of this section.** You verify the new
login path in a *second* terminal before closing the first — if the second login fails
and you've already closed the first, you're locked out with no way back in except OVH's
rescue console.

```bash
# root@vps
adduser deploy          # set a password when prompted — used for local sudo, not SSH
usermod -aG sudo deploy
```

Get your public key onto the new account. Easiest from your laptop, since it already
has the key:

```bash
# laptop
ssh-copy-id -i ~/.ssh/id_ed25519.pub deploy@VPS_IP
```

Now, in a **second terminal**, confirm the new account works end to end — key login and
sudo — before changing anything about root or password auth:

```bash
# laptop, second terminal
ssh deploy@VPS_IP
sudo whoami   # should print "root" after your account password
```

Only once that works, harden `sshd`. Drop an override into
`/etc/ssh/sshd_config.d/` rather than editing the shipped file.

**Name it `00-`, not `60-` or `99-`.** In `sshd_config` the **first** occurrence of a
keyword wins, not the last, and drop-ins are read in lexical order — so a
higher-numbered file *loses*. The OVH image ships `50-cloud-init.conf` containing
`PasswordAuthentication yes`, which beats anything numbered above it. This was
originally written as `99-`; on the real box it applied cleanly, `sshd -t` passed,
`systemctl reload` succeeded, and `PasswordAuthentication` stayed `yes`. Always
confirm with `sshd -T`, never with "the file is there".

```bash
# root@vps, first terminal — still open
cat <<'EOF' > /etc/ssh/sshd_config.d/00-hardening.conf
PermitRootLogin no
PasswordAuthentication no
KbdInteractiveAuthentication no
EOF
sshd -t              # syntax check — must print nothing
systemctl reload ssh || systemctl reload sshd
sshd -T | grep -E '^(permitrootlogin|passwordauthentication)'   # the real check
```

Verify in the second terminal (or a fresh one) **before** closing the root session:

```bash
# laptop, new terminal
ssh deploy@VPS_IP "echo still in"
ssh root@VPS_IP     # should now be refused
```

Only close the original root@vps terminal once both of those behave as expected.

---

## 3. Firewall

Default deny incoming, SSH only, with connection-rate limiting as a first line against
brute force (fail2ban in the next section does the actual banning):

```bash
# root@vps
ufw default deny incoming
ufw default allow outgoing
ufw limit OpenSSH        # rate-limited allow on 22/tcp
ufw enable
ufw status verbose
```

80 and 443 stay closed on purpose. The app is reachable only through the Cloudflare
Tunnel (section 8) — `cloudflared` on this box dials *out* to Cloudflare and Cloudflare
proxies inbound traffic to it over that outbound connection, so there is never an
inbound HTTP/HTTPS port to open, scan, or patch.

SSH is deliberately **not** routed through the tunnel, even though it could be. A box
you can only reach via a tunnel daemon is a box with no recovery path if that daemon
ever wedges or its config is wrong — a locked-out server with no console access is a
worse failure mode than a key-only, rate-limited, fail2ban'd port 22. OVH's rescue
console is the real fallback either way, but there's no reason to remove the cheap one.

---

## 4. fail2ban

```bash
# root@vps
apt install -y fail2ban
cat <<'EOF' > /etc/fail2ban/jail.local
[sshd]
enabled = true
port = 22
maxretry = 4
findtime = 10m
bantime = 1h
EOF
systemctl restart fail2ban
systemctl enable fail2ban
fail2ban-client status sshd
```

---

## 5. Unattended security upgrades

```bash
# root@vps
apt install -y unattended-upgrades apt-listchanges
dpkg-reconfigure -f noninteractive unattended-upgrades
```

That writes `/etc/apt/apt.conf.d/20auto-upgrades`. Confirm it actually says what you
think:

```bash
# root@vps
cat /etc/apt/apt.conf.d/20auto-upgrades
```

Expected:

```
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
```

"Installed" is not "configured" — do a dry run to see it actually resolve and apply
updates, and check the log it leaves behind:

```bash
# root@vps
unattended-upgrade --dry-run --debug
cat /var/log/unattended-upgrades/unattended-upgrades.log
```

This box has no user watching a console at 3am, so leave automatic reboots off (the
default) rather than have the app stack restart unattended — you'll reboot by hand when
`/var/run/reboot-required` shows up, same check as section 1.

---

## 6. Swap

4 GB of RAM running Postgres plus a FastAPI container plus a worker container is
workable but has no slack — the failure mode without swap is the OOM killer picking off
Postgres under a burst of memory pressure, which is a much worse afternoon than a bit of
swap thrashing.

```bash
# root@vps
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
cat <<'EOF' > /etc/sysctl.d/60-swappiness.conf
vm.swappiness=10
EOF
sysctl --system
swapon --show
cat /proc/sys/vm/swappiness   # expect 10
```

Low swappiness (10 vs. the default 60) tells the kernel to prefer reclaiming page cache
over swapping out process memory — you want swap as a shock absorber for spikes, not as
a place Postgres's working set lives day to day.

---

## 7. Docker Engine + Compose plugin

The Ubuntu-archive `docker.io` package lags upstream releases; use Docker's own repo.

```bash
# root@vps
apt install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list

apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
usermod -aG docker deploy
```

```bash
# laptop
ssh deploy@VPS_IP   # fresh login so the docker group membership takes effect
```

```bash
# deploy@vps
docker run hello-world
docker compose version
```

**The one gotcha that matters for this whole design:** Docker manages its own iptables
rules and inserts them *ahead of* ufw's. A `ports: ["8000:8000"]` in a compose file
publishes on every interface, including the public one — ufw's `deny incoming` does
**not** stop it, because Docker's rules run first. The only "no inbound ports" guarantee
that's actually true is binding published ports to loopback explicitly:

```yaml
ports:
  - "127.0.0.1:8000:8000"   # correct — reachable only from the box itself
# ports:
#   - "8000:8000"           # wrong here — reachable from the internet regardless of ufw
```

Every container in this stack (Postgres now, the app and worker later) should publish
that way, since `cloudflared` reaches them over `localhost` anyway.

---

## 8. Cloudflare Tunnel

This is the new piece. The shape of it: `cloudflared` runs on the box as a systemd
service, holds an outbound connection to Cloudflare's edge, and Cloudflare proxies
`https://app.jobs4life.hiltonlabs.org` down that connection to `http://localhost:8000`.
No listener, no port forward, no origin certificate to manage — the tunnel *is* the
origin as far as Cloudflare is concerned.

```bash
# root@vps
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg -o /usr/share/keyrings/cloudflare-main.gpg
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared $(lsb_release -cs) main" \
  > /etc/apt/sources.list.d/cloudflared.list
apt update
apt install -y cloudflared
```

Authenticate against the zone — this opens a URL you authorise **in a browser**, not on
the box:

```bash
# root@vps
cloudflared tunnel login
```

It prints a `https://dash.cloudflare.com/argotunnel?...` URL.

```
# browser, on your laptop
Open the printed URL, log in, and pick the hiltonlabs.org zone to authorise.
```

That writes `/root/.cloudflared/cert.pem`. Now create the named tunnel:

```bash
# root@vps
cloudflared tunnel create jobs4life-app
```

Note the **Tunnel ID** it prints and the credentials file path it writes,
`/root/.cloudflared/<TUNNEL_ID>.json` — that file is a bearer credential for the tunnel,
treat it like a private key (root-only, never committed, never logged). Move it
somewhere `cloudflared`'s config expects and lock it down:

```bash
# root@vps
mkdir -p /etc/cloudflared
mv /root/.cloudflared/*.json /etc/cloudflared/credentials.json
chmod 600 /etc/cloudflared/credentials.json
```

Write the routing config:

```bash
# root@vps
cat <<'EOF' > /etc/cloudflared/config.yml
tunnel: jobs4life-app
credentials-file: /etc/cloudflared/credentials.json

ingress:
  - hostname: app.jobs4life.hiltonlabs.org
    service: http://localhost:8000
  - service: http_status:404
EOF
```

The trailing `http_status:404` catch-all is required by `cloudflared` — every ingress
list needs a final rule with no hostname.

Create the DNS record. This adds a **new** CNAME (`app.jobs4life` →
`<TUNNEL_ID>.cfargotunnel.com`); it does not read, modify, or remove the existing
`jobs4life` or `www` records:

```bash
# root@vps
cloudflared tunnel route dns jobs4life-app app.jobs4life.hiltonlabs.org
```

Install and start it as a service:

```bash
# root@vps
cloudflared service install
systemctl enable --now cloudflared
systemctl status cloudflared --no-pager
```

### Prove it end to end with a placeholder

Nothing is listening on 8000 yet, so bring one up just to prove the path:

```bash
# root@vps
docker run -d --rm --name placeholder -p 127.0.0.1:8000:80 nginx:alpine
```

```bash
# laptop
curl -I https://app.jobs4life.hiltonlabs.org
```

Expect `HTTP/2 200`. Tear it down once confirmed:

```bash
# root@vps
docker stop placeholder
```

### What actually goes wrong here

- **Tunnel shows connected, but the hostname 502s.** Nothing is listening on
  `localhost:8000` (this is the expected state until the app is deployed — that's what
  the placeholder above is for). `curl -I http://localhost:8000` on the box itself to
  confirm.
- **Wrong hostname in `config.yml`.** A typo in the `hostname:` line means Cloudflare
  routes fine but `cloudflared` can't match the ingress rule and returns 404, not 502.
  Diff `config.yml` against the DNS record's name.
- **Credentials file unreadable by the service.** `cloudflared` as a systemd unit
  usually runs as root, so this mostly bites if you move or re-permission the file later
  and forget `chmod 600` + root ownership. `journalctl -u cloudflared -n 50` will say
  so plainly ("failed to read credentials file / unable to decrypt token").

---

## 9. Postgres

Running in Docker with a named volume, bound to loopback only per the rule in section 7:

```bash
# root@vps
mkdir -p /opt/jobs4life
cd /opt/jobs4life
openssl rand -base64 32 > pgpassword.txt
chmod 600 pgpassword.txt
cat <<EOF > .env
POSTGRES_USER=jobs4life
POSTGRES_DB=jobs4life
POSTGRES_PASSWORD=$(cat pgpassword.txt)
EOF
chmod 600 .env

cat <<'EOF' > docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg17
    restart: unless-stopped
    env_file: .env
    ports:
      - "127.0.0.1:5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
EOF

docker compose up -d
docker compose ps
```

This compose file is server-side infrastructure, separate from the repo's local dev
`docker-compose.yml` (port 5433) — don't confuse the two.

### Logical backups

The provider's daily snapshot backs up the **machine** — disk image, point in time,
useful for "the VPS died." It cannot restore one table, and it can't move you to a
different provider or a different disk layout. A `pg_dump` is the thing that actually
lets you do either, so keep both:

```bash
# root@vps
mkdir -p /opt/jobs4life/backups
cat <<'EOF' > /opt/jobs4life/backup.sh
#!/usr/bin/env bash
set -euo pipefail
cd /opt/jobs4life
STAMP=$(date +%Y%m%d-%H%M%S)
docker compose exec -T postgres pg_dump -U jobs4life jobs4life | gzip > "backups/jobs4life-${STAMP}.sql.gz"
find backups -name '*.sql.gz' -mtime +14 -delete
EOF
chmod 700 /opt/jobs4life/backup.sh

( crontab -l 2>/dev/null; echo "17 3 * * * /opt/jobs4life/backup.sh" ) | crontab -
crontab -l
```

Daily at 03:17, keeps 14 days, rotates the rest. Run it once by hand now so you know it
works rather than finding out in three weeks:

```bash
# root@vps
/opt/jobs4life/backup.sh
ls -lh /opt/jobs4life/backups
```

---

## 10. Verification checklist

Run these in order. Each should match the "expect" line — if one doesn't, stop and fix
that layer before moving to the next; later checks assume earlier ones passed.

```bash
# 1. SSH: root is refused, key-only deploy login works
ssh root@VPS_IP                          # expect: Permission denied
ssh deploy@VPS_IP "echo ok"              # expect: ok

# 2. Firewall: only 22 open, rate-limited
ssh deploy@VPS_IP "sudo ufw status verbose"
# expect: Default: deny (incoming) ... 22/tcp LIMIT ...

# 3. fail2ban: sshd jail active
ssh deploy@VPS_IP "sudo fail2ban-client status sshd"
# expect: Status for the jail: sshd ... Currently banned: 0 (or more, not an error)

# 4. Unattended upgrades: configured
ssh deploy@VPS_IP "cat /etc/apt/apt.conf.d/20auto-upgrades"
# expect: both Update-Package-Lists and Unattended-Upgrade set to "1"

# 5. Swap: 2G active, low swappiness
ssh deploy@VPS_IP "swapon --show && cat /proc/sys/vm/swappiness"
# expect: /swapfile listed, size 2G; swappiness 10

# 6. Docker: engine + compose plugin, deploy user in the docker group
ssh deploy@VPS_IP "docker run --rm hello-world && docker compose version"
# expect: "Hello from Docker!" and a compose version string, no sudo needed

# 7. Postgres: container healthy, reachable only on loopback
ssh deploy@VPS_IP "cd /opt/jobs4life && docker compose ps"
ssh deploy@VPS_IP "docker compose -f /opt/jobs4life/docker-compose.yml exec -T postgres pg_isready"
# expect: postgres Up; pg_isready: accepting connections
nmap -p 5432 VPS_IP   # from your laptop, if you have nmap; expect: filtered/closed, not open

# 8. Backup: a dump exists and is non-trivial size
ssh deploy@VPS_IP "ls -lh /opt/jobs4life/backups"
# expect: at least one .sql.gz file, more than a few KB

# 9. Tunnel: service up, DNS resolves, placeholder reachable
ssh deploy@VPS_IP "sudo systemctl is-active cloudflared"
# expect: active
dig +short app.jobs4life.hiltonlabs.org
# expect: a cloudflare-owned hostname/IP, not empty
# then bring the placeholder up (section 8) and:
curl -I https://app.jobs4life.hiltonlabs.org
# expect: HTTP/2 200 while the placeholder runs, 502 once you stop it — both are "working", just different states

# 10. No public ports beyond SSH
nmap -p 22,80,443,8000 VPS_IP   # from your laptop
# expect: 22 open, 80/443/8000 filtered or closed
```

---

## Learned by actually running this (2026-09-08)

Five things this runbook could not have predicted, all hit on the real box. They are
recorded here because each one *looked* like it had worked.

**1. sshd drop-ins: first match wins.** Covered above, but it is the most dangerous of
these because the failure is silent — config applied, reload succeeded, setting ignored.

**2. Cloudflare's free Universal SSL covers the apex and ONE label.** `*.hiltonlabs.org`
does not match `app.jobs4life.hiltonlabs.org`. DNS resolves to Cloudflare's edge, the
tunnel is healthy, and TLS fails the handshake with no certificate ever issued. Deeper
subdomains need paid Advanced Certificate Manager. Use a single-label hostname —
`app-jobs4life.hiltonlabs.org` — unless you intend to buy ACM.

**3. `ufw limit OpenSSH` will block your own deploy script.** It denies a source opening
6 or more connections in 30 seconds. A deploy that runs several `ssh` commands in
sequence trips it and locks itself out part-way, leaving the app built but not restarted.
Fix the script with SSH multiplexing (`ControlMaster=auto`, `ControlPersist`), not the
firewall — see `deploy/deploy.sh`.

**4. A venv's console scripts carry an absolute shebang.** Building the virtualenv at one
path in a Docker builder stage and copying it to another leaves every entry point
(`alembic`, `uvicorn`) pointing at an interpreter that does not exist. The error names the
script — `exec /app/.venv/bin/alembic: no such file or directory` — so it reads as though
the package was never installed. Build at the path it will run from.

**5. Never run the app at DEBUG log level.** Authlib logs the PKCE `code_verifier` at
DEBUG, which undermines exactly what PKCE is for. The Dockerfile pins `--log-level info`
deliberately; do not raise it to debug an OAuth problem, which is precisely when you will
be tempted to.

Also worth noting: the OVH Ubuntu image ships **26.04 LTS with OpenSSH 10.2**, not the
24.04 this runbook assumed, and its default user is `ubuntu` with a password and
passwordless sudo — `root` is locked (`passwd -S root` reports `L`) and cannot log in even
at the console. If a root password does not work, that is why.

## What this does NOT do yet

- No application deployed — no FastAPI container, no worker container, no `alembic
  upgrade`. Postgres is up and empty.
- No TLS certificate to manage on the origin — the tunnel terminates TLS at Cloudflare's
  edge and the box never needs a cert for `app.jobs4life.hiltonlabs.org`.
- No monitoring, alerting, or uptime checks.
- No log shipping — container logs are wherever `docker logs` leaves them, nothing is
  aggregated or retained beyond the daemon's defaults.
- No secrets manager — the Postgres password sits in a `chmod 600` `.env` on disk, fine
  for one box and one operator, not a long-term answer once there's more than one
  environment.
- No CI/CD — everything above was typed by hand over SSH, on purpose, since this
  runbook is scoped to "box is ready," not "box has a deploy pipeline."

---

## Decisions made that the brief left open

- **Docker/ufw port-binding gotcha (section 7).** Not asked for explicitly, but it's the
  one mistake that would silently defeat the entire "no inbound ports" design once the
  app container exists, so it's called out as a hard rule now rather than discovered
  later during the actual app deployment.
- **Backup retention: 14 days, daily at 03:17.** The brief asked for "rotation" without a
  number; 14 days is arbitrary and cheap to change in `backup.sh`.
- **fail2ban thresholds:** `maxretry=4`, `findtime=10m`, `bantime=1h` — reasonable
  defaults, not measured against anything specific to this box.
- **Postgres port bound to `127.0.0.1:5432`** rather than not published at all, so you
  can `psql` from the box itself (or over an SSH tunnel) for ad hoc admin without
  editing compose files later. It is not reachable from any other container unless
  they're attached to the same compose project's default network, which the app's
  compose file will need to join when it's written.
- **cloudflared and its credentials run as root**, not the `deploy` user — it's a
  system-level networking service either way, and keeping its secret file under
  `/etc/cloudflared` owned by root avoided a second ownership handoff.
