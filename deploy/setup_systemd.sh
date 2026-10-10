#!/usr/bin/env bash
# Run LatentBlunder as a systemd service on a Debian or Ubuntu VPS, in Docker.
#
#   sudo deploy/setup_systemd.sh          # from a checkout of this repository
#
# Builds the image from this checkout, keeps the Lichess token in a root-only
# env file (never in the image or the unit), keeps the logs and the game log in
# a host directory, and restarts the bot if it exits. Run it again after
# pulling changes to rebuild and restart.
#
# Only one copy of the bot may run per account: stop the one on any other
# machine first, or the two keep knocking each other off Lichess.
set -euo pipefail

NAME=latentblunder
IMAGE="${NAME}:latest"
ENV_FILE="/etc/${NAME}/bot.env"
DATA_DIR="/var/lib/${NAME}"
UNIT="/etc/systemd/system/${NAME}.service"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BOT_UID=10001   # the image's unprivileged user

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root: sudo $0" >&2
  exit 1
fi
DOCKER="$(command -v docker || true)"
if [ -z "${DOCKER}" ]; then
  echo "Docker is not installed: https://docs.docker.com/engine/install/" >&2
  exit 1
fi
for needed in models/trap_policy.pth src/engine/books/repertoire.bin src/engine/books/standard.bin; do
  if [ ! -f "${REPO}/${needed}" ]; then
    echo "missing ${REPO}/${needed} (the .pth weights are not in git; copy them over)" >&2
    exit 1
  fi
done

install -d -m 0700 "/etc/${NAME}"
if [ ! -s "${ENV_FILE}" ]; then
  read -r -s -p "Lichess API token (bot:play scope): " token
  echo
  if [ -z "${token}" ]; then
    echo "no token given" >&2
    exit 1
  fi
  (umask 077 && printf 'LICHESS_API_TOKEN=%s\n' "${token}" > "${ENV_FILE}")
  unset token
fi
chmod 0600 "${ENV_FILE}"

install -d -m 0755 "${DATA_DIR}"
chown "${BOT_UID}:${BOT_UID}" "${DATA_DIR}"

"${DOCKER}" build -t "${IMAGE}" "${REPO}"

cat > "${UNIT}" <<EOF
[Unit]
Description=LatentBlunder Lichess bot
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Restart=always
RestartSec=15
TimeoutStopSec=60
ExecStartPre=-${DOCKER} rm -f ${NAME}
ExecStart=${DOCKER} run --rm --name ${NAME} --env-file ${ENV_FILE} -v ${DATA_DIR}:/app/build ${IMAGE}
ExecStop=${DOCKER} stop --time 30 ${NAME}

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "${NAME}" >/dev/null
systemctl restart "${NAME}"
echo "running. logs: journalctl -u ${NAME} -f   game log: ${DATA_DIR}/live_games.jsonl"
