#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/op/OmniVoice"
SYSTEMD_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/omnivoice"

mkdir -p "${SYSTEMD_DIR}" "${CONFIG_DIR}"

install -m 0644 "${ROOT}/deploy/systemd/omnivoice.service" "${SYSTEMD_DIR}/omnivoice.service"
install -m 0644 "${ROOT}/deploy/systemd/omnivoice-celery.service" "${SYSTEMD_DIR}/omnivoice-celery.service"
install -m 0644 "${ROOT}/deploy/systemd/omnivoice-gradio.service" "${SYSTEMD_DIR}/omnivoice-gradio.service"
install -m 0644 "${ROOT}/deploy/systemd/omnivoice-pool.env" "${CONFIG_DIR}/omnivoice-pool.env"

systemctl --user daemon-reload
systemctl --user enable --now omnivoice-celery.service omnivoice.service omnivoice-gradio.service
