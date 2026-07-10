#!/usr/bin/env bash
# Demo: closed-loop live motor imitation (hear -> gated playback -> capture -> measure).
# Measurement-only (tag_only); no signal subtraction claim unless the gate passes.
#
# Usage:
#   ./scripts/demo_imitation.sh [extra args...]
#
# Examples:
#   ./scripts/demo_imitation.sh                       # basic imitation loop
#   ./scripts/demo_imitation.sh --with-attention      # + active-attention steer loop
#   ./scripts/demo_imitation.sh --with-self-subtraction --held-out-trials 3
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_PY="${REPO_ROOT}/.venv/bin/python"

exec "$VENV_PY" -m cora_infans --listen-live \
  --with-motor-imitation \
  --real-io \
  --duration-ms 8000 \
  --volume-db 12 \
  "$@"
