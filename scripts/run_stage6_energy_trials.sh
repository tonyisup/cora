#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="/private/tmp/cora-stage6-energy-$STAMP"
mkdir -p "$OUT"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing workspace Python: $PYTHON" >&2
  exit 2
fi

echo "Stage 6 energy trials need read-only powermetrics access."
sudo -v

printf '%s\n' \
  "protocol=docs/evidence/2026-07-10-stage6-energy-preregistration.json" \
  "workspace=$ROOT" \
  "git_commit=$(git -C "$ROOT" rev-parse HEAD)" \
  "started_utc=$STAMP" \
  "powermetrics_sample_interval_ms=500" \
  "powermetrics_samples=24" \
  "workload_duration_seconds=10" > "$OUT/manifest.txt"

run_trial() {
  local trial="$1"
  local condition="$2"
  local prefix
  prefix="$(printf '%02d_%s' "$trial" "$condition")"

  echo "[$trial/9] $condition"
  sudo -n /usr/bin/powermetrics \
    --samplers cpu_power \
    --sample-rate 500 \
    --sample-count 24 \
    > "$OUT/${prefix}_powermetrics.txt" \
    2> "$OUT/${prefix}_powermetrics.stderr" &
  local power_pid=$!

  # Two 500 ms samples precede the fixed 10 second workload. The final two
  # samples follow it; only the central 20 are eligible under the preregistration.
  sleep 1
  if [[ "$condition" == "idle" ]]; then
    local idle_started idle_ended
    idle_started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sleep 10
    idle_ended="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '{"processor":"idle","duration_requested_seconds":10,"started_utc":"%s","ended_utc":"%s"}\n' \
      "$idle_started" "$idle_ended" > "$OUT/${prefix}_workload.json"
  else
    "$PYTHON" -m cora_infans \
      --energy-workload \
      --mode "$condition" \
      --duration-seconds 10 \
      > "$OUT/${prefix}_workload.json"
  fi

  wait "$power_pid"
}

# Frozen paired order from the preregistration.
run_trial 1 idle
run_trial 2 conventional
run_trial 3 cora
run_trial 4 idle
run_trial 5 cora
run_trial 6 conventional
run_trial 7 idle
run_trial 8 conventional
run_trial 9 cora

shasum -a 256 "$OUT"/* > "$OUT/sha256.txt"
echo "complete: $OUT"
