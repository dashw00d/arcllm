#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.xpu.sh"

LOG_DIR="${LOG_DIR:-$ROOT/logs}"
API_HOST="${API_HOST:-127.0.0.1}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2}"
WORKER_PORTS_CSV="${WORKER_PORTS:-8001,8002,8003}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPU_IDS_ARR <<< "$GPU_IDS_CSV"
IFS=',' read -r -a WORKER_PORTS_ARR <<< "$WORKER_PORTS_CSV"

if [ "${#GPU_IDS_ARR[@]}" -ne "${#WORKER_PORTS_ARR[@]}" ]; then
  echo "GPU_IDS and WORKER_PORTS must have the same number of entries" >&2
  exit 1
fi

cleanup() {
  jobs -p | xargs -r kill
}
trap cleanup EXIT

WORKER_URL_LIST=()

for idx in "${!GPU_IDS_ARR[@]}"; do
  gpu_id="${GPU_IDS_ARR[$idx]}"
  port="${WORKER_PORTS_ARR[$idx]}"
  WORKER_URL_LIST+=("http://${API_HOST}:${port}")
  "$ROOT/scripts/run_worker.sh" "$gpu_id" "$port" >"$LOG_DIR/worker-${gpu_id}.log" 2>&1 &
done

export WORKER_URLS
WORKER_URLS="$(IFS=,; echo "${WORKER_URL_LIST[*]}")"
"$ROOT/scripts/run_router.sh" "$ROUTER_PORT" >"$LOG_DIR/router.log" 2>&1 &

echo "workers: ${WORKER_PORTS_CSV}"
echo "router:  ${ROUTER_PORT}"
echo "logs:    $LOG_DIR"
wait
