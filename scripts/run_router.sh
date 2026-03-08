#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_HOST="${API_HOST:-127.0.0.1}"
PORT="${1:-${ROUTER_PORT:-8000}}"
export WORKER_URLS="${WORKER_URLS:-http://${API_HOST}:8001,http://${API_HOST}:8002,http://${API_HOST}:8003}"

source "$ROOT/env.xpu.sh"

exec uvicorn api.router:app \
  --app-dir "$ROOT" \
  --host "$API_HOST" \
  --port "$PORT"
