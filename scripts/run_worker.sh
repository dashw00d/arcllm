#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-0}"
PORT="${2:-8001}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
API_HOST="${API_HOST:-127.0.0.1}"

source "$ROOT/env.xpu.sh"
export ZE_AFFINITY_MASK="$GPU_ID"
export MODEL_ID

exec uvicorn api.worker:app \
  --app-dir "$ROOT" \
  --host "$API_HOST" \
  --port "$PORT"
