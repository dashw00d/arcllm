#!/usr/bin/env bash
# arcllm-server — Ollama-compatible lazy-loading proxy for 3x Intel Arc A770
#
# Usage:
#   arcllm-server start            Start the proxy (no model loaded until requested)
#   arcllm-server stop             Stop proxy and any loaded model
#   arcllm-server status           Check what's running/loaded
#   arcllm-server logs             Tail server logs
#   arcllm-server models           List available models
#   arcllm-server load <model>     Pre-load a model (optional, normally auto-loaded)
#   arcllm-server unload           Unload current model (free VRAM)
#
# Like Ollama: listens immediately on port 11435, loads models on demand,
# unloads after idle timeout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PROXY="$SCRIPT_DIR/arcllm-proxy.py"
LOGFILE="/tmp/arcllm-server.log"
PIDFILE="/tmp/arcllm-server.pid"
PORT=${ARCLLM_PORT:-11435}

# ── Environment ────────────────────────────────────────────────────────────
setup_env() {
  if [[ -z "${ZE_AFFINITY_MASK:-}" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT/env.sglang-xpu.sh"
  fi
  export GGML_SYCL_DISABLE_GRAPH=0
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
}

# ── Commands ───────────────────────────────────────────────────────────────
cmd_start() {
  if cmd_is_running; then
    echo "Proxy already running (PID $(cat "$PIDFILE"))"
    exit 1
  fi

  setup_env

  echo "Starting arcllm-proxy..."
  echo "  Port:   $PORT"
  echo "  Log:    $LOGFILE"
  echo "  Mode:   lazy (no model loaded until requested)"

  ARCLLM_PORT="$PORT" python3 "$PROXY" &
  local pid=$!
  echo "$pid" > "$PIDFILE"
  echo "  PID:    $pid"

  # Proxy starts instantly (no model to load)
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    echo ""
    echo "Ready! Proxy listening on http://localhost:$PORT"
    echo "Send a request with any model to auto-load it."
    echo "Available models: curl http://localhost:$PORT/v1/models"
  else
    echo "Proxy died on startup."
    rm -f "$PIDFILE"
    exit 1
  fi
}

cmd_stop() {
  if [[ ! -f "$PIDFILE" ]]; then
    echo "No PID file found."
    pkill -f "arcllm-proxy" 2>/dev/null && echo "Killed stray proxy" || true
    pkill -f "llama-server.*--port 18400" 2>/dev/null && echo "Killed stray backend" || true
    return 0
  fi

  local pid
  pid=$(cat "$PIDFILE")
  echo "Stopping proxy (PID $pid)..."
  kill "$pid" 2>/dev/null || true
  sleep 2
  kill -9 "$pid" 2>/dev/null || true
  # Also kill any backend llama-server
  pkill -f "llama-server.*--port 18400" 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "Stopped."
}

cmd_status() {
  if cmd_is_running; then
    local pid
    pid=$(cat "$PIDFILE")
    echo "arcllm-proxy running (PID $pid) on port $PORT"
    curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
status = d.get('status', 'unknown')
model = d.get('model', 'none')
if status == 'loaded':
    pid = d.get('pid', '?')
    idle = d.get('idle_seconds', 0)
    print(f'  Model: {model} (PID {pid}, idle {idle}s)')
else:
    print('  No model loaded (VRAM free)')
" 2>/dev/null || echo "  (health check failed)"
  else
    echo "arcllm-proxy is not running"
  fi
}

cmd_logs() {
  tail -f "$LOGFILE"
}

cmd_models() {
  if cmd_is_running; then
    curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Available models:')
for m in d.get('data', []):
    status = '(loaded)' if m.get('loaded') else ('(ready)' if m.get('available') else '(not downloaded)')
    aliases = ', '.join(m.get('aliases', []))
    alias_str = f' aka {aliases}' if aliases else ''
    print(f\"  {m['id']}{alias_str} {status}\")
" 2>/dev/null
  else
    echo "Proxy not running. Start with: $0 start"
  fi
}

cmd_load() {
  local model="${1:?Usage: $0 load <model>}"
  if ! cmd_is_running; then
    echo "Proxy not running. Start with: $0 start"
    exit 1
  fi
  echo "Loading $model..."
  curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
    >/dev/null 2>&1 && echo "Loaded!" || echo "Failed — check logs: $LOGFILE"
}

cmd_unload() {
  echo "Unloading model (killing backend)..."
  pkill -f "llama-server.*--port 18400" 2>/dev/null && echo "Unloaded." || echo "No model was loaded."
}

cmd_is_running() {
  [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

# ── Main ───────────────────────────────────────────────────────────────────
case "${1:-help}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  status)  cmd_status ;;
  logs)    cmd_logs ;;
  models)  cmd_models ;;
  load)    cmd_load "${2:-}" ;;
  unload)  cmd_unload ;;
  restart) cmd_stop; sleep 2; cmd_start ;;
  *)
    echo "Usage: $0 {start|stop|status|logs|models|load|unload|restart} [model]"
    ;;
esac
