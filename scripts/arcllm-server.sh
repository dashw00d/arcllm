#!/usr/bin/env bash
# arcllm-server — Operations toolkit for 3x Intel Arc A770 LLM stack
#
# Server:
#   arcllm-server start            Start the proxy
#   arcllm-server stop             Stop proxy and backend
#   arcllm-server restart          Stop + start
#   arcllm-server status           What's running/loaded
#   arcllm-server logs             Tail server logs
#   arcllm-server models           List available models
#   arcllm-server load <model>     Pre-load a model
#   arcllm-server unload           Unload current model
#
# Health:
#   arcllm-server canary           Verify model output is coherent
#   arcllm-server dashboard        Full system status
#
# GPU:
#   arcllm-server gpu-check        Check GPU visibility (sycl-ls)
#   arcllm-server gpu-reset        Sysfs reset + driver rebind
#   arcllm-server gpu-nuke         PCI remove+rescan (clears VRAM corruption)
#
# Cache:
#   arcllm-server cache-status     Show cache sizes
#   arcllm-server cache-flush      Clear JIT + slot caches
#
# Recovery:
#   arcllm-server recover          Full recovery: stop → flush → nuke → start → canary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PROXY="$SCRIPT_DIR/arcllm-proxy.py"
LOGFILE="/tmp/arcllm-server.log"
PIDFILE="/tmp/arcllm-server.pid"
PORT=${ARCLLM_PORT:-11435}
CACHE_DIR="$ROOT/cache"
PCI_ADDRS=(0000:19:00.0 0000:67:00.0 0000:b5:00.0)

# ── Environment ────────────────────────────────────────────────────────────
setup_env() {
  if [[ -z "${ZE_AFFINITY_MASK:-}" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT/env.sglang-xpu.sh"
  fi
  export GGML_SYCL_DISABLE_GRAPH=1  # graph recording causes DG2 corruption
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
}

# ── Server commands ────────────────────────────────────────────────────────
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

  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    echo ""
    echo "Ready! Proxy listening on http://localhost:$PORT"
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

# ── Health commands ────────────────────────────────────────────────────────
cmd_canary() {
  if ! cmd_is_running; then
    echo "FAIL: proxy not running"
    return 1
  fi

  echo -n "Canary test (2+2=?)... "
  local result
  result=$(curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen35","messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],"max_tokens":10,"chat_template_kwargs":{"enable_thinking":false}}' \
    2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
c = d.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
ascii_ratio = sum(1 for ch in c if ord(ch) < 128) / max(len(c), 1)
if '4' in c and ascii_ratio > 0.7:
    print('PASS')
else:
    print(f'FAIL: {c[:80]}')
" 2>/dev/null) || result="FAIL: no response"

  echo "$result"
  [[ "$result" == "PASS" ]]
}

cmd_dashboard() {
  setup_env

  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║              LLM Stack Dashboard                        ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  echo ""

  # GPUs
  echo "── GPUs ──"
  local gpu_count
  gpu_count=$(sycl-ls 2>/dev/null | grep -c "level_zero:gpu" || echo 0)
  if [[ "$gpu_count" -eq 3 ]]; then
    echo "  ✓ $gpu_count GPUs visible"
  else
    echo "  ✗ Only $gpu_count/3 GPUs visible — run: $0 gpu-reset"
  fi
  echo ""

  # Server
  echo "── Server ──"
  cmd_status
  echo ""

  # Cache
  echo "── Cache ──"
  cmd_cache_status
  echo ""

  # Docker
  echo "── Docker (arc-tools) ──"
  if command -v docker &>/dev/null; then
    docker compose -f "$ROOT/arc-tools/docker-compose.yml" ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null || echo "  (docker compose not available)"
  else
    echo "  (docker not installed)"
  fi
}

# ── GPU commands ───────────────────────────────────────────────────────────
cmd_gpu_check() {
  setup_env
  local count
  count=$(sycl-ls 2>/dev/null | grep -c "level_zero:gpu" || echo 0)
  echo "$count/3 GPUs visible"
  sycl-ls 2>/dev/null | grep "level_zero:gpu" || true
  [[ "$count" -eq 3 ]]
}

cmd_gpu_reset() {
  echo "Step 1: Sysfs hardware reset..."
  for p in /sys/class/drm/card*/device/reset; do
    echo 1 | sudo tee "$p" >/dev/null 2>&1 && echo "  Reset $(dirname "$p")" || true
  done
  sleep 3

  setup_env
  local count
  count=$(sycl-ls 2>/dev/null | grep -c "level_zero:gpu" || echo 0)
  if [[ "$count" -eq 3 ]]; then
    echo "✓ $count GPUs recovered"
    return 0
  fi

  echo "Step 2: Driver rebind (sysfs reset insufficient)..."
  for pci in "${PCI_ADDRS[@]}"; do
    echo "$pci" | sudo tee /sys/bus/pci/drivers/i915/unbind >/dev/null 2>&1 || true
  done
  sleep 2
  for pci in "${PCI_ADDRS[@]}"; do
    echo "$pci" | sudo tee /sys/bus/pci/drivers/i915/bind >/dev/null 2>&1 || true
  done
  sleep 5

  count=$(sycl-ls 2>/dev/null | grep -c "level_zero:gpu" || echo 0)
  if [[ "$count" -eq 3 ]]; then
    echo "✓ $count GPUs recovered (after rebind)"
    return 0
  fi

  echo "✗ Only $count/3 GPUs — try: $0 gpu-nuke"
  return 1
}

cmd_gpu_nuke() {
  echo "Nuclear GPU recovery: PCI remove + rescan..."
  echo "  Killing all GPU consumers..."
  pkill -9 -f llama-server 2>/dev/null || true
  pkill -9 -f arcllm-proxy 2>/dev/null || true
  rm -f "$PIDFILE"
  sleep 2

  echo "  Removing GPUs from PCI bus..."
  for pci in "${PCI_ADDRS[@]}"; do
    echo 1 | sudo tee "/sys/bus/pci/devices/$pci/remove" >/dev/null 2>&1 && echo "    Removed $pci" || true
  done
  sleep 3

  echo "  Rescanning PCI bus..."
  echo 1 | sudo tee /sys/bus/pci/rescan >/dev/null 2>&1
  sleep 5

  echo "  Rebinding i915 driver..."
  for pci in "${PCI_ADDRS[@]}"; do
    echo "$pci" | sudo tee /sys/bus/pci/drivers/i915/bind >/dev/null 2>&1 || true
  done
  sleep 5

  setup_env
  local count
  count=$(sycl-ls 2>/dev/null | grep -c "level_zero:gpu" || echo 0)
  if [[ "$count" -eq 3 ]]; then
    echo "✓ $count GPUs recovered (VRAM cleared)"
  else
    echo "✗ Only $count/3 GPUs — reboot may be required"
    return 1
  fi
}

# ── Cache commands ─────────────────────────────────────────────────────────
cmd_cache_status() {
  echo "  JIT (live):"
  if [[ -d /tmp/neo_compiler_cache ]]; then
    du -sh /tmp/neo_compiler_cache 2>/dev/null | awk '{print "    " $1 " " $2}'
  else
    echo "    (none)"
  fi

  echo "  JIT (backups):"
  local found=0
  for d in "$CACHE_DIR"/neo_compiler_cache_*; do
    [[ -d "$d" ]] && du -sh "$d" 2>/dev/null | awk '{print "    " $1 " " $2}' && found=1
  done
  [[ $found -eq 0 ]] && echo "    (none)"

  echo "  Slot caches:"
  if ls "$CACHE_DIR/slots/"*.bin &>/dev/null 2>&1; then
    ls -lh "$CACHE_DIR/slots/"*.bin 2>/dev/null | awk '{print "    " $5 " " $9}'
  else
    echo "    (none)"
  fi
}

cmd_cache_flush() {
  local target="${2:-all}"
  echo "Flushing caches..."

  # JIT caches
  rm -rf /tmp/neo_compiler_cache /tmp/opencl_cache ~/.cache/neo_compiler_cache
  echo "  ✓ JIT caches cleared"

  # Slot caches
  if [[ "$target" == "all" ]]; then
    rm -f "$CACHE_DIR/slots/"*.bin 2>/dev/null
    echo "  ✓ All slot caches cleared"
  else
    rm -f "$CACHE_DIR/slots/${target}".slot*.bin 2>/dev/null
    echo "  ✓ Slot caches cleared for: $target"
  fi
}

# ── Recovery ───────────────────────────────────────────────────────────────
cmd_recover() {
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║           Full Recovery Sequence                         ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  echo ""

  echo "1/5 Stopping server..."
  cmd_stop 2>/dev/null || true
  sleep 2

  echo "2/5 Flushing all caches..."
  cmd_cache_flush
  echo ""

  echo "3/5 Nuclear GPU reset (PCI remove+rescan)..."
  cmd_gpu_nuke || { echo "GPU recovery failed — reboot required"; exit 1; }
  echo ""

  echo "4/5 Starting server..."
  cmd_start
  echo ""

  echo "5/5 Loading model and running canary..."
  cmd_load qwen35
  sleep 3
  if cmd_canary; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║           Recovery COMPLETE — system healthy             ║"
    echo "╚══════════════════════════════════════════════════════════╝"
  else
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  Recovery FAILED — model still garbled, reboot needed   ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    exit 1
  fi
}

# ── Main ───────────────────────────────────────────────────────────────────
case "${1:-help}" in
  start)        cmd_start ;;
  stop)         cmd_stop ;;
  status)       cmd_status ;;
  logs)         cmd_logs ;;
  models)       cmd_models ;;
  load)         cmd_load "${2:-}" ;;
  unload)       cmd_unload ;;
  restart)      cmd_stop; sleep 2; cmd_start ;;
  canary)       cmd_canary ;;
  dashboard)    cmd_dashboard ;;
  gpu-check)    cmd_gpu_check ;;
  gpu-reset)    cmd_gpu_reset ;;
  gpu-nuke)     cmd_gpu_nuke ;;
  cache-status) cmd_cache_status ;;
  cache-flush)  cmd_cache_flush "$@" ;;
  recover)      cmd_recover ;;
  *)
    echo "Usage: $0 <command> [args]"
    echo ""
    echo "Server:  start | stop | restart | status | logs | models | load <model> | unload"
    echo "Health:  canary | dashboard"
    echo "GPU:     gpu-check | gpu-reset | gpu-nuke"
    echo "Cache:   cache-status | cache-flush [model|all]"
    echo "Recovery: recover (full: stop → flush → nuke → start → canary)"
    ;;
esac
