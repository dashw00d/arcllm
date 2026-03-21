#!/usr/bin/env bash
# Extra parallel tests based on round 1-5 findings:
# - Q4_K_M is the clear winner
# - cmdlist=0 gives +7.5%
# - np=8 is better than np=4
# - Push higher: np=12, np=16, np=8 + cmdlist=0
#
# Usage: sg render -c 'bash ./scripts/bench_parallel_extra.sh'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
SERVER="$ROOT/llama.cpp/build-sycl/bin/llama-server"
FIRE="$SCRIPT_DIR/bench_fire_requests.py"
Q4_MODEL="$ROOT/models/Qwen/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf"
RESULTS="/tmp/bench_parallel_results.tsv"
LOGDIR="/tmp/bench_parallel_logs"
PORT=8400
HOST="127.0.0.1"
HEALTH_TIMEOUT=300
REQUEST_TIMEOUT=300
MAX_TOKENS=200

# shellcheck disable=SC1091
source "$ROOT/env.sglang-xpu.sh"
HOST="127.0.0.1"
mkdir -p "$LOGDIR"

kill_server() {
  local pids
  pids=$(pgrep -f "llama-server.*--port $PORT" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "  Stopping server (PIDs: $pids)..."
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 2
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
}

wait_for_health() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT))
  echo -n "  Waiting for health"
  while ((SECONDS < deadline)); do
    if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      echo " ready (${SECONDS}s)"
      return 0
    fi
    if ! pgrep -f "llama-server.*--port $PORT" >/dev/null 2>&1; then
      echo " FAILED (server died)"
      return 1
    fi
    echo -n "."
    sleep 5
  done
  echo " TIMEOUT"
  return 1
}

fire_and_record() {
  local test_name="$1"
  local concurrent="$2"
  local notes="$3"

  echo "  Firing $concurrent concurrent requests..."
  local tmpfile
  tmpfile=$(mktemp /tmp/bench_fire_XXXXXX.json)

  python3 "$FIRE" \
    --url "http://$HOST:$PORT" \
    --concurrent "$concurrent" \
    --model "Qwen3-32B-Q4_K_M" \
    --max-tokens "$MAX_TOKENS" \
    --timeout "$REQUEST_TIMEOUT" \
    > "$tmpfile" 2>/dev/null

  if [[ -s "$tmpfile" ]]; then
    local wall tps completed failed tokens
    wall=$(python3 -c "import json; d=json.load(open('$tmpfile')); print(f\"{d['wall_time_s']:.1f}\")")
    tps=$(python3 -c "import json; d=json.load(open('$tmpfile')); print(f\"{d['total_tps']:.1f}\")")
    completed=$(python3 -c "import json; d=json.load(open('$tmpfile')); print(d['completed'])")
    failed=$(python3 -c "import json; d=json.load(open('$tmpfile')); print(d['failed'])")
    tokens=$(python3 -c "import json; d=json.load(open('$tmpfile')); print(d['total_tokens'])")

    echo "  Results: ${completed}/${concurrent} ok, ${tokens} tokens, ${wall}s wall, ${tps} t/s total"
    echo -e "${test_name}\t${concurrent}\t${completed}/${concurrent}\t${tokens}\t${wall}\t${tps}\t-\t${notes}" >> "$RESULTS"
    cp "$tmpfile" "$LOGDIR/${test_name}_result.json"
  else
    echo "  ERROR: fire_requests returned empty"
    echo -e "${test_name}\t${concurrent}\tERR\tERR\tERR\tERR\t-\t${notes} (fire failed)" >> "$RESULTS"
  fi
  rm -f "$tmpfile"
}

run_extra_test() {
  local test_name="$1"
  local concurrent="$2"
  local notes="$3"
  shift 3
  local server_args=("$@")

  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  TEST: $test_name (concurrent=$concurrent)"
  echo "  Args: ${server_args[*]}"
  echo "═══════════════════════════════════════════════════════════════"

  kill_server

  local log="$LOGDIR/${test_name}.log"
  "$SERVER" \
    --model "$Q4_MODEL" \
    --host "$HOST" --port "$PORT" \
    "${server_args[@]}" \
    > "$log" 2>&1 &
  echo "  Server PID: $!, log: $log"

  if ! wait_for_health; then
    echo "  SKIPPED: server failed"
    echo -e "${test_name}\t${concurrent}\tSKIP\tSKIP\tSKIP\tSKIP\t-\t${notes} (server failed)" >> "$RESULTS"
    kill_server
    return
  fi

  fire_and_record "$test_name" "$concurrent" "$notes"
  kill_server
  sleep 3
}

echo ""
echo "=== Extra Parallel Tests (Q4_K_M + cmdlist=0) ==="
echo ""

# Best combo from round 5: cmdlist=0 gave 8.5 t/s at np=4
# Now test with np=8, np=12, np=16

export GGML_SYCL_DISABLE_GRAPH=0
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0

run_extra_test "X1_np8_cmdlist0" 8 "Q4KM np=8 + cmdlist=0 (best combo)" \
  --split-mode layer -ngl 999 --tensor-split "1,1,1" -fa on \
  -c 16384 -np 8 -b 2048 -ub 512

run_extra_test "X2_np12_cmdlist0" 12 "Q4KM np=12 + cmdlist=0 (push higher)" \
  --split-mode layer -ngl 999 --tensor-split "1,1,1" -fa on \
  -c 24576 -np 12 -b 2048 -ub 512

run_extra_test "X3_np16_cmdlist0" 16 "Q4KM np=16 + cmdlist=0 (max push)" \
  --split-mode layer -ngl 999 --tensor-split "1,1,1" -fa on \
  -c 32768 -np 16 -b 2048 -ub 512

run_extra_test "X4_np8_cmdlist0_bigctx" 8 "Q4KM np=8 cmdlist=0 c=32768 (big context)" \
  --split-mode layer -ngl 999 --tensor-split "1,1,1" -fa on \
  -c 32768 -np 8 -b 2048 -ub 512

# Test np=4 with cmdlist=0 + graph disabled (isolate cmdlist effect)
export GGML_SYCL_DISABLE_GRAPH=1
run_extra_test "X5_np4_cmdlist0_nograph" 4 "Q4KM np=4 cmdlist=0 NO graph (isolate cmdlist)" \
  --split-mode layer -ngl 999 --tensor-split "1,1,1" -fa on \
  -c 8192 -np 4 -b 2048 -ub 512

# Test the abliterated model (what the proxy actually uses)
export GGML_SYCL_DISABLE_GRAPH=0
ABLITERATED="$ROOT/models/Qwen/Qwen3-32B-abliterated-GGUF/Qwen3-32B-abliterated.Q8_0.gguf"
if [[ -f "$ABLITERATED" ]]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  TEST: X6 — Abliterated Q8_0 (production model)"
  echo "═══════════════════════════════════════════════════════════════"
  kill_server
  log="$LOGDIR/X6_abliterated_np4.log"
  "$SERVER" \
    --model "$ABLITERATED" \
    --host "$HOST" --port "$PORT" \
    --split-mode layer -ngl 999 \
    -c 8192 -fa on -np 4 --reasoning-budget 0 \
    > "$log" 2>&1 &
  echo "  Server PID: $!, log: $log"

  if wait_for_health; then
    fire_and_record "X6_abliterated_np4" 4 "Abliterated Q8_0 np=4 cmdlist=0 (production model)"
  else
    echo -e "X6_abliterated_np4\t4\tSKIP\tSKIP\tSKIP\tSKIP\t-\tserver failed" >> "$RESULTS"
  fi
  kill_server
  sleep 3

  # Also test np=2 (current production config)
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  TEST: X7 — Abliterated Q8_0 np=2 (current prod)"
  echo "═══════════════════════════════════════════════════════════════"
  kill_server
  log="$LOGDIR/X7_abliterated_np2.log"
  "$SERVER" \
    --model "$ABLITERATED" \
    --host "$HOST" --port "$PORT" \
    --split-mode layer -ngl 999 \
    -c 49152 -fa on -np 2 --cache-reuse 256 --reasoning-budget 0 --threads 10 \
    > "$log" 2>&1 &
  echo "  Server PID: $!, log: $log"

  if wait_for_health; then
    fire_and_record "X7_abliterated_np2" 2 "Abliterated Q8_0 np=2 (CURRENT production config)"
  else
    echo -e "X7_abliterated_np2\t2\tSKIP\tSKIP\tSKIP\tSKIP\t-\tserver failed" >> "$RESULTS"
  fi
  kill_server
  sleep 3
fi

echo ""
echo "=== Extra tests complete ==="
column -t -s $'\t' < "$RESULTS"
