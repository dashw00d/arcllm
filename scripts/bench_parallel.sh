#!/usr/bin/env bash
# Comprehensive Parallel Throughput Benchmark — Find Max Concurrent Capacity
# Hardware: 3x Intel Arc A770 (48GB VRAM), 64GB RAM, i9-7900X
# Model: Qwen3-32B (Q8_0 + Q4_K_M)
# Binary: build-sycl (clean SYCL build)
#
# Usage:
#   ./scripts/bench_parallel.sh              Run all rounds
#   ./scripts/bench_parallel.sh 1            Run round 1 only
#   ./scripts/bench_parallel.sh 2.1 2.3      Run specific tests
#   ./scripts/bench_parallel.sh 1 2 3        Run rounds 1-3
#
# Results: /tmp/bench_parallel_results.tsv
# GPU log: /tmp/bench_parallel_gpu.jsonl

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

# Use clean build-sycl binary
SERVER="$ROOT/llama.cpp/build-sycl/bin/llama-server"
BENCH_BIN="$ROOT/llama.cpp/build-sycl/bin/llama-bench"
FIRE="$SCRIPT_DIR/bench_fire_requests.py"
MONITOR="$SCRIPT_DIR/bench_gpu_monitor.py"

# Models
Q8_MODEL="$ROOT/models/Qwen/Qwen3-32B-GGUF/Qwen3-32B-Q8_0.gguf"
Q4_MODEL="$ROOT/models/Qwen/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf"
Q8_ABLITERATED="$ROOT/models/Qwen/Qwen3-32B-abliterated-GGUF/Qwen3-32B-abliterated.Q8_0.gguf"

# Output
RESULTS="/tmp/bench_parallel_results.tsv"
GPU_LOG="/tmp/bench_parallel_gpu.jsonl"
LOGDIR="/tmp/bench_parallel_logs"

PORT=8400
HOST="127.0.0.1"
HEALTH_TIMEOUT=300   # 5 min model load
REQUEST_TIMEOUT=300  # 5 min per request
MAX_TOKENS=200

# ── Environment ────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$ROOT/env.sglang-xpu.sh"
HOST="127.0.0.1"  # conda overrides HOST

mkdir -p "$LOGDIR"

# ── Helpers ────────────────────────────────────────────────────────────────

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
  echo -n "  Waiting for server health"
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
  echo " TIMEOUT after ${HEALTH_TIMEOUT}s"
  return 1
}

start_monitor() {
  # Clear old log
  > "$GPU_LOG"
  python3 "$MONITOR" --log "$GPU_LOG" --interval 0.5 &
  MONITOR_PID=$!
  sleep 1  # let it warm up
  echo "  GPU monitor started (PID $MONITOR_PID)"
}

stop_monitor() {
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

get_gpu_summary() {
  # Get average GPU utilization for the time window [$1, $2]
  local after="${1:-0}"
  local before="${2:-$(python3 -c 'import time; print(time.time())')}"
  python3 "$MONITOR" --summarize --log "$GPU_LOG" --after "$after" --before "$before" 2>/dev/null || echo "{}"
}

# Fire concurrent requests and parse results
fire_requests() {
  local concurrent="$1"
  local model="${2:-qwen3-32b}"
  local max_tokens="${3:-$MAX_TOKENS}"
  local records_per_call="${4:-1}"
  local extra_args="${5:-}"

  local tmpfile
  tmpfile=$(mktemp /tmp/bench_fire_XXXXXX.json)

  python3 "$FIRE" \
    --url "http://$HOST:$PORT" \
    --concurrent "$concurrent" \
    --model "$model" \
    --max-tokens "$max_tokens" \
    --records-per-call "$records_per_call" \
    --timeout "$REQUEST_TIMEOUT" \
    $extra_args \
    > "$tmpfile" 2>/dev/null

  # Parse results
  if [[ -s "$tmpfile" ]]; then
    _FIRE_RESULT=$(cat "$tmpfile")
    _FIRE_WALL=$(python3 -c "import json; d=json.load(open('$tmpfile')); print(f\"{d['wall_time_s']:.1f}\")")
    _FIRE_TOTAL_TPS=$(python3 -c "import json; d=json.load(open('$tmpfile')); print(f\"{d['total_tps']:.1f}\")")
    _FIRE_COMPLETED=$(python3 -c "import json; d=json.load(open('$tmpfile')); print(d['completed'])")
    _FIRE_FAILED=$(python3 -c "import json; d=json.load(open('$tmpfile')); print(d['failed'])")
    _FIRE_TOKENS=$(python3 -c "import json; d=json.load(open('$tmpfile')); print(d['total_tokens'])")
  else
    _FIRE_RESULT="{}"
    _FIRE_WALL="ERR"
    _FIRE_TOTAL_TPS="ERR"
    _FIRE_COMPLETED="0"
    _FIRE_FAILED="0"
    _FIRE_TOKENS="0"
  fi
  rm -f "$tmpfile"
}

# Run a single server-based test
run_test() {
  local test_name="$1"
  local model_path="$2"
  local concurrent="$3"
  shift 3

  local env_vars=()
  local server_args=()
  local notes=""
  local max_tokens="$MAX_TOKENS"
  local records_per_call=1

  # Parse arguments
  while [[ $# -gt 0 ]]; do
    case "$1" in
      env:*)    env_vars+=("${1#env:}") ;;
      note:*)   notes="${1#note:}" ;;
      tokens:*) max_tokens="${1#tokens:}" ;;
      rpc:*)    records_per_call="${1#rpc:}" ;;
      *)        server_args+=("$1") ;;
    esac
    shift
  done

  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  TEST: $test_name"
  echo "  Model: $(basename "$model_path")"
  echo "  Concurrent: $concurrent  MaxTokens: $max_tokens  Records/call: $records_per_call"
  echo "  Args: ${server_args[*]:-<default>}"
  echo "  Env:  ${env_vars[*]:-<default>}"
  echo "═══════════════════════════════════════════════════════════════"

  kill_server

  # Build environment
  local log="$LOGDIR/${test_name}.log"
  local env_exports=""
  for ev in "${env_vars[@]:-}"; do
    [[ -n "$ev" ]] && env_exports="$env_exports export $ev;"
  done

  # Start server
  echo "  Starting server..."
  (
    eval "$env_exports"
    "$SERVER" \
      --model "$model_path" \
      --host "$HOST" \
      --port "$PORT" \
      "${server_args[@]}" \
      2>&1
  ) > "$log" 2>&1 &
  local server_pid=$!
  echo "  Server PID: $server_pid, log: $log"

  if ! wait_for_health; then
    echo "  SKIPPED: server failed to start"
    echo -e "${test_name}\t${concurrent}\tSKIP\tSKIP\tSKIP\tSKIP\t${notes} (server failed)" >> "$RESULTS"
    kill_server
    return 0
  fi

  # Mark time window start
  local ts_start
  ts_start=$(python3 -c "import time; print(time.time())")

  # Fire concurrent requests
  echo "  Firing $concurrent concurrent requests..."
  fire_requests "$concurrent" "$(basename "$model_path" .gguf)" "$max_tokens" "$records_per_call"

  # Mark time window end
  local ts_end
  ts_end=$(python3 -c "import time; print(time.time())")

  # Get GPU utilization for this window
  local gpu_summary
  gpu_summary=$(get_gpu_summary "$ts_start" "$ts_end")
  local gpu0_avg gpu1_avg gpu2_avg cpu_avg
  gpu0_avg=$(echo "$gpu_summary" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('gpu0_avg','?'))" 2>/dev/null || echo "?")
  gpu1_avg=$(echo "$gpu_summary" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('gpu1_avg','?'))" 2>/dev/null || echo "?")
  gpu2_avg=$(echo "$gpu_summary" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('gpu2_avg','?'))" 2>/dev/null || echo "?")
  cpu_avg=$(echo "$gpu_summary" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('cpu_avg','?'))" 2>/dev/null || echo "?")

  echo "  Results: ${_FIRE_COMPLETED}/${concurrent} ok, ${_FIRE_TOKENS} tokens, ${_FIRE_WALL}s wall, ${_FIRE_TOTAL_TPS} t/s total"
  echo "  GPU util: GPU0:${gpu0_avg}% GPU1:${gpu1_avg}% GPU2:${gpu2_avg}% CPU:${cpu_avg}%"

  # Record result
  echo -e "${test_name}\t${concurrent}\t${_FIRE_COMPLETED}/${concurrent}\t${_FIRE_TOKENS}\t${_FIRE_WALL}\t${_FIRE_TOTAL_TPS}\tGPU:${gpu0_avg}/${gpu1_avg}/${gpu2_avg} CPU:${cpu_avg}\t${notes}" >> "$RESULTS"

  # Save full JSON result
  echo "$_FIRE_RESULT" > "$LOGDIR/${test_name}_result.json"

  kill_server
  sleep 3  # let GPU memory release
}

# ── Results Header ─────────────────────────────────────────────────────────
if [[ ! -f "$RESULTS" ]]; then
  echo -e "test\tconcurrent\tok\ttokens\twall_s\ttotal_tps\tgpu_cpu\tnotes" > "$RESULTS"
fi

echo ""
echo "Qwen3-32B Parallel Throughput Benchmark (build-sycl)"
echo "Results:  $RESULTS"
echo "GPU log:  $GPU_LOG"
echo "Logs:     $LOGDIR"
echo "Started:  $(date -Iseconds)"

# ── Validate binaries ─────────────────────────────────────────────────────
if [[ ! -x "$SERVER" ]]; then
  echo "ERROR: llama-server not found at $SERVER"
  exit 1
fi

# Build llama-bench in build-sycl if missing
if [[ ! -f "$BENCH_BIN" ]]; then
  echo ""
  echo "Building llama-bench in build-sycl..."
  (cd "$ROOT/llama.cpp/build-sycl" && cmake --build . --target llama-bench -j"$(nproc)" 2>&1 | tail -5)
  if [[ ! -f "$BENCH_BIN" ]]; then
    echo "WARNING: llama-bench build failed, skipping llama-bench tests"
  fi
fi

# ── Parse which tests to run ──────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
  REQUESTED=("$@")
else
  REQUESTED=(1 2 3 4 5 6 7)
fi

should_run() {
  local test_id="$1"
  local round="${test_id%%.*}"
  for r in "${REQUESTED[@]}"; do
    # Exact match (e.g., "2.3") or round match (e.g., "2")
    if [[ "$r" == "$test_id" || "$r" == "$round" ]]; then
      return 0
    fi
  done
  return 1
}

# ── Start GPU Monitor ─────────────────────────────────────────────────────
start_monitor

# ── Common flags ──────────────────────────────────────────────────────────
LAYER_BASE=(--split-mode layer -ngl 999 --tensor-split "1,1,1" -fa on)
ROW_BASE=(--split-mode row -ngl 999 --tensor-split "1,1,1" -fa on)
SYCL_GRAPH="env:GGML_SYCL_DISABLE_GRAPH=0"
SYCL_CMDLIST0="env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0"
SYCL_CMDLIST1="env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1"

# ═══════════════════════════════════════════════════════════════════════════
# ROUND 1: Baseline (Layer-Split, Serial)
# ═══════════════════════════════════════════════════════════════════════════

if should_run "1.1"; then
  run_test "1.1_q8_baseline" "$Q8_MODEL" 1 \
    "${LAYER_BASE[@]}" -c 4096 -np 1 -b 2048 -ub 512 \
    "$SYCL_GRAPH" \
    "note:Q8_0 layer-split baseline, clean build-sycl"
fi

if should_run "1.2"; then
  run_test "1.2_q4km_baseline" "$Q4_MODEL" 1 \
    "${LAYER_BASE[@]}" -c 4096 -np 1 -b 2048 -ub 512 \
    "$SYCL_GRAPH" \
    "note:Q4_K_M layer-split baseline, clean build-sycl"
fi

# ═══════════════════════════════════════════════════════════════════════════
# ROUND 2: Parallel Slots (Layer-Split, -np > 1)
# This is what crashed before. Test with clean build.
# ═══════════════════════════════════════════════════════════════════════════

if should_run "2.1"; then
  run_test "2.1_q8_np2" "$Q8_MODEL" 2 \
    "${LAYER_BASE[@]}" -c 4096 -np 2 -b 2048 -ub 512 \
    "$SYCL_GRAPH" \
    "note:Q8_0 layer-split np=2, minimum parallel"
fi

if should_run "2.2"; then
  run_test "2.2_q8_np4" "$Q8_MODEL" 4 \
    "${LAYER_BASE[@]}" -c 4096 -np 4 -b 2048 -ub 512 \
    "$SYCL_GRAPH" \
    "note:Q8_0 layer-split np=4, medium parallel"
fi

if should_run "2.3"; then
  run_test "2.3_q8_np4_b4096" "$Q8_MODEL" 4 \
    "${LAYER_BASE[@]}" -c 8192 -np 4 -b 4096 -ub 512 \
    "$SYCL_GRAPH" \
    "note:Q8_0 layer-split np=4 b=4096, larger batch"
fi

if should_run "2.4"; then
  run_test "2.4_q4km_np4" "$Q4_MODEL" 4 \
    "${LAYER_BASE[@]}" -c 8192 -np 4 -b 2048 -ub 512 \
    "$SYCL_GRAPH" \
    "note:Q4_K_M layer-split np=4, more KV room"
fi

if should_run "2.5"; then
  run_test "2.5_q4km_np8" "$Q4_MODEL" 8 \
    "${LAYER_BASE[@]}" -c 16384 -np 8 -b 2048 -ub 512 \
    "$SYCL_GRAPH" \
    "note:Q4_K_M layer-split np=8, push it"
fi

# ═══════════════════════════════════════════════════════════════════════════
# ROUND 3: Row-Split (Clean Build)
# Row-split was 0.77 t/s before (DMMV), test with clean build.
# ═══════════════════════════════════════════════════════════════════════════

if should_run "3.1"; then
  run_test "3.1_q8_row" "$Q8_MODEL" 1 \
    "${ROW_BASE[@]}" -c 4096 -np 1 \
    "$SYCL_GRAPH" \
    "note:Q8_0 row-split baseline, clean build"
fi

if should_run "3.2"; then
  run_test "3.2_q4km_row" "$Q4_MODEL" 1 \
    "${ROW_BASE[@]}" -c 4096 -np 1 \
    "$SYCL_GRAPH" \
    "note:Q4_K_M row-split baseline"
fi

if should_run "3.3"; then
  run_test "3.3_q4km_row_np4" "$Q4_MODEL" 4 \
    "${ROW_BASE[@]}" -c 8192 -np 4 \
    "$SYCL_GRAPH" \
    "note:Q4_K_M row-split np=4, parallel"
fi

# ═══════════════════════════════════════════════════════════════════════════
# ROUND 4: Batch/Ubatch Tuning
# These affect how the server processes multiple tokens per iteration.
# ═══════════════════════════════════════════════════════════════════════════

if should_run "4.1"; then
  run_test "4.1_q4km_small_batch" "$Q4_MODEL" 4 \
    "${LAYER_BASE[@]}" -c 8192 -np 4 -b 512 -ub 128 \
    "$SYCL_GRAPH" \
    "note:Q4_K_M np=4 b=512 ub=128, small batches"
fi

if should_run "4.2"; then
  run_test "4.2_q4km_large_batch" "$Q4_MODEL" 4 \
    "${LAYER_BASE[@]}" -c 8192 -np 4 -b 4096 -ub 1024 \
    "$SYCL_GRAPH" \
    "note:Q4_K_M np=4 b=4096 ub=1024, large batches"
fi

if should_run "4.3"; then
  run_test "4.3_q4km_balanced" "$Q4_MODEL" 4 \
    "${LAYER_BASE[@]}" -c 8192 -np 4 -b 2048 -ub 256 \
    "$SYCL_GRAPH" \
    "note:Q4_K_M np=4 b=2048 ub=256, balanced"
fi

# ═══════════════════════════════════════════════════════════════════════════
# ROUND 5: SYCL Env Tuning
# Test graph capture and command list modes.
# ═══════════════════════════════════════════════════════════════════════════

# Pick the best config from rounds 2-4 for env testing.
# Default to Q4_K_M np=4 (most likely winner).

if should_run "5.1"; then
  run_test "5.1_graph_enabled" "$Q4_MODEL" 4 \
    "${LAYER_BASE[@]}" -c 8192 -np 4 -b 2048 -ub 512 \
    "env:GGML_SYCL_DISABLE_GRAPH=0" \
    "note:SYCL graph enabled (DISABLE_GRAPH=0)"
fi

if should_run "5.2"; then
  run_test "5.2_cmdlist_immediate" "$Q4_MODEL" 4 \
    "${LAYER_BASE[@]}" -c 8192 -np 4 -b 2048 -ub 512 \
    "$SYCL_GRAPH" "$SYCL_CMDLIST1" \
    "note:immediate command lists ON"
fi

if should_run "5.3"; then
  run_test "5.3_cmdlist_batched" "$Q4_MODEL" 4 \
    "${LAYER_BASE[@]}" -c 8192 -np 4 -b 2048 -ub 512 \
    "$SYCL_GRAPH" "$SYCL_CMDLIST0" \
    "note:batched command lists (cmdlist=0)"
fi

if should_run "5.4"; then
  run_test "5.4_graph_disabled" "$Q4_MODEL" 4 \
    "${LAYER_BASE[@]}" -c 8192 -np 4 -b 2048 -ub 512 \
    "env:GGML_SYCL_DISABLE_GRAPH=1" \
    "note:SYCL graph disabled (DISABLE_GRAPH=1), control"
fi

# ═══════════════════════════════════════════════════════════════════════════
# ROUND 6: Multi-Record Batch Prompts
# Pack N records into one LLM call to amortize overhead.
# Uses winning server config (defaults to Q4_K_M np=4).
# ═══════════════════════════════════════════════════════════════════════════

if should_run "6.1"; then
  run_test "6.1_batch3" "$Q4_MODEL" 4 \
    "${LAYER_BASE[@]}" -c 8192 -np 4 -b 2048 -ub 512 \
    "$SYCL_GRAPH" \
    "tokens:2048" "rpc:3" \
    "note:3 records/call, 2048 max tokens"
fi

if should_run "6.2"; then
  run_test "6.2_batch5" "$Q4_MODEL" 4 \
    "${LAYER_BASE[@]}" -c 8192 -np 4 -b 2048 -ub 512 \
    "$SYCL_GRAPH" \
    "tokens:2048" "rpc:5" \
    "note:5 records/call, 2048 max tokens"
fi

if should_run "6.3"; then
  run_test "6.3_batch10" "$Q4_MODEL" 4 \
    "${LAYER_BASE[@]}" -c 16384 -np 4 -b 2048 -ub 512 \
    "$SYCL_GRAPH" \
    "tokens:3072" "rpc:10" \
    "note:10 records/call, 3072 max tokens"
fi

if should_run "6.4"; then
  run_test "6.4_batch20" "$Q4_MODEL" 4 \
    "${LAYER_BASE[@]}" -c 32768 -np 4 -b 2048 -ub 512 \
    "$SYCL_GRAPH" \
    "tokens:4096" "rpc:20" \
    "note:20 records/call, 4096 max tokens"
fi

# ═══════════════════════════════════════════════════════════════════════════
# ROUND 7: Multi-Instance (if single-instance can't reach 15-20 concurrent)
# Run separate model instances, one per GPU.
# ═══════════════════════════════════════════════════════════════════════════

if should_run "7.1"; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  TEST 7.1: Single-GPU fit test (Q4_K_M on 1 GPU)"
  echo "═══════════════════════════════════════════════════════════════"

  kill_server

  # Test if Q4_K_M fits on a single GPU (19GB model, 16GB GPU = NO)
  # This is expected to fail, but documents it
  log="$LOGDIR/7.1_single_gpu.log"
  (
    export ZE_AFFINITY_MASK=0
    "$SERVER" \
      --model "$Q4_MODEL" \
      --host "$HOST" --port "$PORT" \
      -ngl 999 -c 2048 -fa on -np 1 \
      2>&1
  ) > "$log" 2>&1 &
  srv_pid=$!
  echo "  PID: $srv_pid (single GPU, ZE_AFFINITY_MASK=0)"

  # Give it 60s to try
  deadline=$((SECONDS + 60))
  started=false
  while ((SECONDS < deadline)); do
    if ! kill -0 "$srv_pid" 2>/dev/null; then
      echo "  EXPECTED: Server died (Q4_K_M 32B doesn't fit in 16GB)"
      break
    fi
    if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      echo "  SURPRISE: Server started! Q4_K_M fits on single GPU"
      started=true
      break
    fi
    sleep 3
  done

  if $started; then
    fire_requests 1 "$(basename "$Q4_MODEL" .gguf)" "$MAX_TOKENS" 1
    echo "  Result: ${_FIRE_TOTAL_TPS} t/s"
    echo -e "7.1_single_gpu\t1\t${_FIRE_COMPLETED}/1\t${_FIRE_TOKENS}\t${_FIRE_WALL}\t${_FIRE_TOTAL_TPS}\t-\tQ4_K_M on 1 GPU" >> "$RESULTS"
  else
    echo -e "7.1_single_gpu\t1\tSKIP\tSKIP\tSKIP\tSKIP\t-\tQ4_K_M doesn't fit 1 GPU" >> "$RESULTS"
  fi

  kill_server
  sleep 2
fi

if should_run "7.2" || should_run "7.3"; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  ROUND 7.2/7.3: Multi-Instance (smaller models, 1 per GPU)"
  echo "  NOTE: Requires Qwen3-8B or Qwen3-14B GGUF downloads."
  echo "  Skipping automatically if model files not found."
  echo "═══════════════════════════════════════════════════════════════"

  # Multi-instance requires launching 3 servers on different ports
  # and a modified fire_requests that distributes across them.
  # This is complex enough to be a separate script if needed.
  echo "  TODO: Implement multi-instance testing if single-instance results are insufficient"
  echo -e "7.2_multi_instance\t-\tSKIP\tSKIP\tSKIP\tSKIP\t-\tnot yet implemented" >> "$RESULTS"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Stop monitor & print summary
# ═══════════════════════════════════════════════════════════════════════════

stop_monitor

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  RESULTS SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [[ -f "$RESULTS" ]]; then
  column -t -s $'\t' < "$RESULTS"
  echo ""

  # Find best total_tps (col 6, excluding header/SKIP/ERR)
  best_line=$(tail -n +2 "$RESULTS" | grep -v 'SKIP\|ERR' | \
    sort -t$'\t' -k6 -rn | head -1 || true)
  if [[ -n "$best_line" ]]; then
    best_test=$(echo "$best_line" | cut -f1)
    best_concurrent=$(echo "$best_line" | cut -f2)
    best_tps=$(echo "$best_line" | cut -f6)
    echo "Best throughput: $best_test @ ${best_tps} total t/s (${best_concurrent} concurrent)"
  fi
fi

echo ""
echo "Full results: $RESULTS"
echo "Server logs:  $LOGDIR/"
echo "GPU log:      $GPU_LOG"
echo "Finished:     $(date -Iseconds)"
