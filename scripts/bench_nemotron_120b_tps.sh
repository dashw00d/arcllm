#!/usr/bin/env bash
# Nemotron 3 Super 120B TPS Optimization — find the best CLI flags & env vars
# Hardware: 3x Intel Arc A770 (48GB VRAM), 64GB RAM
# Model: Q2_K (~55GB), MoE 512 experts / 12B active
#
# Usage:  ./scripts/bench_nemotron_120b_tps.sh [test_number...]
#   No args  → run all tests 1-9
#   1 3 5    → run only tests 1, 3, 5

set -euo pipefail

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
SERVER="$ROOT/llama.cpp/build/bin/llama-server"
MODEL="$ROOT/models/NVIDIA/Nemotron-3-Super-120B-A12B-GGUF/nvidia_Nemotron-3-Super-120B-A12B-Q2_K/nvidia_Nemotron-3-Super-120B-A12B-Q2_K-00001-of-00002.gguf"
RESULTS="/tmp/tps_results.txt"
LOGDIR="/tmp/nemotron_bench_logs"

PORT=8400
HOST="127.0.0.1"
HEALTH_TIMEOUT=600   # 10 min — loading 55GB model over PCIe takes a while
REQUEST_TIMEOUT=600  # 10 min — generation at ~2 t/s × 200 tokens ≈ 100s, plus prompt
PROMPT="Explain quantum computing in 3 sentences."
MAX_TOKENS=200

# ── environment ────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$ROOT/env.sglang-xpu.sh"
# Conda sets HOST to the compiler triple — override it back
HOST="127.0.0.1"

mkdir -p "$LOGDIR"

# ── helpers ────────────────────────────────────────────────────────────────

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
    # Check if server process died
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

run_benchmark() {
  # Use the /completion endpoint which returns timings directly
  local response tmpfile
  tmpfile=$(mktemp /tmp/nemotron_bench_XXXXXX.json)
  # Note: stderr goes to /dev/null to avoid conda libcurl warnings polluting response
  if ! curl -sf --max-time "$REQUEST_TIMEOUT" \
    -o "$tmpfile" \
    "http://$HOST:$PORT/completion" \
    -H "Content-Type: application/json" \
    -d "$(cat <<'PAYLOAD'
{
  "prompt": "Explain quantum computing in 3 sentences.",
  "n_predict": 200,
  "temperature": 0.0,
  "cache_prompt": false
}
PAYLOAD
)" 2>/dev/null; then
    echo "  ERROR: curl failed"
    cat "$tmpfile" 2>/dev/null | tail -5
    rm -f "$tmpfile"
    return 1
  fi
  # Extract all timings in one python call — read from temp file to avoid shell quoting issues
  local prompt_tps gen_tps content tokens_predicted tokens_evaluated
  local parsed
  parsed=$(python3 -c "
import json
with open('$tmpfile') as f:
    d = json.load(f)
t = d.get('timings', {})
print(f\"{t.get('prompt_per_second', 0):.2f}\")
print(f\"{t.get('predicted_per_second', 0):.2f}\")
print(t.get('prompt_n', '?'))
print(t.get('predicted_n', '?'))
c = d.get('content', '')
print(c[:120].replace(chr(10), ' '))
" 2>/dev/null) || parsed=""
  rm -f "$tmpfile"

  if [[ -n "$parsed" ]]; then
    prompt_tps=$(sed -n '1p' <<< "$parsed")
    gen_tps=$(sed -n '2p' <<< "$parsed")
    tokens_evaluated=$(sed -n '3p' <<< "$parsed")
    tokens_predicted=$(sed -n '4p' <<< "$parsed")
    content=$(sed -n '5p' <<< "$parsed")
  else
    prompt_tps="ERR"; gen_tps="ERR"; tokens_evaluated="?"; tokens_predicted="?"; content="(parse error)"
  fi

  echo "  Prompt: ${prompt_tps} t/s (${tokens_evaluated} tokens)"
  echo "  Gen:    ${gen_tps} t/s (${tokens_predicted} tokens)"
  echo "  Output: ${content}"

  # Validate output is non-empty and sensible
  if [[ "$content" == "(parse error)" || -z "$content" ]]; then
    echo "  WARNING: empty or unparseable output"
  fi

  # Return values via globals (bash limitation)
  _PROMPT_TPS="$prompt_tps"
  _GEN_TPS="$gen_tps"
  _CONTENT="$content"
}

run_test() {
  local test_name="$1"
  shift
  local env_vars=()
  local server_args=()
  local notes=""

  # Parse arguments: env:KEY=VAL for env vars, note:TEXT for notes, rest are server args
  while [[ $# -gt 0 ]]; do
    case "$1" in
      env:*)  env_vars+=("${1#env:}") ;;
      note:*) notes="${1#note:}" ;;
      *)      server_args+=("$1") ;;
    esac
    shift
  done

  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  TEST: $test_name"
  echo "  Args: ${server_args[*]:-<default>}"
  echo "  Env:  ${env_vars[*]:-<default>}"
  echo "═══════════════════════════════════════════════════════════════"

  kill_server

  # Build environment for this test
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
      --model "$MODEL" \
      --host "$HOST" \
      --port "$PORT" \
      "${server_args[@]}" \
      2>&1
  ) > "$log" 2>&1 &
  local server_pid=$!
  echo "  Server PID: $server_pid, log: $log"

  if ! wait_for_health; then
    echo "  SKIPPED: server failed to start"
    echo -e "${test_name}\tSKIP\tSKIP\tserver failed to start" >> "$RESULTS"
    kill_server
    return 0
  fi

  # Run benchmark
  echo "  Running benchmark..."
  if run_benchmark; then
    echo -e "${test_name}\t${_PROMPT_TPS}\t${_GEN_TPS}\t${notes}" >> "$RESULTS"
  else
    echo -e "${test_name}\tERR\tERR\tbenchmark failed" >> "$RESULTS"
  fi

  kill_server
  sleep 3  # let GPU memory fully release
}

# ── results header ─────────────────────────────────────────────────────────
if [[ ! -f "$RESULTS" ]]; then
  echo -e "test_name\tprompt_tps\tgen_tps\tnotes" > "$RESULTS"
fi

echo ""
echo "Nemotron 3 Super 120B — TPS Optimization Benchmark"
echo "Model: $MODEL"
echo "Results: $RESULTS"
echo "Logs: $LOGDIR"
echo "Started: $(date -Iseconds)"

# ── which tests to run ─────────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
  TESTS=("$@")
else
  TESTS=(1 2 3 4 5 6 7 8)
fi

# ── common flags ───────────────────────────────────────────────────────────
# Base config: layer-split, all layers on GPU, even split, expert offload
BASE_ARGS=(
  --split-mode layer
  -ngl 999
  --tensor-split "1,1,1"
  --override-tensor 'blk\.\d+\.ffn_(up|down)_exps\.weight=CPU'
  -c 2048
)

for t in "${TESTS[@]}"; do
  case "$t" in

  1)
    run_test "01_baseline" \
      "${BASE_ARGS[@]}" \
      "note:baseline layer-split, expert offload up+down"
    ;;

  2)
    run_test "02_sycl_graph" \
      "${BASE_ARGS[@]}" \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:enable SYCL graph recording for pipelined dispatch"
    ;;

  3)
    run_test "03_ctx512" \
      --split-mode layer \
      -ngl 999 \
      --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_(up|down)_exps\.weight=CPU' \
      -c 512 \
      "note:smaller context (512 vs 2048)"
    ;;

  4)
    run_test "04_uneven_split" \
      --split-mode layer \
      -ngl 999 \
      --tensor-split "1.5,1,1" \
      --override-tensor 'blk\.\d+\.ffn_(up|down)_exps\.weight=CPU' \
      -c 2048 \
      "note:GPU0 gets more layers (1.5:1:1)"
    ;;

  5)
    run_test "05_offload_down_only" \
      --split-mode layer \
      -ngl 999 \
      --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 \
      "note:only offload ffn_down (keep ffn_up on GPU)"
    ;;

  6)
    run_test "06_offload_up_down_gate" \
      --split-mode layer \
      -ngl 999 \
      --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_(up|down|gate)_exps\.weight=CPU' \
      -c 2048 \
      "note:offload up+down+gate experts to CPU"
    ;;

  7)
    run_test "07_ngl80" \
      --split-mode layer \
      -ngl 80 \
      --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_(up|down)_exps\.weight=CPU' \
      -c 2048 \
      "note:80 GPU layers (8 on CPU)"
    ;;

  8)
    run_test "08_batch_tuning" \
      "${BASE_ARGS[@]}" \
      -b 512 -ub 256 \
      "note:explicit batch 512, ubatch 256"
    ;;

  9)
    # Test 9: combined best — user runs after reviewing results from 1-8
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  TEST 9: Combined Best Settings"
    echo "  Review results so far and edit this section with winners."
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    # Parse current results to find best gen_tps
    if [[ -f "$RESULTS" ]]; then
      echo "Current results:"
      echo ""
      column -t -s $'\t' < "$RESULTS"
      echo ""

      best_test=$(tail -n +2 "$RESULTS" | grep -v 'ERR\|SKIP' | \
        sort -t$'\t' -k3 -rn | head -1 | cut -f1)
      best_gen=$(tail -n +2 "$RESULTS" | grep -v 'ERR\|SKIP' | \
        sort -t$'\t' -k3 -rn | head -1 | cut -f3)
      echo "Best so far: $best_test @ ${best_gen} t/s gen"
    fi

    echo ""
    echo "To run a combined test, edit the script or run manually:"
    echo "  $0 9"
    # Combined best: SYCL graph (test 2, +56% gen) + down-only offload (test 5, +7% gen)
    run_test "09_combined_best" \
      --split-mode layer \
      -ngl 999 \
      --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:SYCL graph + down-only offload"
    ;;

  # ── Round 2: build on combined best (SYCL graph + down-only offload) ──

  10)
    # Flash attention — reduces KV cache memory, faster attention kernels
    run_test "10_flash_attn" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:combined + flash attention"
    ;;

  11)
    # Quantized KV cache — q8_0 frees VRAM for more compute/weights
    run_test "11_kv_q8" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 -ctk q8_0 -ctv q8_0 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:combined + KV cache q8_0"
    ;;

  12)
    # Aggressive KV quantization — q4_0 for max VRAM savings
    run_test "12_kv_q4" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 -ctk q4_0 -ctv q4_0 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:combined + KV cache q4_0"
    ;;

  13)
    # Flash attn + q4 KV + NO expert offload — try to fit everything on GPU
    # Model is ~55GB, VRAM is 48GB — will likely OOM but worth trying
    run_test "13_no_offload_fa_q4" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      -c 2048 -fa on -ctk q4_0 -ctv q4_0 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:NO expert offload + flash attn + q4 KV (may OOM)"
    ;;

  14)
    # Enable batched matrix multiply (disabled by default for SYCL)
    run_test "14_batched_mm" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "env:GGML_SYCL_DISABLE_BATCHED_MM=0" \
      "note:combined + batched MM enabled"
    ;;

  15)
    # mlock — pin CPU-side weights in RAM, prevent any swap/compress
    run_test "15_mlock" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 --mlock \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:combined + mlock (pin CPU weights in RAM)"
    ;;

  16)
    # Thread tuning — i9-7900X has 10c/20t, default uses 10 threads
    # More threads may speed up CPU-side expert compute
    run_test "16_threads_20" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 -t 20 -tb 20 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:combined + 20 threads (all HT cores)"
    ;;

  17)
    # Flash attn + q4 KV — keep down-only offload but maximize freed VRAM
    run_test "17_fa_q4_down_only" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 4096 -fa on -ctk q4_0 -ctv q4_0 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:combined + flash attn + q4 KV + 4k context"
    ;;

  18)
    # Combined round 2 best — placeholder, fill in after reviewing 10-17
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  TEST 18: Combined Round 2 Best"
    echo "  Review results from 10-17 and customize."
    echo "═══════════════════════════════════════════════════════════════"
    if [[ -f "$RESULTS" ]]; then
      echo ""; column -t -s $'\t' < "$RESULTS"; echo ""
    fi
    ;;

  # ── Round 3: squeeze more from 21GB free VRAM + CPU-side tuning ──
  # Base: test 10 winner (SYCL graph + down-only offload + flash attn)

  19)
    # Single slot — saves ~500 MiB recurrent state (4 slots → 1)
    run_test "19_single_slot" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:combined best + single slot (-np 1)"
    ;;

  20)
    # Partial offload — only offload layers 0-53 ffn_down, keep 54-87 on GPU
    # ~34 layers × ~580MB ≈ 20GB fits in 21GB free VRAM
    run_test "20_partial_offload_54" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.([0-4]\d|5[0-3])\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:offload ffn_down layers 0-53 only (keep 54-87 on GPU)"
    ;;

  21)
    # More aggressive partial — offload only layers 0-43 (keep 44-87 on GPU)
    # ~44 layers × ~580MB ≈ 25GB — may OOM
    run_test "21_partial_offload_44" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.([0-3]\d|4[0-3])\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:offload ffn_down layers 0-43 only (keep 44-87 on GPU, may OOM)"
    ;;

  22)
    # no-mmap — preload all weights to RAM, no demand paging from NVMe
    run_test "22_no_mmap" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on --no-mmap \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:combined best + no-mmap (preload to RAM)"
    ;;

  23)
    # Fewer threads — 20 was worse, try 4 (less contention with GPU dispatch)
    run_test "23_threads_4" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on -t 4 -tb 4 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:combined best + 4 threads (reduce HT contention)"
    ;;

  24)
    # Single slot + partial offload — combine VRAM savings
    run_test "24_np1_partial_54" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.([0-4]\d|5[0-3])\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:single slot + partial offload (layers 0-53)"
    ;;

  25)
    # Single slot + partial offload + no-mmap — all VRAM + RAM optimizations
    run_test "25_np1_partial_nommap" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.([0-4]\d|5[0-3])\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on -np 1 --no-mmap \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:single slot + partial offload + no-mmap"
    ;;

  # ── Round 4: conservative partial offload (OOM'd at 34 layers, try fewer) ──

  26)
    # Keep 20 layers on GPU (offload 0-67), ~11.6 GB — should fit
    run_test "26_partial_keep20" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.([0-5]\d|6[0-7])\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:np1 + offload ffn_down 0-67 (keep 20 layers on GPU)"
    ;;

  27)
    # Keep 10 layers on GPU (offload 0-77), ~5.8 GB — safe fit
    run_test "27_partial_keep10" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.([0-6]\d|7[0-7])\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:np1 + offload ffn_down 0-77 (keep 10 layers on GPU)"
    ;;

  28)
    # Keep 25 layers on GPU (offload 0-62) + np1 — push the boundary
    run_test "28_partial_keep25" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.([0-5]\d|6[0-2])\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:np1 + offload ffn_down 0-62 (keep 25 layers on GPU)"
    ;;

  29)
    # Keep 15 layers on GPU (offload 0-72)
    run_test "29_partial_keep15" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.([0-6]\d|7[0-2])\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:np1 + offload ffn_down 0-72 (keep 15 layers on GPU)"
    ;;

  # ── Round 5: smart offload — swap WHICH tensor goes to CPU ──

  30)
    # Offload ffn_up instead of ffn_down — 17 GB on CPU vs 30 GB
    # GPU: 33.8 GB model + 3 GB overhead = 36.8 GB (fits in 48 GB)
    run_test "30_offload_up_only" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_up_exps\.weight=CPU' \
      -c 2048 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:offload ffn_UP instead (17GB CPU vs 30GB), stuffs GPU to 37GB"
    ;;

  31)
    # Same but with 4 slots (default) — more recurrent state but more throughput potential
    run_test "31_offload_up_np4" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_up_exps\.weight=CPU' \
      -c 2048 -fa on \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:offload ffn_UP, 4 slots (default)"
    ;;

  32)
    # Offload ffn_up + partial ffn_down — squeeze in as much as possible
    # Keep some ffn_down layers on GPU too, offload rest
    # GPU budget: 48 - 3 overhead - 4.3 non-expert - 17.2 ffn_up_on_cpu... wait
    # We offload ffn_up (17GB to CPU) and keep ALL ffn_down (30GB on GPU)
    # That's 34.8 GB model on GPU. We have ~11 GB headroom.
    # Can we keep some ffn_up layers on GPU too?
    # 40 layers of ffn_up, each ~441 MB. 11 GB / 0.441 = ~25 layers
    # Keep layers 0-24 ffn_up on GPU, offload 25-87 to CPU
    run_test "32_partial_up_offload" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.([2-7]\d|8[0-7])\.ffn_up_exps\.weight=CPU' \
      -c 2048 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:offload ffn_up layers 20-87 only (keep 0-19 on GPU)"
    ;;

  33)
    # Even more aggressive — offload only half the ffn_up layers
    # Keep 20 ffn_up layers on GPU = 20 × 441 MB = 8.8 GB extra
    # Total GPU: 33.8 + 8.8 = 42.6 GB + 3 GB overhead = 45.6 GB (tight!)
    run_test "33_half_up_offload" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.(1\d|[2-7]\d|8[0-7])\.ffn_up_exps\.weight=CPU' \
      -c 2048 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:offload ffn_up layers 10-87 (keep 1-9 on GPU, ~4GB extra)"
    ;;

  # ── Round 6: system tuning (CPU perf governor, ASPM off, SYCL env vars) ──

  34)
    # Re-baseline with CPU perf governor + C-states disabled + ASPM off
    run_test "34_sys_tuned_baseline" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "note:system tuned (perf governor + C-states off + ASPM off)"
    ;;

  35)
    # USM host pointer import — promotes malloc'd memory to pinned/USM
    run_test "35_usm_hostptr" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "env:SYCL_USM_HOSTPTR_IMPORT=1" \
      "note:system tuned + USM host pointer import (pinned memory)"
    ;;

  36)
    # Immediate command lists OFF — batched submission, may help many short kernels
    run_test "36_cmdlist_off" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0" \
      "note:system tuned + immediate cmdlists OFF"
    ;;

  37)
    # Immediate command lists ON
    run_test "37_cmdlist_on" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1" \
      "note:system tuned + immediate cmdlists ON"
    ;;

  38)
    # All SYCL env vars combined
    run_test "38_all_sycl_env" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "env:SYCL_USM_HOSTPTR_IMPORT=1" \
      "env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0" \
      "note:system tuned + all SYCL env vars"
    ;;

  39)
    # Combined: SYCL graph side-effects + cmdlists OFF (no USM hostptr - crashes)
    run_test "39_graph_cmdlist0" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 2048 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0" \
      "note:SYCL graph effects + cmdlists OFF (best combo)"
    ;;

  40)
    # Partial offload: offload layers 1-76, keep 79-87 on GPU
    # 5 MoE layers on GPU × 756 MB = 3.8 GB — safe fit
    run_test "40_partial5_rebased" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.([1-6]\d|7[0-6])\.ffn_down_exps\.weight=CPU' \
      -c 8192 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0" \
      "note:partial: 5 MoE layers ffn_down on GPU (3.8 GB)"
    ;;

  41)
    # Keep 16 MoE layers on GPU (offload 1-59, keep 61-87)
    # 16 MoE layers × 756 MB = 12 GB
    run_test "41_partial16_rebased" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.([1-5]\d)\.ffn_down_exps\.weight=CPU' \
      -c 8192 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0" \
      "note:partial: 16 MoE layers ffn_down on GPU (12 GB)"
    ;;

  42)
    # Keep 10 MoE layers on GPU (offload 1-67, keep 70-87)
    # 10 MoE layers × 756 MB = 7.6 GB
    run_test "42_partial10_rebased" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.([1-5]\d|6[0-7])\.ffn_down_exps\.weight=CPU' \
      -c 8192 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0" \
      "note:partial: 10 MoE layers ffn_down on GPU (7.6 GB)"
    ;;

  43)
    # Keep 13 MoE layers on GPU (offload 1-63, keep 65-87)
    # 13 MoE layers × 756 MB = 9.8 GB
    run_test "43_partial13_rebased" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.([1-5]\d|6[0-3])\.ffn_down_exps\.weight=CPU' \
      -c 8192 -fa on -np 1 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0" \
      "note:partial: 13 MoE layers ffn_down on GPU (9.8 GB)"
    ;;

  # ── Prompt eval optimization tests ──

  44)
    run_test "44_batch4096" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 8192 -fa on -np 1 -b 4096 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0" \
      "note:bigger batch -b 4096"
    ;;

  45)
    run_test "45_batch4096_ub2048" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.\d+\.ffn_down_exps\.weight=CPU' \
      -c 8192 -fa on -np 1 -b 4096 -ub 2048 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0" \
      "note:bigger batch+ubatch -b 4096 -ub 2048"
    ;;

  46)
    # Partial offload + bigger batch
    run_test "46_partial_batch" \
      --split-mode layer -ngl 999 --tensor-split "1,1,1" \
      --override-tensor 'blk\.([1-5]\d|6[0-7])\.ffn_down_exps\.weight=CPU' \
      -c 8192 -fa on -np 1 -b 4096 \
      "env:GGML_SYCL_DISABLE_GRAPH=0" \
      "env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0" \
      "note:partial offload + -b 4096"
    ;;

  *)
    echo "Unknown test: $t (valid: 1-46)"
    ;;
  esac
done

# ── summary ────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  RESULTS SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [[ -f "$RESULTS" ]]; then
  column -t -s $'\t' < "$RESULTS"
  echo ""

  # Find best gen_tps (excluding header, ERR, SKIP)
  best_line=$(tail -n +2 "$RESULTS" | grep -v 'ERR\|SKIP' | \
    sort -t$'\t' -k3 -rn | head -1 || true)
  if [[ -n "$best_line" ]]; then
    best_test=$(echo "$best_line" | cut -f1)
    best_prompt=$(echo "$best_line" | cut -f2)
    best_gen=$(echo "$best_line" | cut -f3)
    echo "Best generation:  $best_test @ ${best_gen} t/s"
    echo "Best prompt eval: check column 2 above"
  fi
fi

echo ""
echo "Full results: $RESULTS"
echo "Server logs:  $LOGDIR/"
echo "Finished: $(date -Iseconds)"
