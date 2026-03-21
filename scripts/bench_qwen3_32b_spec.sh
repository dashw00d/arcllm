#!/usr/bin/env bash
# Benchmark Qwen3-32B Q4_K_M: speculative decoding + row-split
# Usage: ./scripts/bench_qwen3_32b_spec.sh [test|bench|row|all]
#
# Modes:
#   test   — Quick spec decoding test via arcllm-proxy (default)
#   bench  — Build & run llama-bench (layer-split baseline + row-split)
#   row    — Row-split comparison via llama-server directly
#   all    — Run everything
#
# Requires: 3x Arc A770, conda env with SYCL

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Source env if not already set up
if [[ -z "${ZE_AFFINITY_MASK:-}" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT/env.sglang-xpu.sh"
fi

MODEL="$ROOT/models/Qwen/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf"
DRAFT="$ROOT/models/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
BENCH="$ROOT/llama.cpp/build/bin/llama-bench"
SERVER="$ROOT/llama.cpp/build/bin/llama-server"
LOGFILE="/tmp/arcllm-bench.log"

export GGML_SYCL_DISABLE_GRAPH=0
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0

MODE="${1:-test}"

echo "=== Qwen3-32B Q4_K_M Optimization Benchmark ==="
echo "Date: $(date)"
echo "Mode: $MODE"
echo ""

# ── Helper: send a test prompt and extract t/s ────────────────────────────
send_test_prompt() {
    local port="${1:-11435}"
    local model="${2:-qwen3-32b-fast}"
    local prompt="${3:-Write a Python function to compute fibonacci numbers recursively with memoization. Include type hints and a docstring.}"

    echo "  Sending test prompt to $model on port $port..."
    local start end resp
    start=$(date +%s%N)
    resp=$(curl -sf "http://127.0.0.1:$port/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":512,\"stream\":false}" \
        2>/dev/null) || { echo "  ERROR: Request failed"; return 1; }
    end=$(date +%s%N)

    local elapsed_ms=$(( (end - start) / 1000000 ))
    local tokens
    tokens=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "?")
    echo "  Completed: $tokens tokens in ${elapsed_ms}ms"
    if [[ "$tokens" != "?" && "$elapsed_ms" -gt 0 ]]; then
        local tps
        tps=$(python3 -c "print(f'{$tokens / ($elapsed_ms / 1000):.1f}')")
        echo "  Throughput: ~${tps} t/s (wall clock)"
    fi
}

# ── Test: Speculative decoding via arcllm-proxy ───────────────────────────
do_test() {
    echo "=== Step 1: Speculative Decoding Test ==="
    echo ""

    # Check if proxy is running
    if ! curl -sf http://127.0.0.1:11435/health >/dev/null 2>&1; then
        echo "arcllm-proxy not running. Start it first:"
        echo "  ./scripts/arcllm-server.sh start"
        echo ""
        return 1
    fi

    echo "--- Test: qwen3-32b-fast (draft + ngram-mod) ---"
    send_test_prompt 11435 "qwen3-32b-fast"
    echo ""

    echo "Check speculative stats in server log:"
    echo "  grep -iE 'draft|accept|spec|ngram' /tmp/arcllm-server.log | tail -20"
    echo ""

    echo "--- Tuning Guide ---"
    echo "Edit scripts/arcllm-proxy.py qwen3-32b-fast entry:"
    echo "  --draft-max N    (try 4, 8, 12, 16)"
    echo "  --draft-p-min P  (try 0.5, 0.75, 0.8, 0.9)"
    echo "  --draft-min N    (try 1, 2, 4)"
    echo ""
}

# ── Bench: llama-bench per-op timing ──────────────────────────────────────
do_bench() {
    echo "=== Step 4: llama-bench Diagnostics ==="
    echo ""

    # Build if needed
    if [ ! -f "$BENCH" ]; then
        echo "Building llama-bench..."
        cd "$ROOT/llama.cpp/build"
        cmake --build . --target llama-bench -j"$(nproc)" 2>&1 | tail -5
        cd "$ROOT"
        echo ""
    fi

    if [ ! -f "$BENCH" ]; then
        echo "ERROR: llama-bench build failed"
        return 1
    fi

    echo "--- Layer-split baseline (pp128, tg128, 3 runs) ---"
    "$BENCH" \
        -m "$MODEL" \
        -ngl 999 --split-mode layer --tensor-split 1,1,1 \
        -p 128 -n 128 -r 3 \
        -fa 1
    echo ""

    echo "--- Row-split comparison (pp128, tg128, 3 runs) ---"
    "$BENCH" \
        -m "$MODEL" \
        -ngl 999 --split-mode row --tensor-split 1,1,1 \
        -p 128 -n 128 -r 3 \
        -fa 1
    echo ""
}

# ── Row: Direct row-split server test ─────────────────────────────────────
do_row() {
    echo "=== Step 3: Row-Split Test ==="
    echo ""

    local PORT=18500

    # Kill any stale test server
    pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
    sleep 1

    echo "--- Starting row-split server on port $PORT ---"
    "$SERVER" \
        -m "$MODEL" \
        --split-mode row -ngl 999 --tensor-split 1,1,1 \
        -c 8192 -fa on -np 1 -ctk q8_0 -ctv q8_0 \
        --host 127.0.0.1 --port "$PORT" \
        > "$LOGFILE" 2>&1 &
    local srv_pid=$!
    echo "  PID: $srv_pid"

    # Wait for health
    echo "  Waiting for server..."
    local deadline=$((SECONDS + 300))
    while [ $SECONDS -lt $deadline ]; do
        if ! kill -0 "$srv_pid" 2>/dev/null; then
            echo "  ERROR: Server died during startup"
            tail -20 "$LOGFILE"
            return 1
        fi
        if curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | python3 -c "import sys,json; assert json.load(sys.stdin)['status']=='ok'" 2>/dev/null; then
            echo "  Server ready!"
            break
        fi
        sleep 2
    done

    echo ""
    echo "--- Row-split generation test ---"
    send_test_prompt "$PORT"
    echo ""

    echo "--- Row-split + DISABLE_GRAPH=1 test ---"
    kill "$srv_pid" 2>/dev/null; wait "$srv_pid" 2>/dev/null || true
    sleep 2

    export GGML_SYCL_DISABLE_GRAPH=1
    "$SERVER" \
        -m "$MODEL" \
        --split-mode row -ngl 999 --tensor-split 1,1,1 \
        -c 8192 -fa on -np 1 -ctk q8_0 -ctv q8_0 \
        --host 127.0.0.1 --port "$PORT" \
        > "$LOGFILE" 2>&1 &
    srv_pid=$!
    echo "  PID: $srv_pid (DISABLE_GRAPH=1)"

    deadline=$((SECONDS + 300))
    while [ $SECONDS -lt $deadline ]; do
        if ! kill -0 "$srv_pid" 2>/dev/null; then
            echo "  ERROR: Server died"; tail -20 "$LOGFILE"; return 1
        fi
        if curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | python3 -c "import sys,json; assert json.load(sys.stdin)['status']=='ok'" 2>/dev/null; then
            echo "  Server ready!"
            break
        fi
        sleep 2
    done

    send_test_prompt "$PORT"
    echo ""

    # Cleanup
    kill "$srv_pid" 2>/dev/null; wait "$srv_pid" 2>/dev/null || true
    export GGML_SYCL_DISABLE_GRAPH=0
    echo "Row-split test server stopped."
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────────────
case "$MODE" in
    test)  do_test ;;
    bench) do_bench ;;
    row)   do_row ;;
    all)   do_test; do_bench; do_row ;;
    *)
        echo "Usage: $0 {test|bench|row|all}"
        exit 1
        ;;
esac

echo "=== Done ==="
