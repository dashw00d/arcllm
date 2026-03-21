#!/bin/bash
# Test suite for expert padding — run various expert counts and configs
# Usage: bash scripts/test-expert-padding.sh

set -e
cd /home/ryan/llm-stack
source env.sglang-xpu.sh

MODEL_27B="models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf"
MODEL_2B="models/Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF/Qwen1.5-MoE-A2.7B-Chat.Q2_K.gguf"
SERVER="./llama.cpp-stable/build-sycl/bin/llama-server"
PAD_SCRIPT="scripts/pad-experts-gguf.py"
PORT=18404
PROMPT='{"model":"test","messages":[{"role":"user","content":"What is the capital of France? Reply in one sentence."}],"max_tokens":60,"temperature":0}'
RESULTS_FILE="/tmp/expert-padding-results.txt"

> "$RESULTS_FILE"

test_model() {
    local label="$1"
    local model_path="$2"
    local extra_args="$3"
    
    echo "=== TEST: $label ==="
    echo "=== TEST: $label ===" >> "$RESULTS_FILE"
    
    pkill -x llama-server 2>/dev/null; sleep 2
    
    GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
        $SERVER -m "$model_path" --split-mode layer -ngl 99 -np 1 -c 256 \
        --port $PORT --no-warmup -fa off $extra_args \
        > /tmp/pad-test-server.log 2>&1 &
    
    # Wait for ready
    local ready=0
    for i in $(seq 1 60); do
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && ready=1 && break
        sleep 3
    done
    
    if [ $ready -eq 0 ]; then
        echo "  FAILED TO START"
        echo "  FAILED TO START" >> "$RESULTS_FILE"
        tail -5 /tmp/pad-test-server.log
        tail -5 /tmp/pad-test-server.log >> "$RESULTS_FILE"
        return
    fi
    
    local result=$(curl -s --max-time 120 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$PROMPT" 2>/dev/null)
    
    local thinking=$(echo "$result" | python3 -c "import sys,json; m=json.load(sys.stdin)['choices'][0]['message']; print(m.get('reasoning_content','')[:200])" 2>/dev/null || echo "PARSE_ERROR")
    local content=$(echo "$result" | python3 -c "import sys,json; m=json.load(sys.stdin)['choices'][0]['message']; print(m.get('content','')[:200])" 2>/dev/null || echo "PARSE_ERROR")
    local tps=$(echo "$result" | python3 -c "import sys,json; t=json.load(sys.stdin).get('timings',{}); print(f\"{t.get('predicted_per_second',0):.1f}\")" 2>/dev/null || echo "?")
    
    echo "  THINKING: ${thinking:0:120}"
    echo "  CONTENT:  ${content:0:120}"
    echo "  SPEED:    ${tps} t/s"
    echo "  THINKING: ${thinking:0:120}" >> "$RESULTS_FILE"
    echo "  CONTENT:  ${content:0:120}" >> "$RESULTS_FILE"
    echo "  SPEED:    ${tps} t/s" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    pkill -x llama-server 2>/dev/null; sleep 1
}

pad_model() {
    local input="$1"
    local output="$2"
    local n_gpus="$3"
    
    rm -f "$output"
    python3 "$PAD_SCRIPT" "$input" "$output" --n-gpus "$n_gpus" 2>&1 | grep -E "Padding:|Done!"
}

echo "=========================================="
echo "Expert Padding Test Suite"
echo "=========================================="
echo ""

# --- 2.7B TESTS (fast, 60 experts) ---
echo "--- 2.7B MoE (60 experts, 4 active) ---"

echo "[1] 2.7B original (60 experts) — baseline"
test_model "2.7B-original-60exp" "$MODEL_2B"

echo "[2] 2.7B padded to 63 (÷3)"
pad_model "$MODEL_2B" "/tmp/2.7b-63exp.gguf" 3
test_model "2.7B-padded-63exp" "/tmp/2.7b-63exp.gguf"

echo "[3] 2.7B padded to 64 (power of 2)"
pad_model "$MODEL_2B" "/tmp/2.7b-64exp.gguf" 64
test_model "2.7B-padded-64exp" "/tmp/2.7b-64exp.gguf"

echo "[4] 2.7B padded to 61 (prime)"
pad_model "$MODEL_2B" "/tmp/2.7b-61exp.gguf" 61
test_model "2.7B-padded-61exp" "/tmp/2.7b-61exp.gguf"

# --- 2.7B CPU-ONLY TESTS ---
echo ""
echo "--- 2.7B MoE CPU-only (no SYCL) ---"

echo "[5] 2.7B original CPU-only"
test_model "2.7B-original-CPU" "$MODEL_2B" "-ngl 0"

echo "[6] 2.7B padded 63 CPU-only"
test_model "2.7B-padded-63-CPU" "/tmp/2.7b-63exp.gguf" "-ngl 0"

# --- 30B TESTS (slower) ---
echo ""
echo "--- 30B MoE (128 experts, 8 active) ---"

echo "[7] 30B original (128 experts) — baseline"
test_model "30B-original-128exp" "$MODEL_27B"

echo "[8] 30B padded to 129 (÷3)"
pad_model "$MODEL_27B" "/tmp/30b-129exp.gguf" 3
test_model "30B-padded-129exp" "/tmp/30b-129exp.gguf"

# Cleanup
echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
cat "$RESULTS_FILE"

rm -f /tmp/2.7b-63exp.gguf /tmp/2.7b-64exp.gguf /tmp/2.7b-61exp.gguf /tmp/30b-129exp.gguf
pkill -x llama-server 2>/dev/null
echo "Done. Full results in $RESULTS_FILE"
