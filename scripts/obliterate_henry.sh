#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════
#  OBLITERATE HENRY — Nuclear abliteration of Qwen3-32B
#
#  This downloads the base Qwen3-32B-Instruct, runs OBLITERATUS in
#  nuclear mode with custom prompts, then quantizes to Q8_0 GGUF.
#
#  Expected runtime: 2-4 hours on i9-7900X (CPU-only abliteration)
#  RAM required: ~64 GB (will use swap for overflow)
# ═══════════════════════════════════════════════════════════════════════

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sglang-xpu.sh"

MODEL_ID="Qwen/Qwen3-32B-Instruct"
OUTPUT_DIR="$ROOT/models/henry-obliterated"
CUSTOM_PROMPTS="$ROOT/scripts/henry_prompts.txt"

echo "═══════════════════════════════════════════════════════"
echo " Step 1: Nuclear abliteration"
echo " Model: $MODEL_ID"
echo " Method: nuclear (max force)"
echo " Custom prompts: $CUSTOM_PROMPTS"
echo "═══════════════════════════════════════════════════════"

# Check for custom prompts file
if [[ -f "$CUSTOM_PROMPTS" ]]; then
    echo "Found $(wc -l < "$CUSTOM_PROMPTS") custom prompts"
else
    echo "WARNING: No custom prompts at $CUSTOM_PROMPTS"
    echo "Create it with one prompt per line to add your own refusal targets."
fi

# Kill llama-server to free RAM
pkill -f llama-server 2>/dev/null && echo "Killed llama-server to free RAM" && sleep 3 || true

obliteratus obliterate "$MODEL_ID" \
    --method nuclear \
    --device cpu \
    --dtype float16 \
    --output-dir "$OUTPUT_DIR" \
    --large-model \
    --refinement-passes 3 \
    --n-directions 6

echo ""
echo "═══════════════════════════════════════════════════════"
echo " Step 2: Convert to GGUF Q8_0"
echo "═══════════════════════════════════════════════════════"

# Find the output safetensors
OBLITERATED_PATH="$OUTPUT_DIR"
if [[ -d "$OBLITERATED_PATH/obliterated" ]]; then
    OBLITERATED_PATH="$OUTPUT_DIR/obliterated"
fi

python3 "$ROOT/llama.cpp/convert_hf_to_gguf.py" \
    "$OBLITERATED_PATH" \
    --outfile "$OUTPUT_DIR/henry-32b-f16.gguf" \
    --outtype f16

"$ROOT/llama.cpp/build/bin/llama-quantize" \
    "$OUTPUT_DIR/henry-32b-f16.gguf" \
    "$OUTPUT_DIR/henry-32b-Q8_0.gguf" \
    Q8_0

echo ""
echo "═══════════════════════════════════════════════════════"
echo " DONE. Henry's new brain: $OUTPUT_DIR/henry-32b-Q8_0.gguf"
echo ""
echo " To deploy:"
echo "   Update arcllm-proxy.py model path to:"
echo "   $OUTPUT_DIR/henry-32b-Q8_0.gguf"
echo "═══════════════════════════════════════════════════════"
