#!/usr/bin/env bash
# Benchmark: system prompt size vs response latency
# Measures time-to-first-token and total time across different system prompt sizes

set -euo pipefail

PORT=11435
HOST="127.0.0.1"
RESULTS="/tmp/prompt_size_results.txt"

echo -e "test\tsys_tokens\tuser_tokens\ttotal_tokens\tprompt_ms\tgen_ms\ttotal_ms\tprompt_tps\tgen_tps" > "$RESULTS"

run_test() {
  local name="$1"
  local sys_prompt="$2"
  local user_msg="${3:-hello}"
  local max_tokens="${4:-20}"

  local tmpfile
  tmpfile=$(mktemp /tmp/prompt_bench_XXXXXX.json)

  # Build JSON payload
  local payload
  if [ -z "$sys_prompt" ]; then
    payload=$(python3 -c "
import json
print(json.dumps({
  'model': 'nemotron-3-super-120b',
  'messages': [{'role': 'user', 'content': '$user_msg'}],
  'max_tokens': $max_tokens,
  'temperature': 0
}))
")
  else
    payload=$(python3 -c "
import json, sys
sys_prompt = sys.stdin.read()
print(json.dumps({
  'model': 'nemotron-3-super-120b',
  'messages': [
    {'role': 'system', 'content': sys_prompt},
    {'role': 'user', 'content': '$user_msg'}
  ],
  'max_tokens': $max_tokens,
  'temperature': 0
}))
" <<< "$sys_prompt")
  fi

  curl -s --max-time 300 -o "$tmpfile" \
    "http://$HOST:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$payload" 2>/dev/null

  python3 -c "
import json
with open('$tmpfile') as f:
    d = json.load(f)
t = d.get('timings', {})
u = d.get('usage', {})
prompt_ms = t.get('prompt_ms', 0)
gen_ms = t.get('predicted_ms', 0)
prompt_n = t.get('prompt_n', u.get('prompt_tokens', 0))
gen_n = t.get('predicted_n', u.get('completion_tokens', 0))
total_n = prompt_n + gen_n
prompt_tps = prompt_n / (prompt_ms/1000) if prompt_ms > 0 else 0
gen_tps = gen_n / (gen_ms/1000) if gen_ms > 0 else 0
total_ms = prompt_ms + gen_ms

print(f'  {\"$name\":.<40} {prompt_n:>5} prompt tok → {prompt_ms:>7.0f}ms prompt, {gen_ms:>6.0f}ms gen = {total_ms:>8.0f}ms total  ({prompt_tps:.1f} p/s, {gen_tps:.1f} g/s)')
print(f'$name\t{prompt_n}\t{gen_n}\t{total_n}\t{prompt_ms:.0f}\t{gen_ms:.0f}\t{total_ms:.0f}\t{prompt_tps:.1f}\t{gen_tps:.1f}', file=open('$RESULTS', 'a'))
" 2>/dev/null

  rm -f "$tmpfile"
}

echo ""
echo "System Prompt Size vs Latency Benchmark"
echo "========================================"
echo ""

# Generate filler text of various sizes
make_filler() {
  python3 -c "
words = 'The quick brown fox jumps over the lazy dog near the river bank where flowers grow in spring and birds sing their morning songs while the sun rises over the distant mountains casting golden light across the peaceful valley below '.split()
import itertools
out = []
for w in itertools.cycle(words):
    out.append(w)
    if len(out) >= $1:
        break
print(' '.join(out))
"
}

# Test 1: No system prompt
echo "--- No system prompt ---"
run_test "no_system" "" "hello" 20

# Test 2: Tiny system prompt (~20 tokens)
echo "--- Tiny (~20 tokens) ---"
run_test "tiny_20tok" "You are a helpful assistant. Be concise." "hello" 20

# Test 3: Small system prompt (~100 tokens)
echo "--- Small (~100 tokens) ---"
SMALL=$(make_filler 80)
run_test "small_100tok" "You are a helpful assistant. Here is context: $SMALL" "hello" 20

# Test 4: Medium system prompt (~500 tokens)
echo "--- Medium (~500 tokens) ---"
MEDIUM=$(make_filler 400)
run_test "medium_500tok" "You are a helpful assistant. Here is context: $MEDIUM" "hello" 20

# Test 5: Large system prompt (~2000 tokens)
echo "--- Large (~2000 tokens) ---"
LARGE=$(make_filler 1600)
run_test "large_2k" "You are a helpful assistant. Here is context: $LARGE" "hello" 20

# Test 6: Very large (~5000 tokens)
echo "--- Very Large (~5000 tokens) ---"
VLARGE=$(make_filler 4000)
run_test "vlarge_5k" "You are a helpful assistant. Here is context: $VLARGE" "hello" 20

# Test 7: Huge (~10000 tokens)
echo "--- Huge (~10000 tokens) ---"
HUGE=$(make_filler 8000)
run_test "huge_10k" "You are a helpful assistant. Here is context: $HUGE" "hello" 20

# Test 8: Massive (~20000 tokens) — OpenCode Commander size
echo "--- Massive (~20000 tokens) — OpenCode Commander size ---"
MASSIVE=$(make_filler 16000)
run_test "massive_20k" "You are a helpful assistant. Here is context: $MASSIVE" "hello" 20

echo ""
echo "========================================"
echo "Results:"
echo ""
column -t -s $'\t' < "$RESULTS"
echo ""
echo "Full results: $RESULTS"
