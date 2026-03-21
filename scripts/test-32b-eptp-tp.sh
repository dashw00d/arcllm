#!/bin/bash
# Test Qwen3-32B dense on eptp build with tensor-split (TP mode)
# Compares against known baselines:
#   - stable layer-split np=16 FUSED_MMQ: 21.7 t/s
#   - stable layer-split np=16 no FUSED_MMQ: 17.7 t/s
#
# Usage: bash scripts/test-32b-eptp-tp.sh

set -euo pipefail
cd /home/ryan/llm-stack
source env.sglang-xpu.sh

MODEL="models/Qwen/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf"
SERVER="./llama.cpp-eptp/build-sycl/bin/llama-server"
PORT=18404

echo "=== Qwen3-32B Dense on EPTP Tensor-Split ==="
echo ""

pkill -x llama-server 2>/dev/null; sleep 2

# np=1 tensor-split (pure TP, no concurrent slots)
echo "[1] np=1, tensor-split, c=512"
GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
  $SERVER -m $MODEL --split-mode tensor -ngl 99 -np 1 -c 512 --port $PORT --no-warmup \
  > /tmp/32b-tp-np1.log 2>&1 &

for i in $(seq 1 90); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; sleep 3; done

# Warmup
curl -s --max-time 60 -X POST http://127.0.0.1:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"Hi"}],"max_tokens":10,"temperature":0}' > /dev/null 2>&1

# Actual test
curl -s --max-time 120 -X POST http://127.0.0.1:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"Explain quantum entanglement in 3 sentences."}],"max_tokens":100,"temperature":0}' 2>/dev/null | python3 -c "
import sys,json; r=json.load(sys.stdin); m=r['choices'][0]['message']; t=r['timings']
print(f'  Gen: {t[\"predicted_per_second\"]:.1f} t/s ({t[\"predicted_n\"]} tokens)')
print(f'  PP:  {t.get(\"prompt_per_second\",0):.1f} t/s')
content = m.get('content','') or m.get('reasoning_content','')[:200]
print(f'  Output: {content[:150]}')
"

pkill -x llama-server; sleep 2

# np=4 tensor-split
echo ""
echo "[2] np=4, tensor-split, c=2048"
GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
  $SERVER -m $MODEL --split-mode tensor -ngl 99 -np 4 -c 2048 --port $PORT --no-warmup \
  > /tmp/32b-tp-np4.log 2>&1 &

for i in $(seq 1 90); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; sleep 3; done

# Fire 4 concurrent
rm -f /tmp/32b_tp_*.json
for i in $(seq 1 4); do
  curl -s --max-time 120 -X POST http://127.0.0.1:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"What is $i * $i?\"}],\"max_tokens\":50,\"temperature\":0}" \
    -o /tmp/32b_tp_${i}.json 2>/dev/null &
done
wait

python3 -c "
import json, glob
results = []
for f in sorted(glob.glob('/tmp/32b_tp_*.json')):
    try:
        r = json.load(open(f))
        t = r['timings']
        results.append(t['predicted_per_second'])
    except: pass
if results:
    print(f'  Per-slot avg: {sum(results)/len(results):.1f} t/s')
    print(f'  Aggregate: {sum(results):.1f} t/s')
    print(f'  Slots: {len(results)}/4')
"

pkill -x llama-server; sleep 2

echo ""
echo "=== Baselines (stable build, layer-split) ==="
echo "  np=16 FUSED_MMQ=1: 21.7 t/s aggregate"
echo "  np=16 FUSED_MMQ=0: 17.7 t/s aggregate"
echo ""
echo "Done. Logs in /tmp/32b-tp-np*.log"
