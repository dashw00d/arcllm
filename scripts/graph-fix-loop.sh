#!/bin/bash
# Self-healing graph fix agent loop.
# Runs in tmux, restarts on context limit or crash.
# Each restart reads TaskList to find unfinished work.

LOG="/tmp/graph-fix-agent.log"
ITERATION=0

while true; do
    ITERATION=$((ITERATION + 1))
    echo "=== ITERATION $ITERATION === $(date)" | tee -a "$LOG"

    claude -p --dangerously-skip-permissions \
        "You are the 'fixer' agent for the graph-fix team. Your job is to fix SYCL graph recording for tensor parallelism.

## First: Check what's already done
Run TaskList to see current task status. Claim the first unblocked, unowned pending task. If a task is in_progress with no owner, claim it.

## Context
- Working directory: /home/ryan/llm-stack/llama.cpp-eptp/
- Build dir: /home/ryan/llm-stack/llama.cpp-eptp/build-sycl/
- Build: source /home/ryan/llm-stack/env.sglang-xpu.sh && cmake --build . --target llama-server -j\$(nproc)
- Read OPTIMIZATION_PLAN.md for the full plan
- Key file: ggml/src/ggml-sycl/ggml-sycl.cpp (graph_compute ~line 4490)

## What we proved
- Graph replay gives 8ms tokens vs 400ms without (50x speedup)
- Crash on request 2 is stale tensor data pointers
- MUL_MAT_ID ops block graph recording (host memcpy inside)

## Test model
- 0.6B: /home/ryan/llm-stack/models/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
- 30B MoE: /home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf
- Server: GGML_SYCL_DISABLE_GRAPH=0 ./bin/llama-server -m MODEL --split-mode tensor -ngl 99 -np 1 -c 512 --port 18404
- Request: curl -s --max-time 60 -X POST http://127.0.0.1:18404/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":10}'
- Kill: pkill -f 'llama-server.*18404'

## Rules
- ALWAYS read files before editing
- ALWAYS build after changes
- Do NOT regress to queue.wait() hacks
- Use GGML_LOG_INFO not fprintf for logging
- Mark tasks completed when done, then check TaskList for next
- If ALL tasks are completed, exit cleanly

Work through every available task until all are done or you hit an unsolvable blocker." \
        2>&1 | tee -a "$LOG"

    EXIT_CODE=$?
    echo "=== ITERATION $ITERATION ENDED (exit=$EXIT_CODE) === $(date)" | tee -a "$LOG"

    # Check if all tasks are done
    if grep -q "ALL.*COMPLETE\|all tasks.*completed\|no more tasks" "$LOG" 2>/dev/null; then
        echo "=== ALL TASKS COMPLETE — STOPPING ===" | tee -a "$LOG"
        break
    fi

    echo "Restarting in 10 seconds..." | tee -a "$LOG"
    sleep 10
done

echo "Graph fix loop finished at $(date)" | tee -a "$LOG"
