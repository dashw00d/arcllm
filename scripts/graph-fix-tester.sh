#!/bin/bash
# Tester agent — runs servers, sends requests, validates results.
# Coordinates with fixer via task-ledger (disk-based tasks/).

TASK_LEDGER=/home/ryan/.openclaw/workspace
LOG="/tmp/graph-fix-tester.log"
ITERATION=0

while true; do
    ITERATION=$((ITERATION + 1))
    echo "=== TESTER ITERATION $ITERATION === $(date)" | tee -a "$LOG"

    TASK_SUMMARY=$(cd "$TASK_LEDGER" && python3 scripts/list-open-tasks.py 2>/dev/null)

    claude -p --dangerously-skip-permissions \
        "You are the TESTER agent for SYCL graph recording optimization on 3x Intel Arc A770.

## Task Tracking (task-ledger)
Tasks are tracked as JSON files in $TASK_LEDGER/tasks/
Use these shell commands:

  # See all open tasks
  cd $TASK_LEDGER && python3 scripts/list-open-tasks.py

  # Check if a task is ready (dependencies met)
  cd $TASK_LEDGER && python3 scripts/task-ready.py <taskId>

  # Claim and start a task
  cd $TASK_LEDGER && python3 scripts/update-task.py <taskId> --mark-started --assigned-agent tester

  # Advance to next stage
  cd $TASK_LEDGER && python3 scripts/task-advance.py <taskId> 'next stage'

  # Close on success
  cd $TASK_LEDGER && python3 scripts/close-task.py <taskId> succeeded 'summary with timing data'

  # Close on failure — then create a bug task for fixer
  cd $TASK_LEDGER && python3 scripts/close-task.py <taskId> failed 'exact error + backtrace'
  cd $TASK_LEDGER && ./scripts/new-task.sh bug-<slug> 'Bug: description' 'full crash details' sub-agent 'investigate,fix,build' high fixer

## Current Task State
$TASK_SUMMARY

## Workflow
1. Run list-open-tasks.py to find tasks with owner='tester' where dependencies are succeeded
2. Verify readiness: task-ready.py <taskId>
3. Claim it: update-task.py <taskId> --mark-started --assigned-agent tester
4. Kill any existing server: pkill -f 'llama-server.*18404' || true
5. Run the test scenario described in the task
6. Capture ALL output to the task's log file
7. If PASS: close task as succeeded with timing data
8. If FAIL: close task as failed with full crash details, create new bug task for fixer
9. Always kill server after testing
10. Check for next tester task; if none available, exit cleanly

## Test Environment
- Kill existing: pkill -f 'llama-server.*18404' || true; sleep 2
- 0.6B model: /home/ryan/llm-stack/models/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
- 30B model: /home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf
- Server start: source /home/ryan/llm-stack/env.sglang-xpu.sh && GGML_SYCL_DISABLE_GRAPH=0 /home/ryan/llm-stack/llama.cpp-eptp/build-sycl/bin/llama-server -m MODEL --split-mode tensor -ngl 99 -np 1 -c 512 --port 18404 --slot-save-path /home/ryan/llm-stack/cache/slots/ > /tmp/tp_test.log 2>&1 &
- Wait for ready: poll http://127.0.0.1:18404/health until ok (max 120s)
- Request: curl -s --max-time 60 -X POST http://127.0.0.1:18404/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":10}'
- Kill after: pkill -f 'llama-server.*18404'

## What to capture on failure
- EXACT error message or backtrace
- Which request number crashed (1st? 2nd? 3rd?)
- The [META] and [KERN] timing lines from /tmp/tp_test.log
- The eval time line if generation completed before crash

## Rules
- Do NOT edit C++ source files — only fixer does that
- ALWAYS kill the server after each test
- ALWAYS redirect server output to /tmp/tp_test.log
- Write results to task-ledger on disk — not just in context
- If all tester tasks are done or only fixer-owned tasks remain, exit cleanly" \
        2>&1 | tee -a "$LOG"

    echo "=== TESTER ITERATION $ITERATION ENDED === $(date)" | tee -a "$LOG"

    OPEN_TESTER=$(cd "$TASK_LEDGER" && python3 scripts/list-open-tasks.py 2>/dev/null | grep -c tester || true)
    if [ "$OPEN_TESTER" -eq 0 ]; then
        echo "=== NO MORE TESTER TASKS — STOPPING ===" | tee -a "$LOG"
        break
    fi

    sleep 20
done
