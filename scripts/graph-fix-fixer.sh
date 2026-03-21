#!/bin/bash
# Fixer agent — implements code changes, builds, marks tasks done via task-ledger.
# Picks up tasks owned by 'fixer' that are pending and unblocked.

TASK_LEDGER=/home/ryan/.openclaw/workspace
LOG="/tmp/graph-fix-fixer.log"
ITERATION=0

while true; do
    ITERATION=$((ITERATION + 1))
    echo "=== FIXER ITERATION $ITERATION === $(date)" | tee -a "$LOG"

    # Find next available fixer task
    NEXT_TASK=$(cd "$TASK_LEDGER" && python3 scripts/list-open-tasks.py 2>/dev/null \
        | grep -E '"owner".*fixer|fixer' \
        | head -1)

    # Show full open task list for agent context
    TASK_SUMMARY=$(cd "$TASK_LEDGER" && python3 scripts/list-open-tasks.py 2>/dev/null)

    claude -p --dangerously-skip-permissions \
        "You are the FIXER agent for SYCL graph recording optimization on 3x Intel Arc A770.

## Task Tracking (task-ledger)
Tasks are tracked as JSON files in $TASK_LEDGER/tasks/
Use these shell commands to manage tasks (run them directly):

  # See all open tasks
  cd $TASK_LEDGER && python3 scripts/list-open-tasks.py

  # Check if a task is ready (dependencies met)
  cd $TASK_LEDGER && python3 scripts/task-ready.py <taskId>

  # Mark a task started
  cd $TASK_LEDGER && python3 scripts/update-task.py <taskId> --mark-started --assigned-agent fixer

  # Advance stage
  cd $TASK_LEDGER && python3 scripts/task-advance.py <taskId> 'next stage description'

  # Close task when done
  cd $TASK_LEDGER && python3 scripts/close-task.py <taskId> succeeded 'summary of what was done'

  # Mark blocked (if you hit an unsolvable issue)
  cd $TASK_LEDGER && python3 scripts/update-task.py <taskId> --mark-blocked --blocked-reason 'reason'

  # Create a new bug task when you find something unexpected
  cd $TASK_LEDGER && ./scripts/new-task.sh bug-<slug> 'Bug: description' 'details' sub-agent 'investigate,fix,build' high fixer

## Current Task State
$TASK_SUMMARY

## Workflow
1. Run list-open-tasks.py to find tasks with owner='fixer' that are pending/in_progress and not blocked
2. Check task is ready: task-ready.py <taskId>
3. Claim it: update-task.py <taskId> --mark-started --assigned-agent fixer
4. Read the relevant source files
5. Implement the changes
6. Build: cd /home/ryan/llm-stack/llama.cpp-eptp/build-sycl && source /home/ryan/llm-stack/env.sglang-xpu.sh && cmake --build . --target llama-server -j\$(nproc)
7. If build fails, fix and rebuild
8. When build succeeds, advance stage then close task as succeeded
9. Check for next available fixer task
10. If no unblocked fixer tasks exist, exit cleanly

## Project Context
- Working dir: /home/ryan/llm-stack/llama.cpp-eptp/
- Key file: ggml/src/ggml-sycl/ggml-sycl.cpp (graph_compute ~line 4490)
- Key file: ggml/src/ggml-sycl/common.hpp (graph_cache_entry struct)
- Read OPTIMIZATION_PLAN.md and ORCHESTRATOR_BRIEF.md for full context
- Graph replay gives 8ms vs 400ms (50x). Crash is stale tensor pointers.
- Segmented graph recording already implemented (task-2 done) — build is green

## Rules
- ALWAYS read files before editing
- ALWAYS build after changes
- Do NOT test (no starting servers, no curl requests) — tester does that
- Do NOT regress to queue.wait() hacks
- Use GGML_LOG_INFO not fprintf
- Write task updates to disk — do NOT rely on in-context memory for state
- If all fixer tasks are done or only tester-owned tasks remain, exit" \
        2>&1 | tee -a "$LOG"

    echo "=== FIXER ITERATION $ITERATION ENDED === $(date)" | tee -a "$LOG"

    # Check if any fixer tasks remain open
    OPEN_FIXER=$(cd "$TASK_LEDGER" && python3 scripts/list-open-tasks.py 2>/dev/null | grep -c fixer || true)
    if [ "$OPEN_FIXER" -eq 0 ]; then
        echo "=== NO MORE FIXER TASKS — STOPPING ===" | tee -a "$LOG"
        break
    fi

    sleep 15
done
