# llm-stack Bootstrap — 2026-03-20

## What Happened Today

### Orchestrator Setup
- Cloned `dashw00d/no-nonsense-orchestration` to `/home/ryan/projects/no-nonsense-orchestration`
- Set up parallel mode with `spawn_workers.sh` that spawns X concurrent agents
- Added `claimed_at` and `agent_id` columns to tasks DB for parallel awareness
- Cron job every 10 min: `/home/ryan/projects/no-nonsense-orchestration/scripts/spawn_workers.sh 3`
- **Cron is currently STOPPED** — we were debugging

### REAM Model Discovery
**FINDING: `Qwen3-30B-A3B-REAM-heretic-i1` produces garbled output on BOTH stable and eptp builds.**

This is NOT an EP bug — it's a model/quantization issue in that specific heretic-i1 merge variant.

**Models:**
- `Qwen3-30B-A3B-abliterated` (96→128 experts, Q4_K_M) — WORKS, clean output
- `Qwen3-30B-A3B-REAM-heretic-i1` (96 experts, Q4_K_M) — BROKEN, garbled

**The 96÷3=32 divisibility is still valid** — abliterated model proves MoE works.

### Build Fixes (2026-03-20 continued)
- **`config.py`**: Fixed `server_args()` to pass `-fa off` when `flash_attn=False`. Previously the flag was only added when `True`, causing `False` to default to `auto` which enables flash attention.
- **`llama.cpp/build-sycl`**: Symlinked to `llama.cpp-stable/build-sycl`. Old build crashed on MoE (IGC internal compiler error with MoE + flash attn).
- **`test_frontier.py`**: `TestMoEFrontier` now sets `flash_attn=False` (IGC crashes on MoE + flash attention).

### MoE Benchmark Results (qwen30b-ablit-q4km, c=512, layer split)

| Config | Result | Notes |
|--------|--------|-------|
| np=4 | **11.2 t/s** | 4 slots |
| np=8 | **14.4 t/s** | 8 slots |
| np=16 | **25.7 t/s** | 16 slots |

### Dense Benchmark Results (Q4_K_M, c=32768)

| Config | Result | Notes |
|--------|--------|-------|
| np=16 FUSED_MMQ=0 | **17.7 t/s** | Stable build, no FUSED_MMQ |
| np=16 FUSED_MMQ=1 | **22.1 t/s** | Old build with FUSED_MMQ |

Note: Stable build (Mar 20) doesn't have FUSED_MMQ. Old build (Mar 17) had it but crashed on MoE. MoE works at 25.9 t/s without FUSED_MMQ.

### EP Status
**EP (tensor split mode) does NOT work in stable build.** Crashes with:
```
GGML_ASSERT(split_state.ne[j] % tensor->src[i]->ne[src_split_states[i].axis] == 0)
```
EP-specific code (`handle_mul_mat_id` with SPLIT_AXIS_2) exists only in `llama.cpp-eptp`, not in stable.

### Files Updated
- `llama.cpp-stable/docs/EP-DEBUG.md` — Updated status, marked REAM as broken
- `CLAUDE.md` — Added qwen30b-ablit-q4km to key results (~28 t/s at np=16)
- `scripts/bench/config.py` — Added `qwen30b-ablit-q4km` preset, added `no_warmup` flag, **fixed -fa off flag**
- `scripts/bench/tests/test_frontier.py` — Added `TestMoEFrontier` class with np=4/8/16 tests, **flash_attn=False**
- **`llama.cpp/build-sycl` → `llama.cpp-stable/build-sycl`** (symlink)

### GPU Incident
GPU driver got stuck after hard reset during debugging. **System was rebooted.**

---

## How to Continue

### 1. Verify GPUs Are Working
```bash
cd /home/ryan/llm-stack
source env.sglang-xpu.sh
sycl-ls
# Should show 3x Intel Arc A770
```

### 2. Kill Any Stale Servers
```bash
pkill -x llama-server
```

### 3. Quick Sanity Test
```bash
cd /home/ryan/llm-stack
source env.sglang-xpu.sh
./llama.cpp-stable/build-sycl/bin/llama-server \
  -m models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf \
  --split-mode layer -ngl 99 -np 1 -c 512 --no-warmup --port 18450 &
sleep 15
curl -s -X POST http://localhost:18450/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}'
# Should get clean output
```

### 4. Run MoE Benchmark
```bash
cd /home/ryan/llm-stack/scripts
python3 -m bench moefrontier.np16
```

### 5. Restart Cron (when ready)
```bash
(crontab -l 2>/dev/null; echo "*/10 * * * * /home/ryan/projects/no-nonsense-orchestration/scripts/spawn_workers.sh 3 >> /home/ryan/orchestrator-cron.log 2>&1") | crontab -
```

---

## Task DB: SQLite

**Location:** `~/.no-nonsense/tasks.db`

### Common Commands
```bash
export NO_NONSENSE_TASKS_DB=~/.no-nonsense/tasks.db

# List all tasks
/home/ryan/projects/no-nonsense-orchestration/scripts/task_list.sh

# List by status
/home/ryan/projects/no-nonsense-orchestration/scripts/task_list.sh -s todo
/home/ryan/projects/no-nonsense-orchestration/scripts/task_list.sh -s in-progress

# Show task details
/home/ryan/projects/no-nonsense-orchestration/scripts/task_show.sh <id>

# Add task
/home/ryan/projects/no-nonsense-orchestration/scripts/task_add.sh "Title" -r role -t "tag1,tag2,gpu"

# Move task
/home/ryan/projects/no-nonsense-orchestration/scripts/task_move.sh <id> -s done

# Reset stuck task
sqlite3 ~/.no-nonsense/tasks.db "UPDATE tasks SET agent_id='', claimed_at=NULL, status='todo' WHERE id=<id>;"
```

### Key Task: #42
- **Title:** Debug EP np=1 still garbled after SPLIT_AXIS_2 fix
- **Tags:** ep,critical,seq:6b,gpu
- **Status:** done (REAM model bug, not EP)

### Task: #31
- **Title:** Test Option B refactor: verify subgraph count ~49 and numerical correctness
- **Tags:** seq:9,gpu
- **Status:** blocked (EP not working in stable build)
- **Context:** EP tensor split crashes in stable. EP code exists only in llama.cpp-eptp.

### Task: #45
- **Title:** Verify EP works on abliterated model
- **Tags:** ep,gpu,verification,moe
- **Status:** blocked (EP crashes in stable build, GGML_ASSERT in split_state)

### Task: #44
- **Title:** Fix bench framework: explicit -fa off flag, stable build for MoE
- **Tags:** bench,gpu,housekeeping
- **Status:** done

### Schema
```sql
CREATE TABLE tasks (
  id INTEGER PRIMARY KEY,
  title TEXT,
  description TEXT,
  status TEXT DEFAULT 'todo',  -- todo|backlog|in-progress|done|blocked
  role TEXT,                    -- planner|breaker|implementer|tester|reviewer
  assignee TEXT,
  tags TEXT,
  context TEXT,
  parent TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  -- Parallel mode columns:
  claimed_at DATETIME,
  agent_id TEXT
);
```

---

## Project Structure

```
/home/ryan/projects/no-nonsense-orchestration/
├── orchestrator.md           # Original single-agent orchestrator
├── orchestrator_parallel.md  # Parallel mode orchestrator
├── migrations/
│   ├── 001_initial_schema.sql
│   └── 002_parallel_mode.sql  # claimed_at, agent_id
├── scripts/
│   ├── task_list.sh          # --unassigned, --stale filters
│   ├── task_claim.sh         # Atomic claim
│   ├── spawn_workers.sh      # Spawn X parallel agents
│   └── ...
└── roles/
    ├── planner.md
    ├── breaker.md
    ├── implementer.md
    ├── tester.md
    └── reviewer.md
```

---

## Critical Files

| File | Purpose |
|------|---------|
| `/home/ryan/llm-stack/CLAUDE.md` | Main project doc — model notes, bench results |
| `/home/ryan/llm-stack/llama.cpp-stable/docs/EP-DEBUG.md` | EP corruption investigation (REAM is broken) |
| `/home/ryan/llm-stack/scripts/bench/config.py` | Benchmark config presets |
| `/home/ryan/llm-stack/scripts/bench/tests/test_frontier.py` | Frontier + MoE benchmarks |
| `/home/ryan/projects/no-nonsense-orchestration/` | Orchestrator codebase |

---

## Bench Framework

```bash
cd /home/ryan/llm-stack/scripts

# List tests
python3 -m bench help

# Run specific test
python3 -m bench moefrontier.np16
python3 -m bench frontier.np16

# Run suite
python3 -m bench moefrontier
python3 -m bench parallel

# Results saved to /tmp/bench_results.json
```

---

## Current Priority

1. **EP is blocked** — stable build doesn't have EP support. EPTP build has it but is older.
2. **MoE layer-split works** — 25.9 t/s at np=16 on abliterated model
3. **Next: non-EP backlog tasks** — #7 (MUL_MAT_ID opt), #9-12, etc.
4. **Restart cron** when ready to resume autonomous task orchestration
5. **Task 31** blocked until EP works in stable or EPTP is rebuilt

---

## Notes

- REAM-heretic-i1 is **NOT usable** for MoE testing
- Use `Qwen3-30B-A3B-abliterated` for all MoE work
- EP (tensor split) is **EP-specific code only in llama.cpp-eptp** — stable build crashes
- **IGC crashes on MoE + flash attention** — always use `flash_attn=False` for MoE
- The parallel orchestrator setup is ready but cron is paused during debugging
- `llama.cpp/build-sycl` symlink points to stable — enables bench framework to use working build
