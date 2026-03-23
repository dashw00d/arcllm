# Agent Briefs — Arc-Tools Pipeline Overnight Run

## Shared Context

- **Project**: /home/ryan/llm-stack/arc-tools/
- **Branch**: overnight (already checked out)
- **Henry (local LLM)**: http://localhost:11435 — Qwen3-32B at np=4, c=32768
- **Docker stack**: `cd /home/ryan/llm-stack/arc-tools && docker compose ...`
- **DB**: Postgres at localhost:5432 (user=temporal, pass=temporal)
  - `ghostgraph` DB: site-auditor + grabber tables
  - `churner` DB: churner tables (missions, raw_ingests, entities, etc.)
- **Task tracker**: `~/.openclaw/workspace/skills/no-nonsense-tasks/scripts/`
  - List: `task_list.sh --status todo`
  - Move to in-progress: `task_move.sh <id> --status in-progress`
  - Mark done: `task_move.sh <id> --status done`
  - Filter: `task_filter.sh arc-tools`
- **Git**: Commit your changes with descriptive messages. Push to `overnight` branch.
- **Style**: Python 3.12. No tests, no docstrings, no type annotations unless already present. Keep it simple.
- **agent-browser**: installed at /home/ryan/.nvm/versions/node/v24.14.0/bin/agent-browser (v0.21.4)

## Rules
1. Before starting a task, move it to `in-progress`
2. When done, move it to `done`
3. Commit after each completed task
4. If blocked, add a note to the task description and move to next task
5. Do NOT restart Henry/llama-server — it's shared across all agents
6. Do NOT run `docker compose down` — only restart individual services
7. Check `docker compose ps` before touching Docker services
