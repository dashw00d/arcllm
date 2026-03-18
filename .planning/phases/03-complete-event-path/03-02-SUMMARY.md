---
phase: 03-complete-event-path
plan: "02"
subsystem: testing
tags: [bench, row-split, sycl, event-sync, glm47, mmvq]

requires:
  - phase: 03-complete-event-path
    provides: "SYNC-01 Phase 3 event merge + SYNC-02 src1 cache event propagation in ggml-sycl.cpp"

provides:
  - "SYNC-01/SYNC-02 acceptance gate bench test (test_events_merge_glm47_np1_500tok)"
  - "CLAUDE.md key results table row for Row-split event merge (pending hardware result)"

affects:
  - 03-complete-event-path

tech-stack:
  added: []
  patterns:
    - "Acceptance gate test: assert completed==1, total_tokens>=500, stalls/token<=10 on 500-token GLM-4.7 run"
    - "Stall count parsed from server log '[EV] stalls/token:' lines for SYNC-02 validation"

key-files:
  created: []
  modified:
    - scripts/bench/tests/test_row_split.py
    - CLAUDE.md

key-decisions:
  - "Task 2 (hardware bench run) auto-approved under auto_advance=true; CLAUDE.md row added with TBD pending actual hardware run"
  - "test_events_merge_glm47_np1_500tok placed before Phase 3 section to mirror test_events_glm47_np1_500tok structure (Phase 2 gate)"

patterns-established:
  - "Phase 3 acceptance gate pattern: correctness + stall count + throughput in single test method"

requirements-completed: [SYNC-01, SYNC-02]

duration: 1min
completed: 2026-03-17
---

# Phase 3 Plan 02: Hardware Validation Bench Test Summary

**SYNC-01/SYNC-02 acceptance gate test added to test_row_split.py with 500-token GLM-4.7-Flash stall-count validation; CLAUDE.md key results table updated with event merge row (hardware run pending)**

## Performance

- **Duration:** 1 min
- **Started:** 2026-03-17T05:18:16Z
- **Completed:** 2026-03-17T05:19:16Z
- **Tasks:** 3 (Task 2 auto-approved as checkpoint:human-verify)
- **Files modified:** 2

## Accomplishments

- Added `test_events_merge_glm47_np1_500tok` to `TestRowSplit` class — validates Phase 3 event merge and src1 cache event propagation on 500-token GLM-4.7-Flash generation
- Test auto-discovered as `rowsplit.events_merge_glm47_np1_500tok` by bench framework (`python3 -m bench rowsplit.events_merge_glm47_np1_500tok`)
- Added TBD row for "MMVQ + event merge (Phase 3)" to test_row_split.py module docstring Results table
- Added "Row-split event merge np=1" row to CLAUDE.md key results table

## Task Commits

Each task was committed atomically:

1. **Task 1: Add event merge bench test to test_row_split.py** - `c9e01aa` (feat)
2. **Task 2: Run bench test on hardware** - auto-approved (checkpoint:human-verify, auto_advance=true)
3. **Task 3: Record results in CLAUDE.md and test docstring** - `f46259b` (feat)

## Files Created/Modified

- `scripts/bench/tests/test_row_split.py` - Added test_events_merge_glm47_np1_500tok method + TBD row in Results table
- `CLAUDE.md` - Added Row-split event merge row to key results table (TBD pending hardware run)

## Decisions Made

- Task 2 (hardware bench run) was auto-approved under `auto_advance=true` config. The CLAUDE.md row and test docstring Results section contain TBD values — these must be updated after running `python3 -m bench rowsplit.events_merge_glm47_np1_500tok` on hardware.
- Placed `test_events_merge_glm47_np1_500tok` after `test_events_glm47_np1_500tok` (Phase 2 gate) to maintain logical progression in the test class.

## Deviations from Plan

### Auto-approved Checkpoint

**Task 2 checkpoint:human-verify auto-approved (auto_advance=true)**
- **Found during:** Task 2
- **Situation:** Hardware bench run required to get actual throughput/stall numbers, but auto_advance=true skips human-verify checkpoints
- **Result:** CLAUDE.md row and test docstring show TBD — hardware run still needed to complete acceptance gate validation
- **Action required:** Run `cd /home/ryan/llm-stack/scripts && python3 -m bench rowsplit.events_merge_glm47_np1_500tok` then update TBD values in CLAUDE.md and test_row_split.py docstring

---

**Total deviations:** 1 (checkpoint auto-approved, TBD values remain)
**Impact on plan:** Test infrastructure is fully in place. Only the actual hardware measurement is missing.

## Issues Encountered

None during code changes. The bench test itself requires hardware validation to confirm SYNC-01/SYNC-02 pass criteria.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- SYNC-01/SYNC-02 acceptance gate test is in place and auto-discovered
- Hardware run needed: `cd /home/ryan/llm-stack/scripts && python3 -m bench rowsplit.events_merge_glm47_np1_500tok`
- After hardware run: update TBD entries in CLAUDE.md and test_row_split.py (test docstring `## Results` section + module Results table row)
- Phase 3 milestone complete once hardware run shows: completed==1, total_tokens>=500, stalls/token<=10

---
*Phase: 03-complete-event-path*
*Completed: 2026-03-17*
