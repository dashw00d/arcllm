# Roadmap: Event-Based Multi-GPU Sync for SYCL Row-Split

## Overview

Phases 1-2 validate and harden the existing OOO queue + event chain infrastructure (already implemented). Phase 3 completes the event path by converting the last blocking stall site (Phase 3 merge) to event-based chains and propagating events through the src1 copy cache. The result: GPU pipeline stays full, GLM-4.7-Flash hits 10+ t/s on 3x A770 row-split with ~3 host stalls per token instead of ~4000.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Harden Infrastructure** - Assert correct OOO queue context and enforce immediate command list requirement in code (Plan 01-01 complete)
- [x] **Phase 2: Validate Existing Event Path** - Confirm Phase 1-2 of the dispatch loop is correct before adding Phase 3 work (completed 2026-03-18)
- [x] **Phase 3: Complete Event Path** - Convert Phase 3 merge to event-based chains, propagate events through src1 cache, characterize throughput (completed 2026-03-18)

## Phase Details

### Phase 1: Harden Infrastructure
**Goal**: The OOO queue infrastructure is provably correct — misconfigured environments crash loudly instead of silently producing wrong results
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02
**Success Criteria** (what must be TRUE):
  1. Starting llama.cpp with `GGML_SYCL_ROW_EVENTS=1` but without `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` aborts with a clear error message
  2. Starting llama.cpp with an OOO queue constructed from a different SYCL context than the in-order streams aborts with a clear error message
  3. Both assertions trigger before any matmul is dispatched (startup, not first inference)
**Plans:** 1/1 plans complete
Plans:
- [x] 01-01-PLAN.md — Add cmdlist + context assertions to SYCL init path, create bench test for validation

### Phase 2: Validate Existing Event Path
**Goal**: The existing Phase 1-2 dispatch loop (OOO queues, event chains, barrier completion) is confirmed correct on GLM-4.7-Flash before Phase 3 is added
**Depends on**: Phase 1
**Requirements**: CORR-01, CORR-02, SYNC-03
**Success Criteria** (what must be TRUE):
  1. GLM-4.7-Flash Q4_K_M generates 500+ tokens at 2048 context on 3x A770 without desync or crash
  2. Host stall counter reports ~3 waits per token (one per device at Phase 2 barrier), not ~4000
  3. The stall count is visible in bench output when `GGML_SYCL_ROW_EVENTS=1` is set
**Plans:** 1/1 plans complete
Plans:
- [x] 02-01-PLAN.md — Add stall counter instrumentation (SYNC-03), run GLM 500-token correctness bench (CORR-01/CORR-02)

### Phase 3: Complete Event Path
**Goal**: The last blocking stall site is eliminated — Phase 3 merge uses event-based chains, src1 cache propagates completion events, and throughput is characterized
**Depends on**: Phase 2
**Requirements**: SYNC-01, SYNC-02
**Success Criteria** (what must be TRUE):
  1. GLM-4.7-Flash row-split throughput reaches 10+ t/s on 3x A770 (measured by bench framework)
  2. Phase 3 merge no longer calls `dev2dev_memcpy_staged_sync` — uses `handler::depends_on()` chaining
  3. src1 cache hit path returns a completion event that downstream matmuls can chain `depends_on()` on — no silent sync gap
  4. Benchmark results recorded in `test_row_split.py` docstring and `CLAUDE.md` key results table
**Plans:** 2/2 plans complete
Plans:
- [ ] 03-01-PLAN.md — SYNC-02 src1 cache event propagation + SYNC-01 event-based Phase 3 merge
- [ ] 03-02-PLAN.md — Bench test, hardware validation, results recording

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Harden Infrastructure | 1/1 | Complete    | 2026-03-18 |
| 2. Validate Existing Event Path | 1/1 | Complete    | 2026-03-18 |
| 3. Complete Event Path | 2/2 | Complete   | 2026-03-18 |
