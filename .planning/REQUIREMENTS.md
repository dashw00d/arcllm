# Requirements: Event-Based Multi-GPU Sync

**Defined:** 2026-03-17
**Core Value:** GPU pipeline stays full — GPU 1 begins computing the moment GPU 0's output is available, without ever draining either device.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Infrastructure

- [x] **INFRA-01**: System asserts `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` in C++ when `GGML_SYCL_ROW_EVENTS=1` is enabled
- [x] **INFRA-02**: System verifies OOO queue context matches in-order stream context at startup, crashes loudly if mismatched

### Event Sync

- [x] **SYNC-01**: Phase 3 merge copies use `handler::depends_on(kernel_completion_event)` instead of blocking `dev2dev_memcpy_staged_sync`
- [x] **SYNC-02**: src1 copy cache stores last copy completion event per device and returns it to downstream matmuls via `depends_on()`
- [x] **SYNC-03**: Host stall counter instruments `queue.wait()` calls and reports count per token when `GGML_SYCL_ROW_EVENTS=1`

### Correctness

- [x] **CORR-01**: GLM-4.7-Flash Q4_K_M generates 500+ tokens at 2048 context on 3x A770 without desync or crash
- [x] **CORR-02**: Host stall count per token is ~3 (one per device at Phase 2), not ~4000

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Validation

- **VAL-01**: Perplexity matches single-GPU reference within 0.01
- **VAL-02**: Debug fprintf statements removed from Phase 2 hot path

### Upstream

- **UPST-01**: Code cleaned for upstream PR (error handling, cross-platform guards, documentation)
- **UPST-02**: Event path generalized to work with any row-split model (not just GLM-4.7-Flash)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Worker threads for device dispatch | Tested and reverted — 25% slower due to queue contention |
| SYCL graph capture | SIGABRT at startup — graph recording doesn't support MoE ops |
| Silent fallback to host-sync | Masks bugs; use explicit GGML_SYCL_ROW_EVENTS gate instead |
| Layer-split event sync | Different root cause (single-device L0 ordering), separate milestone |
| np > 1 stability | Crashes at 56-83s even with host-stall sync — separate investigation |
| Batched command list support | Deadlocks cross-queue depends_on — confirmed experimentally |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Complete (2026-03-18) |
| INFRA-02 | Phase 1 | Complete (2026-03-18) |
| SYNC-03 | Phase 2 | Complete |
| CORR-01 | Phase 2 | Complete |
| CORR-02 | Phase 2 | Complete |
| SYNC-01 | Phase 3 | Complete |
| SYNC-02 | Phase 3 | Complete |

**Coverage:**
- v1 requirements: 7 total
- Mapped to phases: 7
- Unmapped: 0

---
*Requirements defined: 2026-03-17*
*Last updated: 2026-03-18 — INFRA-01, INFRA-02 complete (plan 01-01)*
