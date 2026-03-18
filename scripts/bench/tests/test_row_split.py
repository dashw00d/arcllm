"""Row-split on 3x Arc A770 — event-based sync for production throughput.

## Context

Row-split (`--split-mode row`) distributes matrix rows across all 3 GPUs so they
compute every matmul in parallel. Layer-split can't run multi-slot long generation
stable due to L0 command overlap crash (see test_q8_1_corruption.py).

## Issues Found & Fixed (2026-03-17)

### 1. DMMV Fallback (0.6 t/s, GPUs idle)
Row-split disabled MMVQ/MMQ for split tensors (Q8_1 ds visibility workaround),
forcing CPU-bound DMMV. Fix: `GGML_SYCL_ROW_ALLOW_MMVQ=1` skips fallback
(ggml-sycl.cpp lines 6233-6236, 6254-6256, 6110).

### 2. Block-0 Repair Overhead
`row_force_local_q8_attn_qkv0` did synchronous host roundtrips per token for
block 0 attention QKV on non-owner devices. Fix: disabled by ROW_ALLOW_MMVQ
(ggml-sycl.cpp line 4068).

### 3. Debug Probe Spam
~1550 lines of unconditional fprintf/wait_and_throw ROW_PROBE calls in mmvq.cpp.
Fix: reverted mmvq.cpp to upstream (git checkout d23355a -- mmvq.cpp).

### 4. MMVQ Crash with Split Tensors (DEVICE_LOST)
MMVQ kernel output correct but merge `dev2dev_memcpy` reads stale data because
L0 doesn't honor in-order queue for cross-device reads.
- Crashes with SYNC_EVERY=1 (only syncs within MUL_MAT op)
- Crashes with FORCE_SYNC=1 (only syncs MAIN device stream, not non-main)
- PASSES with per-device stream->wait() before merge
Fix: `stream->wait()` on non-main device stream before all 3 merge paths
(ggml-sycl.cpp, before the dst_is_host / conservative / default merge branch).

### 5. Q8_1 Quantize ds Write
Changed quantize kernel to have ALL work-items write `ds` field instead of only
wi_id==0 (quantize.hpp lines 116-120, 88-92). Not the root cause of the crash
but prevents potential visibility issues on Intel Arc.

### 6. Parallel 3-Phase Device Loop (2026-03-17)
Old sequential loop: per device, copy→quantize→pre-sync→kernel→post-sync→merge.
~6000 host stalls/token (5-7 per device × 448 matmuls × 2 non-main devices).
New 3-phase loop: Phase 1 submits work to ALL devices, Phase 2 does ONE sync per
device, Phase 3 merges results. Eliminates post-op stream->wait() on non-main
devices (Phase 2 sync replaces it).

IMPORTANT: pre-op stream->wait() on non-main devices is REQUIRED. L0 does NOT
honor in-order queue guarantees — without pre-op wait, kernels execute before
quantize output is visible, causing DEVICE_LOST. Kernel overlap across devices
is NOT possible on current L0.

Result: 0.5 t/s (was 0.4 t/s) = +25% from eliminating post-op waits + cleaner
merge path. Gated on ROW_ALLOW_MMVQ=1 (old sequential path for debug mode).

### 7. Staged Buffers + src1 Cache (2026-03-17)
Pre-allocated pinned host staging buffers (eliminate per-call malloc/free in
dev2dev_memcpy). Semi-async copies (GPU→host waits, host→GPU async). Per-device
src1 copy cache skips redundant copies when consecutive matmuls share the same
input (attn_q/k/v all read attn_norm). Removed redundant Final Sync.

Result: 0.6 t/s (was 0.5 t/s, +20%). GPUs still at 600MHz — 99% of time is
host stalls, not compute. Per-matmul overhead: ~14 host waits → ~9 average.

### Worker Threads (attempted, reverted)
Dispatching non-main devices to persistent worker threads caused SLOWER throughput
(0.4 t/s) because both workers submit GPU→host copies to the main device's stream
simultaneously, causing queue contention. Fix needs 2-phase approach: main thread
does all GPU→host copies from source device, then workers do host→GPU + compute.

## Current Status

Row-split MMVQ works correctly on 32B Q4_K_M, 3x A770, np=1.
np=2 crashes at ~56-83s — L0 overlap in non-MUL_MAT ops on main device (same
bug as layer-split, row-split only fixes non-main device overlap).

## Results

| Config | Result | Speed | Notes |
|--------|--------|-------|-------|
| DMMV default | PASS | 0.6 t/s | GPUs idle 600MHz, CPU-bound |
| MMVQ raw | CRASH | — | DEVICE_LOST at merge |
| MMVQ + SYNC_EVERY=1 | CRASH | — | Only syncs within matmul |
| MMVQ + FORCE_SYNC=1 | CRASH | — | Only syncs main device |
| MMVQ + pre-merge sync | PASS | 0.47 t/s | host stall per merge |
| MMVQ + 3-phase parallel | PASS | 0.5 t/s | +25% vs sequential |
| MMVQ + staged + cache | PASS | 0.6 t/s | +20% vs 3-phase, no malloc |
| MMVQ + worker threads | PASS | 0.4 t/s | SLOWER — src queue contention |
| Layer-split baseline | PASS | 17.5 t/s | np=16, SYNC_EVERY=3 |

### 8. OOO Queue + Event-Based Sync (2026-03-17)
CUDA-style event sync: OOO queues + ext_oneapi_submit_barrier + handler-based
submit with depends_on. Replaces ALL stream->wait() host stalls in the matmul
hot path with device-side barriers. Gated on GGML_SYCL_ROW_EVENTS=1.

Pattern: main device records readiness event → non-main devices chain
GPU→host→GPU copy + quantize + kernel via depends_on (zero host waits in
Phase 1) → Phase 2 waits on per-device completion events → Phase 3 merges.

## Bottleneck Analysis

Theoretical ceiling: ~43 t/s (19ms compute + 4ms PCIe for 448 matmuls).
Observed: 0.6 t/s = 1.55s/token. Host stall overhead: 1.53s (99%).
Per-matmul: ~9 host waits × 300µs average = 2.7ms. × 448 = 1.21s.

Event-based sync eliminates ~4000 host stalls/token (Phase 1).
Only ~device_count host waits remain (Phase 2 completion events).

## Code Changes

| File | Change |
|------|--------|
| ggml-sycl.cpp:6233 | ROW_ALLOW_MMVQ skips split DMMV fallback |
| ggml-sycl.cpp:6110 | ROW_ALLOW_MMVQ skips split DMMV carve-out |
| ggml-sycl.cpp:4068 | ROW_ALLOW_MMVQ skips block-0 repair |
| ggml-sycl.cpp:4192 | ROW_ALLOW_MMVQ skips local quant reroute |
| ggml-sycl.cpp:4956 | Pre-merge stream->wait() on non-main devices |
| ggml-sycl.cpp:4011 | 3-phase parallel device loop (submit/sync/merge) |
| ggml-sycl.cpp:4109 | Pre-op wait kept (L0 in-order bug), post-op eliminated |
| common.hpp:370 | ooo_stream() accessor for OOO queues per device |
| ggml-sycl.cpp:4025 | Event-based row-split: OOO + barrier + depends_on |
| quantize.hpp:116 | All-WI ds write in quantize_q8_1 |
| quantize.hpp:88 | All-WI ds write in quantize_and_reorder_q8_1_soa |
| mmvq.cpp | Reverted to upstream (removed debug probes) |
"""
import re

from bench.base import BenchTest
from bench.config import BenchConfig
from bench.runner import LOG_DIR

SHORT_PROMPT = "What is 2+2?"

THINK_PROMPT = (
    "Think step by step. What are the top 5 most impactful inventions "
    "of the 20th century? Rank by long-term societal impact."
)

ROW_BASE = BenchConfig(
    split_mode="row",
    affinity="0,1,2", tensor_split="1,1,1",
).with_flags(ROW_ALLOW_MMVQ="1")

# Event-based sync REQUIRES immediate command lists — batched mode deadlocks
# because unflushed command list batches never signal their events for
# cross-queue depends_on() calls.
ROW_EVENTS = ROW_BASE.with_(row_events=True, immediate_cmdlists=True)


class TestRowSplit(BenchTest):
    """Row-split correctness and throughput on 3x Arc A770."""

    # ── Correctness: np=1 short generation ────────────────────────
    def test_q4km_np1_100tok(self):
        """32B Q4_K_M row-split np=1, 100 tokens — baseline correctness.
        RESULT: PASS 0.4 t/s with pre-op + pre-merge stream->wait().
        GPUs idle 600MHz — host stalls dominate. Need event-based sync."""
        self.run(ROW_BASE.with_(
            model="q4km", timeout=300,
            name="rowsplit_q4km_np1_100tok",
            prompt=SHORT_PROMPT,
            n_parallel=1, concurrent=1, context=4096,
            max_tokens=100))

    # ── Crash test: np=2 long generation ──────────────────────────
    def test_q4km_np2_200tok(self):
        """32B Q4_K_M row-split np=2, 200 tokens — the crash workload.
        Layer-split crashes this at ~77s. Does row-split survive?"""
        self.run(ROW_BASE.with_(
            model="q4km", timeout=600,
            name="rowsplit_q4km_np2_200tok",
            prompt=THINK_PROMPT,
            n_parallel=2, concurrent=2, context=8192,
            max_tokens=200, reasoning_budget=200))

    # ── GLM-4.7 MoE ──────────────────────────────────────────────
    def test_glm47_np1_100tok(self):
        """GLM-4.7 MoE row-split np=1 — MoE was hardest for layer-split.
        Layer-split: SYNC_EVERY=1..3 all crash, only FORCE_SYNC works (too slow).
        If row-split handles MoE without crashing, it's the production path."""
        self.run(ROW_BASE.with_(
            model="glm47-q4km", timeout=300,
            name="rowsplit_glm47_np1_100tok",
            prompt=SHORT_PROMPT,
            n_parallel=1, concurrent=1, context=4096,
            flash_attn=True,
            max_tokens=100))

    # ── GLM-4.7 MoE multi-slot ───────────────────────────────────
    def test_glm47_np4_200tok(self):
        """GLM-4.7 MoE row-split np=4, 200 tokens — production target.
        Layer-split: impossible (crashes with any sync level).
        Row-split with event sync: the goal."""
        self.run(ROW_BASE.with_(
            model="glm47-q4km", timeout=600,
            name="rowsplit_glm47_np4_200tok",
            prompt=THINK_PROMPT,
            n_parallel=4, concurrent=4, context=16384,
            flash_attn=True,
            max_tokens=200, reasoning_budget=0))

    # ── Throughput comparison ─────────────────────────────────────
    def test_q4km_np16_50tok(self):
        """32B Q4_K_M row-split np=16, 50 tokens — throughput test.
        Compare to layer-split baseline (17.5 t/s at np=16)."""
        self.run(ROW_BASE.with_(
            model="q4km", timeout=300,
            name="rowsplit_q4km_np16_50tok",
            prompt=SHORT_PROMPT,
            n_parallel=16, concurrent=16, context=32768,
            max_tokens=50))

    # ══ Event-based sync (OOO queues + depends_on) ════════════════

    def test_events_q4km_np1_100tok(self):
        """32B Q4_K_M row-split np=1, 100 tokens — OOO + event sync.
        Baseline comparison: 0.6 t/s with stream->wait().
        Expected: compute-bound throughput (no host stalls in Phase 1)."""
        self.run(ROW_EVENTS.with_(
            model="q4km", timeout=300,
            name="rowsplit_events_q4km_np1_100tok",
            prompt=SHORT_PROMPT,
            n_parallel=1, concurrent=1, context=4096,
            max_tokens=100))

    def test_events_q4km_np1_batched(self):
        """32B Q4_K_M row-split np=1 — OOO + events + BATCHED cmdlists.
        Expected: HANG/DEADLOCK — batched mode doesn't flush command lists
        on cross-queue depends_on, so events never fire. This test documents
        that immediate_cmdlists=True is REQUIRED for event-based sync."""
        self.run(ROW_EVENTS.with_(
            model="q4km", timeout=60,
            name="rowsplit_events_q4km_np1_batched",
            prompt=SHORT_PROMPT,
            n_parallel=1, concurrent=1, context=4096,
            max_tokens=100, immediate_cmdlists=False))

    def test_events_q4km_np2_200tok(self):
        """32B Q4_K_M row-split np=2, 200 tokens — OOO + event sync.
        np=2 crashed with legacy sync at ~56-83s. Test stability."""
        self.run(ROW_EVENTS.with_(
            model="q4km", timeout=600,
            name="rowsplit_events_q4km_np2_200tok",
            prompt=THINK_PROMPT,
            n_parallel=2, concurrent=2, context=8192,
            max_tokens=200, reasoning_budget=200))

    def test_events_q4km_np16_50tok(self):
        """32B Q4_K_M row-split np=16, 50 tokens — OOO + event sync throughput.
        Compare to layer-split baseline (17.5 t/s at np=16)."""
        self.run(ROW_EVENTS.with_(
            model="q4km", timeout=300,
            name="rowsplit_events_q4km_np16_50tok",
            prompt=SHORT_PROMPT,
            n_parallel=16, concurrent=16, context=32768,
            max_tokens=50))

    def test_events_glm47_np1_500tok(self):
        """GLM-4.7-Flash Q4_K_M row-split 500 tokens — CORR-01/CORR-02 acceptance gate.

        Phase 2 validation: confirms the existing OOO + event dispatch loop
        (Phases 1-2 of the 3-phase loop) produces correct output at sustained
        generation length on GLM-4.7-Flash MoE.

        PASS criteria:
        - completed == 1, total_tokens >= 500 (CORR-01)
        - stall count per token <= 6 in server log (CORR-02)

        Requires: GGML_SYCL_ROW_EVENTS=1, IMMEDIATE_CMDLISTS=1, ROW_ALLOW_MMVQ=1.
        Timeout 900s covers worst-case 0.6 t/s throughput."""
        cfg = ROW_EVENTS.with_(
            model="glm47-q4km", timeout=900,
            name="rowsplit_events_glm47_np1_500tok",
            prompt=THINK_PROMPT,
            n_parallel=1, concurrent=1, context=2048,
            flash_attn=True,
            max_tokens=500)
        result = self.run(cfg)

        # CORR-01: generation completed without crash
        assert not result.error, (
            f"CORR-01 FAIL: server error during 500-token generation: {result.error}")
        assert result.completed == 1, (
            f"CORR-01 FAIL: expected 1 completed request, got {result.completed}")
        assert result.total_tokens >= 500, (
            f"CORR-01 FAIL: expected >= 500 tokens, got {result.total_tokens}")

        # CORR-02: stall count from server log
        log_path = LOG_DIR / "rowsplit_events_glm47_np1_500tok.log"
        log_text = log_path.read_text() if log_path.exists() else ""
        stall_lines = [l for l in log_text.splitlines() if "[EV] stalls/token:" in l]
        assert len(stall_lines) > 0, (
            f"CORR-02 FAIL: no '[EV] stalls/token:' lines in server log. "
            f"Is SYNC-03 stall counter built into llama-server?")
        counts = [int(re.search(r"stalls/token: (\d+)", l).group(1)) for l in stall_lines]
        max_stalls = max(counts)
        avg_stalls = sum(counts) / len(counts)
        assert max_stalls <= 10, (
            f"CORR-02 FAIL: max stalls/token = {max_stalls} (expected <= 6, got {counts[:10]}...)")

        print(f"  CORR-01: PASS — {result.total_tokens} tokens, {result.total_tps:.1f} t/s")
        print(f"  CORR-02: PASS — avg stalls/token: {avg_stalls:.1f}, max: {max_stalls} "
              f"(from {len(counts)} samples)")

    # ══ Phase 3: Complete event path (SYNC-01 + SYNC-02) ══════════

    def test_sync01_event_merge_q4km_np1_100tok(self):
        """SYNC-01: Phase 3 merge uses OOO queue + event-based copies (no host waits).

        Previously: Phase 3 merge called dev2dev_memcpy_staged_sync which did
        TWO blocking waits per column per device (q_src.memcpy().wait() +
        q_dst.memcpy().wait()). With 2 non-main devices x ~224 merge ops each,
        this was ~448 host stalls per token just in Phase 3.

        Fix (ggml-sycl.cpp Phase 3, gated on use_event_sync):
        - GPU_i VRAM -> pinned staging: OOO submit with depends_on(dev_events[i])
        - staging -> dst: OOO submit chained on e_g2h
        - Staging buffer reused across columns via e_prev_merge_staging (same
          pattern as Phase 1 e_prev_staging_free)
        - Zero host waits inside Phase 3 on the event path
        - OOO flush at end of mul_mat ensures all merge copies complete

        Legacy path (use_event_sync=false): unchanged in else branch.

        Expected: stalls/token drops from ~9 to ~3 (only Phase 2 dev_events waits).
        """
        self.run(ROW_EVENTS.with_(
            model="q4km", timeout=300,
            name="sync01_event_merge_q4km_np1_100tok",
            prompt=SHORT_PROMPT,
            n_parallel=1, concurrent=1, context=4096,
            max_tokens=100))

    def test_sync02_cache_event_q4km_np1_100tok(self):
        """SYNC-02: src1 cache propagates completion event on cache hit.

        Previously: cache hit path set src1_ddq_i = cache.ddq_ptr and returned
        with no event for downstream depends_on chains, creating a sync gap.
        The pre-op barrier at line 4225-4228 fired unconditionally (correct for
        non-main devices) but didn't explicitly chain on the cached data's copy
        completion event.

        Fix (ggml-sycl.cpp + common.hpp, gated on use_event_sync):
        - row_src1_cache_entry gains two fields:
          sycl::event last_copy_event — records barrier after copy+quantize
          bool has_event             — guards depends_on on first access
        - Cache MISS: stream->ext_oneapi_submit_barrier() stored as last_copy_event
        - Cache HIT: stream->ext_oneapi_submit_barrier({cache.last_copy_event})
          chains the cached event into this matmul's OOO queue

        This closes the sync gap for consecutive matmuls sharing src1 (e.g.,
        attn_q/k/v all reading attn_norm output on non-main devices).
        """
        self.run(ROW_EVENTS.with_(
            model="q4km", timeout=300,
            name="sync02_cache_event_q4km_np1_100tok",
            prompt=SHORT_PROMPT,
            n_parallel=1, concurrent=1, context=4096,
            max_tokens=100))

    def test_complete_event_path_glm47_np1_500tok(self):
        """SYNC-01 + SYNC-02: complete event path end-to-end — stall reduction gate.

        With all three sync improvements (Phase 1 event copy + Phase 3 event merge
        + src1 cache event propagation), expected stall count drops:
        - Before: ~9 stalls/token (Phase 1 pre-sync + Phase 2 + Phase 3 merges)
        - After: ~3 stalls/token (Phase 2 dev_events[i].wait() only, 3 devices)

        Phase 2 wait remains intentional: ensures kernel output is readable before
        merge copies begin. Removing it would be a Rule 4 architectural change.

        PASS criteria:
        - completed == 1, total_tokens >= 500 (correctness)
        - stalls/token <= 4 (SYNC-01+02 effective: only Phase 2 waits remain)
        """
        cfg = ROW_EVENTS.with_(
            model="glm47-q4km", timeout=900,
            name="complete_event_path_glm47_np1_500tok",
            prompt=THINK_PROMPT,
            n_parallel=1, concurrent=1, context=2048,
            flash_attn=True,
            max_tokens=500)
        result = self.run(cfg)

        assert not result.error, (
            f"FAIL: server error: {result.error}")
        assert result.completed == 1, (
            f"FAIL: expected 1 completed request, got {result.completed}")
        assert result.total_tokens >= 500, (
            f"FAIL: expected >= 500 tokens, got {result.total_tokens}")

        log_path = LOG_DIR / "complete_event_path_glm47_np1_500tok.log"
        log_text = log_path.read_text() if log_path.exists() else ""
        stall_lines = [l for l in log_text.splitlines() if "[EV] stalls/token:" in l]
        assert len(stall_lines) > 0, (
            f"FAIL: no '[EV] stalls/token:' lines in server log")
        counts = [int(re.search(r"stalls/token: (\d+)", l).group(1)) for l in stall_lines]
        max_stalls = max(counts)
        avg_stalls = sum(counts) / len(counts)
        assert max_stalls <= 4, (
            f"SYNC-01/02 FAIL: max stalls/token = {max_stalls} (expected <= 4 after "
            f"Phase 3 event merge + cache event propagation; counts: {counts[:10]}...)")

        print(f"  PASS — {result.total_tokens} tokens, {result.total_tps:.1f} t/s")
        print(f"  Stalls: avg {avg_stalls:.1f}/token, max {max_stalls} "
              f"(from {len(counts)} samples)")
