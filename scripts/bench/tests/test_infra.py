"""Startup assertion validation for SYCL row-split infrastructure.

## Context

INFRA-01 and INFRA-02 are startup assertions added to the SYCL backend to catch
misconfigured environments before any GPU work is dispatched. Without them, a bad
env produces a silent hang (INFRA-01) or silent correctness failure (INFRA-02)
that can waste 5-20 minutes of debugging.

### INFRA-01: Cmdlist assertion (ggml-sycl.cpp, inside ggml_check_sycl())

ROW_EVENTS=1 requires SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1.
In batched cmdlist mode, command batches are never flushed when another queue
polls for an event — cross-queue depends_on() deadlocks indefinitely.

Assertion placement: immediately after g_ggml_sycl_row_events is read at
line 287. Fires within seconds of server startup (before model loading).

### INFRA-02: Context mismatch assertion (common.hpp, inside ooo_stream())

OOO queue must share the same SYCL context as the in-order stream. Cross-context
event dependencies (handler::depends_on) silently fail on Level Zero — kernels
proceed without waiting for the dependency.

Assertion placement: on first ooo_stream(device) call per device (before first
matmul). Expected to always pass with current dpct construction; exists as a
regression guard for future queue-construction changes.

## Test Approach

test_row_events_no_cmdlist — INFRA-01 negative test:
  Sets ROW_EVENTS=1 with IMMEDIATE_COMMANDLISTS=0.
  Server should abort within seconds with the error message in the log.
  Validates that BenchResult.error == "server failed" and log contains the abort string.

test_row_events_correct_env — INFRA-01/02 positive test:
  Sets ROW_EVENTS=1 with IMMEDIATE_COMMANDLISTS=1 (correct configuration).
  Server should start, run inference, and return completed > 0.
  Also implicitly validates INFRA-02 (OOO queues share context with in-order streams).

## Results

INFRA-01 negative test: [not yet run on hardware]
INFRA-02 positive test: [not yet run on hardware]

## Relevant Files

- llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp — INFRA-01 assertion in ggml_check_sycl()
- llama.cpp/ggml/src/ggml-sycl/common.hpp — INFRA-02 assertion in ooo_stream()
"""
from pathlib import Path

from bench.base import BenchTest
from bench.config import BenchConfig
from bench.runner import LOG_DIR

SHORT_PROMPT = "What is 2+2?"

INFRA_BASE = BenchConfig(
    model="0.6b-q8",
    split_mode="row",
    n_parallel=1,
    concurrent=1,
    context=4096,
    flash_attn=False,
)


class TestInfra(BenchTest):
    """Startup assertion validation for SYCL row-split infrastructure."""

    base = INFRA_BASE

    def test_row_events_no_cmdlist(self):
        """INFRA-01 negative test: ROW_EVENTS=1 without IMMEDIATE_COMMANDLISTS aborts.

        Config: GGML_SYCL_ROW_EVENTS=1, SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
        Expected: server exits non-zero within ~10s; log contains abort message.
        """
        cfg = INFRA_BASE.with_(
            name="infra_row_events_no_cmdlist",
            row_events=True,
            immediate_cmdlists=False,
            max_tokens=1,
            timeout=30,
            prompt=SHORT_PROMPT,
        )
        result = self.run(cfg)

        log_path = LOG_DIR / "infra_row_events_no_cmdlist.log"
        log_text = log_path.read_text() if log_path.exists() else ""

        assert "server failed" in (result.error or ""), (
            f"Expected server abort but got error={result.error!r}; "
            f"completed={result.completed}"
        )
        assert "GGML_SYCL_ROW_EVENTS=1 requires" in log_text, (
            f"Expected INFRA-01 abort message in log but got:\n{log_text[-500:]}"
        )
        print(f"  INFRA-01 abort confirmed: server exited early with correct message")

    def test_row_events_correct_env(self):
        """INFRA-01/02 positive test: ROW_EVENTS=1 with correct env starts cleanly.

        Config: GGML_SYCL_ROW_EVENTS=1, SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
        Expected: server starts, completes inference, no errors.
        Implicitly validates INFRA-02 (OOO queue context matches in-order stream).
        """
        cfg = INFRA_BASE.with_(
            name="infra_row_events_correct_env",
            row_events=True,
            immediate_cmdlists=True,
            max_tokens=10,
            timeout=120,
            prompt=SHORT_PROMPT,
        )
        result = self.run(cfg)

        assert not result.error, (
            f"Expected clean startup but got error={result.error!r}"
        )
        assert result.completed > 0, (
            f"Expected completed > 0 but got completed={result.completed}"
        )
        print(f"  INFRA-01/02 positive path confirmed: {result.completed}/1 ok, "
              f"{result.total_tokens} tok, {result.total_tps} t/s")
