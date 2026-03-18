---
phase: 2
slug: validate-existing-event-path
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-17
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | python3 -m bench (custom benchmark framework) |
| **Config file** | scripts/bench/config.py |
| **Quick run command** | `python3 -m bench validate` |
| **Full suite command** | `python3 -m bench validate` |
| **Estimated runtime** | ~900 seconds (500-token generation at potentially 0.6 t/s) |

---

## Sampling Rate

- **After every task commit:** Run quick compile check
- **After every plan wave:** Run `python3 -m bench validate`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 900 seconds (hardware-bound)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 1 | SYNC-03 | integration | `cmake --build . && python3 -m bench validate.stall_counter` | ❌ W0 | ⬜ pending |
| 2-01-02 | 01 | 1 | CORR-01, CORR-02 | integration | `python3 -m bench validate.500tok_2048ctx` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `scripts/bench/tests/test_validate.py` — bench suite for SYNC-03 stall counter and CORR-01/02 500-token generation
- [ ] Stall counter C++ implementation must be in place before test can run

*Test file created as part of plan execution, not a prerequisite.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 500-token output coherence | CORR-01 | Desync produces garbage text, not detectable by token count alone | Read generated text, verify it's coherent English |
| Stall count interpretation | CORR-02 | "~3 per token" is approximate, actual may be 4 | Check stall counter output, verify it's in the 3-4 range not 4000+ |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 900s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
