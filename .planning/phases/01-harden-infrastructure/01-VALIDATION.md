---
phase: 1
slug: harden-infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-17
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | python3 -m bench (custom benchmark framework) |
| **Config file** | scripts/bench/config.py |
| **Quick run command** | `python3 -m bench infra` |
| **Full suite command** | `python3 -m bench infra` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python3 -m bench infra`
- **After every plan wave:** Run `python3 -m bench infra`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | INFRA-01 | integration | `python3 -m bench infra.cmdlist_assertion` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | INFRA-02 | integration | `python3 -m bench infra.context_assertion` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `scripts/bench/tests/test_infra.py` — negative-path tests for INFRA-01 (bad env → abort) and INFRA-02 (context check)
- [ ] Subprocess wrapper for expect-abort test pattern (BenchRunner may not support non-zero exit natively)

*Research flagged: BenchRunner "expect non-zero exit" mode may need a simple subprocess wrapper outside normal BenchTest.run() path.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Error message clarity | INFRA-01, INFRA-02 | Subjective readability | Read stderr output, verify message explains what's wrong and how to fix it |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
