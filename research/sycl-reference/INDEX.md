# SYCL / Level Zero Reference Index

Reference documentation for Intel Arc A770 SYCL/Level Zero development on llm-stack.

## External Documentation

| Document | Project Use Case |
|----------|-----------------|
| [l0-events-sync.md](l0-events-sync.md) | Event pool sizing, cross-device event signaling, event lifecycle API — needed for row-split OOO queue + event-based sync implementation |
| [multi-card-sycl.md](multi-card-sycl.md) | USM allocation scoping, queue creation patterns, cross-device memory — architecture guide for 3x A770 multi-GPU setup |
| [async-data-transfer.md](async-data-transfer.md) | Overlapping H2D transfers with compute, event chaining patterns — pattern reference for 3-phase dispatch loop in row-split |
| [multi-queue-kernels.md](multi-queue-kernels.md) | In-order vs OOO queue benchmarks, concurrent kernel execution — context for L0 in-order queue overlap bug |
| [l0-backend-guide.md](l0-backend-guide.md) | All SYCL_PI_LEVEL_ZERO_* environment variables, command list config, copy engine config — configuration reference for env.sglang-xpu.sh tuning |

## Project-Derived References

| Document | Project Use Case |
|----------|-----------------|
| [cuda-to-sycl-mapping.md](cuda-to-sycl-mapping.md) | CUDA-to-SYCL primitive translation table — reference when porting CUDA patterns (Marlin, CUTLASS) to SYCL |
| [known-failure-modes.md](known-failure-modes.md) | Catalog of all discovered failure modes with root causes, symptoms, and workarounds — prevent re-investigation of known issues |

## Quick Links by Topic

### Event-Based Sync (row-split)
- Event API: [l0-events-sync.md](l0-events-sync.md)
- Event chaining pattern: [async-data-transfer.md](async-data-transfer.md)
- Batched cmdlist deadlock: [known-failure-modes.md](known-failure-modes.md#6-event-based-sync-deadlock-with-batched-command-lists)
- L0 in-order queue overlap: [known-failure-modes.md](known-failure-modes.md#1-level-zero-in-order-queue-overlap-critical)

### Multi-GPU Configuration
- Queue patterns: [multi-card-sycl.md](multi-card-sycl.md)
- Environment variables: [l0-backend-guide.md](l0-backend-guide.md)
- Cross-device memory: [multi-card-sycl.md](multi-card-sycl.md#cross-device-memory)

### Kernel Development
- CUDA to SYCL mapping: [cuda-to-sycl-mapping.md](cuda-to-sycl-mapping.md)
- XMX / joint_matrix limitations: [known-failure-modes.md](known-failure-modes.md#9-xmx-joint_matrix-compiler-crash)
- Concurrent kernel execution: [multi-queue-kernels.md](multi-queue-kernels.md)
