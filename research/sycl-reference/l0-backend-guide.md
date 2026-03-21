# Level Zero Backend Environment Variables & Configuration

- **Source:** https://intel.github.io/llvm/EnvironmentVariables.html
- **Fetched:** 2026-03-18

## Command List Configuration

### SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS

| Value | Behavior |
|-------|----------|
| 0 | Batched command lists (default on Arc/non-PVC) |
| 1 | Unique immediate command list per SYCL queue |
| 2 | Unique immediate command list per host thread per queue |

**Default:** 1 on Intel Data Center GPU Max Series (PVC) / Linux; 0 elsewhere (including Arc A770).

With immediate command lists, all commands are immediately submitted to the device. With batched mode, commands are accumulated and submitted in batches.

**llm-stack finding:** Batched mode (0) gives +7.5% throughput over immediate (1) for normal inference. BUT event-based cross-queue `depends_on()` DEADLOCKS with batched mode because unflushed command batches never signal their events. Event-based sync REQUIRES immediate mode.

### SYCL_PI_LEVEL_ZERO_BATCH_SIZE
- **Default:** 0 (dynamic adjustment)
- Controls number of compute commands batched before execution

### SYCL_PI_LEVEL_ZERO_COPY_BATCH_SIZE
- **Default:** 0 (dynamic adjustment)
- Controls number of copy commands batched before execution

## Memory / USM Configuration

### SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR
- **Default:** `EnableBuffers=1, MaxPoolSize=unlimited`
- **Format:** `1;32M;host:1M,4,64K;device:1M,4,64K`
- Controls pooling for SYCL buffers and USM memory allocations

### SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR
- Set any non-null value to disable USM allocator pooling
- All memory requests bypass pooling

### SYCL_PI_LEVEL_ZERO_USM_RESIDENT
- **Default:** 0x002 (forces residency for device allocations)
- **Format:** Bit-mask (0xHSD)
- Controls if/where to make USM allocations resident at allocation time

### SYCL_PI_LEVEL_ZERO_SINGLE_ROOT_DEVICE_BUFFER_MIGRATION
- **Default:** 1
- Set to 0 to use single root-device allocation for all context devices

## Copy Engine Configuration

### SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE
- **Default:** `0:0` with immediate commandlists; `1` otherwise
- **Format:** Integer or `lower_index:upper_index`
- Enables use of copy engines for data transfer and fill operations

### SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY
- **Default:** 0
- **Status:** Experimental
- Enables copy engine for device-to-device transfers

### SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_FILL
- **Default:** 0
- Enables copy engine for memory fill operations

### SYCL_PI_LEVEL_ZERO_USE_COMPUTE_ENGINE
- **Default:** 0
- Routes compute commands to specified engine index; negative allows all engines

## Event & Synchronization Configuration

### SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS
- **Default:** 0 (host-visible events)
- Mode 1: device-scope events with host proxies
- Mode 2: adds proxy at submission end
- **Ignored with immediate commandlists**

### SYCL_PI_LEVEL_ZERO_REUSE_DISCARDED_EVENTS
- **Default:** 1
- Enables reuse of discarded events in in-order queues based on dependency chains

### SYCL_PI_LEVEL_ZERO_FILTER_EVENT_WAIT_LIST
- **Default:** 0
- Controls filtering of signaled events from wait lists

## Command List Management

### SYCL_PI_LEVEL_ZERO_USE_MULTIPLE_COMMANDLIST_BARRIERS
- **Default:** 1
- Uses multiple commandlists when submitting barriers

### SYCL_PI_LEVEL_ZERO_COMMANDLISTS_CLEANUP_THRESHOLD
- **Default:** 20
- Triggers cleanup of completed command lists when threshold exceeded

### SYCL_PI_LEVEL_ZERO_IMMEDIATE_COMMANDLISTS_EVENT_CLEANUP_THRESHOLD
- **Default:** 1000
- Triggers signaled event recycling when threshold exceeded

## Threading

### SYCL_PI_LEVEL_ZERO_SINGLE_THREAD_MODE
- **Default:** 0
- Single-threaded apps can enable (>0) to avoid mutex locking overhead

## Device Hierarchy

### ZE_FLAT_DEVICE_HIERARCHY
Controls how Level Zero exposes root devices to SYCL. Affects sub-device/tile exposure.

### ZE_AFFINITY_MASK
Controls which devices/sub-devices are visible. Syntax: comma-separated device indices.

### ONEAPI_DEVICE_SELECTOR
- **Format:** `level_zero:*` or `level_zero:0,1,2`
- Supports `level_zero:*.*` for sub-device partitioning
- Works with ZE_FLAT_DEVICE_HIERARCHY

## V2 Adapter (Xe2+ GPUs only)

### SYCL_UR_USE_LEVEL_ZERO_V2
- **Default:** Auto-enabled on Xe2+ GPUs (Battlemage, Lunar Lake, Arrow Lake)
- Redesigned adapter for optimized immediate/batched queue modes
- **Not applicable to Arc A770 (Alchemist architecture)**

### UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD
- V2 only, default 0
- Prevents copy offload to dedicated engines

### UR_L0_V2_FORCE_BATCHED
- V2 only
- Forces batched submission mode

## Deprecated Variables

| Variable | Status |
|----------|--------|
| `SYCL_PI_LEVEL_ZERO_EXPOSE_CSLICE_IN_AFFINITY_PARTITIONING` | Immediately deprecated |
| `SYCL_ENABLE_PCI` | Immediately deprecated |

## llm-stack Active Configuration

From `env.sglang-xpu.sh`:
```bash
SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0  # batched: +7.5% throughput
GGML_SYCL_DISABLE_GRAPH=1                         # safer; graph=0 untested at long sequences
GGML_SYCL_ROW_EVENTS=1                            # OOO queue + event-based sync for row-split
```

When `ROW_EVENTS=1` is active, `IMMEDIATE_COMMANDLISTS` must be 1 (overrides the default 0).
