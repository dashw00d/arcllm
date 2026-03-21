# Multi-Card SYCL Programming with Level Zero

- **Source:** https://intel.github.io/llvm/MultiTileCardWithLevelZero.html
- **Fetched:** 2026-03-18

## Device Discovery

### Root Devices
Intel GPUs appear as SYCL GPU devices (root-devices). Multiple physical GPUs appear as separate root-devices on Linux (same platform).

Filter: `ONEAPI_DEVICE_SELECTOR=level_zero:*` or `sycl::ext::oneapi::filter_selector("level_zero")`.

Emulate multiple root devices: `CreateMultipleRootDevices=N NEOReadDebugKeys=1`

### Sub-Devices (Tiles)
Multi-tile hardware can be partitioned from root-devices:

```cpp
vector<device> SubDevices = RootDevice.create_sub_devices<
    sycl::info::partition_property::partition_by_affinity_domain>(
    sycl::info::partition_affinity_domain::next_partitionable);
```

Only `partition_by_affinity_domain` is supported for Intel GPUs. Both `next_partitionable` and `numa` are functionally equivalent.

`ZE_AFFINITY_MASK` controls which sub-devices Level Zero UMD exposes.

Emulate: `CreateMultipleSubDevices=N NEOReadDebugKeys=1`

## Context Configurations

A SYCL context may contain multiple devices (root or sub-devices from the same platform).

**Key constraint:** A SYCL program (kernel_bundle) created against a context with multiple devices will be built to each of the root-devices in the context. For sub-devices of the same root-device, only a single build is needed.

## USM Memory Allocation

### malloc_device
- Device-exclusive access, fastest execution
- Requires explicit copies for host/other-device synchronization
- **Memory allocated against a root-device is accessible by all of its sub-devices (tiles)**

### malloc_host
- Host and all context devices access via PCI
- No explicit copies needed
- Slower due to PCI access

### malloc_shared
- Host and single specified device
- Driver-managed migration
- Explicit copies needed for other context devices

**Critical note:** For multi-tile same-root contexts, use `malloc_device` on the root-device instead of slower `malloc_host`. Root-device memory is accessible by all sub-devices.

## Queue Creation Patterns

### Pattern A: Single sub-device context (best per-tile performance)

```cpp
vector<device> SubDevices = ...;
for (auto &D : SubDevices) {
    auto Q = queue(D);
    Q.submit([&](handler& cgh) { ... });
}
```

Best per-tile performance; no cross-queue data sharing.

### Pattern B: Multi-tile (multiple sub-devices, same root)

```cpp
vector<device> SubDevices = ...;
auto C = context(SubDevices);
for (auto &D : SubDevices) {
    auto Q = queue(C, D);
    Q.submit([&](handler& cgh) { ... });
}
```

Explicit scaling. **Avoid adding root-device to context** -- creates implicit scaling conflict.

### Pattern C: Single root-device context (implicit scaling)

```cpp
auto Q = queue(gpu_selector_v);
Q.submit([&](handler& cgh) { ... });
```

Driver handles multi-tile distribution. Simplest but no tile-level targeting.

### Pattern D: Multi-card (multiple root-devices)

```cpp
auto P = platform(gpu_selector_v);
auto RootDevices = P.get_devices();
auto C = context(RootDevices);
for (auto &D : RootDevices) {
    auto Q = queue(C, D);
    Q.submit([&](handler& cgh) { ... });
}
```

Maximum flexibility. **Requires explicit copying for cross-card data.**

## Cross-Device Memory

- No automatic migration between different root-devices
- `malloc_shared` only works within host-device pairs on same root
- Multi-card data sharing requires host-mediated copies or `malloc_host` (PCI penalty)
- No P2P (peer-to-peer) DMA between discrete GPUs on current Intel platforms

## Buffer Allocation Behavior

Mapping depends on context composition:
- **Integrated devices:** allocations on host, accessible without copying
- **Sub-devices of same root:** allocated on root-device, accessible by all sub-devices, implicit map/unmap copies
- **Multi-root-device contexts:** allocated on host for universal accessibility

## Key Constraints for llm-stack (3x Arc A770)

1. Three separate root-devices (3 physical cards), no tiles
2. No P2P -- all cross-GPU data goes through host RAM
3. GPU2 is PCIe 3.0 x8 (half bandwidth of GPU0/1)
4. Must use Pattern D (multi-card) with explicit host-mediated copies
5. `malloc_device` per GPU for weights; host staging buffers for cross-GPU communication
