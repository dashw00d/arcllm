# CUDA to SYCL/Level Zero Mapping

- **Created:** 2026-03-18
- **Purpose:** Quick reference for translating CUDA patterns to SYCL equivalents, focused on llm-stack use cases

## Core Primitives

| CUDA | SYCL | Level Zero | Notes |
|------|------|------------|-------|
| `cudaStream_t` | `sycl::queue` | `ze_command_queue_handle_t` | SYCL queues are in-order by default with `sycl::property::queue::in_order()` |
| `cudaEvent_t` | `sycl::event` | `ze_event_handle_t` | Returned by queue operations; used with `depends_on()` |
| `cudaStreamCreate()` | `sycl::queue q(dev)` | `zeCommandQueueCreate()` | |
| `cudaStreamSynchronize()` | `q.wait()` | `zeCommandQueueSynchronize()` | Host-blocking |
| `cudaEventCreate()` | (implicit) | `zeEventCreate()` | SYCL events auto-created by queue ops |
| `cudaEventRecord(event, stream)` | `auto e = q.submit(...)` | `zeCommandListAppendSignalEvent()` | |
| `cudaStreamWaitEvent(stream, event)` | `h.depends_on(e)` | `zeCommandListAppendWaitOnEvents()` | Device-side wait |
| `cudaEventSynchronize(event)` | `e.wait()` | `zeEventHostSynchronize()` | Host-blocking |
| `cudaEventQuery(event)` | `e.get_info<...>()` | `zeEventQueryStatus()` | Non-blocking poll |

## Memory Operations

| CUDA | SYCL | Notes |
|------|------|-------|
| `cudaMalloc()` | `sycl::malloc_device()` | Device-only memory |
| `cudaMallocHost()` | `sycl::malloc_host()` | Pinned host memory for DMA |
| `cudaMallocManaged()` | `sycl::malloc_shared()` | Driver-managed migration |
| `cudaMemcpy()` | `q.memcpy().wait()` | Blocking copy |
| `cudaMemcpyAsync()` | `q.memcpy()` | Returns event; no host block |
| `cudaFree()` | `sycl::free(ptr, ctx)` | |
| `cudaMemset()` | `q.fill()` | Returns event |
| `cudaMemcpy2D()` | `q.ext_oneapi_memcpy2d()` | Requires `SYCL_PI_LEVEL_ZERO_USE_NATIVE_USM_MEMCPY2D=1` |

## Synchronization Patterns

| CUDA Pattern | SYCL Equivalent |
|--------------|-----------------|
| `cudaDeviceSynchronize()` | `q.wait()` (per queue) or `ctx.wait()` (all queues) |
| `__syncthreads()` | `item.barrier(sycl::access::fence_space::local_space)` |
| `cudaStreamWaitEvent(s2, e1)` | `q2.submit([&](auto& h){ h.depends_on(e1); ... })` |
| `cudaEventRecord + cudaStreamWaitEvent` | `auto e = q1.ext_oneapi_submit_barrier(); q2.submit([&](h){ h.depends_on(e); ... })` |

## Kernel Launch

| CUDA | SYCL | Notes |
|------|------|-------|
| `kernel<<<grid, block>>>()` | `q.parallel_for(nd_range{grid*block, block}, ...)` | |
| `blockIdx.x` | `item.get_group(0)` | |
| `threadIdx.x` | `item.get_local_id(0)` | |
| `blockDim.x` | `item.get_local_range(0)` | |
| `gridDim.x` | `item.get_group_range(0)` | |
| `__shared__` | `sycl::local_accessor` | SLM: 64KB on Arc A770 |
| `warpSize` | Sub-group size | 8, 16, or 32 on Arc A770; WARP_SIZE=16 in llama.cpp |

## Warp/Sub-group Operations

| CUDA | SYCL | Notes |
|------|------|-------|
| `__shfl_down_sync()` | `sg.shuffle_down()` | sg = sub-group |
| `__shfl_xor_sync()` | `sg.shuffle_xor()` | |
| `__ballot_sync()` | `sycl::ext::oneapi::group_ballot()` | |
| `__reduce_add_sync()` | `sycl::reduce_over_group(sg, val, plus<>())` | |
| `atomicAdd()` | `sycl::atomic_ref<...>(...).fetch_add()` | |

## Multi-GPU Patterns

| CUDA | SYCL | Notes |
|------|------|-------|
| `cudaSetDevice(i)` | Create queue per device: `q[i] = queue(devices[i])` | No global device state |
| `cudaMemcpyPeer()` | Host-staged copy (no P2P on Intel) | `q_src.memcpy(host_buf, dev_src); q_dst.memcpy(dev_dst, host_buf)` |
| `cudaEventRecord(e, s1); cudaStreamWaitEvent(s2, e)` | `auto e = q1.ext_oneapi_submit_barrier(); q2.submit([](h){ h.depends_on(e); })` | Cross-device event sync |
| `cudaDeviceEnablePeerAccess()` | Not available | Intel Arc has no P2P DMA |
| `cudaStreamCreateWithFlags(CUDA_STREAM_NON_BLOCKING)` | `sycl::queue q(dev)` | SYCL OOO queues are non-blocking by default |

## XMX / Tensor Core Mapping

| CUDA | SYCL | Notes |
|------|------|-------|
| `wmma::mma_sync()` | `joint_matrix_mad()` | Intel XMX |
| `wmma::load_matrix_sync()` | `joint_matrix_load()` | Requires sg=8 on Arc A770 |
| `wmma::store_matrix_sync()` | `joint_matrix_store()` | |
| Tensor Core shapes | INT8: 8x8x32, BF16: 8x8x16 | sg=16/32 CRASH on Arc A770 (ICE) |

## dp4a (INT8 Dot Product)

| CUDA | SYCL |
|------|------|
| `__dp4a(a, b, c)` | `sycl::ext::intel::math::dp4a(a, b, c)` |

Available at sub-group sizes 8 and 16 on Arc A770. Used by the fused kernel for Q4_K/Q8_0 matmul.

## Key Behavioral Differences

1. **No implicit synchronization:** CUDA has some implicit sync points (e.g., `cudaMemcpy` is synchronous). SYCL `q.memcpy()` is always async unless you call `.wait()`.

2. **Queue ≠ Stream ordering guarantees:** CUDA streams guarantee in-order execution within a stream. SYCL in-order queues SHOULD guarantee this, but **Level Zero batched command lists can violate this** (see test_q8_1_corruption.py). Use explicit events for safety.

3. **No global device state:** CUDA has `cudaSetDevice()` for implicit device context. SYCL requires explicit device/queue/context management.

4. **Events are lightweight in SYCL:** CUDA events must be explicitly created/destroyed. SYCL events are returned by every queue operation and managed automatically.

5. **No P2P on Intel discrete GPUs:** All cross-GPU communication goes through host memory. Plan for PCIe bandwidth as the bottleneck.
