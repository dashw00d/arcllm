# Asynchronous and Overlapping Data Transfers

- **Source:** https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/asynchronous-and-overlapping-data-transfers.html
- **Fetched:** 2026-03-18

## Core Principle

Overlap host-to-device (H2D) transfers with GPU computation by using separate hardware copy engines and compute engines concurrently. Divide work into independent chunks processed in a pipeline.

## Event-Based Dependency Chaining

```cpp
for (int c = 0; c < num_chunks; c++) {
    // Step 1: copy chunk to device
    auto copy_in = q.memcpy(device_data[c], host_data[c],
                            sizeof(float) * chunk_size);

    // Step 2: compute on chunk (depends on copy_in)
    auto compute = q.parallel_for(chunk_size, copy_in, add_one);

    // Step 3: copy result back (depends on compute)
    auto cg = [=](auto &h) {
        h.depends_on(compute);
        h.memcpy(host_data[c], device_data[c],
                 sizeof(float) * chunk_size);
    };
    auto copy_out = q.submit(cg);
}
q.wait();
```

**Key pattern:** `q.memcpy()` returns an `sycl::event`. Pass that event as a dependency to `q.parallel_for()` or `h.depends_on()`. No host-side blocking between pipeline stages.

## Copy Engine vs Compute Engine

Intel GPUs have dedicated copy engines separate from compute EUs:
- "Between two memory copies, where one is executed by the GPU EUs and one by a copy engine, or both are executed by copy engines"
- "Between a memory copy and a compute kernel, where the memory copy is executed by the copy engine and the compute kernel by the GPU EUs"

This enables TRUE parallelism: H2D transfer on copy engine while previous chunk's kernel runs on compute engine.

## Single Queue Pattern

The example uses a **single queue** with explicit event dependencies (not multiple queues). Overlapping occurs because:
- Copy engines and compute engines are separate hardware
- Event dependencies allow out-of-order execution within a single queue
- Independent chunks from different loop iterations execute concurrently on different hardware units

## Limitation: Concurrent Kernel Execution

> "In the example above, we cannot have two kernels (even though they are independent) executing concurrently because we only have one GPU."

GPU partitioning or out-of-order queues required for concurrent kernel execution (see multi-queue-kernels.md).

## Complete Example

```cpp
#include <CL/sycl.hpp>
#define NITERS 10
#define KERNEL_ITERS 10000
#define NUM_CHUNKS 10
#define CHUNK_SIZE 10000000

int main() {
    const int num_chunks = NUM_CHUNKS;
    const int chunk_size = CHUNK_SIZE;

    sycl::queue q;

    // Allocate host pinned memory
    float *host_data[num_chunks];
    for (int c = 0; c < num_chunks; c++) {
        host_data[c] = sycl::malloc_host<float>(chunk_size, q);
        for (int i = 0; i < chunk_size; i++)
            host_data[c][i] = (float)c;
    }

    // Allocate device memory
    float *device_data[num_chunks];
    for (int c = 0; c < num_chunks; c++) {
        device_data[c] = sycl::malloc_device<float>(chunk_size, q);
        q.fill<float>(device_data[c], 1000.0f, chunk_size);
    }
    q.wait();

    // Pipeline execution with overlap
    for (int it = 0; it < NITERS; it++) {
        for (int c = 0; c < num_chunks; c++) {
            auto add_one = [=](auto id) {
                for (int i = 0; i < KERNEL_ITERS; i++)
                    device_data[c][id] += 1.0;
            };

            auto copy_in = q.memcpy(device_data[c], host_data[c],
                                    sizeof(float) * chunk_size);
            auto compute = q.parallel_for(chunk_size, copy_in, add_one);
            auto cg = [=](auto &h) {
                h.depends_on(compute);
                h.memcpy(host_data[c], device_data[c],
                         sizeof(float) * chunk_size);
            };
            auto copy_out = q.submit(cg);
        }
        q.wait();
    }
}
```

## Best Practices

1. **Keep data in local memory** -- avoid H2D/D2H transfers when possible
2. **Keep intermediate results on device** -- execute all kernels on accelerator even if some are less efficient, to avoid data movement
3. **Chunk work into independent pieces** -- small enough for responsive pipelining, large enough to amortize copy overhead
4. **Use `sycl::malloc_host` for host buffers** -- pinned memory enables async DMA transfers
5. **Chain events, not host waits** -- use `depends_on()` instead of `q.wait()` between pipeline stages

## Relevance to llm-stack

The row-split implementation uses this pattern for the 3-phase dispatch loop:
- Phase 1: submit copies + quantize + kernel to all devices (event-chained, no host waits)
- Phase 2: wait on per-device completion events (minimal host waits)
- Phase 3: merge results from non-main devices to main device

The key insight is that `q.memcpy()` and kernel submission return events that can be chained via `depends_on()`, eliminating host-side `stream->wait()` calls.
