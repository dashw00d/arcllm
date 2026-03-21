# Multi-Queue Concurrent Kernel Execution

- **Source:** https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/executing-multiple-kernels-on-the-device-at-the.html
- **Fetched:** 2026-03-18

## Queue Types

### In-Order Queue
```cpp
sycl::property_list q_prop{sycl::property::queue::in_order()};
sycl::queue q1(d_selector, q_prop);
```

Kernels execute in submission order. No parallel execution within the queue.

### Out-of-Order Queue (default)
```cpp
sycl::queue q2(d_selector);  // OOO by default
```

Kernels can execute in arbitrary order (subject to dependency constraints). Enables concurrent execution.

## Performance: Sequential vs Concurrent

- In-order queue: "larger total execution time for all the kernels"
- Out-of-order queue: "the overall execution time is much lower, indicating that the machine is able to execute different kernels from the queue at the same time"

Profiling with **onetrace** confirms "kernels submitted to the out-of-order queue are being executed in parallel."

## Concurrency Constraints

> "Not all three kernels are executed in parallel all the time. How many kernels are executed in parallel is affected by multiple factors such as the availability of hardware resources, the time gap between kernel submissions, etc."

Factors:
- Available hardware resources (EUs, SLM)
- Time gap between kernel submissions
- Work-group sizing and resource usage per kernel
- Hardware copy engine availability

## Sub-Device Partitioning Alternative

Using `create_sub_devices` provides explicit control over kernel-to-resource assignment, but sacrifices runtime scheduling flexibility.

## Best Practice

> "In situations where kernels do not scale strongly and therefore cannot effectively utilize full machine compute resources, it is better to allocate only the required compute units through appropriate selection of work-group/work-item values and try to execute multiple kernels at the same time."

Strategic work-group sizing is essential for achieving parallelism benefits.

## Relevance to llm-stack

The row-split event-based sync uses OOO queues (`sycl::queue` without `in_order` property) for the event dispatch path. This allows:
1. Copy + quantize + kernel submissions to overlap across devices
2. `depends_on()` chains to express dependencies without host waits
3. Multiple matmul operations to potentially overlap on the same device

**Key discovery from project:** Level Zero does NOT reliably honor in-order queue semantics for kernel execution ordering. Kernels submitted to an in-order queue may execute concurrently when the L0 runtime determines they have no data dependencies. This causes data corruption in llama.cpp's SYCL backend (see test_q8_1_corruption.py). The fix is explicit event-based synchronization or periodic `stream->wait()`.
