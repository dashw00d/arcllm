# Level Zero Events & Synchronization Reference

- **Source:** https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html + api.html
- **Fetched:** 2026-03-18

## Event Pool

### ze_event_pool_desc_t

```c
typedef struct ze_event_pool_desc_t {
    ze_structure_type_t stype;   // ZE_STRUCTURE_TYPE_EVENT_POOL_DESC
    const void* pNext;
    ze_event_pool_flags_t flags;
    uint32_t count;              // number of events in pool (indices 0..count-1)
} ze_event_pool_desc_t;
```

### ze_event_pool_flag_t

| Flag | Meaning |
|------|---------|
| `ZE_EVENT_POOL_FLAG_HOST_VISIBLE` | All events in pool are visible to host (enables host sync/query) |
| `ZE_EVENT_POOL_FLAG_IPC` | Pool can be shared across processes |
| `ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP` | Events record kernel start/end timestamps |
| `ZE_EVENT_POOL_FLAG_KERNEL_MAPPED_TIMESTAMP` | Kernel-mapped timestamps |

### zeEventPoolCreate

```c
ze_result_t zeEventPoolCreate(
    ze_context_handle_t hContext,        // context owning the pool
    const ze_event_pool_desc_t* desc,    // pool descriptor (count + flags)
    uint32_t numDevices,                 // number of devices (0 = host-only)
    ze_device_handle_t* phDevices,       // device handles for multi-device pools
    ze_event_pool_handle_t* phEventPool  // [out] pool handle
);
```

Multi-device pools: pass multiple device handles to enable cross-device event signaling.

### zeEventPoolDestroy

```c
ze_result_t zeEventPoolDestroy(ze_event_pool_handle_t hEventPool);
```

Destroys pool and invalidates ALL contained events. Must not be called while events are in use.

## Event Descriptor

### ze_event_desc_t

```c
typedef struct ze_event_desc_t {
    ze_structure_type_t stype;     // ZE_STRUCTURE_TYPE_EVENT_DESC
    const void* pNext;             // can chain ze_event_sync_mode_desc_t
    uint32_t index;                // index within pool (0 to pool.count - 1)
    ze_event_scope_flags_t signal; // coherency scope on signal
    ze_event_scope_flags_t wait;   // coherency scope on wait
} ze_event_desc_t;
```

### ze_event_scope_flag_t

| Flag | Meaning |
|------|---------|
| `ZE_EVENT_SCOPE_FLAG_SUBDEVICE` | Sub-device scope |
| `ZE_EVENT_SCOPE_FLAG_DEVICE` | Device scope: flush device caches on signal |
| `ZE_EVENT_SCOPE_FLAG_HOST` | Host scope: memory visible to host after signal |
| (none) | Execution-only semantics, no memory coherency guarantee |

## Event Lifecycle

### Create

```c
ze_result_t zeEventCreate(
    ze_event_pool_handle_t hEventPool,
    const ze_event_desc_t* desc,
    ze_event_handle_t* phEvent
);
```

### Signal from Command List

```c
ze_result_t zeCommandListAppendSignalEvent(
    ze_command_list_handle_t hCommandList,
    ze_event_handle_t hEvent
);
```

Signals event upon completion of all prior commands in the command list.

### Wait in Command List

```c
ze_result_t zeCommandListAppendWaitOnEvents(
    ze_command_list_handle_t hCommandList,
    uint32_t numEvents,
    ze_event_handle_t* phEvents
);
```

Stalls command list execution until ALL specified events are signaled.

### Host Signal

```c
ze_result_t zeEventHostSignal(ze_event_handle_t hEvent);
```

Allows host thread to trigger device-waiting operations.

### Host Wait (Blocking)

```c
ze_result_t zeEventHostSynchronize(
    ze_event_handle_t hEvent,
    uint64_t timeout    // nanoseconds; UINT64_MAX = infinite
);
```

Returns `ZE_RESULT_SUCCESS` when signaled, `ZE_RESULT_NOT_READY` on timeout.

### Poll (Non-blocking)

```c
ze_result_t zeEventQueryStatus(ze_event_handle_t hEvent);
```

Returns `ZE_RESULT_SUCCESS` if signaled, `ZE_RESULT_NOT_READY` if pending.

### Reset

```c
ze_result_t zeEventHostReset(ze_event_handle_t hEvent);
```

**Events do NOT implicitly reset.** Must explicitly reset before re-signaling.

Device-side reset:

```c
ze_result_t zeCommandListAppendEventReset(
    ze_command_list_handle_t hCommandList,
    ze_event_handle_t hEvent
);
```

### Destroy

```c
ze_result_t zeEventDestroy(ze_event_handle_t hEvent);
```

### Kernel Timestamps

```c
ze_result_t zeEventQueryKernelTimestamp(
    ze_event_handle_t hEvent,
    ze_kernel_timestamp_result_t* pResult
);
```

Contains `kernelStart` and `kernelEnd` (context + global). **Counters are 32-bit -- application must handle wrapping.**

## Barriers

```c
ze_result_t zeCommandListAppendBarrier(
    ze_command_list_handle_t hCommandList,
    ze_event_handle_t hSignalEvent,       // optional: signal on completion
    uint32_t numWaitEvents,               // optional: wait before barrier
    ze_event_handle_t* phWaitEvents
);

ze_result_t zeCommandListAppendMemoryRangesBarrier(
    ze_command_list_handle_t hCommandList,
    uint32_t numRanges,
    const size_t* pRangeSizes,
    const void** pRanges,
    ze_event_handle_t hSignalEvent,
    uint32_t numWaitEvents,
    ze_event_handle_t* phWaitEvents
);

ze_result_t zeContextSystemBarrier(
    ze_context_handle_t hContext,
    ze_device_handle_t hDevice
);
```

## Fences vs Events

| Property | Fence | Event |
|----------|-------|-------|
| Scope | Host-only wait, tied to single queue | Device-to-device, cross-process |
| Granularity | Command list completion | Fine-grained per-operation |
| Reusable | Yes (reset) | Yes (explicit reset required) |
| IPC | No | Yes (with ZE_EVENT_POOL_FLAG_IPC) |
| Use case | Coarse completion tracking | Fine-grained pipeline control |

## Counter-Based Events (CB Events)

Alternative to pool-based events. No pool allocation needed.

```c
ze_result_t zeEventCounterBasedCreate(
    ze_context_handle_t hContext,
    ze_device_handle_t hDevice,
    const ze_event_counter_based_desc_t* desc,
    ze_event_handle_t* phEvent
);
```

Properties:
- Initially marked as completed after creation
- Counter state updates implicitly on append calls
- No need to wait for completion before reusing/destroying
- Supports IPC (one-directional: opened events support waiting only)

### ze_event_counter_based_flag_t

| Flag | Meaning |
|------|---------|
| `ZE_EVENT_COUNTER_BASED_FLAG_IMMEDIATE` | For immediate command lists |
| `ZE_EVENT_COUNTER_BASED_FLAG_NON_IMMEDIATE` | For regular command lists |
| `ZE_EVENT_COUNTER_BASED_FLAG_HOST_VISIBLE` | Host can observe |
| `ZE_EVENT_COUNTER_BASED_FLAG_IPC` | IPC-shareable |
| `ZE_EVENT_COUNTER_BASED_FLAG_DEVICE_TIMESTAMP` | Device timestamps |
| `ZE_EVENT_COUNTER_BASED_FLAG_HOST_TIMESTAMP` | Host timestamps |

## Event Sync Modes

Set via `ze_event_sync_mode_desc_t` in `pNext` of `ze_event_desc_t`:

| Mode | Behavior |
|------|----------|
| `ZE_EVENT_SYNC_MODE_FLAG_LOW_POWER_WAIT` | OS sleep instead of active polling |
| `ZE_EVENT_SYNC_MODE_FLAG_SIGNAL_INTERRUPT` | GPU programs interrupt on signal |
| `ZE_EVENT_SYNC_MODE_FLAG_EXTERNAL_INTERRUPT_WAIT` | OS interrupt for completion (CB events only) |

## Usage Pattern

```c
// Create pool
ze_event_pool_desc_t poolDesc = {
    ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
    ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
    1  // one event
};
ze_event_pool_handle_t hPool;
zeEventPoolCreate(hContext, &poolDesc, 0, nullptr, &hPool);

// Create event with device scope
ze_event_desc_t eventDesc = {
    ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr,
    0,  // index
    ZE_EVENT_SCOPE_FLAG_DEVICE,  // signal scope
    0   // wait scope
};
ze_event_handle_t hEvent;
zeEventCreate(hPool, &eventDesc, &hEvent);

// Signal after kernel
zeCommandListAppendLaunchKernel(hCmdList, hKernel, &args, hEvent, 0, nullptr);

// Wait on host
zeEventHostSynchronize(hEvent, UINT64_MAX);

// Reset for reuse
zeEventHostReset(hEvent);
```

## Error Codes

| Code | Meaning |
|------|---------|
| `ZE_RESULT_SUCCESS` | Operation completed |
| `ZE_RESULT_NOT_READY` | Event not yet signaled |
| `ZE_RESULT_ERROR_INVALID_ARGUMENT` | Invalid event index or pool exhausted |
| `ZE_RESULT_ERROR_INVALID_NULL_HANDLE` | Invalid handle |
| `ZE_RESULT_ERROR_DEVICE_LOST` | Device reset invalidates all events in context |

## Critical Warning

> "There are no protections against events causing deadlocks, such as circular wait scenarios. These problems are left to the application to avoid."

## Cross-Device Events

An event can be shared across devices and processes. Create the event pool with multiple device handles in `zeEventPoolCreate` to enable multi-device synchronization. Host-visible events enable inter-device dependency tracking.

## IPC Event Pool Sharing

```c
ze_result_t zeEventPoolGetIpcHandle(
    ze_event_pool_handle_t hEventPool,
    ze_ipc_event_pool_handle_t* phIpc);

ze_result_t zeEventPoolOpenIpcHandle(
    ze_context_handle_t hContext,
    ze_ipc_event_pool_handle_t hIpc,
    ze_event_pool_handle_t* phEventPool);

ze_result_t zeEventPoolCloseIpcHandle(
    ze_event_pool_handle_t hEventPool);
```
