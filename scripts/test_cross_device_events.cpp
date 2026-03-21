// Test: can L0 do cross-device event waits via ext_oneapi_submit_barrier?
// Build: icpx -fsycl test_cross_device_events.cpp -o test_cross_device_events
// Run: sg render -c ./test_cross_device_events

#include <sycl/sycl.hpp>
#include <vector>
#include <cstdio>
#include <chrono>

int main() {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    printf("Found %zu GPU devices\n", devices.size());
    if (devices.size() < 2) {
        printf("Need at least 2 GPUs\n");
        return 1;
    }

    // Create in-order queues on device 0 and device 1
    sycl::queue q0(devices[0], sycl::property::queue::in_order());
    sycl::queue q1(devices[1], sycl::property::queue::in_order());

    printf("Device 0: %s\n", devices[0].get_info<sycl::info::device::name>().c_str());
    printf("Device 1: %s\n", devices[1].get_info<sycl::info::device::name>().c_str());

    // Allocate buffers
    constexpr int N = 1024;
    float *buf0 = sycl::malloc_device<float>(N, q0);
    float *buf1 = sycl::malloc_device<float>(N, q1);
    float *host = sycl::malloc_host<float>(N, q0);

    // Initialize on device 0
    q0.fill(buf0, 42.0f, N).wait();
    printf("Initialized device 0 buffer\n");

    // Test 1: Basic cross-device event wait
    printf("\n=== Test 1: Record event on dev0, wait on dev1 ===\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    // Record event on device 0 after some work
    q0.fill(buf0, 7.0f, N);
    sycl::event e0 = q0.ext_oneapi_submit_barrier();

    // Device 1 waits on device 0's event
    q1.ext_oneapi_submit_barrier({e0});
    q1.fill(buf1, 3.0f, N);
    q1.wait();

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Cross-device event wait: %.2f ms — ", ms);
    printf(ms < 1000 ? "OK\n" : "SLOW/HUNG\n");

    // Test 2: Cross-device copy via host staging + event
    printf("\n=== Test 2: GPU0->host->GPU1 with events (no host wait) ===\n");
    t0 = std::chrono::high_resolution_clock::now();

    // GPU0 → host (get event, don't host-wait)
    sycl::event e_to_host = q0.memcpy(host, buf0, N * sizeof(float));

    // GPU1 waits on that event, then host → GPU1
    q1.ext_oneapi_submit_barrier({e_to_host});
    q1.memcpy(buf1, host, N * sizeof(float));
    q1.wait();

    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Async cross-device copy: %.2f ms — ", ms);

    // Verify
    float verify[N];
    q1.memcpy(verify, buf1, N * sizeof(float)).wait();
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (verify[i] != 7.0f) { ok = false; break; }
    }
    printf(ok ? "CORRECT\n" : "WRONG DATA\n");

    // Test 3: Many iterations to check stability
    printf("\n=== Test 3: 1000 iterations of cross-device event sync ===\n");
    t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000; iter++) {
        q0.fill(buf0, (float)iter, N);
        sycl::event e = q0.ext_oneapi_submit_barrier();
        q1.ext_oneapi_submit_barrier({e});
        q1.fill(buf1, (float)(iter + 1), N);
    }
    q0.wait();
    q1.wait();
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("1000 cross-device event syncs: %.2f ms (%.2f us/sync)\n", ms, ms * 1000 / 1000);

    // Test 4: Reverse direction (record on dev1, wait on dev0)
    printf("\n=== Test 4: Record on dev1, wait on dev0 ===\n");
    t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000; iter++) {
        q1.fill(buf1, (float)iter, N);
        sycl::event e = q1.ext_oneapi_submit_barrier();
        q0.ext_oneapi_submit_barrier({e});
        q0.fill(buf0, (float)(iter + 1), N);
    }
    q0.wait();
    q1.wait();
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("1000 reverse event syncs: %.2f ms (%.2f us/sync)\n", ms, ms * 1000 / 1000);

    sycl::free(buf0, q0);
    sycl::free(buf1, q1);
    sycl::free(host, q0);

    printf("\nDone.\n");
    return 0;
}
