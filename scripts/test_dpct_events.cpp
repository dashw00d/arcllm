// Test cross-device events with dpct queues (same as llama.cpp uses)
#include <sycl/sycl.hpp>
#include <cstdio>
#include "dpct/helper.hpp"

int main() {
    int count = dpct::dev_mgr::instance().device_count();
    printf("dpct device count: %d\n", count);
    if (count < 2) { printf("Need 2+ devices\n"); return 1; }

    auto &q0 = dpct::get_device(0).default_queue();
    auto &q1 = dpct::get_device(1).default_queue();

    printf("q0 device: %s\n", q0.get_device().get_info<sycl::info::device::name>().c_str());
    printf("q1 device: %s\n", q1.get_device().get_info<sycl::info::device::name>().c_str());

    // Check if same context
    bool same_ctx = (q0.get_context() == q1.get_context());
    printf("Same SYCL context: %s\n", same_ctx ? "YES" : "NO");

    constexpr int N = 1024;
    float *buf0 = sycl::malloc_device<float>(N, q0);
    float *buf1 = sycl::malloc_device<float>(N, q1);

    q0.fill(buf0, 42.0f, N).wait();

    printf("\n=== Test: ext_oneapi_submit_barrier across dpct queues ===\n");
    q0.fill(buf0, 7.0f, N);
    sycl::event e0 = q0.ext_oneapi_submit_barrier();

    printf("Event recorded on q0, waiting on q1...\n");
    fflush(stdout);

    q1.ext_oneapi_submit_barrier({e0});
    q1.fill(buf1, 3.0f, N);
    q1.wait();

    printf("OK — no hang\n");

    sycl::free(buf0, q0);
    sycl::free(buf1, q1);
    return 0;
}
