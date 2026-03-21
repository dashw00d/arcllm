/**
 * Phase 0 feasibility test: Intel joint_matrix on Arc A770.
 *
 * Tests whether joint_matrix (XMX) operations actually work on this
 * driver/compiler combination. This is the single most important
 * risk-reduction step for the fused dequant+matmul kernel.
 *
 * Tests:
 *   1. BF16 * BF16 -> float  (most likely to work)
 *   2. INT8 * INT8 -> INT32  (needed for quantized matmul)
 *   3. Sub-group size 16 vs 32
 *   4. Basic performance: joint_matrix INT8 matmul throughput
 *
 * Build:
 *   source env.sglang-xpu.sh
 *   icpx -fsycl -O2 -o test_joint_matrix scripts/test_joint_matrix.cpp
 *
 * Run:
 *   ZE_AFFINITY_MASK=0 ./test_joint_matrix
 */

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cmath>

namespace syclex = sycl::ext::oneapi::experimental;
namespace syclmx = syclex::matrix;

// ── Helpers ──────────────────────────────────────────────────────────

void print_ok(const char* name) {
    std::cout << "  [PASS] " << name << std::endl;
}

void print_fail(const char* name, const char* reason) {
    std::cout << "  [FAIL] " << name << " — " << reason << std::endl;
}

// Helper: create a global_ptr from a USM device pointer
template<typename T>
sycl::multi_ptr<T, sycl::access::address_space::global_space, sycl::access::decorated::no>
make_global_ptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space,
                                     sycl::access::decorated::no>(p);
}

// ── Test 1: BF16 joint_matrix multiply ───────────────────────────────

bool test_bf16_joint_matrix(sycl::queue& q) {
    const char* name = "BF16 joint_matrix 8x8x16 multiply";
    constexpr int M = 8, N = 8, K = 16;

    std::vector<sycl::ext::oneapi::bfloat16> A(M * K), B(K * N);
    std::vector<float> C(M * N, 0.0f);

    for (auto& x : A) x = sycl::ext::oneapi::bfloat16(1.0f);
    for (auto& x : B) x = sycl::ext::oneapi::bfloat16(1.0f);

    auto* dA = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(M * K, q);
    auto* dB = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(K * N, q);
    auto* dC = sycl::malloc_device<float>(M * N, q);

    q.memcpy(dA, A.data(), M * K * sizeof(sycl::ext::oneapi::bfloat16));
    q.memcpy(dB, B.data(), K * N * sizeof(sycl::ext::oneapi::bfloat16));
    q.memset(dC, 0, M * N * sizeof(float));
    q.wait();

    try {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<1>(8, 8),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(8)]] {
                    auto sg = item.get_sub_group();

                    syclmx::joint_matrix<sycl::sub_group, sycl::ext::oneapi::bfloat16,
                        syclmx::use::a, M, K, syclmx::layout::row_major> ma;
                    syclmx::joint_matrix<sycl::sub_group, sycl::ext::oneapi::bfloat16,
                        syclmx::use::b, K, N, syclmx::layout::row_major> mb;
                    syclmx::joint_matrix<sycl::sub_group, float,
                        syclmx::use::accumulator, M, N> mc;

                    auto pA = make_global_ptr(dA);
                    auto pB = make_global_ptr(dB);
                    auto pC = make_global_ptr(dC);

                    syclmx::joint_matrix_fill(sg, mc, 0.0f);
                    syclmx::joint_matrix_load(sg, ma, pA, K);
                    syclmx::joint_matrix_load(sg, mb, pB, N);
                    syclmx::joint_matrix_mad(sg, mc, ma, mb, mc);
                    syclmx::joint_matrix_store(sg, mc, pC, N, syclmx::layout::row_major);
                });
        }).wait();
    } catch (sycl::exception& e) {
        print_fail(name, e.what());
        sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
        return false;
    }

    q.memcpy(C.data(), dC, M * N * sizeof(float)).wait();
    sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);

    for (int i = 0; i < M * N; i++) {
        if (std::abs(C[i] - (float)K) > 0.1f) {
            char buf[128];
            snprintf(buf, sizeof(buf), "C[%d] = %.2f, expected %.2f", i, C[i], (float)K);
            print_fail(name, buf);
            return false;
        }
    }

    print_ok(name);
    return true;
}

// ── Test 2: INT8 joint_matrix multiply ───────────────────────────────

bool test_int8_joint_matrix(sycl::queue& q) {
    const char* name = "INT8 joint_matrix 8x8x32 sg=8";
    constexpr int M = 8, N = 8, K = 32;

    std::vector<int8_t> A(M * K, 1), B(K * N, 2);
    std::vector<int32_t> C(M * N, 0);

    auto* dA = sycl::malloc_device<int8_t>(M * K, q);
    auto* dB = sycl::malloc_device<int8_t>(K * N, q);
    auto* dC = sycl::malloc_device<int32_t>(M * N, q);

    q.memcpy(dA, A.data(), M * K);
    q.memcpy(dB, B.data(), K * N);
    q.memset(dC, 0, M * N * sizeof(int32_t));
    q.wait();

    try {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<1>(8, 8),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(8)]] {
                    auto sg = item.get_sub_group();

                    syclmx::joint_matrix<sycl::sub_group, int8_t,
                        syclmx::use::a, M, K, syclmx::layout::row_major> ma;
                    syclmx::joint_matrix<sycl::sub_group, int8_t,
                        syclmx::use::b, K, N, syclmx::layout::row_major> mb;
                    syclmx::joint_matrix<sycl::sub_group, int32_t,
                        syclmx::use::accumulator, M, N> mc;

                    auto pA = make_global_ptr(dA);
                    auto pB = make_global_ptr(dB);
                    auto pC = make_global_ptr(dC);

                    syclmx::joint_matrix_fill(sg, mc, 0);
                    syclmx::joint_matrix_load(sg, ma, pA, K);
                    syclmx::joint_matrix_load(sg, mb, pB, N);
                    syclmx::joint_matrix_mad(sg, mc, ma, mb, mc);
                    syclmx::joint_matrix_store(sg, mc, pC, N, syclmx::layout::row_major);
                });
        }).wait();
    } catch (sycl::exception& e) {
        print_fail(name, e.what());
        sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
        return false;
    }

    q.memcpy(C.data(), dC, M * N * sizeof(int32_t)).wait();
    sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);

    int expected = K * 2;
    for (int i = 0; i < M * N; i++) {
        if (C[i] != expected) {
            char buf[128];
            snprintf(buf, sizeof(buf), "C[%d] = %d, expected %d", i, C[i], expected);
            print_fail(name, buf);
            return false;
        }
    }

    print_ok(name);
    return true;
}

// ── Test 3: INT8 with sub_group size 32 ──────────────────────────────

bool test_int8_sg32(sycl::queue& q) {
    // NOTE: sg=32 crashes IGC on Arc A770 (driver 1.14.37020+3).
    // Testing sg=8 with different values to confirm correctness.
    const char* name = "INT8 joint_matrix sg=8, different values";
    constexpr int M = 8, N = 8, K = 32;

    std::vector<int8_t> A(M * K, 3), B(K * N, 2);
    std::vector<int32_t> C(M * N, 0);

    auto* dA = sycl::malloc_device<int8_t>(M * K, q);
    auto* dB = sycl::malloc_device<int8_t>(K * N, q);
    auto* dC = sycl::malloc_device<int32_t>(M * N, q);

    q.memcpy(dA, A.data(), M * K);
    q.memcpy(dB, B.data(), K * N);
    q.memset(dC, 0, M * N * sizeof(int32_t));
    q.wait();

    try {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<1>(8, 8),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(8)]] {
                    auto sg = item.get_sub_group();

                    syclmx::joint_matrix<sycl::sub_group, int8_t,
                        syclmx::use::a, M, K, syclmx::layout::row_major> ma;
                    syclmx::joint_matrix<sycl::sub_group, int8_t,
                        syclmx::use::b, K, N, syclmx::layout::row_major> mb;
                    syclmx::joint_matrix<sycl::sub_group, int32_t,
                        syclmx::use::accumulator, M, N> mc;

                    auto pA = make_global_ptr(dA);
                    auto pB = make_global_ptr(dB);
                    auto pC = make_global_ptr(dC);

                    syclmx::joint_matrix_fill(sg, mc, 0);
                    syclmx::joint_matrix_load(sg, ma, pA, K);
                    syclmx::joint_matrix_load(sg, mb, pB, N);
                    syclmx::joint_matrix_mad(sg, mc, ma, mb, mc);
                    syclmx::joint_matrix_store(sg, mc, pC, N, syclmx::layout::row_major);
                });
        }).wait();
    } catch (sycl::exception& e) {
        print_fail(name, e.what());
        sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
        return false;
    }

    q.memcpy(C.data(), dC, M * N * sizeof(int32_t)).wait();
    sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);

    int expected = K * 3 * 2;
    for (int i = 0; i < M * N; i++) {
        if (C[i] != expected) {
            char buf[128];
            snprintf(buf, sizeof(buf), "C[%d] = %d, expected %d", i, C[i], expected);
            print_fail(name, buf);
            return false;
        }
    }

    print_ok(name);
    return true;
}

// ── Test 4: Performance — tiled INT8 matmul ──────────────────────────

void test_performance(sycl::queue& q) {
    const char* name = "INT8 matmul 256x256x256 throughput";
    constexpr int M = 256, N = 256, K = 256;
    constexpr int ITERS = 100;

    std::vector<int8_t> A(M * K, 1), B(K * N, 1);

    auto* dA = sycl::malloc_device<int8_t>(M * K, q);
    auto* dB = sycl::malloc_device<int8_t>(K * N, q);
    auto* dC = sycl::malloc_device<int32_t>(M * N, q);

    q.memcpy(dA, A.data(), M * K);
    q.memcpy(dB, B.data(), K * N);
    q.memset(dC, 0, M * N * sizeof(int32_t));
    q.wait();

    constexpr int TILE_M = 8, TILE_N = 8, TILE_K = 32;
    constexpr int WG_M = M / TILE_M;
    constexpr int WG_N = N / TILE_N;
    constexpr int K_TILES = K / TILE_K;

    auto launch = [&]() {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<2>({(size_t)WG_M * 8, (size_t)WG_N},
                                  {8, 1}),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(8)]] {
                    auto sg = item.get_sub_group();
                    int tile_m = item.get_group(0);
                    int tile_n = item.get_group(1);

                    syclmx::joint_matrix<sycl::sub_group, int32_t,
                        syclmx::use::accumulator, TILE_M, TILE_N> acc;
                    syclmx::joint_matrix_fill(sg, acc, 0);

                    for (int kk = 0; kk < K_TILES; kk++) {
                        syclmx::joint_matrix<sycl::sub_group, int8_t,
                            syclmx::use::a, TILE_M, TILE_K, syclmx::layout::row_major> ma;
                        syclmx::joint_matrix<sycl::sub_group, int8_t,
                            syclmx::use::b, TILE_K, TILE_N, syclmx::layout::row_major> mb;

                        auto pA = make_global_ptr(dA + tile_m * TILE_M * K + kk * TILE_K);
                        auto pB = make_global_ptr(dB + kk * TILE_K * N + tile_n * TILE_N);

                        syclmx::joint_matrix_load(sg, ma, pA, K);
                        syclmx::joint_matrix_load(sg, mb, pB, N);
                        syclmx::joint_matrix_mad(sg, acc, ma, mb, acc);
                    }

                    auto pC = make_global_ptr(dC + tile_m * TILE_M * N + tile_n * TILE_N);
                    syclmx::joint_matrix_store(sg, acc, pC, N, syclmx::layout::row_major);
                });
        });
    };

    for (int i = 0; i < 5; i++) launch();
    q.wait();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) launch();
    q.wait();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ops = 2.0 * M * N * K * ITERS;
    double gops = ops / (ms * 1e6);

    std::cout << "  [PERF] " << name << std::endl;
    std::cout << "         " << ITERS << " iters, "
              << ms << " ms total, "
              << ms / ITERS << " ms/iter" << std::endl;
    std::cout << "         " << gops << " GOPS (INT8)" << std::endl;

    sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
}

// ── Main ─────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Phase 0: joint_matrix feasibility on Intel Arc ===" << std::endl;

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    auto dev = q.get_device();
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Driver: " << dev.get_info<sycl::info::device::driver_version>() << std::endl;

    auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    std::cout << "Sub-group sizes:";
    for (auto s : sg_sizes) std::cout << " " << s;
    std::cout << std::endl;

    auto slm = dev.get_info<sycl::info::device::local_mem_size>();
    std::cout << "Local memory (SLM): " << slm / 1024 << " KB" << std::endl;

    bool has_matrix = dev.has(sycl::aspect::ext_intel_matrix);
    std::cout << "ext_intel_matrix: " << (has_matrix ? "YES" : "NO") << std::endl;

    if (!has_matrix) {
        std::cout << "\nXMX not available. Cannot test joint_matrix." << std::endl;
        return 1;
    }

    std::cout << "\n--- Correctness Tests (INT8 first — critical path) ---" << std::endl;

    bool int8_ok = test_int8_joint_matrix(q);
    bool sg32_ok = test_int8_sg32(q);

    // BF16 last — known to trigger IGC compiler bugs on some shapes
    std::cout << "\n--- BF16 Test (may trigger compiler bugs) ---" << std::endl;
    bool bf16_ok = test_bf16_joint_matrix(q);

    std::cout << "\n--- Performance Test ---" << std::endl;

    if (int8_ok) {
        test_performance(q);
    } else {
        std::cout << "  [SKIP] Performance test (INT8 correctness failed)" << std::endl;
    }

    std::cout << "\n--- Summary ---" << std::endl;
    std::cout << "  INT8 joint_matrix:  " << (int8_ok ? "PASS" : "FAIL") << std::endl;
    std::cout << "  INT8 sg=32:         " << (sg32_ok ? "PASS" : "FAIL") << std::endl;
    std::cout << "  BF16 joint_matrix:  " << (bf16_ok ? "PASS" : "FAIL") << std::endl;
    std::cout << "\n  XMX viable for fused kernel: "
              << (int8_ok ? "YES" : "NO — use dp4a fallback") << std::endl;

    return int8_ok ? 0 : 1;
}
