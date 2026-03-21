// Test: OOO queue + handler-based submit + depends_on for row-split pattern.
// Validates the FULL CUDA-style event sync pattern on 3 A770s before
// touching llama.cpp integration.
//
// Build: icpx -fsycl test_ooo_row_split.cpp -o test_ooo_row_split
// Run:   sg render -c ./test_ooo_row_split
// Also:  SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 sg render -c ./test_ooo_row_split

#include <sycl/sycl.hpp>
#include <vector>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main() {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    printf("Found %zu GPU devices\n", devices.size());
    if (devices.size() < 3) {
        printf("Need at least 3 GPUs for row-split test\n");
        return 1;
    }

    for (size_t i = 0; i < devices.size(); i++) {
        printf("  Device %zu: %s\n", i,
               devices[i].get_info<sycl::info::device::name>().c_str());
    }

    // ── Create OOO queues (one per device) ────────────────────────
    printf("\n=== Creating OOO queues ===\n");
    sycl::queue oq0(devices[0]);  // OOO = no in_order property
    sycl::queue oq1(devices[1]);
    sycl::queue oq2(devices[2]);
    sycl::queue* oqs[] = {&oq0, &oq1, &oq2};
    printf("OOO queues created (no in_order property)\n");

    // ── Allocate buffers ──────────────────────────────────────────
    constexpr int N = 4096;  // simulate a row chunk
    constexpr int ROWS = 64;
    constexpr int TOTAL = N * ROWS;

    // Per-device: src (on dev0), dst (result on each dev), staging (host)
    float *src0 = sycl::malloc_device<float>(TOTAL, oq0);
    float *dst1 = sycl::malloc_device<float>(TOTAL, oq1);
    float *dst2 = sycl::malloc_device<float>(TOTAL, oq2);
    float *result0 = sycl::malloc_device<float>(TOTAL, oq0);
    float *result1 = sycl::malloc_device<float>(TOTAL, oq1);
    float *result2 = sycl::malloc_device<float>(TOTAL, oq2);

    // Pinned host staging buffers (one per non-main device)
    float *staging1 = sycl::malloc_host<float>(TOTAL, oq0);
    float *staging2 = sycl::malloc_host<float>(TOTAL, oq0);

    // Merge staging (results back to dev0)
    float *merge_staging1 = sycl::malloc_host<float>(TOTAL, oq0);
    float *merge_staging2 = sycl::malloc_host<float>(TOTAL, oq0);
    float *merged1 = sycl::malloc_device<float>(TOTAL, oq0);
    float *merged2 = sycl::malloc_device<float>(TOTAL, oq0);

    // Initialize src on device 0
    std::vector<float> host_init(TOTAL);
    for (int i = 0; i < TOTAL; i++) host_init[i] = (float)(i % 100) * 0.01f;
    oq0.memcpy(src0, host_init.data(), TOTAL * sizeof(float)).wait();
    printf("Initialized %d floats on device 0\n", TOTAL);

    // ══════════════════════════════════════════════════════════════
    // Test 1: Full row-split pattern with OOO queues + events
    // ══════════════════════════════════════════════════════════════
    printf("\n=== Test 1: Full 3-device row-split pattern (OOO + events) ===\n");
    auto t0 = now_ms();

    // Phase 1: Main device (dev0) records readiness
    sycl::event e_main = oq0.ext_oneapi_submit_barrier();

    // Phase 2a: Dev1 — GPU0→host→GPU1 copy via handler submit + depends_on
    sycl::event e_g2h_1 = oq0.submit([&](sycl::handler& h) {
        h.depends_on(e_main);
        h.memcpy(staging1, src0, TOTAL * sizeof(float));
    });
    sycl::event e_h2g_1 = oq1.submit([&](sycl::handler& h) {
        h.depends_on(e_g2h_1);
        h.memcpy(dst1, staging1, TOTAL * sizeof(float));
    });

    // Phase 2b: Dev2 — GPU0→host→GPU2 copy
    sycl::event e_g2h_2 = oq0.submit([&](sycl::handler& h) {
        h.depends_on(e_main);
        h.memcpy(staging2, src0, TOTAL * sizeof(float));
    });
    sycl::event e_h2g_2 = oq2.submit([&](sycl::handler& h) {
        h.depends_on(e_g2h_2);
        h.memcpy(dst2, staging2, TOTAL * sizeof(float));
    });

    // Phase 2c: Barrier on each dev queue before "kernel" (simulated as scale-by-2)
    oq0.ext_oneapi_submit_barrier({e_main});
    oq0.parallel_for(sycl::range<1>(TOTAL), [=](sycl::id<1> i) {
        result0[i] = src0[i] * 2.0f;
    });
    sycl::event e_done0 = oq0.ext_oneapi_submit_barrier();

    oq1.ext_oneapi_submit_barrier({e_h2g_1});
    oq1.parallel_for(sycl::range<1>(TOTAL), [=](sycl::id<1> i) {
        result1[i] = dst1[i] * 2.0f;
    });
    sycl::event e_done1 = oq1.ext_oneapi_submit_barrier();

    oq2.ext_oneapi_submit_barrier({e_h2g_2});
    oq2.parallel_for(sycl::range<1>(TOTAL), [=](sycl::id<1> i) {
        result2[i] = dst2[i] * 2.0f;
    });
    sycl::event e_done2 = oq2.ext_oneapi_submit_barrier();

    // Phase 3: Merge — copy results from dev1,dev2 back to dev0 via host staging
    // Dev1 → host (depends on dev1 kernel completion)
    sycl::event e_merge_d2h_1 = oq1.submit([&](sycl::handler& h) {
        h.depends_on(e_done1);
        h.memcpy(merge_staging1, result1, TOTAL * sizeof(float));
    });
    // Host → dev0 (depends on d2h)
    sycl::event e_merge_h2d_1 = oq0.submit([&](sycl::handler& h) {
        h.depends_on(e_merge_d2h_1);
        h.memcpy(merged1, merge_staging1, TOTAL * sizeof(float));
    });

    // Dev2 → host
    sycl::event e_merge_d2h_2 = oq2.submit([&](sycl::handler& h) {
        h.depends_on(e_done2);
        h.memcpy(merge_staging2, result2, TOTAL * sizeof(float));
    });
    // Host → dev0
    sycl::event e_merge_h2d_2 = oq0.submit([&](sycl::handler& h) {
        h.depends_on(e_merge_d2h_2);
        h.memcpy(merged2, merge_staging2, TOTAL * sizeof(float));
    });

    // Single host wait at the very end: dev0 done + both merges complete
    e_done0.wait();
    e_merge_h2d_1.wait();
    e_merge_h2d_2.wait();

    auto t1 = now_ms();
    printf("Full pattern: %.2f ms — ", t1 - t0);

    // Verify all results
    std::vector<float> verify(TOTAL);
    bool ok = true;

    // Dev0 result
    oq0.memcpy(verify.data(), result0, TOTAL * sizeof(float)).wait();
    for (int i = 0; i < TOTAL && ok; i++) {
        float expected = host_init[i] * 2.0f;
        if (std::fabs(verify[i] - expected) > 1e-5f) {
            printf("FAIL dev0 [%d]: got %f expected %f\n", i, verify[i], expected);
            ok = false;
        }
    }

    // Merged dev1 result
    oq0.memcpy(verify.data(), merged1, TOTAL * sizeof(float)).wait();
    for (int i = 0; i < TOTAL && ok; i++) {
        float expected = host_init[i] * 2.0f;
        if (std::fabs(verify[i] - expected) > 1e-5f) {
            printf("FAIL merged1 [%d]: got %f expected %f\n", i, verify[i], expected);
            ok = false;
        }
    }

    // Merged dev2 result
    oq0.memcpy(verify.data(), merged2, TOTAL * sizeof(float)).wait();
    for (int i = 0; i < TOTAL && ok; i++) {
        float expected = host_init[i] * 2.0f;
        if (std::fabs(verify[i] - expected) > 1e-5f) {
            printf("FAIL merged2 [%d]: got %f expected %f\n", i, verify[i], expected);
            ok = false;
        }
    }

    printf(ok ? "ALL CORRECT\n" : "DATA MISMATCH\n");

    // ══════════════════════════════════════════════════════════════
    // Test 2: Repeated iterations (simulate 100 tokens)
    // ══════════════════════════════════════════════════════════════
    printf("\n=== Test 2: 100 iterations (simulating 100 tokens) ===\n");
    t0 = now_ms();
    int host_waits = 0;

    for (int iter = 0; iter < 100; iter++) {
        // Phase 1: main readiness
        sycl::event em = oq0.ext_oneapi_submit_barrier();

        // Phase 2: per-device copy + kernel
        // Dev0: kernel directly
        oq0.ext_oneapi_submit_barrier({em});
        oq0.parallel_for(sycl::range<1>(TOTAL), [=](sycl::id<1> i) {
            result0[i] = src0[i] * 2.0f + (float)iter;
        });
        sycl::event ed0 = oq0.ext_oneapi_submit_barrier();

        // Dev1: copy + kernel
        sycl::event eg1 = oq0.submit([&](sycl::handler& h) {
            h.depends_on(em);
            h.memcpy(staging1, src0, TOTAL * sizeof(float));
        });
        sycl::event eh1 = oq1.submit([&](sycl::handler& h) {
            h.depends_on(eg1);
            h.memcpy(dst1, staging1, TOTAL * sizeof(float));
        });
        oq1.ext_oneapi_submit_barrier({eh1});
        oq1.parallel_for(sycl::range<1>(TOTAL), [=](sycl::id<1> i) {
            result1[i] = dst1[i] * 2.0f + (float)iter;
        });
        sycl::event ed1 = oq1.ext_oneapi_submit_barrier();

        // Dev2: copy + kernel
        sycl::event eg2 = oq0.submit([&](sycl::handler& h) {
            h.depends_on(em);
            h.memcpy(staging2, src0, TOTAL * sizeof(float));
        });
        sycl::event eh2 = oq2.submit([&](sycl::handler& h) {
            h.depends_on(eg2);
            h.memcpy(dst2, staging2, TOTAL * sizeof(float));
        });
        oq2.ext_oneapi_submit_barrier({eh2});
        oq2.parallel_for(sycl::range<1>(TOTAL), [=](sycl::id<1> i) {
            result2[i] = dst2[i] * 2.0f + (float)iter;
        });
        sycl::event ed2 = oq2.ext_oneapi_submit_barrier();

        // Phase 3: merge
        sycl::event md1 = oq1.submit([&](sycl::handler& h) {
            h.depends_on(ed1);
            h.memcpy(merge_staging1, result1, TOTAL * sizeof(float));
        });
        sycl::event mm1 = oq0.submit([&](sycl::handler& h) {
            h.depends_on(md1);
            h.memcpy(merged1, merge_staging1, TOTAL * sizeof(float));
        });
        sycl::event md2 = oq2.submit([&](sycl::handler& h) {
            h.depends_on(ed2);
            h.memcpy(merge_staging2, result2, TOTAL * sizeof(float));
        });
        sycl::event mm2 = oq0.submit([&](sycl::handler& h) {
            h.depends_on(md2);
            h.memcpy(merged2, merge_staging2, TOTAL * sizeof(float));
        });

        // ONE host wait per "token" (at the merge boundary, not per matmul)
        ed0.wait();
        mm1.wait();
        mm2.wait();
        host_waits += 3;
    }

    t1 = now_ms();
    double per_iter = (t1 - t0) / 100.0;
    printf("100 iterations: %.1f ms total, %.2f ms/iter, %d total host waits\n",
           t1 - t0, per_iter, host_waits);
    printf("(Compare: current row-split = ~9 host waits × 448 matmuls = 4032/token)\n");

    // Verify last iteration
    oq0.memcpy(verify.data(), result0, TOTAL * sizeof(float)).wait();
    ok = true;
    for (int i = 0; i < TOTAL && ok; i++) {
        float expected = host_init[i] * 2.0f + 99.0f;
        if (std::fabs(verify[i] - expected) > 1e-3f) {
            printf("FAIL iter verify dev0 [%d]: got %f expected %f\n",
                   i, verify[i], expected);
            ok = false;
        }
    }
    printf(ok ? "Final iteration: CORRECT\n" : "Final iteration: WRONG\n");

    // ══════════════════════════════════════════════════════════════
    // Test 3: Barrier ordering on OOO queue (critical for kernel launch)
    // Validates that ext_oneapi_submit_barrier() provides ordering on OOO queues
    // ══════════════════════════════════════════════════════════════
    printf("\n=== Test 3: Barrier ordering on OOO queue ===\n");
    float *a = sycl::malloc_device<float>(N, oq0);
    float *b = sycl::malloc_device<float>(N, oq0);

    // Submit: fill a → barrier → read a + write b
    // If barrier works, b should see the filled values of a
    oq0.fill(a, 42.0f, N);
    oq0.ext_oneapi_submit_barrier();
    oq0.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        b[i] = a[i] + 1.0f;
    });
    oq0.wait();

    std::vector<float> check(N);
    oq0.memcpy(check.data(), b, N * sizeof(float)).wait();
    ok = true;
    for (int i = 0; i < N && ok; i++) {
        if (std::fabs(check[i] - 43.0f) > 1e-5f) {
            printf("FAIL barrier ordering [%d]: got %f expected 43.0\n", i, check[i]);
            ok = false;
        }
    }
    printf("Barrier ordering: %s\n", ok ? "CORRECT" : "BROKEN");

    // ══════════════════════════════════════════════════════════════
    // Test 4: Staging buffer reuse with event chaining
    // Simulates the i0 loop where staging buffer is reused across iterations
    // ══════════════════════════════════════════════════════════════
    printf("\n=== Test 4: Staging buffer reuse with event chaining ===\n");
    constexpr int CHUNK = 1024;
    constexpr int ITERS = 10;
    float *src_chunks = sycl::malloc_device<float>(CHUNK * ITERS, oq0);
    float *dst_chunks = sycl::malloc_device<float>(CHUNK * ITERS, oq1);
    float *staging_shared = sycl::malloc_host<float>(CHUNK, oq0);  // ONE staging buf

    // Init each chunk with different values
    std::vector<float> chunk_init(CHUNK * ITERS);
    for (int i = 0; i < CHUNK * ITERS; i++) chunk_init[i] = (float)(i + 1);
    oq0.memcpy(src_chunks, chunk_init.data(), CHUNK * ITERS * sizeof(float)).wait();

    // Chain copies: each iter reuses staging_shared, must wait for prev h2g
    sycl::event e_prev_h2g;
    for (int iter = 0; iter < ITERS; iter++) {
        float *src_off = src_chunks + iter * CHUNK;
        float *dst_off = dst_chunks + iter * CHUNK;

        // GPU0→host: depends on main readiness + prev h2g (staging buf free)
        sycl::event e_g2h = oq0.submit([&](sycl::handler& h) {
            if (iter > 0) h.depends_on(e_prev_h2g);
            h.memcpy(staging_shared, src_off, CHUNK * sizeof(float));
        });

        // host→GPU1: depends on g2h
        e_prev_h2g = oq1.submit([&](sycl::handler& h) {
            h.depends_on(e_g2h);
            h.memcpy(dst_off, staging_shared, CHUNK * sizeof(float));
        });
    }

    // Wait for all to complete
    e_prev_h2g.wait();

    // Verify
    std::vector<float> dst_verify(CHUNK * ITERS);
    oq1.memcpy(dst_verify.data(), dst_chunks, CHUNK * ITERS * sizeof(float)).wait();
    ok = true;
    for (int i = 0; i < CHUNK * ITERS && ok; i++) {
        if (std::fabs(dst_verify[i] - chunk_init[i]) > 1e-5f) {
            printf("FAIL staging reuse [%d]: got %f expected %f\n",
                   i, dst_verify[i], chunk_init[i]);
            ok = false;
        }
    }
    printf("Staging buffer reuse (%d iters): %s\n", ITERS, ok ? "CORRECT" : "BROKEN");

    // Cleanup
    sycl::free(src0, oq0);
    sycl::free(dst1, oq1);
    sycl::free(dst2, oq2);
    sycl::free(result0, oq0);
    sycl::free(result1, oq1);
    sycl::free(result2, oq2);
    sycl::free(staging1, oq0);
    sycl::free(staging2, oq0);
    sycl::free(merge_staging1, oq0);
    sycl::free(merge_staging2, oq0);
    sycl::free(merged1, oq0);
    sycl::free(merged2, oq0);
    sycl::free(a, oq0);
    sycl::free(b, oq0);
    sycl::free(src_chunks, oq0);
    sycl::free(dst_chunks, oq1);
    sycl::free(staging_shared, oq0);

    printf("\nAll tests complete.\n");
    return 0;
}
