/**
 * Standalone Q8_1 overflow torture test.
 *
 * Tests whether the quantize_q8_1 → MMVQ pipeline produces NaN/inf
 * at various activation magnitudes, WITHOUT loading any model.
 *
 * Build (from llm-stack/):
 *   source env.sglang-xpu.sh
 *   icpx -fsycl -O2 -DGGML_SYCL_WARP_SIZE=16 \
 *     -I llama.cpp/ggml/include -I llama.cpp/ggml/src -I llama.cpp/ggml/src/ggml-sycl \
 *     scripts/test_q8_1_overflow.cpp -o scripts/test_q8_1_overflow
 *
 * Run:
 *   ZE_AFFINITY_MASK=0 ./scripts/test_q8_1_overflow
 *
 * Expected: identifies the exact magnitude at which Q8_1 ds overflows F16.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <sycl/sycl.hpp>

// ─── Minimal types from ggml (avoid pulling in the whole library) ────────

#define QK8_1 32
#define QK_K  256
#define WARP_SIZE 16

typedef struct {
    sycl::half2 ds;     // ds.x = d (scale), ds.y = sum
    int8_t qs[QK8_1];   // quants
} block_q8_1;

typedef struct {
    sycl::half2 dm;     // dm.x = d (super-block scale), dm.y = dmin
    uint8_t scales[12]; // sub-block scales and mins (packed 6-bit)
    uint8_t qs[128];    // 4-bit packed quant values
} block_q4_K;

// ─── Quantize Q8_1 kernel (matches quantize.hpp logic) ──────────────────

void quantize_q8_1_test(sycl::queue &q,
                         const float *src, block_q8_1 *dst,
                         int n_elements, bool clamp_ds) {
    int n_blocks = n_elements / QK8_1;

    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>({(size_t)(n_blocks * WARP_SIZE)}, {(size_t)WARP_SIZE}),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                int block_id = it.get_group(0);
                int wi_id = it.get_local_id(0);
                int elements_per_wi = QK8_1 / WARP_SIZE;  // 32/16 = 2

                float vals[2];
                float amax = 0.0f;
                float sum = 0.0f;

                for (int i = 0; i < elements_per_wi; i++) {
                    int idx = block_id * QK8_1 + wi_id * elements_per_wi + i;
                    vals[i] = src[idx];
                    sum += vals[i];
                    amax = sycl::fmax(amax, sycl::fabs(vals[i]));
                }

                // Sub-group reduction
                sum = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
                amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<float>());

                float d = amax == 0 ? 1.0f : amax / 127.0f;

                // Quantize
                for (int i = 0; i < elements_per_wi; i++) {
                    int idx = wi_id * elements_per_wi + i;
                    dst[block_id].qs[idx] = (int8_t)sycl::round(vals[i] / d);
                }

                d = amax == 0 ? 0.0f : d;

                // Write ds — with or without clamp
                if (wi_id == 0) {
                    if (clamp_ds) {
                        float d_c = sycl::fmin(sycl::fmax(d, -65504.0f), 65504.0f);
                        float s_c = sycl::fmin(sycl::fmax(sum, -65504.0f), 65504.0f);
                        dst[block_id].ds = sycl::half2(sycl::half(d_c), sycl::half(s_c));
                    } else {
                        // Unclamped — matches original code and fattn-common.hpp
                        dst[block_id].ds = sycl::half2(sycl::half(d), sycl::half(sum));
                    }
                }
            });
    }).wait();
}

// ─── Test harness ────────────────────────────────────────────────────────

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};

    auto dev = q.get_device();
    printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    printf("WARP_SIZE=%d  QK8_1=%d\n\n", WARP_SIZE, QK8_1);

    constexpr int K = 8192;     // hidden dim (matches Qwen3-32B)
    constexpr int BATCH = 2;    // multi-slot decode
    constexpr int N = K * BATCH;

    float *src_host = new float[N];
    block_q8_1 *q8_host = new block_q8_1[N / QK8_1];

    float *src_dev = sycl::malloc_device<float>(N, q);
    block_q8_1 *q8_dev = sycl::malloc_device<block_q8_1>(N / QK8_1, q);

    printf("%-12s | %-10s | %-12s | %-12s | %-10s | %-10s | %-10s\n",
           "magnitude", "clamp", "d_f32", "d_f16_back", "sum_f16", "ds_nan", "ds_inf");
    printf("%-12s-+-%-10s-+-%-12s-+-%-12s-+-%-10s-+-%-10s-+-%-10s\n",
           "------------", "----------", "------------", "------------",
           "----------", "----------", "----------");

    float magnitudes[] = {
        1e2f, 1e3f, 1e4f, 5e4f, 6.5e4f, 1e5f, 5e5f,
        1e6f, 5e6f, 8.3e6f, 1e7f, 2.7e7f, 1e8f
    };

    for (int clamp = 0; clamp <= 1; clamp++) {
        for (float mag : magnitudes) {
            // Fill with values spanning [-mag, +mag]
            for (int i = 0; i < N; i++) {
                // Deterministic pseudo-random pattern
                float t = (float)(i % 1000) / 999.0f;  // [0, 1]
                src_host[i] = mag * (2.0f * t - 1.0f);  // [-mag, +mag]
            }

            q.memcpy(src_dev, src_host, N * sizeof(float)).wait();

            quantize_q8_1_test(q, src_dev, q8_dev, N, clamp != 0);

            q.memcpy(q8_host, q8_dev, (N / QK8_1) * sizeof(block_q8_1)).wait();

            // Check all Q8_1 blocks
            int nan_count = 0, inf_count = 0;
            float max_d_f16 = 0;
            for (int b = 0; b < N / QK8_1; b++) {
                float d_back = (float)q8_host[b].ds.x();
                float s_back = (float)q8_host[b].ds.y();

                if (std::isnan(d_back) || std::isnan(s_back)) nan_count++;
                if (std::isinf(d_back) || std::isinf(s_back)) inf_count++;
                max_d_f16 = std::fmax(max_d_f16, std::fabs(d_back));
            }

            float expected_d = mag / 127.0f;

            printf("%-12.1e | %-10s | %-12.4e | %-12.4e | %-10s | %-10d | %-10d\n",
                   mag,
                   clamp ? "YES" : "NO",
                   expected_d,
                   max_d_f16,
                   (float)q8_host[0].ds.y() == 0.0f ? "zero" : "nonzero",
                   nan_count,
                   inf_count);
        }
        if (clamp == 0) printf("\n--- WITH CLAMP ---\n\n");
    }

    // ─── Flash attention inline quantize test ─────────────────────────
    // This simulates fattn-common.hpp:347 quantize_q8_1_to_shared()
    // which does: d = amax/127, make_half2(d, sum) WITHOUT clamp.
    printf("\n=== Flash Attention Inline Q8_1 (fattn-common.hpp:347) ===\n");
    printf("%-12s | %-12s | %-12s | %-12s | %-10s\n",
           "amax", "d=amax/127", "half(d)", "overflow?", "fatal?");
    printf("%-12s-+-%-12s-+-%-12s-+-%-12s-+-%-10s\n",
           "------------", "------------", "------------", "------------", "----------");

    float test_amaxes[] = {
        1e2f, 1e3f, 1e4f, 5e4f, 6.5e4f, 8.3e6f, 1e7f, 2.7e7f, 1e8f
    };

    for (float amax : test_amaxes) {
        float d = amax / 127.0f;
        sycl::half d_h(d);
        float d_back = (float)d_h;
        bool overflow = std::isinf(d_back) || std::isnan(d_back);
        bool fatal = overflow;  // inf in Q8_1 ds → NaN in matmul

        printf("%-12.1e | %-12.4e | %-12.4e | %-12s | %-10s\n",
               amax, d, d_back,
               overflow ? "YES inf!" : "no",
               fatal ? "CRASH" : "ok");
    }

    // ─── Flash attention KQ_max_scale test ─────────────────────────────
    // fattn-tile.hpp:648: KQ_max_scale = exp(KQ_max - KQ_max_new)
    printf("\n=== Flash Attention KQ_max_scale (fattn-tile.hpp:648) ===\n");
    printf("%-12s | %-12s | %-12s | %-10s\n",
           "delta", "exp(delta)", "half(exp)", "overflow?");
    printf("%-12s-+-%-12s-+-%-12s-+-%-10s\n",
           "------------", "------------", "------------", "----------");

    float deltas[] = {1.0f, 5.0f, 10.0f, 11.0f, 11.1f, 12.0f, 15.0f, 20.0f, 50.0f};
    for (float delta : deltas) {
        float val = std::exp(delta);
        sycl::half val_h(val);
        float val_back = (float)val_h;
        printf("%-12.1f | %-12.4e | %-12.4e | %-10s\n",
               delta, val, val_back,
               (std::isinf(val_back) || std::isnan(val_back)) ? "YES inf!" : "no");
    }

    printf("\n=== Summary ===\n");
    printf("F16 max = 65504.0\n");
    printf("Q8_1 ds overflow when activation amax > 65504 * 127 = %.0f (%.1e)\n",
           65504.0f * 127.0f, 65504.0f * 127.0f);
    printf("FA KQ_max_scale overflow when attention score delta > %.1f\n",
           std::log(65504.0f));
    printf("Sniffer found amax=2.7e7 in Qcur output at graph[2].\n");
    printf("  → Q8_1 d = 2.7e7/127 = %.0f (within F16? %s)\n",
           2.7e7f / 127.0f, (2.7e7f / 127.0f <= 65504.0f) ? "YES" : "NO — OVERFLOW!");

    sycl::free(src_dev, q);
    sycl::free(q8_dev, q);
    delete[] src_host;
    delete[] q8_host;

    return 0;
}
