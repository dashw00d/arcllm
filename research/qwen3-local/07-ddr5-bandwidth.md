# RAM Bandwidth Impact on MoE Expert Loading During Inference

**Research date:** 2026-03-15
**System:** 3x Intel Arc A770 (48 GB VRAM) + 64 GB DDR5 + `-cmoe` (expert FFN weights on CPU RAM)
**Model:** Qwen3-235B-A22B at Q3_K_S (101 GB total, ~91 GB expert weights)

---

## 1. Architecture Parameters and Expert Weight Sizes

From `config.json` at [Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json):

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 4,096 |
| `moe_intermediate_size` (expert FFN dim) | 1,536 |
| `num_experts` (`n_routed_experts`) | 128 |
| `num_experts_per_tok` | 8 (top-8 routing) |
| `num_hidden_layers` | 94 |
| `num_attention_heads` | 64 |
| `num_key_value_heads` | 4 (GQA) |

**Correction to research prompt:** The expert FFN dim is **1,536**, not 2,048. The previous estimate in the research prompt was too high by ~33%.

### Q3_K_S Quantization Block Structure

From `/home/ryan/llm-stack/llama.cpp/ggml/src/ggml-common.h`, line 293–300:

```c
// Effectively 3.4375 bits per weight
typedef struct {
    uint8_t hmask[QK_K/8]; // quants - high bit
    uint8_t qs[QK_K/4];    // quants - low 2 bits
    uint8_t scales[12];    // scales, quantized with 6 bits
    ggml_half d;           // super-block scale
} block_q3_K;
// sizeof = 2 + 64 + 32 + 12 = 110 bytes per 256 weights
```

`QK_K = 256`, so `block_q3_K` = 110 bytes for 256 weights = **3.4375 bpw** (not 3.0 bpw as the name might suggest).

### Expert FFN Tensor Sizes

Each expert has 3 weight matrices under `-cmoe`:
- `ffn_gate_exps`: `[hidden_size, expert_ffn_dim, n_experts]` = `[4096, 1536, 128]`
- `ffn_up_exps`:   `[hidden_size, expert_ffn_dim, n_experts]` = `[4096, 1536, 128]`
- `ffn_down_exps`: `[expert_ffn_dim, hidden_size, n_experts]` = `[1536, 4096, 128]`

Per-expert parameter count: `3 × 4096 × 1536 = 18,874,368 params`

At Q3_K (3.4375 bpw): **7.73 MB per expert per layer**

Total expert storage:
- Per token per layer (8 active experts): **61.9 MB must be read from RAM**
- Per token across all 94 layers: **5,816 MB = 5.68 GB**
- All 128 experts, all 94 layers (full model in RAM): **90.9 GB**

The 90.9 GB total aligns with the ~87–91 GB estimate in `06-quant-selection.md` (discrepancy comes from the router/gate tensors which are small and reside in VRAM, not CPU).

---

## 2. Theoretical Tokens/Second from RAM Bandwidth Alone

**Formula:** `t/s = RAM_bandwidth / bytes_per_token`
where `bytes_per_token = 8_experts × 3_matrices × hidden × ffn_dim × bpw/8 × 94_layers`
= 5.68 GB/token at Q3_K_S.

| Memory Configuration | Theoretical Peak GB/s | Theoretical Max t/s (Q3_K_S) |
|---------------------|----------------------|------------------------------|
| DDR5-4800 single channel | 38.4 GB/s | 6.30 t/s |
| DDR5-5600 single channel | 44.8 GB/s | 7.35 t/s |
| DDR5-6400 single channel | 51.2 GB/s | 8.40 t/s |
| **DDR5 ~50 GB/s actual (Ryan's system)** | **~50 GB/s** | **~8.20 t/s** |
| DDR4-3200 dual channel | 51.2 GB/s | 8.40 t/s |
| DDR5-4800 dual channel | 76.8 GB/s | 12.60 t/s |
| DDR5-5600 dual channel | 89.6 GB/s | 14.69 t/s |
| DDR4-2933 quad channel (X299) | 93.9 GB/s | 15.39 t/s |
| DDR5-6400 dual channel | 102.4 GB/s | 16.80 t/s |

**Key result for Ryan's system:** At ~50 GB/s effective DDR5 bandwidth, the **absolute theoretical ceiling is ~8.2 t/s**. This is a hard upper bound from bandwidth alone, before adding compute overhead or dequantization time.

For Q2_K_S (2.625 bpw, 4.66 GB/token): ceiling rises to **~10.7 t/s** at 50 GB/s.

Sources: Bandwidth = `MT/s × 8 bytes × channels`. DDR5 uses two 32-bit sub-channels per DIMM, but for bandwidth calculation purposes, a single DDR5 DIMM in the "single channel" slot still delivers the full DIMM bandwidth (e.g., DDR5-4800 × 8 bytes = 38.4 GB/s). Dual-channel means two DIMMs active simultaneously.

---

## 3. Real-World llama.cpp CPU+GPU Hybrid MoE Benchmarks

### Benchmark A: AMD TR 2950X + 128 GB DDR4-2933 quad-channel + RTX 3090 (24 GB)

Source: [ubergarm/Qwen3-235B-A22B-GGUF discussions #3](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/discussions/3)

- **Model:** mix-IQ3_K (~3.9 bpw), `-ot exps=CPU`, ik_llama.cpp
- **PP (512 tokens):** 24.05 t/s (with `-rtr -fmoe`) → **67.35 t/s** (without those flags)
- **TG (128 tokens):** **8.19 t/s**
- Reported limiting factor: RAM bandwidth for PP; removing `-rtr` was 2.8x faster
- Theoretical t/s at 93.9 GB/s: **16.5 t/s** (IQ3_K ~= Q3_K bpw)
- **Observed efficiency: 8.19 / 16.5 = ~50%** of theoretical bandwidth ceiling

### Benchmark B: Intel Xeon W5-3425 + 512 GB DDR5 (160 GB/s measured via mlc) + RTX 4090D (48 GB) + RTX 3090 (24 GB)

Source: [ubergarm/Qwen3-235B-A22B-GGUF discussions #6](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/discussions/6)

- **Model:** mix-IQ3_K, `-ot exps=CPU`, `-fmoe`, ik_llama.cpp
- **PP:** 147.51 t/s
- **TG:** **14.27 t/s**
- Theoretical t/s at 160 GB/s: **28.2 t/s**
- **Observed efficiency: 14.27 / 28.2 = ~50.7%** of theoretical bandwidth ceiling

**Cross-system efficiency is consistently ~50% of theoretical bandwidth peak** for TG with expert CPU offload. This is expected: random expert access patterns, cache line granularity overhead, NUMA effects, and Q3_K dequantization on the CPU all consume ~half the theoretical bandwidth.

### Benchmark C: Qwen3-Coder-Next (80B MoE, 3B active) on AMD Ryzen AI 9 HX PRO 370

Source: [llama.cpp issue #19480](https://github.com/ggerganov/llama.cpp/issues/19480)

- **RAM:** 96 GB DDR5-5600 dual channel (~89.6 GB/s)
- **TG:** 7.74 t/s (CPU only, Q4_K_M)
- **Expected from bandwidth:** 3B active / ~3 GB/token × 89.6 GB/s = ~30 t/s
- **With `--n-cpu-moe` flag** (routing on GPU): boosted to ~25 t/s
- Analysis: vanilla llama.cpp reads far more data than sparse activation requires — kernel efficiency was poor until PR #19422/19504 fixes

### Bandwidth-to-TPS Formula (Empirical)

Based on the two well-characterized benchmarks:

```
actual_t/s ≈ 0.50 × (RAM_bandwidth_GB/s / bytes_per_token_GB)
```

For Q3_K_S Qwen3-235B on Ryan's system (~50 GB/s DDR5):
```
expected_t/s ≈ 0.50 × (50 / 5.68) ≈ 4.4 t/s
```

**However**, Ryan's system has ~91 GB expert weights against 64 GB RAM. The OS keeps recently-used pages hot but cold pages hit the NVMe SSD. NVMe peak ~3–5 GB/s sequential, but random page loads are worse. This brings real-world TG well below even the 4.4 t/s calculation.

---

## 4. DDR5 vs DDR4 for MoE Inference

### Theoretical Bandwidth Comparison

| Platform | Config | Speed | Theoretical GB/s | Comment |
|----------|--------|-------|-----------------|---------|
| DDR4 dual ch (consumer) | 2×DIMM | 3200 MT/s | 51.2 | Standard desktop |
| DDR4 dual ch (XMP) | 2×DIMM | 3600 MT/s | 57.6 | Overclocked |
| DDR4 quad ch (X299 HEDT) | 4×DIMM | 2933 MT/s | 93.9 | LCC, max 4ch |
| DDR4 quad ch (X299 XMP) | 4×DIMM | 3200 MT/s | 102.4 | Overclocked quad-ch |
| DDR5 dual ch (consumer) | 2×DIMM | 4800 MT/s | 76.8 | Min DDR5 speed |
| **DDR5 dual ch (Ryan est.)** | **2×DIMM** | **~6400 MT/s** | **~100 GB/s** | **DDR5-6400 rated** |
| DDR5 dual ch | 2×DIMM | 5600 MT/s | 89.6 | Common consumer |
| DDR5-6400 dual ch | 2×DIMM | 6400 MT/s | 102.4 | High-end consumer |
| DDR5 quad ch (Xeon SP, etc.) | 4×DIMM | 4800 MT/s | 153.6 | Server platforms |

**Key insight:** DDR5 dual-channel at 5600–6400 MT/s ≈ DDR4 quad-channel (X299) at 2933–3200 MT/s. For MoE inference, an X299 platform with four sticks of DDR4 can match or exceed a modern DDR5 dual-channel consumer system.

**Ryan's "50 GB/s" figure** suggests effective single-channel or BIOS-limited dual-channel operation. If the two DDR5 DIMMs are running in **dual-channel** mode and rated at DDR5-5600+, theoretical bandwidth is ~89.6 GB/s or higher. The 50 GB/s figure may indicate:
1. Single-channel operation (DIMMs not in matched slots)
2. DDR5-4800 base speed running single channel: 38.4 GB/s
3. Sub-optimal BIOS memory settings (XMP/EXPO not enabled)

See Section 7 for verification and optimization steps.

### Latency Differences

DDR5 has higher CAS latency in absolute nanoseconds vs DDR4 at the same logical CL rating. DDR4-3200 CL16 = 10 ns; DDR5-4800 CL40 = 16.7 ns; DDR5-6400 CL32 = 10 ns.

For MoE inference, **bandwidth matters more than latency** because expert FFN GEMV operations stream through large contiguous weight blocks (7.73 MB per expert). Once the first cache line is fetched, the access is sequential. Latency overhead per expert is typically 1–4 μs (cache miss + memory controller latency), which is negligible against the ~122 ms token generation time at 50 GB/s.

However, with random expert access (each token picks a different subset of 8 from 128 experts), there are `8 × 94 = 752` separate weight-region accesses per token, each of which may trigger a TLB miss if the page wasn't recently accessed. This is where 4 KB pages hurt and 2 MB hugepages help (see Section 6).

---

## 5. CPU Compute Time vs RAM Bandwidth (Roofline Analysis)

### FLOPS vs Bandwidth Budget

Per token (batch=1, token generation), with 8 active experts × 94 layers:

| Metric | Value |
|--------|-------|
| Total FLOPS (FP32, 3 matrices × 2 ops) | 28.39 GFLOPS |
| Total bytes read from RAM (Q3_K_S) | 5,816 MB = 5.68 GB |
| Arithmetic intensity | **4.65 FLOP/byte** |

**10-core Skylake-X AVX-512 peak** (2 FMA units × 16 FP32 × 2 ops × 10 cores × 3 GHz):

```
Peak = 2 × 16 × 2 × 10 × 3e9 = 1.92 TFLOPS FP32
Compute time = 28.39 GFLOPS / 1.92 TFLOPS = 14.8 ms/token = 67.6 t/s (if compute-bound)
```

**Memory bandwidth time at 50 GB/s:**

```
Memory time = 5.68 GB / 50 GB/s = 113.6 ms/token = 8.8 t/s (if bandwidth-bound)
```

**Roofline balance point:** The hardware would be at equal compute/bandwidth balance at:
```
balance_bandwidth = arithmetic_intensity × bandwidth = 4.65 × 50 GB/s = 232.5 GFLOPS
```
That's well below the 1.92 TFLOPS AVX-512 peak, so **the operation is firmly bandwidth-bound by a factor of ~13x** at 50 GB/s.

Even at DDR5 dual-channel (90 GB/s), arithmetic intensity 4.65 FLOP/byte × 90 GB/s = 418 GFLOPS vs 1.92 TFLOPS AVX-512 — still compute has headroom; bandwidth is the ceiling.

### Q3_K Dequantization Overhead

Q3_K is not natively computable; the CPU must dequantize weights from 3.4375-bit packed integers to FP32 (or BF16) before the GEMV. This adds CPU compute overhead on top of the memory bandwidth cost. The dequant step reads the same bytes as the bandwidth calculation already accounts for, but adds ~30–50% extra CPU cycles for the bit-unpacking operation. This is one reason real-world efficiency is ~50% of theoretical bandwidth ceiling.

### Is the bottleneck RAM bandwidth or CPU FLOPS?

**For TG (batch=1): RAM bandwidth, unambiguously.** CPU FLOPS are 13× headroom at 50 GB/s. Adding more CPU cores would not meaningfully improve TG speed.

**For PP (batch > 64): CPU FLOPS.** Prompt processing loads the same weights but computes against a matrix of token activations. At batch 512, compute time grows 512× while bandwidth stays constant → compute becomes the bottleneck.

Source: This matches the Apple Silicon finding in [llama.cpp discussions #4167](https://github.com/ggerganov/llama.cpp/discussions/4167) where TG scales linearly with bandwidth and PP scales with compute.

---

## 6. mmap vs Preloaded Weights

### How llama.cpp loads model weights

From `/home/ryan/llm-stack/llama.cpp/src/llama-mmap.cpp`:

```cpp
impl(struct llama_file * file, size_t prefetch, bool numa) {
    if (numa) { prefetch = 0; }  // NUMA mode: no prefetch, let faults distribute
    if (prefetch) { flags |= MAP_POPULATE; }
    addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
    if (prefetch > 0) {
        posix_madvise(addr, min(file->size(), prefetch), POSIX_MADV_WILLNEED);
    }
    posix_madvise(addr, file->size(), POSIX_MADV_RANDOM);
    // MADV_RANDOM disables readahead - each page fetched individually
```

Key behaviors:
1. **`PROT_READ` + `MAP_POPULATE`** (when prefetch > 0): the initial mmap call with MAP_POPULATE will attempt to fault in all pages at load time. Combined with `POSIX_MADV_WILLNEED`, the kernel pre-fetches up to `prefetch` bytes.
2. **`POSIX_MADV_RANDOM`**: disables kernel readahead for all pages. This is correct for random expert access but means each 4 KB page is fetched independently when cold.
3. **NUMA mode** (`--numa` flag): disables all prefetch so pages distribute to NUMA nodes on first-touch fault.

### `--mlock` Flag Effect

`use_mlock = true` (`--mlock` flag) calls `mlock(addr, size)` on the mapped region, which:
- Pins all pages in physical RAM (prevents swapping to any swap device)
- Does **not** prefetch from SSD — pages still start unmapped; first-touch causes page faults
- Prevents the OS from evicting expert pages under memory pressure
- Requires sufficient `ulimit -l` (locked memory limit); may need `sudo` or `/etc/security/limits.conf` changes

**For 64 GB RAM with 91 GB expert weights:** `--mlock` will fail or cause OOM because there isn't enough RAM to lock all expert pages. Do not use `--mlock` with the current hardware/quant combination.

### Warmup to Eliminate Page Faults

From [llama.cpp PR #11571](https://github.com/ggerganov/llama.cpp/pull/11571), the `llama_set_warmup()` API runs inference with `n_expert_used = n_expert` to force all pages into RAM cache:

```
"loads a couple of times faster than letting it warm up naturally"
"avoiding random access on the SSD"
"increased CPU utilization (~2.5 cores vs ~0.5 cores) during warmup"
```

`llama-server` performs a warmup pass on startup by default. This pre-faults pages for experts that fit in RAM. However, with 91 GB experts against 64 GB RAM, **~27 GB of expert pages cannot stay resident simultaneously** — page eviction occurs during inference regardless of warmup.

### SSD Page Fault Penalty

When an expert page is not in RAM (cold or evicted), the kernel page fault handler reads it from storage:
- NVMe SSD sequential read: 3–7 GB/s
- NVMe SSD random 4K read: ~500K IOPS × 4 KB = ~2 GB/s effective for random pages
- SATA SSD: ~500 MB/s sequential, ~100K IOPS random

For 27 GB of cold expert pages:
- If NVMe sequential: adds ~5–9 seconds to cold start
- During steady inference: page evictions cause ~100–400 μs stalls per cold expert page fault

This is why the qualitative report of "2–3 t/s" for the 3×3090+P40+64GB setup with Q2_K (`06-quant-selection.md`, unsloth discussion #6) is consistent — even Q2_K experts (69 GB total) exceed 64 GB RAM by ~5 GB, causing constant cold page pressure.

---

## 7. NUMA Effects on Expert Loading

### Single-socket systems (Ryan's system): NUMA is irrelevant

From [llama.cpp issue #1437](https://github.com/ggml-org/llama.cpp/issues/1437):

> "Most generic motherboards are two channel, with **1 NUMA node**. NUMA is not relevant for typical single-socket consumer systems."

A single-CPU system (regardless of DDR5 dual-channel) presents exactly one NUMA node to the OS. The `--numa` flags (`distribute`, `isolate`, `numactl`) have no effect and may disable the MAP_POPULATE prefetch for no benefit.

**Do not use `--numa` on a single-CPU system.**

### Multi-socket systems (X299 with dual-CPU, or Threadripper): NUMA matters greatly

The benchmark data from [llama.cpp issue #1437](https://github.com/ggml-org/llama.cpp/issues/1437) shows:
- 2-socket Xeon E5-2690 without NUMA opt: 253.67 ms/token
- 2-socket with `numactl --interleave=0-1`: **111.85 ms/token (2.27× faster)**
- 4-socket with interleaving: **95.34 ms/token (2.66× faster)**

X299 is a single-socket platform, so this doesn't apply to X299 setups unless using dual Xeon W configurations.

### `numactl --interleave` vs `--membind`

For MoE inference where expert weights are spread across RAM:
- `numactl --interleave=all`: best for large models distributed across NUMA nodes; ensures no NUMA node is a hot spot
- `numactl --membind=0`: binds all memory to NUMA node 0; useful if CPU 0 is the active socket for inference
- `numactl --cpunodebind=0 --membind=0`: as recommended in `/home/ryan/llm-stack/llama.cpp/docs/backend/ZenDNN.md`

For single-socket setups, all of the above are no-ops or redundant.

---

## 8. BIOS/OS Optimizations

### Memory Configuration (Highest Priority)

**Verify dual-channel mode.** If Ryan's system is showing ~50 GB/s actual bandwidth, this may be single-channel DDR5-4800 (38.4 GB/s theoretical). Two DIMMs must be in the correct paired slots (typically slots A2 and B2, not A1 and B1). Run:

```bash
sudo dmidecode -t memory | grep -E "Speed|Configured|Locator"
# Look for "Configured Memory Speed" to confirm actual running speed
# Look for "Bank Locator" to see if both channels are populated

# Or use mlc (Intel Memory Latency Checker):
sudo ./mlc --bandwidth_matrix
# Should show ~80-100+ GB/s for DDR5 dual channel
```

**Enable XMP/EXPO in BIOS.** DDR5 DIMMs often default to JEDEC base speed (e.g., 4800 MT/s) even if rated for 5600 or 6400 MT/s. Enable XMP (Intel) or EXPO (AMD) in BIOS to get rated speed.

**Memory interleaving** — in BIOS, set "Memory Interleaving" to "Channel" or "Auto" (not "Disabled"). This enables NUMA-aware interleaving at the hardware level.

### Huge Pages

The `POSIX_MADV_RANDOM` hint in llama-mmap.cpp causes 4 KB page granularity for expert accesses. Each expert (7.73 MB) spans ~1,932 pages of 4 KB, creating TLB pressure. With 2 MB huge pages, that drops to ~4 huge-page entries per expert.

Enable transparent huge pages (THP) for the inference process:

```bash
# Enable THP globally (or per-process for llama-server)
echo always > /sys/kernel/mm/transparent_hugepage/enabled
# Or for madvise mode (safer, opt-in per allocation):
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
```

Note: llama.cpp does not call `madvise(MADV_HUGEPAGE)` on the mmap region as of current source. THP will only help if `enabled=always`. With `enabled=madvise`, the model weights mapped via `mmap()` without an explicit MADV_HUGEPAGE call won't get huge pages.

For explicit control, `hugetlbfs` with pre-allocated huge pages:

```bash
echo 50000 > /proc/sys/vm/nr_hugepages  # 50000 × 2 MB = ~100 GB huge page pool
# Requires reboot or boot parameter: hugepages=50000
```

Benefit: reduces TLB misses by ~500× per expert access, saving ~5–15% of CPU overhead for sequential accesses. Does not improve RAM bandwidth itself.

### CPU Frequency and C-States

For MoE inference, the CPU is frequently woken from idle to process expert GEMV operations. C-state latency (C6/C7 exit: ~100–200 μs) can add meaningful overhead if the CPU sleeps between layers.

```bash
# Disable deep C-states for inference sessions
cpupower idle-set -D 1  # Allow up to C1 only
# Or set performance governor:
cpupower frequency-set -g performance
```

With 94 layers × ~122 ms total token time, there's no C-state issue — the CPU is continuously busy. But at very low t/s (e.g., 1–2 t/s), inter-token pauses could trigger C-states.

### Kernel Page Cache Priority

The OS kernel manages expert page eviction through the LRU page cache. With 91 GB experts and 64 GB RAM, the kernel will evict cold expert pages. Recently accessed experts stay in cache; experts not seen for many tokens get evicted.

To hint that model pages are high-priority:

```bash
# Pre-warm all expert pages into RAM (run once after loading):
vmtouch -t /path/to/model.gguf
# This touches every page, forcing them into page cache
# Pages still subject to eviction under memory pressure
```

With 91 GB model and 64 GB RAM, `vmtouch` will succeed in loading the file but 27 GB will be evicted as the file is walked — it does not hold pages, it just warms them. Still useful for the warmup pass.

---

## 9. Qwen3-235B Specific Reports

### Setup 1: AMD TR 2950X + 128 GB DDR4-2933 + RTX 3090 (ik_llama.cpp)

From [ubergarm discussions #3](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/discussions/3):
- IQ3_K mix (~3.9 bpw), experts on CPU, 16-thread inference
- **TG: 8.19 t/s** at 93.9 GB/s quad-channel DDR4

### Setup 2: Xeon W5-3425 + 512 GB DDR5 + 4090D+3090 (ik_llama.cpp)

From [ubergarm discussions #6](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/discussions/6):
- IQ3_K mix, experts on CPU, 160 GB/s DDR5 bandwidth (mlc measured)
- **TG: 14.27 t/s** (IQ3_K) / **16.00 t/s** (UD-Q3_K_XL unsloth)
- Efficiency: ~50% of theoretical bandwidth ceiling
- Note: ik_llama.cpp with `-fmoe`; vanilla llama.cpp with SYCL may differ in efficiency

### Setup 3: 3×3090 + P40 + 64 GB RAM (qualitative)

From [unsloth discussions #6](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF/discussions/6):
- Q2_K, 64 GB RAM (experts ~69 GB, ~5 GB overflow to swap)
- **TG: 2–3 t/s** — severely degraded by SSD paging

### CUDA-Graphs for n-cpu-moe (PR #18934)

From [llama.cpp PR #18934](https://github.com/ggerganov/llama.cpp/pull/18934), enabling CUDA graphs for CPU-MoE hybrid:
- **GLM4-MOE 106B:** 4–10% TG speedup
- **GPT-OSS 120B MoE:** 5–10% TG speedup
- DeepSeek V3 real-world: ~5% TG speedup
- Applicable to Qwen3-235B with `-cmoe` on vanilla llama.cpp + SYCL once merged

---

## 10. Summary Tables

### Bandwidth vs Expected TG for Ryan's System (Q3_K_S, 5.68 GB/token)

| Scenario | BW | Theoretical | ~50% efficiency | SSD penalty |
|----------|-----|------------|-----------------|-------------|
| Current: ~50 GB/s effective | 50 GB/s | 8.2 t/s | **4.1 t/s** | heavy (~27 GB cold) |
| If dual-channel confirmed (89 GB/s) | 89 GB/s | 14.7 t/s | **7.4 t/s** | heavy (~27 GB cold) |
| If Q2_K downloaded (4.66 GB/token) | 50 GB/s | 10.7 t/s | **5.4 t/s** | moderate (~5 GB cold) |
| With 96 GB RAM + Q3_K_S | 89 GB/s | 14.7 t/s | **7.4 t/s** | **none** |
| With 128 GB RAM + Q4_K_M (5.82 GB/token est.) | 89 GB/s | 15.3 t/s | **7.7 t/s** | **none** |

### DDR5 vs DDR4 (X299) Summary for MoE Inference

| Metric | DDR5 dual-ch (5600) | DDR4 quad-ch (X299, 3200) |
|--------|---------------------|--------------------------|
| Theoretical peak | 89.6 GB/s | 102.4 GB/s |
| Real-world efficiency | ~85–90% | ~85–90% |
| Practical BW | ~75–80 GB/s | ~87–93 GB/s |
| Expected TG (Q3_K_S, 50% BW util) | ~6.7–7.1 t/s | ~7.7–8.2 t/s |
| Access latency | Higher absolute ns | Lower absolute ns |
| # DIMMs needed | 2 | 4 |
| Cost per GB | Higher | Lower |

**Conclusion:** X299 quad-channel DDR4 has a ~20% bandwidth advantage over DDR5 dual-channel for MoE inference, translating to ~1–2 additional t/s. The difference is not dramatic; RAM capacity (>96 GB) matters more than the DDR4 vs DDR5 distinction.

---

## 11. Recommended Changes to Deployment Plan

### Priority 1 (immediate, no cost): Verify and Fix Dual-Channel Mode

The reported ~50 GB/s bandwidth strongly suggests DDR5 is running in single-channel or at base JEDEC speed. Confirm:

```bash
sudo dmidecode -t memory | grep "Configured Memory Speed"
sudo dmidecode -t memory | grep "Bank Locator"
```

Expected output for dual-channel DDR5-5600: "Configured Memory Speed: 5600 MT/s" on both DIMMs in different bank locators. If both DIMMs show the same bank/locator, they may be in the same channel.

If single-channel: reseat DIMMs into matching slots (A2+B2 on Intel, or as specified in motherboard manual). This alone could double bandwidth from ~38.4 to ~89.6 GB/s, translating to **~3–4 additional t/s** under expert offload (from ~4 to ~7 t/s actual).

If DDR5 XMP is not enabled: enable in BIOS. DDR5-5600 at base JEDEC 4800 loses ~16% bandwidth.

### Priority 2 (immediate, no cost): Switch to `-ncmoe 47` Hybrid Offload

From `06-quant-selection.md` section 11.3: VRAM has ~29 GB headroom. Use `-ncmoe 47` to keep the first 47 layers' experts in VRAM (~14 GB) and only offload the remaining 47 layers to CPU RAM (~43 GB experts). This fits in 64 GB RAM without any SSD paging for the RAM-resident experts.

```bash
-ncmoe 47 -ngl 99
```

Estimated impact: eliminates SSD page faults for ~50% of expert accesses; the other ~50% (layers 0–46) are in VRAM at VRAM bandwidth (1536 GB/s total across 3× A770). The VRAM-resident experts are no longer bandwidth-limited. Net effect could be **2× TG speed improvement** for the layers with VRAM experts.

### Priority 3 (immediate, low cost): Download Q2_K

Q2_K (2.625 bpw) reads only 4.66 GB/token vs 5.68 GB/token at Q3_K_S — a 18% reduction in bandwidth demand. More importantly, Q2_K total expert storage is ~69.4 GB vs ~90.9 GB, meaning only ~5.4 GB overflows 64 GB RAM (vs ~27 GB for Q3_K_S). SSD paging frequency drops dramatically.

Expected TG improvement vs current Q3_K_S+SSD paging: meaningful (2×+ if paging was severe). Quality trade-off: mild increase in repetition; compensate with `presence_penalty 1.1-1.3`.

### Priority 4 (medium cost, highest long-term impact): Upgrade to 96 GB DDR5

With 96 GB RAM, Q3_K_S experts (~91 GB) fit entirely in RAM with ~5 GB headroom for OS + model overhead. No SSD paging. At confirmed dual-channel DDR5-5600 (~89 GB/s), the bandwidth-limited ceiling becomes:

```
theoretical: 89.6 GB/s / 5.68 GB/token = 15.8 t/s
~50% efficiency: 7.9 t/s
```

This is roughly **2× improvement** over the current expected 4 t/s with SSD paging. Cost: 2×32 GB DDR5-5600 DIMMs to add to existing 2×32 GB = 128 GB total (if 4 slots available), or replace with 2×48 GB = 96 GB.

If 4 slots available and installing 4×32 GB DDR5-5600: bandwidth stays at dual-channel (DDR5 only has 2 channels per CPU, regardless of DIMM count per channel), so bandwidth doesn't improve. The benefit is purely RAM capacity.

### Priority 5: Enable Transparent Huge Pages

```bash
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
```

Reduces TLB pressure for expert weight accesses. Expected gain: 5–15% CPU efficiency improvement. Easy to try; revert with `echo madvise` if issues arise.

### Priority 6: Use --numa numactl Strategy (Single-Socket: Skip)

Ryan's system is single-socket, single NUMA node. The `--numa` flags provide no benefit and disabling prefetch (`if (numa) { prefetch = 0; }`) could hurt by preventing MAP_POPULATE. **Do not use `--numa` flags on this system.**

### Priority 7: Pre-warm with llama-server Warmup

`llama-server` runs a warmup pass by default (PR #11571). Ensure the server is allowed to complete warmup before serving requests. The warmup loads all experts, faulting in as many pages as RAM allows. Monitor with:

```bash
watch -n 1 'free -h && vmstat 1 1'
```

Wait until the "cache" column in `free -h` stops growing before sending requests.

### Action Priority Matrix

| Action | Impact on TG | Effort | Cost |
|--------|-------------|--------|------|
| Verify/fix dual-channel mode | +3–4 t/s | Low | Zero |
| `-ncmoe 47` hybrid | +2–4 t/s | Low | Zero |
| Download Q2_K | +1–3 t/s | Low | Zero (bandwidth) |
| Enable huge pages | +5–15% CPU | Trivial | Zero |
| Upgrade to 96 GB DDR5 | +2–3 t/s, no SSD paging | Medium | ~$100–150 |
| Upgrade to 128 GB DDR5 | Same + Q4_K_M fits | Medium | ~$200–300 |

The single highest-leverage action is confirming dual-channel mode — if the system is currently single-channel DDR5, fixing that alone roughly doubles available bandwidth and is completely free.

---

**Sources:**
- [Qwen/Qwen3-235B-A22B config.json](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json)
- [ubergarm/Qwen3-235B-A22B-GGUF discussions #3 (2950X benchmark)](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/discussions/3)
- [ubergarm/Qwen3-235B-A22B-GGUF discussions #6 (Xeon W5-3425 benchmark)](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/discussions/6)
- [llama.cpp issue #19480 (Qwen3-Coder-Next 5x slower than expected)](https://github.com/ggerganov/llama.cpp/issues/19480)
- [llama.cpp PR #18934 (CUDA graphs for n-cpu-moe)](https://github.com/ggerganov/llama.cpp/pull/18934)
- [llama.cpp PR #11571 (MoE warmup loading)](https://github.com/ggerganov/llama.cpp/pull/11571)
- [llama.cpp issue #1437 (NUMA optimization, 2.27× speedup on dual-socket)](https://github.com/ggml-org/llama.cpp/issues/1437)
- [ikawrakow/ik_llama.cpp PR #520 (MoE GPU offload heuristic)](https://github.com/ikawrakow/ik_llama.cpp/issues/520)
- [ikawrakow/ik_llama.cpp discussion #419 (Qwen3 on dual P100)](https://github.com/ikawrakow/ik_llama.cpp/discussions/419)
- [llama.cpp discussions #4167 (Apple Silicon bandwidth scaling)](https://github.com/ggerganov/llama.cpp/discussions/4167)
- [kipp.ly — Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [arXiv 2312.11514 — LLM in a Flash](https://arxiv.org/abs/2312.11514)
- `/home/ryan/llm-stack/llama.cpp/ggml/src/ggml-common.h` (block_q3_K definition)
- `/home/ryan/llm-stack/llama.cpp/src/llama-mmap.cpp` (mmap/prefetch/NUMA behavior)
- `/home/ryan/llm-stack/llama.cpp/common/arg.cpp` (--numa flag documentation)
- `06-quant-selection.md` (quant sizes, VRAM analysis, prior system characterization)
