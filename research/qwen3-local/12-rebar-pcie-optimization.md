# ReBAR, PCIe, and BIOS Optimizations for Intel Arc A770 LLM Inference

**Target setup:** 3x Intel Arc A770 16GB, Linux, llama.cpp SYCL backend, no PCIe P2P.
**Research date:** 2026-03-15

---

## 1. Resizable BAR (ReBAR) on Intel Arc A770

### 1.1 Does Arc A770 require ReBAR?

Intel's official position is unambiguous: **"Resizable BAR (or Smart Access Memory) must be enabled for optimal performance when using Intel Arc A-Series Graphics."** The card will still initialize without it, but Intel explicitly states the driver's performance and stability characteristics are designed around ReBAR being present. ([Intel support article 000092416](https://www.intel.com/content/www/us/en/support/articles/000092416/graphics.html))

The Tom's Hardware launch benchmark measured **up to 24% lower performance** at 1080p without ReBAR, averaging around 23% at 1440p and 20% at 4K across gaming workloads. ([Tom's Hardware: Arc A770 Loses Up to 24 Percent Performance Without Resizable BAR](https://www.tomshardware.com/news/arc-a770-loses-25-percent-performance-without-resizable-bar))

No direct LLM inference benchmark exists that isolates ReBAR vs. no-ReBAR delta specifically for token generation. However, the architectural reason is clear: without ReBAR, the CPU can only address a 256 MB window of the GPU's 16 GB GDDR6. The i915 driver logs this as:

```
i915: Using a reduced BAR size of 256MiB. Consider enabling 'Resizable BAR' or similar, if available in the BIOS.
i915: Failed to resize BAR2 to 8192M (-EINVAL)
```

([Proxmox Forum: Reduced BAR on Arc A380](https://forum.proxmox.com/threads/reduced-bar-on-8-1-with-intel-arc-a380.144193/))

With reduced BAR, every DMA transfer from GPU framebuffer to CPU must be windowed through that 256 MB aperture. For LLM inference this matters for: (a) initial model weight loading, (b) any KV cache management that involves host-staged copies between GPUs (your exact use case with no P2P).

### 1.2 How the A770 16GB BAR works

The Arc A770 16GB has a full 16 GB prefetchable BAR. With ReBAR enabled and Above 4G Decoding active, `lspci -v` shows:

```
Memory at 6000000000 (64-bit, prefetchable) [size=16G]
```

Without ReBAR, the BAR is negotiated down to 256 MB. The Linux kernel v5.18+ includes patches that handle the "partial BAR" case for DG2/Arc — even if the BAR can't be sized to full VRAM, the driver ensures all local memory remains *GPU-accessible* (not CPU-accessible through BAR, but usable for GPU compute). ([Phoronix: Intel DG2 ReBAR Linux](https://www.phoronix.com/news/Intel-DG2-ReBAR-Linux-Prepare)) This means compute kernels can still use the full 16 GB for matrix weights; it is the host-to-device and staged inter-GPU copies that suffer without full ReBAR.

### 1.3 Verifying ReBAR on Linux

```bash
# Find your Arc device address
lspci | grep -i "VGA\|Display\|Arc"

# Check BAR size for that device (replace XX:00.0)
sudo lspci -vv -s XX:00.0 | grep -A5 "Resizable BAR\|Memory at"
```

With full ReBAR active, you should see `[size=16G]` in the prefetchable memory region. The "Physical Resizable BAR" capability block will show `current size: 16GB, supported: 16GB`.

**Note on Intel Arc Control false negatives:** Intel Arc Control on Windows sometimes reports "General-Not-Supported" even when ReBAR is active — this is a cosmetic display bug, not an accurate status. ([Intel support article 000093975](https://www.intel.com/content/www/us/en/support/articles/000093975/graphics.html)) On Linux, trust `lspci -vv` output over any GUI tool.

### 1.4 Level Zero / compute-runtime and allocation limits

A related issue that was fixed in mid-2025: `CL_DEVICE_MAX_MEM_ALLOC_SIZE` was historically capped at 4 GB on the A770 16GB due to 32-bit index overflow in the compute runtime. Software like FluidX3D (which needs a single contiguous 82% VRAM allocation) was blocked by this. The fix was merged into intel/compute-runtime and marked COMPLETED in May 2025. ([intel/compute-runtime issue #627](https://github.com/intel/compute-runtime/issues/627)) If you are on an older driver stack, verify `CL_DEVICE_MAX_MEM_ALLOC_SIZE` returns > 4 GB:

```bash
clinfo | grep "Max mem alloc"
```

For llama.cpp SYCL this matters if any single tensor exceeds 4 GB — relevant for large Q8_0 models where the embedding table or large FFN layers may approach that limit.

---

## 2. PCIe Lane Count and LLM Performance

### 2.1 Bandwidth arithmetic

| Configuration      | Theoretical BW (each direction) | Notes                        |
|--------------------|----------------------------------|------------------------------|
| PCIe 4.0 x16       | 32 GB/s                          | A770 max spec                |
| PCIe 4.0 x8        | 16 GB/s                          | Common in multi-slot boards  |
| PCIe 4.0 x4        | 8 GB/s                           | Likely for 3rd slot          |
| PCIe 3.0 x16       | 16 GB/s                          | Older platforms              |
| PCIe 3.0 x8        | 8 GB/s                           |                              |
| PCIe 3.0 x4        | 4 GB/s                           | Severe for inter-GPU copies  |

**A770 on-die memory bandwidth: 560 GB/s GDDR6** ([Intel A770 16GB specs](https://www.intel.com/content/www/us/en/products/sku/229151/intel-arc-a770-graphics-16gb/specifications.html))

The PCIe link is **17–70x narrower** than on-die VRAM bandwidth. This ratio determines when PCIe becomes the bottleneck.

### 2.2 Single-GPU inference: PCIe lane count is nearly irrelevant

Once the model is resident in VRAM, token generation (decode phase) reads weights from GDDR6 at up to 560 GB/s. PCIe bandwidth does not participate in weight reads. Lane count only affects:

- Initial model load time (one-shot transfer, not steady-state performance)
- Partial offload scenarios where weights spill to CPU RAM

Puget Systems benchmarked 4x Titan V at x16 vs x8 for machine learning: single-GPU workloads showed **1.6–2.3% difference** between x16 and x8. Even with 4 GPUs, the gap was only 4–7%. ([Puget Systems: PCIe X16 vs X8 with 4x Titan V](https://www.pugetsystems.com/labs/hpc/pcie-x16-vs-x8-with-4-x-titan-v-gpus-for-machine-learning-1167/))

### 2.3 Multi-GPU inference without P2P: PCIe becomes the bottleneck

Your setup uses host-staged inter-GPU copies (no PCIe P2P, no NVLink). In llama.cpp's `--split-mode row`, each matrix multiplication layer is split across all 3 GPUs in parallel, then an AllReduce-equivalent synchronization is needed before proceeding to the next layer. Without P2P, this synchronization goes:

```
GPU0 result → host RAM → GPU1 (and GPU2)
GPU1 result → host RAM → GPU0 (and GPU2)
GPU2 result → host RAM → GPU0 (and GPU1)
```

For a model like Qwen3-30B-MoE at INT4 (roughly 16 GB total), a single forward pass with sequence length 512 moves partial activations whose size depends on hidden dim. With hidden_dim=7168 (30B-class model), a single token's partial activation per GPU is ~28 KB (fp16). But in batch scenarios, this scales linearly. The bottleneck threshold is:

```
max_t/s = PCIe_BW_per_direction / (bytes_transferred_per_token × num_GPUs)
```

For PCIe 3.0 x4 at 4 GB/s (real throughput ~3 GB/s after protocol overhead), with three-GPU AllReduce staging 3× partial activation copies per token, this can easily become the ceiling before GPU compute.

The dual-A770 Ollama report is illustrative: second GPU on a **PCIe 3.0 x4 slot** caused throttling and progressive slowdown. ([ipex-llm issue #10847](https://github.com/intel-analytics/ipex-llm/issues/10847)) The GPU frequency dropped to 100–300 MHz — the PCIe bottleneck caused kernel submission stalls which the GPU power management interpreted as idle, triggering frequency downscaling. This is a known failure mode.

### 2.4 Layer-split vs row-split under PCIe constraint

llama.cpp's `--split-mode layer` assigns whole transformer layers to individual GPUs. Each GPU works independently on its layers; data passes to the next GPU only between layers (one activation tensor per layer boundary). This is one inter-GPU transfer per layer, with each transfer being one activation tensor of size `batch × seq_len × hidden_dim × dtype_bytes`.

`--split-mode row` requires inter-GPU synchronization *within* each layer (after the partial MatMul), which means far more frequent and smaller transfers — potentially much worse under PCIe host-staging overhead.

Historical llama.cpp data: before a regression in commit b2475, row-split delivered **~40% speedup** over layer-split on PCIe-connected A6000 pairs. After the regression, it matched layer-split. With correct `--ubatch-size` tuning (1024 instead of 512 default), row-split recovered to **~50% speedup** over single GPU. ([llama.cpp issue #6476](https://github.com/ggml-org/llama.cpp/issues/6476))

For a 3x A770 without P2P, **layer-split is the safer and likely faster choice** until SYCL backend P2P support matures.

### 2.5 Real-world PCIe bandwidth measurement

Actual unidirectional PCIe 4.0 x16 host-to-device transfer rates observed in practice: **~24 GB/s for A100 PCIe 4.0**, with bidirectional bandwidth dropping to ~18–19 GB/s due to contention. PCIe 4.0 x8 halves these to ~12 GB/s unidirectional. For your x4 slot (if a third card uses one), expect ~6 GB/s real-world unidirectional.

---

## 3. BIOS Settings for Multi-GPU LLM

### 3.1 Above 4G Decoding (mandatory)

**Must be enabled.** With three 16 GB cards, the PCIe memory address space requirements exceed the legacy 4 GB limit. Without Above 4G Decoding, the BIOS cannot map all three cards' 16 GB BARs into the address space, and cards will be mapped with reduced or no BAR access. Additionally, Above 4G Decoding is a prerequisite for Resizable BAR — on most boards, the ReBAR option does not appear in BIOS until Above 4G Decoding is enabled first. ([Intel Quick Start Guide](https://www.intel.com/content/www/us/en/support/articles/000091128/graphics/intel-arc-dedicated-graphics-family.html))

**Enable order:** Above 4G Decoding → Save & Reboot → then enable Re-Size BAR Support.

### 3.2 Resizable BAR (Re-Size BAR Support)

Enable explicitly. The Intel Quick Start Guide lists the required BIOS settings:
- Above 4G Decoding: **Enabled**
- Re-Size BAR Support: **Enabled** (or Auto)
- CSM / Legacy Mode: **Disabled** (UEFI only)

([Intel Arc Desktop Quick Start Guide](https://www.intel.com/content/www/us/en/support/articles/000091128/graphics/intel-arc-dedicated-graphics-family.html))

With three cards, Linux will automatically negotiate the appropriate BAR size for each GPU during boot if Above 4G Decoding is active. With kernel 6.5+ this negotiation is robust.

### 3.3 CSM / Legacy Mode

Must be **disabled**. Arc A770 requires UEFI boot mode. With CSM enabled, some boards cannot complete POST with multiple Arc cards, or Arc will downgrade to PCIe Gen 1 x1 speeds. This was the cause of the documented "PCIe 2.5GT/s, Width x1" reports — solved by disabling CSM and ensuring UEFI mode. ([Intel boot issue article 000092544](https://www.intel.com/content/www/us/en/support/articles/000092544/graphics.html))

### 3.4 PCIe Generation (3.0 vs 4.0 vs 5.0)

**Leave at Auto or 4.0** — do not force 5.0 if the cards or traces don't support it. Arc A770 is rated to PCIe 4.0 x16. If your platform supports PCIe 5.0 on the primary slot, the card will negotiate to 4.0 x16, which is correct. Forcing "PCIe 5.0" in BIOS can cause link training failures or instability on cards that don't support it.

For the secondary and tertiary slots, verify the actual negotiated width with:
```bash
sudo lspci -vv | grep -E "LnkCap|LnkSta"
```
`LnkSta` shows the active link. `LnkCap` shows the maximum supported. If `LnkSta: Speed 8GT/s, Width x4` for your third card, that is PCIe 3.0 x4 = 4 GB/s — the most constrained scenario.

### 3.5 IOMMU / VT-d

**For bare-metal inference, disable or use passthrough mode.** IOMMU adds address translation overhead to every DMA operation. For GPU workloads on bare metal (no VM, no device passthrough needed), IOMMU provides no security benefit and introduces IOTLB invalidation overhead (~2000 CPU cycles per invalidation). NVIDIA's own documentation for DGX/HGX systems recommends disabling IOMMU for latency-sensitive inference. ([NVIDIA Grace Perf Tuning Guide](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html))

If you run VMs and need IOMMU for passthrough, set `iommu=pt` (passthrough mode) in kernel cmdline rather than full IOMMU translation:
```
intel_iommu=on iommu=pt
```
This allows the GPU to do direct DMA without address translation overhead while still enabling VFIO for other devices.

### 3.6 ASPM (Active State Power Management in BIOS)

Intel's Quick Start Guide recommends **Native ASPM: Enabled** for power management in desktop use. However, for inference workloads where latency matters, consider setting PCIe Link State Power Management to Off or Performance in the OS (not necessarily in BIOS). BIOS-level ASPM setting controls whether ASPM is negotiated at all; OS-level controls the policy.

---

## 4. Linux PCIe and OS Settings

### 4.1 `pcie_aspm=off` kernel parameter

ASPM causes link-state transitions when the PCIe device appears idle, adding latency on the first access after a quiet period. Intel's own Ethernet performance guide recommends disabling ASPM for latency-sensitive workloads. ([Intel PCIe Power Management guide](https://edc.intel.com/content/www/us/en/design/products/ethernet/appnote-perf-tuning-guide-700-series-linux/%E2%80%8Bpcie-power-management/))

To disable system-wide in kernel cmdline (`/etc/default/grub`):
```
GRUB_CMDLINE_LINUX_DEFAULT="... pcie_aspm=off"
```

Or, to set per-policy without full disable:
```bash
echo performance | sudo tee /sys/module/pcie_aspm/parameters/policy
```

**Caution:** `pcie_aspm=force` on hardware that does not support ASPM can cause system hangs. Only use `=off` (safe) or `=performance` (safe policy mode).

### 4.2 CPU frequency governor

Use the `performance` governor for inference servers. The `ondemand` and `schedutil` governors scale CPU frequency based on utilization. During LLM inference, the CPU role is primarily kernel submission to the GPU command queue — it appears lightly loaded but must respond with sub-millisecond latency to GPU completion events. Governors that scale down CPU frequency during apparent idle periods introduce jitter:

> "The CPU may scale down its frequency to save CPU power without considering the high utilization of the GPU, which relies on the CPU to quickly feed it the next operators, potentially elongating end-to-end inference latency." ([arxiv: Dissecting the Impact of Mobile DVFS Governors on LLM Inference](https://arxiv.org/html/2507.02135v1))

```bash
# Set performance governor on all cores
sudo cpupower frequency-set -g performance
# Or per-core
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 4.3 PCIe sysfs settings for Arc A770

Check and optionally disable PCIe link power management at the sysfs level for each GPU:
```bash
# Find Arc PCIe device paths
for dev in /sys/bus/pci/devices/*/; do
  if grep -q "8086" "$dev/vendor" 2>/dev/null; then
    echo "$dev: $(cat $dev/power/control 2>/dev/null)"
  fi
done

# Set auto → on (prevent runtime suspend)
echo on | sudo tee /sys/bus/pci/devices/0000:XX:00.0/power/control
```

### 4.4 intel_gpu_top and XPUM monitoring

`intel_gpu_top` does not expose memory bandwidth counters for Arc discrete GPUs. Use Intel XPU Manager (XPUM) instead:
```bash
xpumcli discovery           # find GPU IDs
xpumcli stats -d 0 -j      # JSON stats including memory BW
```
([Intel Community: checking VRAM and bandwidth on A770](https://community.intel.com/t5/Intel-Graphics-Performance/How-to-check-the-usage-of-video-memory-and-bandwidth-of-A770-on/td-p/1633733))

To check active PCIe link speed and width:
```bash
sudo lspci -vv | grep -E "LnkCap|LnkSta"
# Or for a specific device:
sudo lspci -vv -s 00:02.0 | grep Lnk
```

---

## 5. Intel Arc A770 Specific Findings

### 5.1 ReBAR and the 16 GB framebuffer

Without ReBAR, only 256 MB of the 16 GB framebuffer is CPU-addressable. GPU compute kernels can still access all 16 GB internally (they use GPU-side virtual addresses, not the BAR aperture). The practical impact for LLM inference:

- **GPU-only inference (single GPU):** Weights live in GDDR6, SYCL kernels access them via GPU MMU. ReBAR absence has *minimal impact* on pure throughput once model is loaded.
- **Host-staged inter-GPU copies (your case):** Data must pass through host memory. Without full ReBAR, the copy path is: GPU VRAM → [256 MB aperture, chunked] → CPU → [256 MB aperture, chunked] → other GPU VRAM. This serialization and chunking can severely reduce effective inter-GPU bandwidth, potentially from 10+ GB/s to sub-1 GB/s.

### 5.2 Memory bandwidth efficiency ceiling

The A770 achieves **560 GB/s on-die bandwidth** but not all of it is available to a single workload. The Chips and Cheese microbenchmark found that "the A770 cannot saturate its VRAM bandwidth even with all 32 Xe-cores active" at low occupancy. A single Xe-core accesses only ~8 GB/s — inference decode with batch=1 runs at very low occupancy. ([Chips and Cheese: Microbenchmarking Intel's Arc A770](https://chipsandcheese.com/p/microbenchmarking-intels-arc-a770))

Observed decode throughput from llama.cpp discussions: **36–44 t/s for 7B Q4_0 on a single A770** (SYCL backend, tg128). This implies effective memory bandwidth utilization of roughly:

```
7B × 0.5 bytes/param (Q4) = 3.5 GB per token read
36 t/s × 3.5 GB = 126 GB/s effective utilization
= 22% of the 560 GB/s peak
```

This low efficiency (vs NVIDIA's ~70-90%) is the core architectural issue Codeplay identified: the SYCL `mul_mat_vec_q` kernels have suboptimal memory access patterns. Optimizations are planned (DPAS instruction targeting, better quantization data layout). ([llama.cpp discussion #12570](https://github.com/ggml-org/llama.cpp/discussions/12570))

The Vulkan backend shows **44 t/s** for tg128 vs SYCL's 36 t/s on the same A770 — for generation, Vulkan is currently faster. Prompt processing (prefill) is dramatically faster on SYCL: 1616 t/s vs 95 t/s on Vulkan (Linux), because SYCL can use DPAS/XMX matrix engines for batched MatMul while Vulkan cannot yet. ([llama.cpp discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879))

### 5.3 Level Zero memory allocation path

Level Zero (the backend for SYCL on Intel GPUs) allocates device memory in the GPU's local memory pool. When `zeMemAllocDevice` is called, the allocation does not require a BAR mapping — the GPU MMU maps GDDR6 directly. BAR-mapped memory is only used for `zeMemAllocHost` (pinned host memory) or `zeMemAllocShared` (unified memory that may migrate). For inference workloads, model weights are typically device allocations and do not traverse the BAR for compute access. The BAR only matters for explicit `zeCommandListAppendMemoryCopy` host↔device transfers.

---

## 6. Practical Impact on LLM Inference Throughput

### 6.1 Is inter-GPU transfer the bottleneck or GPU compute?

For your row-split 3x A770 configuration running Qwen3-30B-MoE (or similar):

**Scenario A: Model fits entirely in 3× 16 GB = 48 GB VRAM**

Single-GPU decode speed per card: ~15–20 t/s (estimated from scaling; 30B model is ~4× heavier than 7B, so roughly 36/4 = ~9 t/s per card baseline, but row-split reduces per-card weight by 3× giving ~27 t/s theoretical if bandwidth saturates 3× more efficiently).

Inter-GPU transfer per token with row-split requires exchanging partial activation tensors. For Qwen3-30B, hidden_dim ≈ 7168, in bfloat16 = 14 KB per activation vector per GPU pair. With 3 GPUs and 28 layers per expert group, budget per token: ~3 × 28 × 14 KB = ~1.2 MB of inter-GPU traffic per token.

At PCIe 4.0 x8 effective bandwidth (~10 GB/s real unidirectional with overhead):
```
1.2 MB / 10 GB/s ≈ 0.12 ms/token for transfers alone
= maximum ~8,300 t/s from PCIe bandwidth alone
```

This means for *single-user decode*, PCIe is not the bottleneck at x8 — GPU compute/memory bandwidth is. PCIe only becomes the bottleneck at high batch sizes or extremely fast GPUs.

**Scenario B: Third GPU on PCIe 3.0 x4 (real BW ~3 GB/s)**

At 3 GB/s: `1.2 MB / 3 GB/s ≈ 0.40 ms/token` for that link alone. This caps you at ~2,500 t/s theoretical from that inter-GPU path — still far above what a single A770 achieves, so still not the primary bottleneck for single-stream inference.

**The real bottleneck is GPU kernel efficiency.** At ~22% VRAM bandwidth utilization and 560 GB/s peak, there is 4× headroom if the SYCL backend were fully optimized. Fix kernel efficiency before optimizing PCIe topology.

### 6.2 PCIe as bottleneck: when does it matter?

PCIe host-staged inter-GPU copies become the throughput ceiling when:
1. Batch size is high (many concurrent users) — each extra batch item multiplies activation transfer size
2. The third GPU is on PCIe 3.0 x4 and model hidden dimensions are large
3. GPU compute reaches its efficiency ceiling (future optimized SYCL backend)

For now, with batch=1 to batch=4, the GPU is the bottleneck. Verify with `xpumcli stats` — if GPU utilization is below 80% and you see inter-GPU sync waiting in traces, PCIe is contributing. If GPU utilization is 95%+, PCIe is not the current ceiling.

---

## 7. Community Reports: ReBAR + Arc A770 Inference

Direct community reports comparing LLM inference with/without ReBAR on Arc A770 are sparse. The available evidence:

- Gaming benchmarks show consistent 20–24% improvement with ReBAR enabled. ([Tom's Hardware](https://www.tomshardware.com/news/arc-a770-loses-25-percent-performance-without-resizable-bar))
- The Proxmox forum documented that without ReBAR (reduced BAR = 256 MB), the Arc A380 was practically unusable for compute-intensive tasks, with slow boot and degraded performance. ([Proxmox Forum](https://forum.proxmox.com/threads/reduced-bar-on-8-1-with-intel-arc-a380.144193/))
- The dual-A770 Ollama issue documented performance degradation attributed partly to one card being on PCIe 3.0 x4 — driver mismatch was also suspected, highlighting that the full optimization chain (correct PCIe width, correct driver, ReBAR active) must all be satisfied simultaneously. ([ipex-llm #10847](https://github.com/intel-analytics/ipex-llm/issues/10847))
- vLLM on Arc A770: 21.7 t/s for Qwen3-4B-Thinking in single-user mode. ([Roger Ngo: Arc A770 with vLLM](https://www.rogerngo.com/blog/arc-a770-with-vllm)) No ReBAR vs. no-ReBAR comparison was documented.

The absence of a specific LLM-inference ReBAR ablation study reflects that Arc A770 LLM users generally accept ReBAR as a prerequisite and do not test without it.

---

## Recommended Changes to Deployment Plan

### BIOS — Must Verify/Enable

| Setting | Required Value | Priority |
|---------|---------------|----------|
| Above 4G Decoding | **Enabled** | Critical |
| Re-Size BAR Support | **Enabled** | Critical |
| CSM / Legacy Mode | **Disabled** (UEFI only) | Critical |
| PCIe Generation | **Auto** or **Gen 4** | High |
| IOMMU / VT-d | **Disabled** (bare metal) or `iommu=pt` | High |
| ASPM | **Disabled** or set OS-side to performance | Medium |

**Verification after BIOS change:**
```bash
# Confirm ReBAR 16G on each Arc card
sudo lspci -vv | grep -E "Memory at|Resizable BAR" | grep -A1 "Resizable"

# Confirm PCIe gen and width for all three cards
sudo lspci -vv | grep -E "LnkCap|LnkSta"

# Check for i915 reduced BAR warning
sudo dmesg | grep -i "BAR\|rebar\|resiz"
```

### Linux Kernel/OS — Recommended Changes

```bash
# 1. Kernel cmdline: disable ASPM, set IOMMU passthrough
# In /etc/default/grub:
GRUB_CMDLINE_LINUX_DEFAULT="... pcie_aspm=off intel_iommu=on iommu=pt"
# Then: sudo update-grub && sudo reboot

# 2. CPU governor: set to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 3. Prevent PCIe runtime power management for Arc cards
# (replace with actual device addresses from lspci)
for addr in 0000:01:00.0 0000:02:00.0 0000:03:00.0; do
  echo on | sudo tee /sys/bus/pci/devices/$addr/power/control
done

# 4. Verify compute-runtime allocation limit (should be > 4GB after fix)
clinfo | grep "Max mem alloc"
```

### llama.cpp SYCL Backend — Configuration

1. **Use `--split-mode layer`** (not row) for 3x A770 without P2P. Layer-split has one inter-GPU transfer per layer boundary vs. row-split's intra-layer synchronization. Under host-staged PCIe routing, this reduces synchronization frequency significantly.

2. **Verify the third card's PCIe width** with `lspci -vv`. If it is on x4, consider whether the model can fit on 2x A770 (32 GB) instead, eliminating the bandwidth-constrained link.

3. **Set `--ubatch-size 1024`** (not the default 512) when using row-split if you test it — this was the fix for the documented row-split regression.

4. **Monitor GPU efficiency** with `xpumcli stats -d 0 1 2 -j` during inference. Target: GPU engine utilization > 80%. If lower, the bottleneck is kernel dispatch overhead or model architecture, not PCIe.

5. **Consider Vulkan backend** for generation-heavy workloads: Vulkan currently shows slightly higher tg128 speed (44 t/s vs 36 t/s) than SYCL on a single A770. SYCL is superior for prompt processing (prefill). For chat use cases with short prompts, Vulkan may outperform SYCL until the SYCL `mul_mat_vec_q` kernel optimization lands.

### Priority Order of Changes

1. **Immediately verify ReBAR is active** (`lspci -vv`). If not, fix BIOS (Above 4G Decoding → ReBAR) — this is the highest-impact single setting for Arc A770.
2. **Check all three PCIe link widths**. If the third card is on x4, evaluate moving to a different slot or accepting the bandwidth limitation.
3. **Set CPU governor to performance** — low-cost, immediate effect on inference latency.
4. **Add `pcie_aspm=off` to kernel cmdline** — reduces tail latency for inter-GPU synchronization.
5. **Disable IOMMU or set `iommu=pt`** for bare-metal inference — moderate latency improvement for DMA operations.
6. **Verify `CL_DEVICE_MAX_MEM_ALLOC_SIZE > 4 GB`** — required if using large Q8_0 models with single tensors > 4 GB.

---

## Sources

- [Intel: Do You Need a Resizable BAR to Use Intel Arc A-Series Graphics?](https://www.intel.com/content/www/us/en/support/articles/000092416/graphics.html)
- [Tom's Hardware: Arc A770 Loses Up to 24 Percent Performance Without Resizable Bar](https://www.tomshardware.com/news/arc-a770-loses-25-percent-performance-without-resizable-bar)
- [Intel: What Is Resizable BAR and How Do I Enable It?](https://www.intel.com/content/www/us/en/support/articles/000090831/graphics.html)
- [Intel: Arc Control Software Shows Resizable BAR as Not Supported](https://www.intel.com/content/www/us/en/support/articles/000093975/graphics.html)
- [Intel: Arc A-Series Desktop Quick Start Guide](https://www.intel.com/content/www/us/en/support/articles/000091128/graphics/intel-arc-dedicated-graphics-family.html)
- [Intel: Arc A770 16GB Product Specifications](https://www.intel.com/content/www/us/en/products/sku/229151/intel-arc-a770-graphics-16gb/specifications.html)
- [Phoronix: Intel Preparing Resizable BAR Support for Arc on Linux](https://www.phoronix.com/news/Intel-DG2-ReBAR-Linux-Prepare)
- [PC Gamer: Linux users running Intel Arc GPUs will benefit from FPS-boosting Resizable BAR](https://www.pcgamer.com/linux-users-running-intel-arc-gpus-will-benefit-from-fps-boosting-resizable-bar/)
- [Proxmox Forum: Reduced BAR on 8.1 with Intel Arc A380](https://forum.proxmox.com/threads/reduced-bar-on-8-1-with-intel-arc-a380.144193/)
- [Proxmox Forum: How to get Resizable BAR working](https://forum.proxmox.com/threads/how-to-get-resizable-bar-rebar-working.151895/)
- [intel/compute-runtime issue #627: Too low 4GB allocation limit on Intel Arc GPUs](https://github.com/intel/compute-runtime/issues/627)
- [intel/compute-runtime issue #586: Visible Memory on Intel Arc GPUs for OpenCL and Level Zero](https://github.com/intel/compute-runtime/issues/586)
- [Intel: Boot Issue with Intel Arc A750 or A770 Graphics Cards](https://www.intel.com/content/www/us/en/support/articles/000092544/graphics.html)
- [Chips and Cheese: Microbenchmarking Intel's Arc A770](https://chipsandcheese.com/p/microbenchmarking-intels-arc-a770)
- [Puget Systems: PCIe X16 vs X8 with 4x Titan V GPUs for Machine Learning](https://www.pugetsystems.com/labs/hpc/pcie-x16-vs-x8-with-4-x-titan-v-gpus-for-machine-learning-1167/)
- [llama.cpp GitHub Discussion #12570: Current status of Intel Arc GPUs](https://github.com/ggml-org/llama.cpp/discussions/12570)
- [llama.cpp GitHub Discussion #10879: Performance of llama.cpp with Vulkan](https://github.com/ggml-org/llama.cpp/discussions/10879)
- [llama.cpp GitHub Issue #4055: Multi GPU CUDA — 8x performance degradation with tensor split](https://github.com/ggml-org/llama.cpp/issues/4055)
- [llama.cpp GitHub Issue #6476: Multi GPU --split-mode row speed regression](https://github.com/ggml-org/llama.cpp/issues/6476)
- [llama.cpp GitHub Issue #9097: Throughput does not scale with batch sizes on Intel GPUs](https://github.com/ggml-org/llama.cpp/issues/9097)
- [ipex-llm GitHub Issue #10847: Running 2x A770 with Ollama, inference slowdown](https://github.com/intel-analytics/ipex-llm/issues/10847)
- [Roger Ngo: Arc A770 with vLLM benchmark](https://www.rogerngo.com/blog/arc-a770-with-vllm)
- [Intel: Run LLMs on Intel GPUs Using llama.cpp](https://www.intel.com/content/www/us/en/developer/articles/technical/run-llms-on-gpus-using-llama-cpp.html)
- [NVIDIA: Understanding PCIe Configuration for Maximum Performance](https://enterprise-support.nvidia.com/s/article/understanding-pcie-configuration-for-maximum-performance)
- [NVIDIA Grace Performance Tuning Guide: OS Settings](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html)
- [Intel PCIe Power Management Guide for Ethernet 700 Series](https://edc.intel.com/content/www/us/en/design/products/ethernet/appnote-perf-tuning-guide-700-series-linux/%E2%80%8Bpcie-power-management/)
- [Red Hat: Active-State Power Management (ASPM)](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/power_management_guide/aspm)
- [Intel Community: Checking VRAM and bandwidth of A770 on Ubuntu](https://community.intel.com/t5/Intel-Graphics-Performance/How-to-check-the-usage-of-video-memory-and-bandwidth-of-A770-on/td-p/1633733)
- [Intel Community: A770 PCIe link negotiates for x1 on 3.0 x16](https://community.intel.com/t5/Intel-ARC-Graphics/A770-PCIe-link-negotiates-for-x1-on-a-3-0-x16/td-p/1454000)
- [arxiv: Dissecting the Impact of Mobile DVFS Governors on LLM Inference](https://arxiv.org/html/2507.02135v1)
- [Medium: LLM Performance and PCIe Lanes: Key Considerations](https://www.glukhov.org/post/2025/06/llm-performance-and-pci-lanes/)
