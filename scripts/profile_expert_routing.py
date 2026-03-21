#!/usr/bin/env python3
"""Profile expert routing patterns in GLM-4.7-Flash.

Sends prompts to a running llama-server with GGML_SYCL_DEBUG_EXPERT=1
and parses the expert selection logs to build a frequency distribution.

Usage:
    # Terminal 1: start server with expert debug logging
    source env.sglang-xpu.sh
    GGML_SYCL_DEBUG_EXPERT=1 llama.cpp/build-sycl/bin/llama-server \
        -m models/GLM-4.7-Flash-heretic-GGUF/GLM-4.7-Flash-ultimate-irrefusable-heretic-Q4_K_M.gguf \
        -ngl 99 -np 1 -c 2048 --port 18401

    # Terminal 2: run profiler
    python3 scripts/profile_expert_routing.py

Alternative: use gguf metadata + HuggingFace transformers to run
routing-only forward pass (no need for full inference).
"""
import json
import sys
from collections import Counter
from pathlib import Path

# Since we can't easily hook into llama.cpp's MUL_MAT_ID routing
# without C++ changes, let's analyze the model structure first and
# then discuss what profiling approach makes sense.

def analyze_gguf_metadata(model_path: str):
    """Read GGUF metadata to understand MoE structure."""
    import struct

    with open(model_path, "rb") as f:
        # GGUF magic
        magic = f.read(4)
        if magic != b"GGUF":
            print(f"Not a GGUF file: {magic}")
            return

        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        print(f"GGUF v{version}: {n_tensors} tensors, {n_kv} metadata entries")
        print()

        # Read metadata key-value pairs
        moe_keys = {}
        for _ in range(n_kv):
            # Read key
            key_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(key_len).decode("utf-8", errors="replace")

            # Read value type
            vtype = struct.unpack("<I", f.read(4))[0]

            # Read value based on type
            if vtype == 0:  # UINT8
                val = struct.unpack("<B", f.read(1))[0]
            elif vtype == 1:  # INT8
                val = struct.unpack("<b", f.read(1))[0]
            elif vtype == 2:  # UINT16
                val = struct.unpack("<H", f.read(2))[0]
            elif vtype == 3:  # INT16
                val = struct.unpack("<h", f.read(2))[0]
            elif vtype == 4:  # UINT32
                val = struct.unpack("<I", f.read(4))[0]
            elif vtype == 5:  # INT32
                val = struct.unpack("<i", f.read(4))[0]
            elif vtype == 6:  # FLOAT32
                val = struct.unpack("<f", f.read(4))[0]
            elif vtype == 7:  # BOOL
                val = struct.unpack("<B", f.read(1))[0] != 0
            elif vtype == 8:  # STRING
                slen = struct.unpack("<Q", f.read(8))[0]
                val = f.read(slen).decode("utf-8", errors="replace")
            elif vtype == 9:  # ARRAY
                atype = struct.unpack("<I", f.read(4))[0]
                alen = struct.unpack("<Q", f.read(8))[0]
                # Skip array contents
                if atype == 8:  # array of strings
                    val = []
                    for _ in range(alen):
                        slen2 = struct.unpack("<Q", f.read(8))[0]
                        val.append(f.read(slen2).decode("utf-8", errors="replace"))
                elif atype in (0, 1):
                    val = list(f.read(alen))
                elif atype in (2, 3):
                    val = [struct.unpack("<H" if atype == 2 else "<h", f.read(2))[0] for _ in range(alen)]
                elif atype in (4, 5):
                    val = [struct.unpack("<I" if atype == 4 else "<i", f.read(4))[0] for _ in range(alen)]
                elif atype == 6:
                    val = [struct.unpack("<f", f.read(4))[0] for _ in range(alen)]
                else:
                    val = f"<array type={atype} len={alen}>"
                    # Skip unknown array types
                    break
            elif vtype == 10:  # UINT64
                val = struct.unpack("<Q", f.read(8))[0]
            elif vtype == 11:  # INT64
                val = struct.unpack("<q", f.read(8))[0]
            elif vtype == 12:  # FLOAT64
                val = struct.unpack("<d", f.read(8))[0]
            else:
                print(f"  Unknown type {vtype} for key {key}")
                break

            # Filter for MoE-related keys
            if any(k in key.lower() for k in [
                "expert", "moe", "layer", "hidden", "ffn", "intermediate",
                "num_", "n_", "head", "embd", "vocab", "context", "arch",
                "block_count", "dense", "routing"
            ]):
                moe_keys[key] = val

        # Print MoE-relevant metadata
        print("=== Model Architecture ===")
        for key in sorted(moe_keys):
            val = moe_keys[key]
            if isinstance(val, list) and len(val) > 10:
                val = f"[{len(val)} elements]"
            print(f"  {key}: {val}")

        print()

        # Compute expert distribution feasibility
        n_experts = None
        n_expert_used = None
        n_layers = None
        n_ff = None
        n_embd = None

        for key, val in moe_keys.items():
            if "expert_count" in key or "num_local_experts" in key:
                n_experts = val
            elif "expert_used" in key or "num_experts_per_tok" in key:
                n_expert_used = val
            elif "block_count" in key:
                n_layers = val
            elif "feed_forward_length" in key or "intermediate_size" in key:
                n_ff = val
            elif "embedding_length" in key:
                n_embd = val

        if n_experts and n_expert_used:
            print(f"=== Expert Parallelism Analysis ===")
            print(f"  Total experts: {n_experts}")
            print(f"  Active per token: {n_expert_used}")
            print(f"  Layers: {n_layers}")
            print(f"  Hidden dim: {n_embd}")
            print(f"  FFN dim: {n_ff}")
            print()

            # Estimate sizes
            if n_ff and n_embd:
                # Each expert has gate_proj, up_proj, down_proj
                # gate: [n_ff, n_embd], up: [n_ff, n_embd], down: [n_embd, n_ff]
                params_per_expert = 3 * n_ff * n_embd
                bytes_per_expert_f16 = params_per_expert * 2
                bytes_per_expert_q4 = params_per_expert * 0.5625  # ~4.5 bits avg for Q4_K_M

                total_expert_params = params_per_expert * n_experts * (n_layers or 1)
                total_expert_q4 = bytes_per_expert_q4 * n_experts * (n_layers or 1)

                print(f"  Per expert (per layer):")
                print(f"    Parameters: {params_per_expert:,}")
                print(f"    Size F16: {bytes_per_expert_f16 / 1e6:.1f} MB")
                print(f"    Size Q4_K_M: {bytes_per_expert_q4 / 1e6:.1f} MB")
                print()
                print(f"  All experts ({n_experts} × {n_layers} layers):")
                print(f"    Total params: {total_expert_params:,}")
                print(f"    Total Q4_K_M: {total_expert_q4 / 1e9:.2f} GB")
                print()

                # GPU placement analysis
                gpu_budget = 16e9  # 16GB per Arc A770
                backbone_estimate = 4.5e9  # ~4.5GB for non-expert weights

                experts_per_layer = n_experts
                expert_size = bytes_per_expert_q4

                # How many experts fit on GPU 0 with backbone?
                remaining_gpu0 = gpu_budget - backbone_estimate
                experts_on_gpu0 = int(remaining_gpu0 / (expert_size * (n_layers or 1)))
                experts_on_gpu0 = min(experts_on_gpu0, n_experts)

                experts_on_gpu1 = n_experts - experts_on_gpu0
                experts_on_gpu1 = min(experts_on_gpu1, int(gpu_budget / (expert_size * (n_layers or 1))))

                print(f"  === GPU Placement (3x Arc A770 @ 16GB each) ===")
                print(f"  GPU 0: backbone ({backbone_estimate/1e9:.1f}GB) + {experts_on_gpu0}/{n_experts} experts")
                print(f"  GPU 1: {experts_on_gpu1}/{n_experts} remaining experts")
                print(f"  GPU 2: KV cache / hot expert duplicates")
                print()

                # Cross-GPU hit rate estimate
                # If experts are uniformly distributed: P(remote) = experts_on_gpu1 / n_experts
                # With top-k=4: expected remote hits = 4 * P(remote)
                if experts_on_gpu0 < n_experts:
                    p_remote_uniform = experts_on_gpu1 / n_experts
                    expected_remote = n_expert_used * p_remote_uniform
                    print(f"  Cross-GPU analysis (uniform routing):")
                    print(f"    P(expert is remote): {p_remote_uniform:.1%}")
                    print(f"    Expected remote hits per token: {expected_remote:.2f} / {n_expert_used}")
                    print(f"    Expected local hits per token: {n_expert_used - expected_remote:.2f} / {n_expert_used}")
                    print()
                    print(f"  With locality (hot experts on GPU 0):")
                    print(f"    If 80% of hits go to top-20 experts (all on GPU 0):")
                    p_remote_locality = 0.2 * (experts_on_gpu1 / max(1, n_experts - 20))
                    expected_remote_locality = n_expert_used * p_remote_locality
                    print(f"    Expected remote hits per token: {expected_remote_locality:.2f} / {n_expert_used}")
                    print(f"    Cross-GPU transfers per MoE layer: ~{expected_remote_locality:.1f}")
                    print(f"    vs layer-split: 1 full transfer per layer")


if __name__ == "__main__":
    model_path = (
        "/home/ryan/llm-stack/models/GLM-4.7-Flash-heretic-GGUF/"
        "GLM-4.7-Flash-ultimate-irrefusable-heretic-Q4_K_M.gguf"
    )

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    analyze_gguf_metadata(model_path)
