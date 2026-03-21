#!/usr/bin/env python3
"""Nuclear abliteration of Qwen3-32B for Henry.

Run: python scripts/obliterate_henry.py

This will:
1. Download Qwen3-32B-Instruct (~64 GB)
2. Run nuclear abliteration with custom + builtin prompts
3. Save the abliterated model to models/henry-obliterated/
4. You then quantize with: ./llama.cpp/build/bin/llama-quantize ...

Expected: 2-4 hours on CPU, ~64 GB RAM needed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════════════
#  CUSTOM PROMPTS — ADD YOURS HERE
#
#  These are things you want Henry to STOP refusing.
#  The more varied the phrasing, the more thoroughly
#  those refusal patterns get removed from the weights.
# ═══════════════════════════════════════════════════════════════════════

CUSTOM_HARMFUL = [

]

# Matching harmless prompts (same count as CUSTOM_HARMFUL).
# These help the model learn what refusal looks like vs normal behavior.
CUSTOM_HARMLESS = [

]

# ═══════════════════════════════════════════════════════════════════════
#  PIPELINE CONFIG — don't touch unless you know what you're doing
# ═══════════════════════════════════════════════════════════════════════

# Qwen3-8B on pure VRAM — 16 GB FP16 fits on one GPU with room for the
# deepcopy OBLITERATUS needs during PROBE. Nuclear in ~15-30 min.
MODEL_ID = "Qwen/Qwen3-8B"
OUTPUT_DIR = str(ROOT / "models" / "henry-8b-obliterated")
METHOD = "nuclear"  # Maximum force combo


def main():
    from obliteratus.abliterate import AbliterationPipeline, METHODS
    from obliteratus.prompts import BUILTIN_HARMFUL, BUILTIN_HARMLESS

    # Merge custom + builtin prompts
    harmful = list(CUSTOM_HARMFUL) + list(BUILTIN_HARMFUL)
    harmless = list(CUSTOM_HARMLESS) + list(BUILTIN_HARMLESS)

    # Ensure equal length (nuclear uses SVD which needs paired prompts)
    min_len = min(len(harmful), len(harmless))
    harmful = harmful[:min_len]
    harmless = harmless[:min_len]

    print(f"{'═' * 60}")
    print(f" OBLITERATING: {MODEL_ID}")
    print(f" Method: {METHOD} ({METHODS[METHOD]['label']})")
    print(f" Prompts: {len(harmful)} harmful + {len(harmless)} harmless")
    print(f" Custom: {len(CUSTOM_HARMFUL)} custom prompts added")
    print(f" Output: {OUTPUT_DIR}")
    print(f"{'═' * 60}")

    if len(CUSTOM_HARMFUL) != len(CUSTOM_HARMLESS):
        print(f"\n⚠ WARNING: {len(CUSTOM_HARMFUL)} custom harmful but {len(CUSTOM_HARMLESS)} harmless.")
        print("  Each harmful prompt needs a matching harmless one for SVD.")
        print("  Truncating to shorter list.\n")

    pipeline = AbliterationPipeline(
        model_name=MODEL_ID,
        output_dir=OUTPUT_DIR,
        device="xpu",
        dtype="float16",
        method=METHOD,
        refinement_passes=3,
        n_directions=6,
        harmful_prompts=harmful,
        harmless_prompts=harmless,
        large_model_mode=True,
        on_log=lambda msg: print(f"  > {msg}"),
    )

    pipeline.run()

    print(f"\n{'═' * 60}")
    print(f" DONE — abliterated model saved to: {OUTPUT_DIR}")
    print(f"")
    print(f" Next steps:")
    print(f"   1. Convert to GGUF:")
    print(f"      python llama.cpp/convert_hf_to_gguf.py {OUTPUT_DIR} --outfile {OUTPUT_DIR}/henry-f16.gguf --outtype f16")
    print(f"   2. Quantize to Q8_0:")
    print(f"      ./llama.cpp/build/bin/llama-quantize {OUTPUT_DIR}/henry-f16.gguf {OUTPUT_DIR}/henry-Q8_0.gguf Q8_0")
    print(f"   3. Update arcllm-proxy.py model path")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
