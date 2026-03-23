"""Qwen3-32B dense pipeline tuning: context × concurrency for structured output workloads.

## Context

The 32B dense model is the pipeline workhorse — clean JSON output, respects
enable_thinking=false, tool calling works. 19GB model → ~29GB free for KV.
FUSED_MMQ gives +25% at np=16 (global SYCL_ENV has it enabled).

Unlike the 30B MoE (which can't produce structured JSON due to abliteration),
the 32B dense handles all pipeline stages correctly.

Need to find: how much context can we give each slot at various np levels?
The auditor discovery agent accumulates ~2500-3500 tokens per loop iteration.
Churner DOM chunks can be ~1000 tokens. More context = fewer parsing passes.

## Results

(Fill in after running)

## How to run
    cd /home/ryan/llm-stack/scripts
    python3 -m bench dense_pipeline                    # all tests
    python3 -m bench dense_pipeline.np4_c32k           # specific test
"""

from bench.base import BenchTest
from bench.config import BenchConfig

CHURNER_PROMPT = (
    "Extract structured data from this DOM chunk for a wedding venue:\n"
    "<div class='venue-detail'><h1>LBJ Library Lawn</h1>"
    "<p class='desc'>Beautiful outdoor space on the UT campus with views of downtown Austin. "
    "Capacity for up to 200 guests. Features include natural lighting, covered pavilion, "
    "and on-site parking. Popular for weddings, receptions, and corporate events.</p>"
    "<div class='reviews'><div class='review'><span class='rating'>5</span>"
    "<p>Amazing venue, the sunset views were incredible for our ceremony.</p></div>"
    "<div class='review'><span class='rating'>4</span>"
    "<p>Great space but parking can be tricky on game days.</p></div></div>"
    "<div class='pricing'><span class='price'>$3500</span><span class='period'>per event</span></div>"
    "</div>\n\nRespond with JSON containing all extracted fields."
)


class TestDensePipeline(BenchTest):
    """Find optimal context × concurrency for the 32B dense pipeline model."""

    base = BenchConfig(
        model="q4km",
        split_mode="layer",
        n_parallel=4,
        concurrent=4,
        context=32768,
        max_tokens=300,
        immediate_cmdlists=False,
        disable_graph=True,
        extra_flags="-fit off",
    ).with_flags(FUSED_MMQ="1")

    # ── np=16: max throughput (frontier config) ──

    def test_np16_c32k(self):
        """np=16 c=32768 (2048/slot) — proven frontier at 21.7 t/s."""
        self.run(self.base.with_(
            name="dense_np16_c32k",
            n_parallel=16, concurrent=16, context=32768,
        ))

    def test_np16_c48k(self):
        """np=16 c=49152 (3072/slot) — push context."""
        self.run(self.base.with_(
            name="dense_np16_c48k",
            n_parallel=16, concurrent=16, context=49152,
        ))

    def test_np16_c64k(self):
        """np=16 c=65536 (4096/slot) — may OOM."""
        self.run(self.base.with_(
            name="dense_np16_c64k",
            n_parallel=16, concurrent=16, context=65536,
        ))

    # ── np=8: balance throughput and context ──

    def test_np8_c32k(self):
        """np=8 c=32768 (4096/slot) — good for discovery agent."""
        self.run(self.base.with_(
            name="dense_np8_c32k",
            n_parallel=8, concurrent=8, context=32768,
        ))

    def test_np8_c48k(self):
        """np=8 c=49152 (6144/slot) — generous context."""
        self.run(self.base.with_(
            name="dense_np8_c48k",
            n_parallel=8, concurrent=8, context=49152,
        ))

    def test_np8_c64k(self):
        """np=8 c=65536 (8192/slot) — may need -b 256."""
        self.run(self.base.with_(
            name="dense_np8_c64k",
            n_parallel=8, concurrent=8, context=65536, batch=256,
        ))

    # ── np=4: current proxy config, max context per slot ──

    def test_np4_c32k(self):
        """np=4 c=32768 (8192/slot) — current proxy config."""
        self.run(self.base.with_(
            name="dense_np4_c32k",
            n_parallel=4, concurrent=4, context=32768,
        ))

    def test_np4_c48k(self):
        """np=4 c=49152 (12288/slot) — heavy discovery agent."""
        self.run(self.base.with_(
            name="dense_np4_c48k",
            n_parallel=4, concurrent=4, context=49152,
        ))

    def test_np4_c64k(self):
        """np=4 c=65536 (16384/slot) — max push, -b 256."""
        self.run(self.base.with_(
            name="dense_np4_c64k",
            n_parallel=4, concurrent=4, context=65536, batch=256,
        ))

    def test_np4_c96k(self):
        """np=4 c=98304 (24576/slot) — extreme push, -b 256."""
        self.run(self.base.with_(
            name="dense_np4_c96k",
            n_parallel=4, concurrent=4, context=98304, batch=256,
        ))

    # ── Pipeline-realistic: churner prompt at best configs ──

    def test_np8_churner(self):
        """np=8 with churner DOM extraction prompts."""
        self.run(self.base.with_(
            name="dense_np8_churner",
            n_parallel=8, concurrent=8, context=49152,
            prompt=CHURNER_PROMPT,
        ))

    def test_np4_churner(self):
        """np=4 with churner DOM extraction prompts."""
        self.run(self.base.with_(
            name="dense_np4_churner",
            n_parallel=4, concurrent=4, context=49152,
            prompt=CHURNER_PROMPT,
        ))
