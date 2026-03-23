"""MoE pipeline tuning: context × concurrency balance for data pipeline workloads.

## Context

The data pipeline has 4 concurrent workload types hitting Henry simultaneously:
- **Discord**: short Q&A, ~20 token prompts, ~100 token responses
- **Site Auditor**: a11y tree analysis, ~200 token prompts, ~300 token responses
- **Grabber**: extraction validation, ~50 token prompts, ~100 token JSON
- **Churner**: DOM→structured data, ~500-1000 token prompts, ~300 token JSON

All need to run concurrently. Qwen3-30B-A3B MoE is the flagship at 26.8 t/s
(test_moe_churn.py) — no Q8_1 concurrent bug, supports np=16.

Key question: what's the optimal context:slot balance?
- More context per slot = longer prompts (churner DOM chunks)
- More slots = higher aggregate throughput
- Model is 17.3 GB → ~30.7 GB free for KV across 3x A770

From test_moe_churn.py: context has ZERO throughput cost up to c=4096 at np=16.
But the pipeline needs longer prompts (churner sends ~1000 tokens). Need to find
where context starts to cost throughput and pick the right np×context tradeoff.

## Results (2026-03-23)

| Config | tok/slot | Agg t/s | Status |
|--------|----------|---------|--------|
| np=16 c=8k   | 512  | 24.7 | too small for discovery |
| np=16 c=16k  | 1024 | 23.0 | tight for discovery |
| np=16 c=24k  | 1536 | 20.7 | ✅ LOCKED — covers all stages |
| np=16 c=32k  | 2048 | timeout | OOM (b=2048) |
| np=8  c=24k  | 3072 | 13.3 | comfortable but slow |
| np=8  c=32k  | 4096 | 12.4 | generous but half throughput |
| np=8  c=48k  | 6144 | crash | OOM even with -b 256 |
| np=4  c=32k  | 8192 | crash | OOM even with -b 256 |

**Decision: np=16 c=24576 → 20.7 t/s, 1536 tok/slot.**
Proxy updated. Discovery agent compacts to ~1500 tok via context.py.

## How to run
    cd /home/ryan/llm-stack/scripts
    python3 -m bench moe_pipeline                   # all tests
    python3 -m bench moe_pipeline.np16_c8k           # specific test
"""

from bench.base import BenchTest
from bench.config import BenchConfig

# Representative pipeline prompts
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
)  # ~200 tokens

AUDITOR_PROMPT = (
    "You are a web research expert. Based on this page snapshot:\n"
    "- heading 'Venues' [level=1]\n- link 'LBJ Library' [ref=e1]\n"
    "- link 'Barton Creek' [ref=e2]\n- link 'Next Page' [ref=e3]\n"
    "Which elements are entity listings? Respond with JSON."
)  # ~80 tokens


class TestMoEPipeline(BenchTest):
    """Find optimal context × concurrency for the data pipeline on 30B MoE."""

    base = BenchConfig(
        model="qwen30b-ablit-q4km",
        split_mode="layer",
        n_parallel=16,
        concurrent=16,
        context=8192,
        max_tokens=300,
        immediate_cmdlists=False,
        disable_graph=True,
        no_warmup=True,
        flash_attn=False,
    )

    # ── Context scaling at np=16 (extend beyond test_moe_churn c=4096) ──

    def test_np16_c8k(self):
        """np=16 c=8192 (512 tok/slot) — does context start costing here?"""
        self.run(self.base.with_(name="pipeline_np16_c8k", context=8192))

    def test_np16_c16k(self):
        """np=16 c=16384 (1024 tok/slot) — enough for churner prompts."""
        self.run(self.base.with_(name="pipeline_np16_c16k", context=16384))

    def test_np16_c32k(self):
        """np=16 c=32768 (2048 tok/slot) — headroom for large DOM chunks."""
        self.run(self.base.with_(name="pipeline_np16_c32k", context=32768))

    def test_np16_c24k(self):
        """np=16 c=24576 (1536 tok/slot) — between 16k and 32k ceiling."""
        self.run(self.base.with_(name="pipeline_np16_c24k", context=24576))

    # ── Fewer slots, more context per slot ──
    # Round 1 (b=2048) crashed at c=32k — OOM on compute buffers.
    # Round 2: -b 256 halves compute buffer allocation (proven on 35B).

    def test_np8_c24k(self):
        """np=8 c=24576 (3072 tok/slot) — discovery agent sweet spot?"""
        self.run(self.base.with_(
            name="pipeline_np8_c24k",
            n_parallel=8, concurrent=8, context=24576, batch=256,
        ))

    def test_np8_c32k(self):
        """np=8 c=32768 (4096 tok/slot) — with -b 256 to avoid OOM."""
        self.run(self.base.with_(
            name="pipeline_np8_c32k",
            n_parallel=8, concurrent=8, context=32768, batch=256,
        ))

    def test_np8_c48k(self):
        """np=8 c=49152 (6144 tok/slot) — max push with -b 256."""
        self.run(self.base.with_(
            name="pipeline_np8_c48k",
            n_parallel=8, concurrent=8, context=49152, batch=256,
        ))

    def test_np4_c32k(self):
        """np=4 c=32768 (8192 tok/slot) — Discord + 3 pipeline slots."""
        self.run(self.base.with_(
            name="pipeline_np4_c32k",
            n_parallel=4, concurrent=4, context=32768, batch=256,
        ))

    def test_np4_c32k(self):
        """np=4 c=32768 (8192 tok/slot) — Discord + 3 pipeline slots."""
        self.run(self.base.with_(
            name="pipeline_np4_c32k",
            n_parallel=4, concurrent=4, context=32768,
        ))

    # ── Pipeline-realistic prompts (not default TCP/UDP) ──

    def test_np16_churner(self):
        """np=16 with churner-sized prompts (DOM extraction)."""
        self.run(self.base.with_(
            name="pipeline_np16_churner",
            context=16384,
            prompt=CHURNER_PROMPT,
        ))

    def test_np16_auditor(self):
        """np=16 with auditor-sized prompts (a11y tree analysis)."""
        self.run(self.base.with_(
            name="pipeline_np16_auditor",
            context=16384,
            prompt=AUDITOR_PROMPT,
            max_tokens=200,
        ))

    def test_np8_long_churner(self):
        """np=8 with 5x churner prompt (~1000 tokens) — large DOM extraction."""
        self.run(self.base.with_(
            name="pipeline_np8_long_churner",
            n_parallel=8, concurrent=8, context=32768,
            prompt=CHURNER_PROMPT * 5,
            max_tokens=300,
        ))
