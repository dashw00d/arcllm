"""All tunable parameters in one frozen dataclass."""

from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # llm-stack/

MODELS = {
    "q4km": ROOT / "models/Qwen/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf",
    "9b-q8": ROOT / "models/Qwen/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf",
    "qwen30b-ablit-q4km": ROOT
    / "models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf",
    "qwen35-35b-unsloth": ROOT
    / "models/Qwen/Qwen3.5-35B-A3B-unsloth-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf",
    "qwen35-35b-hauhau": ROOT
    / "models/Qwen/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-GGUF/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
}

DEFAULT_PROMPT = (
    "Explain the difference between TCP and UDP in networking. "
    "Include when you would use each."
)


@dataclass(frozen=True)
class BenchConfig:
    """Every tunable lever. Immutable — use .with_() to derive variants."""

    # Model
    model: str = "q4km"  # key into MODELS, or absolute path

    # Server flags
    split_mode: str = "layer"  # layer | row
    ngl: int = 999
    tensor_split: str = "1,1,1"
    context: int = 32768  # total context across all slots
    flash_attn: bool = True
    n_parallel: int = 1  # -np
    batch: int = 2048  # -b
    ubatch: int = 512  # -ub
    reasoning_budget: int = 0  # 0 = disabled
    cache_reuse: int = 0
    threads: int = 0  # 0 = default
    kv_quant: str = ""  # "" = f16, "q8_0", "q4_0"
    no_warmup: bool = False  # skip warmup run
    extra_flags: str = ""  # raw flags appended to cmd

    # SYCL env
    disable_graph: bool = True  # GGML_SYCL_DISABLE_GRAPH
    immediate_cmdlists: bool = False  # SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS
    affinity: str = "0,1,2"  # ZE_AFFINITY_MASK
    row_events: bool = False  # GGML_SYCL_ROW_EVENTS (OOO queue + event sync)

    # Experimental kernel flags — each maps to a GGML_SYCL_* env var.
    # Add flags here to A/B test kernel changes. Once proven, remove the
    # flag and make the behavior default. The test file docstring documents
    # what each flag does, results, and when it graduated.
    #
    # Example: {"FUSED_MMQ": "1", "FUSED_XMX": "1"}
    # → sets GGML_SYCL_FUSED_MMQ=1, GGML_SYCL_FUSED_XMX=1
    sycl_flags: tuple[tuple[str, str], ...] = ()  # frozen-safe dict

    # Test params
    concurrent: int = 1
    max_tokens: int = 200
    prompt: str = DEFAULT_PROMPT
    timeout: int = 300  # per request

    # Build tracking
    build: str = "build-sycl"  # subdir under llama.cpp/
    patches: tuple[str, ...] = ()  # descriptions of applied patches

    # Metadata
    name: str = ""
    notes: str = ""

    def with_(self, **kw) -> BenchConfig:
        return replace(self, **kw)

    def with_flags(self, **flags: str) -> BenchConfig:
        """Derive a config with additional experimental SYCL flags.

        Usage: config.with_flags(FUSED_MMQ="1", FUSED_XMX="1")
        → sets GGML_SYCL_FUSED_MMQ=1, GGML_SYCL_FUSED_XMX=1
        """
        merged = dict(self.sycl_flags)
        merged.update(flags)
        return replace(self, sycl_flags=tuple(merged.items()))

    @property
    def model_path(self) -> Path:
        return MODELS[self.model] if self.model in MODELS else Path(self.model)

    @property
    def server_bin(self) -> Path:
        return ROOT / "llama.cpp" / self.build / "bin" / "llama-server"

    def server_args(self) -> list[str]:
        a = [
            str(self.server_bin),
            "--model",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            "8400",
            "--split-mode",
            self.split_mode,
            "-ngl",
            str(self.ngl),
            "--tensor-split",
            self.tensor_split,
            "-c",
            str(self.context),
            "-np",
            str(self.n_parallel),
            "-b",
            str(self.batch),
            "-ub",
            str(self.ubatch),
        ]
        if self.flash_attn:
            a += ["-fa", "on"]
        else:
            a += ["-fa", "off"]
        if self.reasoning_budget > 0:
            a += ["--reasoning-budget", str(self.reasoning_budget)]
        if self.cache_reuse > 0:
            a += ["--cache-reuse", str(self.cache_reuse)]
        if self.threads > 0:
            a += ["--threads", str(self.threads)]
        if self.kv_quant:
            a += ["-ctk", self.kv_quant, "-ctv", self.kv_quant]
        if self.no_warmup:
            a += ["--no-warmup"]
        if self.extra_flags:
            a += self.extra_flags.split()
        return a

    def sycl_env(self) -> dict[str, str]:
        env = {
            "GGML_SYCL_DISABLE_GRAPH": "1" if self.disable_graph else "0",
            "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1"
            if self.immediate_cmdlists
            else "0",
            "ZE_AFFINITY_MASK": self.affinity,
            "ZES_ENABLE_SYSMAN": "1",
        }
        if self.row_events:
            env["GGML_SYCL_ROW_EVENTS"] = "1"
        for key, val in self.sycl_flags:
            env[f"GGML_SYCL_{key}"] = val
        return env

    def summary(self) -> str:
        p = [
            f"model={self.model}",
            f"np={self.n_parallel}",
            f"c={self.context}",
            f"×{self.concurrent}",
            f"max_tok={self.max_tokens}",
        ]
        if self.split_mode != "layer":
            p.append(f"split={self.split_mode}")
        if self.reasoning_budget:
            p.append(f"think={self.reasoning_budget}")
        if not self.disable_graph:
            p.append("graph=ON")
        if self.row_events:
            p.append("row_events=ON")
        for key, val in self.sycl_flags:
            p.append(f"{key.lower()}={val}")
        if self.immediate_cmdlists:
            p.append("cmdlist=imm")
        if self.kv_quant:
            p.append(f"kv={self.kv_quant}")
        if self.patches:
            p.append(f"patches={len(self.patches)}")
        return " ".join(p)
