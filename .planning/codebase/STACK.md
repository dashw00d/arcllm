# Technology Stack

**Analysis Date:** 2026-03-17

## Languages

**Primary:**
- Python 3.12 - Main scripting language for proxy, benchmark framework, and utilities
- C++ - Core llama.cpp runtime for LLM inference
- C - GGML tensor operations backend

**Secondary:**
- Bash - Environment setup, wrapper scripts, build orchestration

## Runtime

**Environment:**
- Conda (Miniforge3) with custom environment `.conda-sgl-xpu312` containing Intel SYCL toolchain
- Intel Data Center GPU Flex 170 / Arc A770 runtime via Level Zero driver
- Intel oneAPI toolchain (DPC++ compiler, oneMKL, TBB)

**Package Manager:**
- Python: `pip` via conda environment
- Poetry: For Python project packaging (llama.cpp/pyproject.toml)
- CMake: Build system for llama.cpp

**Build System:**
- CMake 3.14+ configured for SYCL compilation
- Ninja (used in build-sycl directory)
- ccache for incremental compilation

## Frameworks

**Core:**
- llama.cpp (local fork in `llama.cpp/`) - Inference runtime with SYCL backend for Intel Arc GPUs
  - GGML backend with SYCL execution (GGML_SYCL=ON)
  - llama-server HTTP API endpoint (`.cpp/build-sycl/bin/llama-server`)
  - Supports tensor parallelism, layer/row splitting, KV cache quantization

**Testing:**
- Custom benchmark framework at `scripts/bench/` - Python test harness with GPU monitoring
- No external test framework (pure Python orchestration)

**Build/Dev:**
- CMake - Build configuration and compilation
- Intel SYCL compiler (icx/icpx) - GPU code generation
- DPCT/oneMKL - oneAPI Math Kernel Library for optimized operations
- Git (version control)

## Key Dependencies

**Critical (Python):**
- `transformers` ^4.35.2 - Hugging Face model loading and tokenization
- `numpy` ^1.25.0 - Numerical computations (pyproject.toml)
- `torch` ^2.2.0 - PyTorch (CPU-only wheels, source: pytorch.org)
- `huggingface_hub` >=0.34.0 - Model downloading from Hugging Face Hub
- `sentencepiece` >=0.1.98 - Tokenization for Qwen models
- `pydantic` - Data validation (implied by llama.cpp scripts)
- `protobuf` >=4.21.0 - Protocol buffer serialization

**Infrastructure (Intel SYCL/oneAPI):**
- Intel DPC++ compiler (from conda) - SYCL C++ compilation to SPIR-V
- Intel SYCL Runtime (libsycl) - Runtime for SYCL kernels
- Level Zero (libze_loader) - GPU driver abstraction
- oneMKL (libmkl_core, libmkl_sequential) - Optimized math kernels
- TBB (libtbb) - Threading Building Blocks for parallelism
- UMF (Unified Memory Framework) - GPU memory management
- OpenSSL, libssh2, c-ares, zstd - Supporting libraries for conda environment

**Python Utilities:**
- `requests` ~2.32.3 - HTTP client (imported in bench)
- `aiohttp` ~3.9.3 - Async HTTP client
- `prometheus-client` ~0.20.0 - Metrics collection (optional)
- `openai` ~2.14.0 - OpenAI API client for testing compatibility
- `pytest` ~8.3.3 - Testing framework (dev dependency)

## Configuration

**Environment:**
- `env.sglang-xpu.sh` - Primary environment setup script
  - Sources conda activation
  - Sets `CONDA_PREFIX`, `SYCL_ROOT`, `ONEAPI_ROOT`
  - Preloads oneMKL libraries via `LD_PRELOAD`
  - Configures GPU affinity (`ZE_AFFINITY_MASK=0,1,2` for 3 Arc A770s)
  - Sets `HF_HUB_ENABLE_HF_TRANSFER=1` for fast model downloads

**SYCL Runtime Tuning Variables:**
- `GGML_SYCL_DISABLE_GRAPH=1` - Disable command graph optimization (safer)
- `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` - Batched queue mode (better throughput)
- `ZES_ENABLE_SYSMAN=1` - Enable GPU metrics collection
- `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` - Allow full GPU VRAM utilization
- `GGML_SYCL_FUSED_MMQ=1` - Enable fused dequant+matmul kernels (production)
- `GGML_SYCL_ROW_EVENTS=1` - Enable out-of-order queue + event-based sync (experimental)

**Build:**
- `llama.cpp/CMakeLists.txt` - Main build configuration (GGML_SYCL=ON, GGML_SYCL_TARGET=INTEL)
- `llama.cpp/build-sycl/` - Clean SYCL build directory (canonical, not `build/`)
- `CMakePresets.json` - Build presets for different configurations

**Model Configuration:**
- Models stored in `models/` directory with GGUF format
  - Qwen3-32B Q4_K_M (19GB) - Default production model (22.7 t/s @ np=16)
  - Qwen3-32B Q8_0 (33GB) - Higher quality variant (3.3 t/s, bandwidth-limited)
  - Qwen3.5-27B Q8_0 (32GB) - Alternative model
  - Qwen3-0.6B Q8_0 (350MB) - Draft model for speculative decoding
  - GLM-4.7-Flash Q4_K_M (17GB) - MoE model for testing
  - Nemotron-3-Super-120B Q2_K - Large model with CPU offload

## Platform Requirements

**Development:**
- Linux with Intel Arc GPU (A750/A770)
- Render group membership for GPU access (`/dev/dri/renderD*`)
- 64GB+ system RAM
- 24GB+ GPU VRAM per Arc card (3 cards = 72GB total)
- i9-7900X or equivalent processor (for tile-friendly FFT kernels)

**Production:**
- Deployment target: Linux with Intel Arc A770 (3x 24GB cards)
- No CUDA or ROCm dependencies
- Self-contained: All runtime libraries bundled in conda environment

## Key Build Outputs

- `llama.cpp/build-sycl/bin/llama-server` - Main inference server (HTTP API)
- `llama.cpp/build-sycl/bin/llama-cli` - Command-line inference tool
- Python modules in `.conda-sgl-xpu312/` - SYCL runtime, oneMKL, TBB

---

*Stack analysis: 2026-03-17*
