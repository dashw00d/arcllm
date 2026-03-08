#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LLM_STACK_ROOT="$ROOT"
export VENV_PATH="$ROOT/.venv"
export RUNTIME_PATH="$ROOT/runtime-upstream"

export PATH="$VENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$RUNTIME_PATH/usr/lib/x86_64-linux-gnu:$RUNTIME_PATH/usr/lib/x86_64-linux-gnu/intel-opencl:$RUNTIME_PATH/usr/local/lib:$VENV_PATH/lib:$VENV_PATH/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OCL_ICD_VENDORS="$RUNTIME_PATH/etc/OpenCL/vendors"
export OCL_ICD_FILENAMES="$RUNTIME_PATH/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so"
export UR_ADAPTERS_SEARCH_PATH="$VENV_PATH/lib"

# Pin to the first Arc card by default. Override for multi-GPU tests.
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
