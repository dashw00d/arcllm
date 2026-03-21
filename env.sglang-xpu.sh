#!/usr/bin/env bash

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _env_script_path="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  _env_script_path="${(%):-%N}"
else
  _env_script_path="$0"
fi

ROOT="$(cd "$(dirname "$_env_script_path")" && pwd)"
unset _env_script_path
export LLM_STACK_ROOT="$ROOT"
export CONDA_HOME="${CONDA_HOME:-/home/ryan/miniforge3}"
export CONDA_ENV_PATH="${CONDA_ENV_PATH:-$ROOT/.conda-sgl-xpu312}"
export RUNTIME_PATH="${RUNTIME_PATH:-$ROOT/runtime-upstream}"

if [[ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]]; then
  _had_nounset=0
  if [[ $- == *u* ]]; then
    _had_nounset=1
    set +u
  fi
  export SETVARS_CALL="${SETVARS_CALL:-0}"
  # shellcheck disable=SC1091
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_PATH"
  if [[ $_had_nounset -eq 1 ]]; then
    set -u
  fi
  unset _had_nounset
else
  echo "conda.sh not found at $CONDA_HOME/etc/profile.d/conda.sh" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="$CONDA_PREFIX/bin/compiler:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ROOT/.igc28/usr/local/lib:$ROOT/.ocloc26/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export UR_ADAPTERS_SEARCH_PATH="$CONDA_PREFIX/lib"
export SYCL_ROOT="$CONDA_PREFIX"
export CMPLR_ROOT="$CONDA_PREFIX"
export ONEAPI_ROOT="${ONEAPI_ROOT:-$CONDA_PREFIX}"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS="${UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS:-1}"
export ZES_ENABLE_SYSMAN="${ZES_ENABLE_SYSMAN:-1}"

# oneMKL SYCL wheels do not pull the classic MKL service/core symbols into the
# process by themselves. Preload the full runtime set in a stable order.
_mkl_preloads=(
  "$CONDA_PREFIX/lib/libmkl_core.so.2"
  "$CONDA_PREFIX/lib/libmkl_rt.so.2"
  "$CONDA_PREFIX/lib/libmkl_intel_lp64.so.2"
  "$CONDA_PREFIX/lib/libmkl_sequential.so.2"
)
for _mkl_lib in "${_mkl_preloads[@]}"; do
  if [[ -f "$_mkl_lib" ]]; then
    if [[ -n "${LD_PRELOAD:-}" ]]; then
      case ":$LD_PRELOAD:" in
        *":$_mkl_lib:"*) ;;
        *) export LD_PRELOAD="$_mkl_lib:$LD_PRELOAD" ;;
      esac
    else
      export LD_PRELOAD="$_mkl_lib"
    fi
  fi
done
unset _mkl_lib _mkl_preloads

if [[ -d "$RUNTIME_PATH" ]]; then
  export LD_LIBRARY_PATH="$RUNTIME_PATH/usr/lib/x86_64-linux-gnu:$RUNTIME_PATH/usr/lib/x86_64-linux-gnu/intel-opencl:$RUNTIME_PATH/usr/local/lib:$LD_LIBRARY_PATH"
  export OCL_ICD_VENDORS="$RUNTIME_PATH/etc/OpenCL/vendors"
  export OCL_ICD_FILENAMES="$RUNTIME_PATH/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so"
fi

# Use all Arc cards by default. Override per launch if needed.
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1,2}"
