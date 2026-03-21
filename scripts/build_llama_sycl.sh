#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/llama.cpp/build-sycl}"
GGML_SYCL_F16_BUILD="${GGML_SYCL_F16_BUILD-OFF}"
GGML_SYCL_DEVICE_ARCH_BUILD="${GGML_SYCL_DEVICE_ARCH_BUILD-}"
BUILD_CLEAN="${BUILD_CLEAN:-0}"

source "$ROOT/env.sglang-xpu.sh"

ensure_mkl_dev() {
  if [[ -f "$CONDA_PREFIX/include/mkl.h" && -f "$CONDA_PREFIX/include/oneapi/mkl.hpp" && -f "$CONDA_PREFIX/lib/cmake/mkl/MKLConfig.cmake" ]]; then
    return 0
  fi

  python -m pip install \
    "mkl-devel-dpcpp==2025.2.0" \
    "tbb==2022.2.0" \
    "tbb-devel==2022.2.0"
}

ensure_mkl_dev

cd "$ROOT/llama.cpp"
if [[ "$BUILD_CLEAN" == "1" ]]; then
  rm -rf "$BUILD_DIR"
fi

cmake -B "$BUILD_DIR" -G Ninja \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16="$GGML_SYCL_F16_BUILD" \
  -DGGML_SYCL_DNN=OFF \
  -DGGML_SYCL_GRAPH=ON \
  -DGGML_SYCL_DEVICE_ARCH="$GGML_SYCL_DEVICE_ARCH_BUILD" \
  -DGGML_CPU=ON \
  -DGGML_CPU_REPACK=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=OFF \
  -DGGML_AVX2=OFF \
  -DGGML_AVX_VNNI=OFF \
  -DGGML_BMI2=OFF \
  -DGGML_F16C=OFF \
  -DGGML_FMA=OFF \
  -DGGML_AVX512=OFF \
  -DGGML_AVX512_VBMI=OFF \
  -DGGML_AVX512_VNNI=OFF \
  -DGGML_AVX512_BF16=OFF \
  -DGGML_AMX_TILE=OFF \
  -DGGML_AMX_INT8=OFF \
  -DGGML_AMX_BF16=OFF \
  -DCMAKE_MODULE_PATH="$ROOT/cmake" \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release

printf 'SYCL build config:\n'
printf '  CONDA_PREFIX=%s\n' "$CONDA_PREFIX"
printf '  ONEAPI_ROOT=%s\n' "$ONEAPI_ROOT"
printf '  BUILD_DIR=%s\n' "$BUILD_DIR"
printf '  GGML_SYCL_F16=%s\n' "$GGML_SYCL_F16_BUILD"
printf '  GGML_SYCL_DEVICE_ARCH=%s\n' "$GGML_SYCL_DEVICE_ARCH_BUILD"

cmake --build "$BUILD_DIR" -j "${BUILD_JOBS:-12}" --target llama-cli llama-server llama-bench
