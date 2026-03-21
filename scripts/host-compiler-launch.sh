#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
shift || true
orig_compiler="${1:-}"
shift || true

args=()
for arg in "$@"; do
  case "$arg" in
    -fiopenmp)
      args+=("-fopenmp")
      ;;
    *)
      args+=("$arg")
      ;;
  esac
done

case "$mode" in
  c)
    exec /usr/bin/gcc "${args[@]}"
    ;;
  cxx)
    exec /usr/bin/g++ "${args[@]}"
    ;;
  *)
    echo "usage: $0 <c|cxx> <compiler-args...>" >&2
    exit 2
    ;;
esac
