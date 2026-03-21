#!/bin/bash
# cache-manager.sh — Ensures all caches are warm before llama-server starts.
# Called by arcllm-proxy or manually before server launch.
# Usage: source env.sglang-xpu.sh && ./cache-manager.sh [save|restore] [immediate|batched]

set -euo pipefail

CACHE_ROOT="/home/ryan/llm-stack/cache"
L0_LIVE="/tmp/neo_compiler_cache"
ACTION="${1:-restore}"
MODE="${2:-immediate}"

resolve_cache_source() {
    local dir="$1"
    local nested="$dir/neo_compiler_cache"

    if [ -d "$nested" ] && [ "$(ls -A "$nested" 2>/dev/null)" ]; then
        echo "$nested"
    else
        echo "$dir"
    fi
}

copy_cache_dir() {
    local src_raw="$1"
    local dst="$2"
    local src
    src="$(resolve_cache_source "$src_raw")"

    rm -rf "$dst"
    mkdir -p "$dst"
    cp -a "$src"/. "$dst"/
}

clear_live_cache() {
    rm -rf "$L0_LIVE"
}

# Determine mode from env if not specified
if [ "$MODE" = "auto" ]; then
    if [ "${SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS:-1}" = "0" ]; then
        MODE="batched"
    else
        MODE="immediate"
    fi
    # Append graph dimension
    if [ "${GGML_SYCL_DISABLE_GRAPH:-1}" = "0" ]; then
        MODE="${MODE}_graph"
    else
        MODE="${MODE}_nograph"
    fi
fi

L0_PERSISTENT="$CACHE_ROOT/neo_compiler_cache_${MODE}"

case "$ACTION" in
    restore)
        echo "[cache-manager] Restoring caches for $MODE mode"

        # Always clear live cache first so mode isolation is real.
        clear_live_cache

        # Layer 1: L0 compiler cache (mode-specific only)
        if [ -d "$L0_PERSISTENT" ] && [ "$(ls -A "$L0_PERSISTENT" 2>/dev/null)" ]; then
            copy_cache_dir "$L0_PERSISTENT" "$L0_LIVE"
            echo "[cache-manager] L0 compiler cache restored from $L0_PERSISTENT ($(du -sh "$L0_LIVE" | cut -f1))"
        else
            echo "[cache-manager] No L0 compiler cache for $MODE mode — cold JIT compile will happen"
        fi

        # Layer 0: Pre-split weight snapshots (future)
        # TODO: check for tp-shards/<model-hash>/gpu{0,1,2}.bin
        # If exists, pass --tp-shard-dir to llama-server (needs server support)

        echo "[cache-manager] Restore complete"
        ;;

    save)
        echo "[cache-manager] Saving caches for $MODE mode"

        # Layer 1: L0 compiler cache
        if [ -d "$L0_LIVE" ] && [ "$(ls -A "$L0_LIVE" 2>/dev/null)" ]; then
            rm -rf "$L0_PERSISTENT"
            mkdir -p "$L0_PERSISTENT"
            copy_cache_dir "$L0_LIVE" "$L0_PERSISTENT"
            echo "[cache-manager] L0 compiler cache saved to $L0_PERSISTENT ($(du -sh "$L0_PERSISTENT" | cut -f1))"
        else
            echo "[cache-manager] No L0 compiler cache to save"
        fi

        echo "[cache-manager] Save complete"
        ;;

    status)
        echo "[cache-manager] Cache status:"
        echo "  Mode: $MODE"
        echo "  L0 live ($L0_LIVE):"
        if [ -d "$L0_LIVE" ]; then
            echo "    Size: $(du -sh "$L0_LIVE" | cut -f1)"
            echo "    Files: $(find "$L0_LIVE" -type f | wc -l)"
        else
            echo "    EMPTY"
        fi
        echo "  L0 persistent immediate ($CACHE_ROOT/neo_compiler_cache_immediate):"
        if [ -d "$CACHE_ROOT/neo_compiler_cache_immediate" ]; then
            echo "    Size: $(du -sh "$CACHE_ROOT/neo_compiler_cache_immediate" | cut -f1)"
        else
            echo "    EMPTY"
        fi
        echo "  L0 persistent batched ($CACHE_ROOT/neo_compiler_cache_batched):"
        if [ -d "$CACHE_ROOT/neo_compiler_cache_batched" ]; then
            echo "    Size: $(du -sh "$CACHE_ROOT/neo_compiler_cache_batched" | cut -f1)"
        else
            echo "    EMPTY"
        fi
        echo "  KV slots ($CACHE_ROOT/slots/):"
        ls -lh "$CACHE_ROOT/slots/" 2>/dev/null || echo "    EMPTY"
        echo "  TP shards ($CACHE_ROOT/tp-shards/):"
        ls -d "$CACHE_ROOT/tp-shards/"*/ 2>/dev/null || echo "    EMPTY (not implemented)"
        ;;

    *)
        echo "Usage: $0 [restore|save|status] [immediate|batched|auto]"
        exit 1
        ;;
esac
