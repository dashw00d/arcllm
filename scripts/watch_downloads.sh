#!/usr/bin/env bash
# Watch model download progress
# Usage: ./scripts/watch_downloads.sh

DRAFT_DIR="/home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-abliterated-GGUF"
MAIN_DIR="/home/ryan/llm-stack/models/Qwen/Qwen3-235B-A22B-Q3-abliterated"
DRAFT_TARGET="18.6 GB"
MAIN_TARGET="101 GB"

while true; do
    clear
    echo "=== Model Download Progress ($(date +%H:%M:%S)) ==="
    echo ""

    # Draft model
    draft_size=$(du -sh "$DRAFT_DIR" 2>/dev/null | cut -f1)
    draft_gguf=$(find "$DRAFT_DIR" -maxdepth 1 -name "*.gguf" -type f 2>/dev/null | head -1)
    if [[ -n "$draft_gguf" ]]; then
        echo "Draft (30B-A3B Q4_K_M): DONE ✓ ($draft_size)"
    else
        echo "Draft (30B-A3B Q4_K_M): $draft_size / $DRAFT_TARGET"
    fi

    # Main model
    main_size=$(du -sh "$MAIN_DIR" 2>/dev/null | cut -f1)
    main_count=$(find "$MAIN_DIR" -maxdepth 1 -name "Q3_K_S-GGUF-*.gguf" -type f 2>/dev/null | wc -l)
    if [[ "$main_count" -eq 11 ]]; then
        echo "Main (235B Q3_K_S):     DONE ✓ ($main_size, $main_count/11 parts)"
    else
        echo "Main (235B Q3_K_S):     $main_size / $MAIN_TARGET ($main_count/11 parts)"
    fi

    echo ""

    # Check if processes still running
    if ! pgrep -f "huggingface-cli.*download" > /dev/null 2>&1; then
        echo "Downloads completed (or failed). Check files above."
        break
    fi

    echo "Press Ctrl+C to stop watching."
    sleep 10
done
