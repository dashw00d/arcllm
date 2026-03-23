# Prompt Cache — How It Works for Qwen3.5 (Recurrent/DeltaNet)

## What's Working (after seq_rm fix, 2026-03-22)

Three layers, all connected:

### Layer 1: In-Memory Prefix Matching
- Each slot remembers its last conversation's token sequence
- New requests get assigned to the slot with the **Longest Common Prefix** (LCP)
- Server computes `n_past` = matching prefix length, only evals new/changed tokens
- With `np=4`, Discord slot 0 keeps the system prompt + conversation cached between messages
- **Result:** 3.4x prompt eval speedup (4382ms cold → 1300ms cached for 60 tokens)

### Layer 2: Disk Slot Save/Restore
- Proxy saves full recurrent state on shutdown: `cache/slots/qwen35.slot{N}.bin` (~65MB/slot)
- Restores on startup via `/slots/{id}?action=restore`
- **Before seq_rm fix:** state restored but `seq_rm()` returned false → full reprocessing anyway
- **After fix:** restored state is reusable — server finds the closest checkpoint before the new tokens

### Layer 3: In-Memory Checkpoints
- Server maintains periodic snapshots of recurrent state during prompt processing
- When prefix match is partial but tail state is lost (e.g. SWA window), searches backward through checkpoints
- Most sophisticated recovery — avoids full reprocessing on partial cache misses

## Current Limitations

- **No cross-slot sharing:** each slot's cache is independent
- **No conversation library:** disk save captures only the *last* state per slot, not multiple conversations
- **Multi-user interleaving:** if Henry talks to user A on slot 0, then user B takes slot 0, switching back to A loses A's cached state
- **cache_n always reports 0:** the counter isn't wired for recurrent models, but timing proves cache works

## Where It Matters

Henry's pattern (Discord bot, sequential messages on one slot) is the best case:
- System prompt + conversation history stays cached between messages
- Survives server restarts via disk save/restore
- Only the new user message + assistant turn gets processed

The gap is multi-user interleaving — would need a slot-per-user pool or conversation-keyed disk cache to handle concurrent Discord users without thrashing.

## Key Files

| File | What |
|------|------|
| `llama-memory-recurrent.cpp:seq_rm()` | The fix — fuzzy checkpoint search instead of `return false` |
| `tools/server/server-context.cpp:945` | Slot selection by LCP similarity |
| `tools/server/server-context.cpp:2215` | `n_past` computation from common prefix |
| `tools/server/server-context.cpp:2349` | Checkpoint search fallback |
| `scripts/arcllm-proxy.py:280-354` | Disk slot save/restore in proxy |
