from collections import defaultdict, deque

from config import MAX_HISTORY


class HistoryManager:
    """Per-channel sliding window conversation history."""

    def __init__(self, maxlen: int = MAX_HISTORY):
        self._maxlen = maxlen
        self._channels: dict[int, deque] = defaultdict(lambda: deque(maxlen=self._maxlen))

    def add_user(self, channel_id: int, display_name: str, content: str):
        self._channels[channel_id].append({
            "role": "user",
            "content": f"[{display_name}]: {content}",
        })

    def add_assistant(self, channel_id: int, content: str):
        self._channels[channel_id].append({
            "role": "assistant",
            "content": content,
        })

    def get(self, channel_id: int) -> list[dict]:
        return list(self._channels[channel_id])

    def clear(self, channel_id: int):
        self._channels[channel_id].clear()
