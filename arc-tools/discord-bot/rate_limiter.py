import time

from config import COOLDOWN_SECONDS


class RateLimiter:
    """Per-user cooldown to prevent spam."""

    def __init__(self, cooldown: float = COOLDOWN_SECONDS):
        self._cooldown = cooldown
        self._last: dict[int, float] = {}

    def check(self, user_id: int) -> bool:
        """Return True if the user is allowed to send a request."""
        now = time.monotonic()
        last = self._last.get(user_id, 0)
        if now - last < self._cooldown:
            return False
        self._last[user_id] = now
        return True

    def remaining(self, user_id: int) -> float:
        """Seconds until this user can send again."""
        now = time.monotonic()
        last = self._last.get(user_id, 0)
        return max(0, self._cooldown - (now - last))
