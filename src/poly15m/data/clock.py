"""Clock / window manager.

Tracks time-remaining-to-resolution for the currently active 15m window and
emits lifecycle events once, in order, as each boundary is crossed:
window_open, t_minus_5m, t_minus_2m, t_minus_30s, resolved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# (event name, seconds-remaining threshold at/below which it fires)
# "window_open" is handled separately since it fires at t=open, not on
# remaining-time.
_REMAINING_MILESTONES: tuple[tuple[str, float], ...] = (
    ("t_minus_5m", 300.0),
    ("t_minus_2m", 120.0),
    ("t_minus_30s", 30.0),
    ("resolved", 0.0),
)


@dataclass
class WindowClock:
    condition_id: str
    open_ts: float
    close_ts: float
    _emitted: set[str] = field(default_factory=set)

    def time_remaining(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        return max(0.0, self.close_ts - now)

    def is_open(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self.open_ts <= now < self.close_ts

    def poll(self, now: float | None = None) -> list[str]:
        """Return newly-crossed lifecycle events (in order) since the last call."""
        now = time.time() if now is None else now
        crossed: list[str] = []

        if now >= self.open_ts and "window_open" not in self._emitted:
            self._emitted.add("window_open")
            crossed.append("window_open")

        remaining = self.close_ts - now
        for name, threshold in _REMAINING_MILESTONES:
            if remaining <= threshold and name not in self._emitted:
                self._emitted.add(name)
                crossed.append(name)

        return crossed
