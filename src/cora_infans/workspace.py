from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import WorkspaceEvent


class EventWorkspace:
    def __init__(self, window_ms: int) -> None:
        self.window_ms = window_ms
        self._events: deque[WorkspaceEvent] = deque()

    def push(self, event: WorkspaceEvent) -> None:
        self._events.append(event)
        cutoff = int(event.workspace_state["timestamp_ms"]) - self.window_ms
        while self._events and int(self._events[0].workspace_state["timestamp_ms"]) < cutoff:
            self._events.popleft()

    def snapshot(self) -> list[WorkspaceEvent]:
        return list(self._events)

    def focus_path(self) -> list[str]:
        return [
            str(event.workspace_state["focus_label"])
            for event in self._events
            if "focus_label" in event.workspace_state
        ]
