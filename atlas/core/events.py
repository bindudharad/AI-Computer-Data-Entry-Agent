"""Thread-safe publish/subscribe event bus and event model.

The agent emits events at every stage (decision, action, retry, failure,
verification) so that the overlay, dashboard, logs and tests all observe the
same stream.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Canonical agent event topics."""

    # lifecycle
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_PAUSED = "agent.paused"
    AGENT_RESUMED = "agent.resumed"
    STATE_CHANGED = "state.changed"

    # observation
    WINDOW_ATTACHED = "window.attached"
    WINDOW_DETACHED = "window.detached"
    OBSERVED = "observed"
    SCREENSHOT = "screenshot"
    SCREEN_STATE = "screen.state"

    # understanding
    SCENE_ANALYZED = "scene.analyzed"
    FIELD_DISCOVERED = "field.discovered"
    SOURCE_READ = "source.read"

    # reasoning
    MAPPING = "mapping.decided"
    PLAN_CREATED = "plan.created"
    RECOVERY = "recovery.decided"

    # execution
    ACTION_STARTED = "action.started"
    ACTION_COMPLETED = "action.completed"
    ACTION_FAILED = "action.failed"
    ACTION_RETRY = "action.retry"
    VERIFICATION = "verification.result"

    # workflow
    RECORD_STARTED = "record.started"
    RECORD_COMPLETED = "record.completed"
    RECORD_FAILED = "record.failed"
    NEXT_RECORD_WAITING = "next_record.waiting"
    NEXT_RECORD_DETECTED = "next_record.detected"
    NO_RECORD = "no_record.detected"
    WORKFLOW_COMPLETE = "workflow.complete"
    UPLOADING = "uploading"
    UPLOAD_COMPLETED = "upload.completed"

    # logging / diagnostics
    LOG = "log"
    ERROR = "error"


@dataclass(frozen=True)
class Event:
    """An immutable event on the bus."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "data": self.data, "timestamp": self.timestamp}


Listener = Callable[[Event], None]


class EventBus:
    """Minimal thread-safe pub/sub with optional event history."""

    def __init__(self, history_size: int = 500) -> None:
        self._history_size = history_size
        self._subscribers: dict[EventType, list[Listener]] = {}
        self._all_subscribers: list[Listener] = []
        self._history: list[Event] = []
        self._lock = threading.RLock()

    def subscribe(self, event_type: EventType, listener: Listener) -> Callable[[], None]:
        """Subscribe to one event type; returns an unsubscribe callable."""
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(listener)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers[event_type].remove(listener)
                except ValueError:
                    pass

        return _unsubscribe

    def subscribe_all(self, listener: Listener) -> Callable[[], None]:
        """Subscribe to every event; returns an unsubscribe callable."""
        with self._lock:
            self._all_subscribers.append(listener)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._all_subscribers.remove(listener)
                except ValueError:
                    pass

        return _unsubscribe

    def publish(self, event_type: EventType, data: dict[str, Any] | None = None) -> Event:
        """Publish an event to matching subscribers (called under the lock)."""
        event = Event(type=event_type, data=data or {})
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_size:
                del self._history[: len(self._history) - self._history_size]
            targets = list(self._all_subscribers) + list(self._subscribers.get(event_type, []))
        for listener in targets:
            try:
                listener(event)
            except Exception:
                # A listener must never break the bus.
                from loguru import logger

                logger.exception("Event listener failed for {}", event_type)
        return event

    def history(self, event_type: EventType | None = None) -> list[Event]:
        with self._lock:
            if event_type is None:
                return list(self._history)
            return [e for e in self._history if e.type == event_type]

    def clear(self) -> None:
        with self._lock:
            self._history.clear()


_bus: EventBus | None = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Process-wide singleton event bus."""
    global _bus
    with _bus_lock:
        if _bus is None:
            _bus = EventBus()
        return _bus
