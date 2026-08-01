"""Tests for the event bus."""

from __future__ import annotations

from atlas.core.events import EventType, get_event_bus


def test_publish_and_subscribe() -> None:
    bus = get_event_bus()
    bus.clear()
    received = []
    unsubscribe = bus.subscribe(EventType.ACTION_COMPLETED, lambda event: received.append(event))
    bus.publish(EventType.ACTION_COMPLETED, {"action": "type"})
    assert len(received) == 1
    assert received[0].data["action"] == "type"
    unsubscribe()
    bus.publish(EventType.ACTION_COMPLETED, {})
    assert len(received) == 1


def test_subscribe_all() -> None:
    bus = get_event_bus()
    bus.clear()
    received = []
    unsubscribe = bus.subscribe_all(lambda event: received.append(event.type))
    bus.publish(EventType.OBSERVED)
    bus.publish(EventType.RECORD_STARTED)
    assert EventType.OBSERVED in received
    assert EventType.RECORD_STARTED in received
    unsubscribe()


def test_history_and_clear() -> None:
    bus = get_event_bus()
    bus.clear()
    bus.publish(EventType.AGENT_STARTED)
    bus.publish(EventType.AGENT_STARTED)
    assert len(bus.history(EventType.AGENT_STARTED)) == 2
    assert len(bus.history()) >= 2
    bus.clear()
    assert len(bus.history()) == 0


def test_listener_exception_does_not_break_bus() -> None:
    bus = get_event_bus()
    bus.clear()

    def bad_listener(event) -> None:
        raise RuntimeError("boom")

    def good_listener(event) -> None:
        good_listener.seen = True

    good_listener.seen = False
    unsub_bad = bus.subscribe(EventType.LOG, bad_listener)
    unsub_good = bus.subscribe(EventType.LOG, good_listener)
    bus.publish(EventType.LOG, {"message": "x"})
    assert good_listener.seen is True
    unsub_bad()
    unsub_good()
