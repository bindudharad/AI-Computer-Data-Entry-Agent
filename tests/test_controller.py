"""Tests for the assistant controller command layer."""

from __future__ import annotations

from atlas.assistant.controller import Controller
from atlas.config import load_config
from atlas.core.states import AgentState
from atlas.mapping.mapper import SemanticMapper
from atlas.memory.store import MemoryStore
from atlas.target.base import TargetInfo


class FakeTarget:
    info = TargetInfo(name="desktop", title="Fake Window", process_id=1)


class FakeAssistant:
    def __init__(self) -> None:
        self.target = FakeTarget()
        self._state = AgentState.WATCHING
        self.memory = MemoryStore(":memory:")
        self.mapper = SemanticMapper()
        self.config = load_config()
        self.detached = False

    @property
    def state(self) -> str:
        return self._state.value

    def attach_web(self, url=None, browser="chromium", headless=False):
        self.target = FakeTarget()
        return self.target

    def attach_desktop(self, title=None):
        self.target = FakeTarget()
        return self.target

    def detach(self) -> None:
        self.detached = True
        self.target = None  # type: ignore[assignment]

    def run(self, max_records=0):
        from atlas.workflow.loop import WorkflowSummary

        return WorkflowSummary()

    def stop(self): pass
    def pause(self): pass
    def resume(self): pass
    def close(self): self.memory.close()


def test_status() -> None:
    controller = Controller(FakeAssistant())
    result = controller.handle({"command": "status"})
    assert result["ok"] is True
    assert result["attached"] is True
    assert result["state"] == "watching"
    assert result["target"]["title"] == "Fake Window"


def test_attach_and_detach() -> None:
    controller = Controller(FakeAssistant())
    result = controller.handle({"command": "detach"})
    assert result["ok"] is True


def test_learn_alias_and_list() -> None:
    assistant = FakeAssistant()
    controller = Controller(assistant)
    result = controller.handle({"command": "learn_alias", "variant": "dob", "canonical": "date of birth"})
    assert result["ok"] is True
    aliases = controller.handle({"command": "aliases"})["aliases"]
    assert aliases["dob"] == "date of birth"
    assert assistant.mapper.aliases.resolve("dob") == "date of birth"


def test_unknown_command() -> None:
    controller = Controller(FakeAssistant())
    result = controller.handle({"command": "frobnicate"})
    assert result["ok"] is False
    assert "unknown command" in result["error"]


def test_run_returns_summary() -> None:
    controller = Controller(FakeAssistant())
    result = controller.handle({"command": "run", "max_records": 5})
    assert result["ok"] is True
    assert "summary" in result


def test_config() -> None:
    controller = Controller(FakeAssistant())
    result = controller.handle({"command": "config"})
    assert result["ok"] is True
    assert "config" in result
