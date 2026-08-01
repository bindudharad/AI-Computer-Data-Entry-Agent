"""Tests for the agent state machine."""

from __future__ import annotations

import pytest

from atlas.core.states import AgentState, InvalidTransitionError, StateMachine


def test_idle_to_attach_chain() -> None:
    sm = StateMachine()
    assert sm.state == AgentState.IDLE
    sm.transition(AgentState.WAITING_ATTACH)
    sm.transition(AgentState.ATTACHING)
    sm.transition(AgentState.WATCHING)
    sm.transition(AgentState.ANALYZING)
    sm.transition(AgentState.PLANNING)
    sm.transition(AgentState.THINKING)
    sm.transition(AgentState.TYPING)
    sm.transition(AgentState.VERIFYING)
    sm.transition(AgentState.WATCHING)
    assert sm.state == AgentState.WATCHING


def test_pipeline_chain_with_ui_tree_stages() -> None:
    sm = StateMachine()
    sm.transition(AgentState.WAITING_ATTACH)
    sm.transition(AgentState.ATTACHING)
    sm.transition(AgentState.BUILD_UI_TREE)
    sm.transition(AgentState.SCREEN_MODEL)
    sm.transition(AgentState.RECORD_EXTRACTION)
    sm.transition(AgentState.FIELD_MAPPING)
    sm.transition(AgentState.PLANNING)
    sm.transition(AgentState.THINKING)
    sm.transition(AgentState.TYPING)
    sm.transition(AgentState.VERIFYING)
    sm.transition(AgentState.WATCHING)
    sm.transition(AgentState.BUILD_UI_TREE)
    sm.transition(AgentState.SCREEN_MODEL)
    sm.transition(AgentState.RECORD_EXTRACTION)
    sm.transition(AgentState.WATCHING)
    assert sm.state == AgentState.WATCHING


def test_always_transition_to_stopped() -> None:
    sm = StateMachine()
    for state in (
        AgentState.BUILD_UI_TREE,
        AgentState.SCREEN_MODEL,
        AgentState.RECORD_EXTRACTION,
    ):
        sm.force(state)
        sm.transition(AgentState.STOPPED)
        sm.reset()


def test_invalid_transition_raises() -> None:
    sm = StateMachine()
    with pytest.raises(InvalidTransitionError):
        sm.transition(AgentState.WATCHING)  # IDLE -> WATCHING is invalid


def test_pause_and_resume() -> None:
    sm = StateMachine()
    sm.force(AgentState.TYPING)
    assert sm.can_pause()
    sm.transition(AgentState.PAUSED)
    assert sm.state == AgentState.PAUSED
    sm.resume()
    assert sm.state == AgentState.TYPING


def test_stop_from_running() -> None:
    sm = StateMachine()
    sm.force(AgentState.VERIFYING)
    sm.transition(AgentState.STOPPED)
    assert sm.state == AgentState.STOPPED


def test_state_change_listener() -> None:
    sm = StateMachine()
    seen: list[AgentState] = []
    sm.on_change(seen.append)
    sm.force(AgentState.WATCHING)
    sm.force(AgentState.ANALYZING)
    assert seen == [AgentState.WATCHING, AgentState.ANALYZING]


def test_force_ignores_validation() -> None:
    sm = StateMachine()
    sm.force(AgentState.COMPLETED)
    assert sm.state == AgentState.COMPLETED


def test_reset() -> None:
    sm = StateMachine()
    sm.force(AgentState.TYPING)
    sm.transition(AgentState.PAUSED)
    sm.reset()
    assert sm.state == AgentState.IDLE
