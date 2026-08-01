"""Tests for configuration loading."""

from __future__ import annotations

import pytest

from atlas.config import (
    ConfigError,
    VisionConfig,
    load_config,
)


def test_load_config_has_all_sections() -> None:
    config = load_config()
    for attr in (
        "vision", "reasoning", "ocr", "mouse", "typing", "observe",
        "workflow", "overlay", "controller", "memory", "log",
    ):
        assert hasattr(config, attr)
    assert config.log.folder.name == "logs"


def test_load_config_creates_runtime_directories(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOG_FOLDER", str(tmp_path / "logs"))
    monkeypatch.setenv("OBSERVE_SCREENSHOT_DIR", str(tmp_path / "shots"))
    config = load_config()
    assert (tmp_path / "logs").is_dir()
    assert (tmp_path / "shots").is_dir()
    assert config.log.folder == tmp_path / "logs"


def test_env_override_typing(monkeypatch) -> None:
    monkeypatch.setenv("TYPING_CLIPBOARD_MIN_LENGTH", "42")
    config = load_config()
    assert config.typing.clipboard_min_length == 42


def test_env_bool_parsing(monkeypatch) -> None:
    monkeypatch.setenv("DEBUG_MODE", "true")
    assert load_config().debug is True
    monkeypatch.setenv("DEBUG_MODE", "0")
    assert load_config().debug is False


def test_invalid_int_raises(monkeypatch) -> None:
    monkeypatch.setenv("MOUSE_BEZIER_STEPS", "not-a-number")
    with pytest.raises(ConfigError):
        load_config()


def test_clamp_out_of_range_raises(monkeypatch) -> None:
    monkeypatch.setenv("VISION_TIMEOUT", "999999")
    with pytest.raises(ConfigError):
        VisionConfig()


def test_config_to_dict() -> None:
    data = load_config().to_dict()
    assert data["debug"] in (True, False)
    assert "vision" in data and "provider" in data["vision"]
