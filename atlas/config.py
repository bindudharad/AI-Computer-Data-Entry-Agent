"""ATLAS AI - typed configuration.

All settings load from environment variables (a `.env` file is optional) with
sensible defaults. Values are evaluated at instantiation (via ``default_factory``)
so environment overrides take effect at ``load_config()`` time - not just at
import time. Directories are created eagerly so config failures surface at
startup, not mid-run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when configuration is invalid."""


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{name} must be an integer, got {os.getenv(name)!r}") from exc


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{name} must be a number, got {os.getenv(name)!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _clamp(name: str, value: float, low: float, high: float) -> float:
    if not (low <= value <= high):
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}")
    return value


# Field factories - read the environment when the dataclass is instantiated.
def _f(name: str, default: str) -> Any:
    return lambda: _env(name, default)


def _fi(name: str, default: int) -> Any:
    return lambda: _env_int(name, default)


def _ff(name: str, default: float) -> Any:
    return lambda: _env_float(name, default)


def _fb(name: str, default: bool) -> Any:
    return lambda: _env_bool(name, default)


def _fclamp(name: str, default: float, low: float, high: float) -> Any:
    return lambda: _clamp(name, _env_float(name, default), low, high)


@dataclass(frozen=True)
class VisionConfig:
    """Vision Language Model (VLM) configuration - the primary channel."""

    provider: str = field(default_factory=_f("VISION_PROVIDER", "auto"))
    model: str = field(default_factory=_f("VISION_MODEL", ""))
    api_key: str = field(default_factory=_f("VISION_API_KEY", ""))
    api_base: str = field(default_factory=_f("VISION_API_BASE", ""))
    timeout: float = field(default_factory=_fclamp("VISION_TIMEOUT", 60.0, 1.0, 600.0))
    confidence_threshold: float = field(
        default_factory=_fclamp("VISION_CONFIDENCE_THRESHOLD", 0.4, 0.0, 1.0)
    )


@dataclass(frozen=True)
class ReasoningConfig:
    """LLM configuration for planning/decisioning."""

    provider: str = field(default_factory=_f("REASONING_PROVIDER", "auto"))
    model: str = field(default_factory=_f("REASONING_MODEL", ""))
    api_key: str = field(default_factory=_f("REASONING_API_KEY", ""))
    api_base: str = field(default_factory=_f("REASONING_API_BASE", ""))
    timeout: float = field(default_factory=_fclamp("REASONING_TIMEOUT", 60.0, 1.0, 600.0))
    confidence_threshold: float = field(
        default_factory=_fclamp("REASONING_CONFIDENCE_THRESHOLD", 0.5, 0.0, 1.0)
    )


@dataclass(frozen=True)
class OcrConfig:
    """OCR fallback configuration (explicit AI request only)."""

    engine: str = field(default_factory=_f("OCR_ENGINE", "paddle"))
    lang: str = field(default_factory=_f("OCR_LANG", "en"))
    confidence_threshold: float = field(
        default_factory=_fclamp("OCR_CONFIDENCE_THRESHOLD", 0.4, 0.0, 1.0)
    )
    preprocess: bool = field(default_factory=_fb("OCR_PREPROCESS", True))


@dataclass(frozen=True)
class MouseConfig:
    """Human-like mouse behaviour."""

    bezier_steps: int = field(default_factory=_fi("MOUSE_BEZIER_STEPS", 35))
    speed: float = field(default_factory=_ff("MOUSE_SPEED", 0.35))
    min_delay: float = field(default_factory=_ff("MOUSE_MIN_DELAY", 0.05))
    max_delay: float = field(default_factory=_ff("MOUSE_MAX_DELAY", 0.25))
    pause_before_click: float = field(default_factory=_ff("MOUSE_PAUSE_BEFORE_CLICK", 0.08))
    pause_after_click: float = field(default_factory=_ff("MOUSE_PAUSE_AFTER_CLICK", 0.10))
    jitter_px: int = field(default_factory=_fi("MOUSE_JITTER_PX", 3))
    double_click_interval: float = field(default_factory=_ff("MOUSE_DOUBLE_CLICK_INTERVAL", 0.30))


@dataclass(frozen=True)
class TypingConfig:
    """Human-like typing behaviour."""

    min_delay: float = field(default_factory=_ff("TYPING_MIN_DELAY", 0.05))
    max_delay: float = field(default_factory=_ff("TYPING_MAX_DELAY", 0.25))
    pause_after: float = field(default_factory=_ff("TYPING_PAUSE_AFTER", 0.15))
    use_clipboard_for_long: bool = field(default_factory=_fb("TYPING_USE_CLIPBOARD_FOR_LONG", True))
    clipboard_min_length: int = field(default_factory=_fi("TYPING_CLIPBOARD_MIN_LENGTH", 25))
    simulate_typos: bool = field(default_factory=_fb("TYPING_SIMULATE_TYPOS", False))
    typo_rate: float = field(default_factory=_fclamp("TYPING_TYPO_RATE", 0.02, 0.0, 1.0))
    dropdown_wait: float = field(default_factory=_ff("DROPDOWN_ANIMATION_WAIT", 0.35))


@dataclass(frozen=True)
class ObserveConfig:
    """Observation loop configuration."""

    poll_interval: float = field(
        default_factory=_fclamp("OBSERVE_POLL_INTERVAL", 0.8, 0.05, 60.0)
    )
    capture_format: str = field(default_factory=_f("OBSERVE_CAPTURE_FORMAT", "png"))
    screenshot_dir: Path = field(default_factory=lambda: Path(_env("OBSERVE_SCREENSHOT_DIR", "screenshots")))


@dataclass(frozen=True)
class WorkflowConfig:
    """Agent workflow / loop configuration."""

    verify_after_action: bool = field(default_factory=_fb("WORKFLOW_VERIFY_AFTER_ACTION", True))
    max_retries_per_action: int = field(default_factory=_fi("WORKFLOW_MAX_RETRIES_PER_ACTION", 3))
    retry_delay: float = field(default_factory=_ff("WORKFLOW_RETRY_DELAY", 0.8))
    next_record_timeout: float = field(default_factory=_ff("WORKFLOW_NEXT_RECORD_TIMEOUT", 120.0))
    next_record_poll: float = field(default_factory=_ff("WORKFLOW_NEXT_RECORD_POLL", 1.5))
    max_records: int = field(default_factory=_fi("WORKFLOW_MAX_RECORDS", 0))
    log_screenshots: bool = field(default_factory=_fb("WORKFLOW_LOG_SCREENSHOTS", True))


@dataclass(frozen=True)
class OverlayConfig:
    """Floating assistant overlay configuration."""

    enabled: bool = field(default_factory=_fb("OVERLAY_ENABLED", True))
    animation_fps: int = field(default_factory=_fi("OVERLAY_ANIMATION_FPS", 30))
    command_port: int = field(default_factory=_fi("OVERLAY_COMMAND_PORT", 19765))


@dataclass(frozen=True)
class ControllerConfig:
    """Controller (command server) configuration."""

    command_port: int = field(default_factory=_fi("CONTROLLER_COMMAND_PORT", 19768))


@dataclass(frozen=True)
class MemoryConfig:
    """Persistent memory configuration."""

    db_path: str = field(default_factory=_f("MEMORY_DB_PATH", "memory.db"))
    alias_learning: bool = field(default_factory=_fb("MEMORY_ALIAS_LEARNING", True))


@dataclass(frozen=True)
class LogConfig:
    """Logging configuration."""

    level: str = field(default_factory=lambda: _env("LOG_LEVEL", "DEBUG").upper())
    folder: Path = field(default_factory=lambda: Path(_env("LOG_FOLDER", "logs")))


@dataclass(frozen=True)
class PluginsConfig:
    """User plugin loading configuration."""

    enabled: bool = field(default_factory=_fb("PLUGINS_ENABLED", True))
    directory: Path = field(default_factory=lambda: Path(_env("PLUGINS_DIR", "plugins")))


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    debug: bool = field(default_factory=_fb("DEBUG_MODE", False))
    vision: VisionConfig = field(default_factory=VisionConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    typing: TypingConfig = field(default_factory=TypingConfig)
    observe: ObserveConfig = field(default_factory=ObserveConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    log: LogConfig = field(default_factory=LogConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)

    def __post_init__(self) -> None:
        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Create runtime directories without mutating the frozen object."""
        for path in (self.observe.screenshot_dir, self.log.folder):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "debug": self.debug,
            "vision": _asdict(self.vision),
            "reasoning": _asdict(self.reasoning),
            "ocr": _asdict(self.ocr),
            "mouse": _asdict(self.mouse),
            "typing": _asdict(self.typing),
            "observe": _asdict(self.observe),
            "workflow": _asdict(self.workflow),
            "overlay": _asdict(self.overlay),
            "controller": _asdict(self.controller),
            "memory": _asdict(self.memory),
            "log": _asdict(self.log),
            "plugins": _asdict(self.plugins),
        }


def _asdict(obj: Any) -> dict[str, Any]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in obj.__dict__.items()}


def load_config() -> AppConfig:
    """Build the application configuration."""
    return AppConfig()
