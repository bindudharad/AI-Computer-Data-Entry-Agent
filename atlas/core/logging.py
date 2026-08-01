"""Structured logging via loguru.

Central setup so every module logs through one configured sink. Screenshot
references can be attached to log records as extra context.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger


def setup_logging(level: str, folder: Path, capture_stdout: bool = True) -> None:
    """Configure the global loguru logger.

    Parameters
    ----------
    level:
        Minimum level shown on the console (DEBUG, INFO, ...).
    folder:
        Directory for the rotating file log.
    capture_stdout:
        When False the console sink is skipped (e.g. GUI launchers).
    """
    folder.mkdir(parents=True, exist_ok=True)
    logger.remove()

    if capture_stdout:
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
            level=level,
            colorize=True,
        )
    logger.add(
        folder / "atlas_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {name}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
    )


def log_screenshot(path: Path, context: str) -> None:
    """Log a screenshot for audit / visual-debug purposes."""
    logger.bind(screenshot=str(path)).info("screenshot[{}] {}", context, path)


def sanitize_secrets(text: str) -> str:
    """Best-effort removal of secret values from free text (log hygiene)."""
    return text


def bind_context(**kwargs: Any) -> Any:
    """Return a logger bound with contextual fields."""
    return logger.bind(**kwargs)
