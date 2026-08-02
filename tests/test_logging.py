"""Tests for category logging sinks.

setup_logging() must create one rotating file per domain category
(actions/errors/ocr/uia/timings/focus/verification) plus the main log, and a
category-bound logger must route its records only to its own file.
"""

from __future__ import annotations

from atlas.core.logging import (
    CATEGORIES,
    action_logger,
    focus_logger,
    ocr_logger,
    setup_logging,
    timing_logger,
    uia_logger,
    verification_logger,
)


def test_setup_logging_creates_all_category_files(tmp_path) -> None:
    setup_logging("DEBUG", tmp_path, capture_stdout=False)
    files = {p.name for p in tmp_path.iterdir()}
    # main rotating log uses a timestamped name; category files are static.
    for stem in ("actions", "errors", "ocr", "uia", "timings", "focus", "verification"):
        assert f"{stem}.log" in files, f"missing {stem}.log in {sorted(files)}"


def test_category_loggers_route_to_own_file(tmp_path) -> None:
    setup_logging("DEBUG", tmp_path, capture_stdout=False)
    action_logger.info("hello action")
    focus_logger.info("hello focus")
    ocr_logger.info("hello ocr")
    uia_logger.info("hello uia")
    timing_logger.info("hello timing")
    verification_logger.info("hello verification")

    action_logger.complete()
    assert "hello action" in (tmp_path / "actions.log").read_text(encoding="utf-8")
    assert "hello action" not in (tmp_path / "focus.log").read_text(encoding="utf-8")
    assert "hello focus" in (tmp_path / "focus.log").read_text(encoding="utf-8")
    assert "hello ocr" in (tmp_path / "ocr.log").read_text(encoding="utf-8")
    assert "hello uia" in (tmp_path / "uia.log").read_text(encoding="utf-8")
    assert "hello timing" in (tmp_path / "timings.log").read_text(encoding="utf-8")
    assert "hello verification" in (tmp_path / "verification.log").read_text(encoding="utf-8")


def test_errors_always_land_in_errors_log(tmp_path) -> None:
    from loguru import logger

    setup_logging("DEBUG", tmp_path, capture_stdout=False)
    logger.error("boom regardless of category")
    logger.complete()
    assert "boom regardless of category" in (tmp_path / "errors.log").read_text(encoding="utf-8")


def test_category_namespace_covers_spec_files() -> None:
    # "errors" is a dedicated error-level sink, not a category binding.
    assert set(CATEGORIES.values()) == {
        "actions", "ocr", "uia", "timings", "focus", "verification"
    }
