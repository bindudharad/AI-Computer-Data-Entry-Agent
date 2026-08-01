"""MPF diagnostic mode.

Captures everything the agent needs to understand and operate the MPF
(Download and Upload Form) window, writing it to a timestamped folder
(default ``debug/mpf/``):

* ``screen.png``   - full monitor screenshot
* ``window.png``   - the attached window's client area
* ``ui_tree.json`` - native Win32 control hierarchy (hwnd / class / text / rect)
* ``scene.json``   - the agent's structured perception of the window
* ``controls.json``- the editable form controls discovered on the scene
* ``mapping.json`` - how the LEFT source panel maps onto the RIGHT form
* ``summary.json`` - a human-readable diagnosis

Run with ``python main.py diagnose --title MPF``. The diagnosis is read-only -
it never types or clicks anything.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import win32gui

from atlas.assistant import Assistant
from atlas.config import AppConfig, load_config
from atlas.core.logging import logger
from atlas.mapping.mapper import SemanticMapper
from atlas.understanding.fields import discover_fields
from atlas.understanding.source import SourceReader
from atlas.vision.capture import WindowCapture
from atlas.vision.scene import WindowSceneSource


class Diagnostics:
    """Collects a full diagnostic snapshot of an attached target window."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or load_config()
        self._assistant = Assistant(self._config)

    @property
    def assistant(self) -> Assistant:
        return self._assistant

    def run(self, out_dir: str | Path = "debug/mpf", title: str | None = None) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        folder = out / f"diag-{stamp}"
        folder.mkdir(parents=True, exist_ok=True)

        target = self._assistant.attach_desktop(title=title)
        handle = target.info.handle

        summary: dict[str, Any] = {
            "generated_at": stamp,
            "window": target.info.to_dict(),
            "attached": True,
            "notes": [],
        }

        self._dump_full_screen(folder / "screen.png")
        capture = WindowCapture(grabber=self._assistant._grabber)  # noqa: SLF001 - internal reuse
        capture.attach(handle, target.info.title)
        area = capture.capture_until_nonempty(timeout=10.0)
        if area is not None:
            area.save(folder / "window.png")
            summary["client_area"] = {"left": area.left, "top": area.top, "width": area.width, "height": area.height}
        else:
            summary["notes"].append("window client area capture returned empty")

        self._dump_ui_tree(handle, folder / "ui_tree.json", summary)

        # Step 2: write the full UIA diagnostic set to debug/uia/.
        try:
            from atlas.observe.uia import UiaBackend

            uia_summary = UiaBackend.instance().dump_diagnostics(handle, folder / "uia")
            summary["uia"] = uia_summary
        except Exception as exc:
            summary["notes"].append(f"UIA diagnostics failed: {exc}")

        source = WindowSceneSource(capture, self._assistant._analyzer)  # noqa: SLF001
        analysis = source.observe()
        if analysis is None:
            summary["notes"].append("scene analysis failed (capture or VLM error)")
            self._write_json(folder / "summary.json", summary)
            return folder

        self._write_json(folder / "scene.json", analysis.scene.to_dict())

        reader = SourceReader()
        record = reader.read(analysis.scene)
        fields = discover_fields(analysis.scene)
        mapper: SemanticMapper = self._assistant.mapper
        mapping = mapper.map(record, fields)

        self._write_json(folder / "controls.json", {"fields": [f.to_dict() for f in fields]})
        self._write_json(folder / "mapping.json", mapping.to_dict())

        summary.update(
            {
                "record_key": record.record_key,
                "source_pairs": [{"label": k, "value": v} for k, v in record.pairs.items()],
                "fields_visible": len(fields),
                "mappings": len(mapping.mappings),
                "unmapped_source": list(mapping.unmapped_source),
                "unmatched_fields": [f.label for f in mapping.unmatched_fields],
            }
        )
        self._write_json(folder / "summary.json", summary)
        logger.info("diagnostics written to {}", folder)
        return folder

    # -- dumpers --------------------------------------------------------------

    def _dump_full_screen(self, path: Path) -> None:
        try:
            import mss

            with mss.mss() as sct:
                monitor = sct.monitors[0]
                shot = sct.grab(monitor)
            from PIL import Image

            Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX").save(path)
            logger.debug("full-screen screenshot -> {}", path)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("full-screen capture failed: {}", exc)

    def _dump_ui_tree(self, handle: int, path: Path, summary: dict[str, Any]) -> None:
        root = self._collect_controls(handle)
        count = _count_nodes(root)
        summary["ui_controls"] = count
        self._write_json(path, {"root": root, "count": count})
        logger.debug("ui tree -> {} ({} controls)", path, count)

    @staticmethod
    def _collect_controls(handle: int) -> dict[str, Any]:
        children = []

        def _walk(hwnd: int) -> None:
            try:
                class_name = win32gui.GetClassName(hwnd) or ""
                text = win32gui.GetWindowText(hwnd) or ""
                rect = win32gui.GetWindowRect(hwnd)
            except Exception:
                return
            node = {
                "hwnd": hwnd,
                "class": class_name,
                "text": text,
                "rect": list(rect),
            }
            kids = []
            try:
                win32gui.EnumChildWindows(hwnd, lambda h, _: kids.append(h), None)
            except Exception:
                kids = []
            for kid in kids:
                sub = _walk(kid)
                if sub is not None:
                    node.setdefault("children", []).append(sub)
            return node

        return _walk(handle)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def close(self) -> None:
        self._assistant.close()


def _count_nodes(node: dict[str, Any]) -> int:
    total = 1
    for child in node.get("children", []):
        total += _count_nodes(child)
    return total


__all__ = ["Diagnostics"]
