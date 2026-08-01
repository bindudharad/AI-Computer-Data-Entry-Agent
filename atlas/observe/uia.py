"""UI Automation (UIA) helpers.

Thin wrapper over pywinauto/comtypes UIAutomationCore used to (a) resolve the
control the user clicks as the ``StartControl`` anchor, and (b) enumerate the
form's editable controls and the source panel's labels so the agent does not
depend on the VLM for exact field geometry.

Every entry point is defensive: if the UIA provider is unavailable or a call
fails, the helpers degrade to ``None``/``[]`` so the rest of the pipeline keeps
working on vision-only data.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas.core.logging import logger
from atlas.vision.models import BBox, ElementType

#: UIA control types that represent editable form widgets.
EDITABLE_CONTROL_TYPES = {
    "Edit",
    "ComboBox",
    "CheckBox",
    "RadioButton",
    "Calendar",
    "List",
    "ListItem",
    "DataGrid",
    "DataItem",
    "Spinner",
    "Slider",
    "Tree",
    "TreeItem",
}

#: Control types worth inspecting even when no editable widget is found; these
#: tell us the window is a real application form (vs a desktop shell).
INSPECTABLE_CONTROL_TYPES = EDITABLE_CONTROL_TYPES | {
    "Button",
    "SplitButton",
    "ScrollBar",
    "Pane",
    "Custom",
    "Table",
    "Document",
    "Group",
    "List",
    "ListItem",
    "Text",
    "Static",
}

#: UIA control types that carry static text (potential source-panel labels).
TEXT_CONTROL_TYPES = {"Text", "Static", "Document"}

#: Mapping of UIA control type -> agent element type.
_CONTROL_TYPE_MAP: dict[str, ElementType] = {
    "Edit": ElementType.TEXTBOX,
    "ComboBox": ElementType.COMBOBOX,
    "CheckBox": ElementType.CHECKBOX,
    "RadioButton": ElementType.RADIO,
    "Calendar": ElementType.CALENDAR,
    "List": ElementType.LISTBOX,
    "ListItem": ElementType.LISTBOX,
    "DataGrid": ElementType.GRID,
    "DataItem": ElementType.GRID,
    "Spinner": ElementType.TEXTBOX,
    "Slider": ElementType.TEXTBOX,
    "Button": ElementType.BUTTON,
    "SplitButton": ElementType.BUTTON,
    "Text": ElementType.LABEL,
    "Static": ElementType.LABEL,
    "StatusBar": ElementType.STATUS_BAR,
    "ToolBar": ElementType.TOOLBAR,
    "Tab": ElementType.TAB,
    "TabItem": ElementType.TAB,
    "Tree": ElementType.TREE_VIEW,
    "TreeItem": ElementType.TREE_VIEW,
    "Menu": ElementType.MENU,
    "MenuItem": ElementType.MENU,
}


@dataclass
class UiaNode:
    """A UIA element flattened into plain data."""

    name: str = ""
    control_type: str = ""
    automation_id: str = ""
    class_name: str = ""
    handle: int | None = None
    rect: BBox | None = None  # absolute screen coordinates
    value: str | None = None
    enabled: bool = True
    visible: bool = True
    password: bool = False
    options: list[str] = field(default_factory=list)
    type_override: ElementType | None = None

    @property
    def editable(self) -> bool:
        return self.control_type in EDITABLE_CONTROL_TYPES and self.enabled

    @property
    def element_type(self) -> ElementType:
        if self.type_override is not None:
            return self.type_override
        if self.control_type == "Edit" and self.password:
            return ElementType.PASSWORD
        return _CONTROL_TYPE_MAP.get(self.control_type, ElementType.UNKNOWN)

    @property
    def center(self) -> tuple[int, int] | None:
        return self.rect.center if self.rect is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "control_type": self.control_type,
            "automation_id": self.automation_id,
            "class_name": self.class_name,
            "handle": self.handle,
            "rect": self.rect.to_dict() if self.rect else None,
            "value": self.value,
            "enabled": self.enabled,
            "visible": self.visible,
            "password": self.password,
            "options": list(self.options),
            "type_override": self.type_override.value if self.type_override else None,
            "editable": self.editable,
            "element_type": self.element_type.value,
        }


class UiaBackend:
    """Lazily loaded, defensive facade over pywinauto's UIA bindings.

    A single process-wide instance is safe because all UIA work happens on the
    thread that constructs it (the main agent thread).
    """

    _instance: UiaBackend | None = None

    def __init__(self) -> None:
        self._available = False
        self._desktop = None
        self._window = None
        try:
            from pywinauto import Desktop
            from pywinauto.controls.uiawrapper import UIAElementInfo
            from pywinauto.uia_defines import IUIA

            self._desktop = Desktop(backend="uia")
            self._element_info = UIAElementInfo
            self._iuia = IUIA()
            self._available = True
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning("UIA backend unavailable: {}", exc)

    @classmethod
    def instance(cls) -> UiaBackend:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def available(self) -> bool:
        return self._available

    # -- resolution ----------------------------------------------------------

    def element_at(self, x: int, y: int) -> UiaNode | None:
        """Resolve the UIA element under an absolute screen point."""
        if not self._available:
            return None
        try:
            info = self._element_info.from_point(x, y)
            return self._flatten(info)
        except Exception as exc:
            logger.debug("element_at({}, {}) failed: {}", x, y, exc)
            return None

    def focused(self) -> UiaNode | None:
        """Return the currently focused UIA element, if any."""
        if not self._available:
            return None
        try:
            return self._flatten(self._element_info(self._iuia.get_focused_element()))
        except Exception as exc:
            logger.debug("focused() failed: {}", exc)
            return None

    # -- enumeration ---------------------------------------------------------

    def descendants(self, handle: int) -> list[UiaNode]:
        """Return every UIA descendant of a window handle (read order)."""
        if not self._available:
            return []
        window = self._window_for(handle)
        if window is None:
            return []
        try:
            wrappers = window.descendants()
        except Exception as exc:
            logger.debug("descendants({}) failed: {}", handle, exc)
            return []
        nodes = []
        for wrapper in wrappers:
            try:
                node = self._flatten(wrapper.element_info)
            except Exception:
                continue
            if node is not None:
                nodes.append(node)
        nodes.sort(key=lambda n: (n.rect.top, n.rect.left) if n.rect else (10**9, 10**9))
        return nodes

    def editable_fields(self, handle: int) -> list[UiaNode]:
        """Editable form widgets under ``handle``, in reading order."""
        nodes = self.descendants(handle)
        return [n for n in nodes if n.editable and not (n.rect is not None and (n.rect.width <= 0 or n.rect.height <= 0))]

    def text_nodes(self, handle: int) -> list[UiaNode]:
        """Static text controls under ``handle`` (source-panel labels)."""
        nodes = self.descendants(handle)
        result = []
        for n in nodes:
            if n.control_type not in TEXT_CONTROL_TYPES:
                continue
            if not n.name or not n.name.strip():
                continue
            if n.rect is None or n.rect.width <= 0 or n.rect.height <= 0:
                continue
            result.append(n)
        return result

    def buttons(self, handle: int) -> list[UiaNode]:
        """Button-like controls under ``handle``."""
        nodes = self.descendants(handle)
        result = []
        for n in nodes:
            if n.control_type not in {"Button", "SplitButton", "MenuItem"}:
                continue
            if not n.name or not n.name.strip():
                continue
            if n.rect is None or n.rect.width <= 0 or n.rect.height <= 0:
                continue
            result.append(n)
        return result

    # -- geometry ------------------------------------------------------------

    @staticmethod
    def client_origin(handle: int) -> tuple[int, int]:
        """Absolute screen (x, y) of the window's client-area origin."""
        import win32gui

        try:
            return win32gui.ClientToScreen(handle, (0, 0))
        except Exception:
            return (0, 0)

    @staticmethod
    def client_size(handle: int) -> tuple[int, int]:
        """(width, height) of the window's client area."""
        import win32gui

        try:
            rect = win32gui.GetClientRect(handle)
            return (rect[2], rect[3])
        except Exception:
            return (0, 0)

    def scroll_into_view(self, node: UiaNode) -> UiaNode | None:
        """Best-effort ScrollItemPattern.ScrollIntoView + refreshed rect."""
        if not self._available:
            return node
        try:
            from comtypes.gen import UIAutomationClient

            info = self._element_info.from_point(*node.center)
            pattern = info.element.GetCurrentPattern(UIAutomationClient.UIA_ScrollItemPatternId)
            pattern.ScrollIntoView()
            refreshed = self.element_at(*node.center)
            if refreshed is not None and refreshed.rect is not None:
                return refreshed
        except Exception as exc:
            logger.debug("scroll_into_view failed: {}", exc)
        return node

    # -- diagnostics ---------------------------------------------------------

    def dump_tree(self, handle: int) -> dict[str, Any]:
        """Serialisable UIA tree for ``debug/mpf/uia_tree.json``."""
        if not self._available:
            return {"available": False}
        window = self._window_for(handle)
        if window is None:
            return {"available": True, "error": "window not found"}
        try:
            return self._tree_dict(window.element_info)
        except Exception as exc:
            return {"available": True, "error": str(exc)}

    def inspectable_nodes(self, handle: int) -> list[UiaNode]:
        """Every node whose control type we can act on, recursing the full tree.

        Unlike :meth:`descendants` (pywinauto's flat walk), this recurses
        explicitly through ``children()`` so controls nested inside Panes,
        Customs, Groups and Tables are still found.
        """
        if not self._available:
            return []
        window = self._window_for(handle)
        if window is None:
            return []
        nodes: list[UiaNode] = []

        def _walk(info) -> None:
            node = self._flatten(info)
            if node is not None and node.control_type in INSPECTABLE_CONTROL_TYPES:
                nodes.append(node)
            try:
                children = info.children()
            except Exception:
                children = []
            for child in children:
                try:
                    _walk(child)
                except Exception:
                    continue

        try:
            _walk(window.element_info)
        except Exception as exc:
            logger.debug("inspectable_nodes({}) failed: {}", handle, exc)
        nodes.sort(key=lambda n: (n.rect.top, n.rect.left) if n.rect else (10**9, 10**9))
        return nodes

    def dump_diagnostics(self, handle: int, out_dir: str | Path) -> dict[str, Any]:
        """Write the full UIA diagnostic set to ``debug/uia/``.

        Files:
          window.json            - handle / title / pid / exe / class / thread / client area
          tree.json              - full recursive UIA tree
          controls.json          - every inspectable control, flattened
          editable_controls.json - the editable form widgets
          labels.json            - static text nodes (source-panel labels)
          focus.json             - the currently focused element
          bounding_boxes.json    - name/type/bbox for every boxed node

        Returns a summary dict (control counts) for the caller.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary: dict[str, Any] = {"handle": handle}

        nodes = self.descendants(handle)
        editable = self.editable_fields(handle)
        text = self.text_nodes(handle)
        boxes = [n for n in nodes if n.rect is not None]

        window_info = self._window_json(handle)
        self._write_json(out / "window.json", window_info)
        self._write_json(out / "tree.json", self.dump_tree(handle))
        self._write_json(out / "controls.json", {"controls": [n.to_dict() for n in nodes], "count": len(nodes)})
        self._write_json(
            out / "editable_controls.json",
            {"editable_controls": [n.to_dict() for n in editable], "count": len(editable)},
        )
        self._write_json(out / "labels.json", {"labels": [n.to_dict() for n in text], "count": len(text)})
        focus = self.focused()
        self._write_json(out / "focus.json", focus.to_dict() if focus else None)
        self._write_json(
            out / "bounding_boxes.json",
            {"bounding_boxes": [_bbox_dict(n) for n in boxes], "count": len(boxes)},
        )

        summary.update(
            {
                "window": window_info,
                "controls": len(nodes),
                "editable_controls": len(editable),
                "labels": len(text),
                "bounding_boxes": len(boxes),
            }
        )
        logger.info(
            "UIA diagnostics written to {} ({} controls, {} editable)",
            out,
            len(nodes),
            len(editable),
        )
        return summary

    def _window_json(self, handle: int) -> dict[str, Any]:
        import win32gui

        def _safe(fn: Any) -> Any:
            try:
                return fn()
            except Exception:
                return None

        title = _safe(lambda: win32gui.GetWindowText(handle)) or ""
        class_name = _safe(lambda: win32gui.GetClassName(handle)) or ""
        thread_id, pid = 0, 0
        try:
            thread_id, pid = win32gui.GetWindowThreadProcessId(handle)
        except Exception:
            pass
        left, top, width, height = self.client_origin(handle)[0], self.client_origin(handle)[1], *self.client_size(handle)
        return {
            "handle": handle,
            "title": title,
            "class_name": class_name,
            "process_id": pid,
            "thread_id": thread_id,
            "executable": self._executable_for_pid(pid),
            "client_area": {"left": left, "top": top, "width": width, "height": height},
            "generated_at": time.strftime("%Y%m%d-%H%M%S"),
        }

    @staticmethod
    def _executable_for_pid(pid: int) -> str:
        if pid <= 0:
            return ""
        try:
            import psutil

            return psutil.Process(pid).name() or ""
        except Exception:
            return ""

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # -- internals -----------------------------------------------------------

    def _window_for(self, handle: int):
        try:
            return self._desktop.window(handle=handle)
        except Exception as exc:
            logger.debug("desktop.window({}) failed: {}", handle, exc)
            return None

    def _flatten(self, info) -> UiaNode | None:
        try:
            rect = info.rectangle
        except Exception:
            rect = None
        box = None
        if rect is not None:
            try:
                width = max(0, rect.right - rect.left)
                height = max(0, rect.bottom - rect.top)
                box = BBox(int(rect.left), int(rect.top), int(width), int(height))
            except Exception:
                box = None
        try:
            name = info.name or ""
        except Exception:
            name = ""
        try:
            control_type = info.control_type or ""
        except Exception:
            control_type = ""
        try:
            automation_id = info.automation_id or ""
        except Exception:
            automation_id = ""
        try:
            handle = info.handle
        except Exception:
            handle = None
        try:
            class_name = info.class_name or ""
        except Exception:
            class_name = ""
        try:
            enabled = bool(info.enabled)
        except Exception:
            enabled = True
        try:
            visible = bool(info.visible)
        except Exception:
            visible = True
        value: str | None = None
        try:
            value = info.value_pattern().current_value() if hasattr(info, "value_pattern") else None
        except Exception:
            value = None
        return UiaNode(
            name=name,
            control_type=control_type,
            automation_id=automation_id,
            class_name=class_name,
            handle=handle,
            rect=box,
            value=value,
            enabled=enabled,
            visible=visible,
        )

    def _tree_dict(self, info) -> dict[str, Any]:
        try:
            name = info.name or ""
        except Exception:
            name = ""
        try:
            control_type = info.control_type or ""
        except Exception:
            control_type = ""
        try:
            auto_id = info.automation_id or ""
        except Exception:
            auto_id = ""
        try:
            handle = info.handle
        except Exception:
            handle = None
        try:
            r = info.rectangle
            rect = [int(r.left), int(r.top), int(r.right), int(r.bottom)]
        except Exception:
            rect = None
        node: dict[str, Any] = {
            "name": name,
            "control_type": control_type,
            "automation_id": auto_id,
            "handle": handle,
            "rect": rect,
        }
        try:
            children = info.children()
        except Exception:
            children = []
        kids = []
        for child in children:
            try:
                kids.append(self._tree_dict(child))
            except Exception:
                continue
        if kids:
            node["children"] = kids
        return node


def _bbox_dict(node: UiaNode) -> dict[str, Any]:
    """Compact bbox entry for a UIA node."""
    return {
        "name": node.name,
        "control_type": node.control_type,
        "automation_id": node.automation_id,
        "bbox": node.rect.to_dict() if node.rect else None,
        "editable": node.editable,
    }


__all__ = ["UiaBackend", "UiaNode", "EDITABLE_CONTROL_TYPES", "TEXT_CONTROL_TYPES", "INSPECTABLE_CONTROL_TYPES"]
