"""Assistant controller.

A thin command layer over :class:`~atlas.assistant.assistant.Assistant` so the
agent can be driven from the CLI, tests or any JSON client. The ``CommandServer``
exposes the same commands over a localhost HTTP endpoint.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from atlas.core.logging import logger


class Controller:
    """Handles JSON commands for an :class:`Assistant`."""

    def __init__(self, assistant: Any) -> None:
        self._assistant = assistant

    @property
    def assistant(self) -> Any:
        return self._assistant

    def handle(self, command: dict) -> dict:
        """Dispatch a command dict and return a result dict."""
        name = str(command.get("command") or command.get("action") or "").strip().lower()
        handler = getattr(self, f"cmd_{name}", None)
        if handler is None:
            return {"ok": False, "error": f"unknown command {name!r}"}
        try:
            return handler(command)
        except Exception as exc:
            logger.warning("command {} failed: {}", name, exc)
            return {"ok": False, "command": name, "error": str(exc)}

    # -- commands ------------------------------------------------------------

    def cmd_status(self, command: dict) -> dict:
        target = self._assistant.target
        info = None
        if target is not None and getattr(target, "info", None) is not None:
            info = target.info.to_dict()
        return {
            "ok": True,
            "command": "status",
            "attached": target is not None,
            "target": info,
            "state": self._assistant.state,
        }

    def cmd_attach(self, command: dict) -> dict:
        mode = str(command.get("mode", command.get("target", "desktop"))).lower()
        if mode in {"web", "browser", "page"}:
            target = self._assistant.attach_web(
                url=command.get("url"),
                browser=str(command.get("browser", "chromium")),
                headless=bool(command.get("headless", False)),
            )
        else:
            target = self._assistant.attach_desktop(title=command.get("title"))
        return {"ok": True, "command": "attach", "target": target.info.to_dict() if target.info else {}}

    def cmd_detach(self, command: dict) -> dict:
        self._assistant.detach()
        return {"ok": True, "command": "detach"}

    def cmd_run(self, command: dict) -> dict:
        summary = self._assistant.run(max_records=int(command.get("max_records", 0)))
        return {"ok": True, "command": "run", "summary": summary.to_dict()}

    def cmd_stop(self, command: dict) -> dict:
        self._assistant.stop()
        return {"ok": True, "command": "stop"}

    def cmd_pause(self, command: dict) -> dict:
        self._assistant.pause()
        return {"ok": True, "command": "pause"}

    def cmd_resume(self, command: dict) -> dict:
        self._assistant.resume()
        return {"ok": True, "command": "resume"}

    def cmd_aliases(self, command: dict) -> dict:
        return {"ok": True, "command": "aliases", "aliases": self._assistant.memory.all_aliases()}

    def cmd_learn_alias(self, command: dict) -> dict:
        variant = str(command.get("variant", ""))
        canonical = str(command.get("canonical", ""))
        if not variant or not canonical:
            return {"ok": False, "command": "learn_alias", "error": "variant and canonical required"}
        self._assistant.memory.learn_alias(variant, canonical)
        self._assistant.mapper.aliases.learn(variant, canonical)
        return {"ok": True, "command": "learn_alias"}

    def cmd_config(self, command: dict) -> dict:
        return {"ok": True, "command": "config", "config": self._assistant.config.to_dict()}

    def cmd_close(self, command: dict) -> dict:
        self._assistant.close()
        return {"ok": True, "command": "close"}


class _Handler(BaseHTTPRequestHandler):
    server: CommandServer  # type: ignore[assignment]

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b"{}"
        try:
            command = json.loads(body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            command = {}
        result = self.server.controller.handle(command)
        payload = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        payload = json.dumps({"ok": True, "endpoint": "/command"}).encode("utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: Any) -> None:  # silence request logging
        pass


class CommandServer:
    """Localhost HTTP server exposing the controller's commands."""

    def __init__(self, controller: Controller, host: str = "127.0.0.1", port: int = 19768) -> None:
        self._controller = controller
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._host = host
        self._port = port

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = ThreadingHTTPServer((self._host, self._port), _Handler)
        self._server.controller = self._controller  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, name="atlas-controller", daemon=True)
        self._thread.start()
        logger.info("command server listening on {}:{}", self._host, self._port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def controller(self) -> Controller:
        return self._controller

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"


__all__ = ["Controller", "CommandServer"]
