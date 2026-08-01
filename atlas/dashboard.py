"""Live debug dashboard (optional).

A small always-on-top tkinter window that renders the agent's live state while
it operates the MPF form: current agent state, the source record being
processed, the field currently being filled (expected vs observed value and
verification confidence), completed vs missing fields, upload progress and a
log tail.

It consumes the event bus - the agent runs identically without it. Driven by a
thread-safe queue so the tkinter loop never blocks the agent.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

from atlas.core.events import Event, EventType, get_event_bus
from atlas.core.logging import logger

_STATE_LABELS = {
    "idle": "IDLE",
    "waiting_attach": "WAITING FOR WINDOW",
    "attaching": "ATTACHING",
    "waiting_for_start_field": "CLICK THE FIRST FORM FIELD",
    "build_ui_tree": "BUILDING UI TREE",
    "screen_model": "SCREEN MODEL",
    "record_extraction": "EXTRACTING RECORD",
    "field_mapping": "BUILDING UIA FIELD MAP",
    "watching": "WATCHING FOR RECORD",
    "observing": "OBSERVING",
    "understanding": "UNDERSTANDING",
    "analyzing": "ANALYZING SCREEN",
    "planning": "PLANNING",
    "thinking": "THINKING",
    "typing": "TYPING",
    "clicking": "CLICKING",
    "scrolling": "SCROLLING",
    "uploading": "UPLOADING",
    "waiting": "WAITING FOR NEXT RECORD",
    "verifying": "VERIFYING",
    "completed": "COMPLETED",
    "paused": "PAUSED",
    "stopped": "STOPPED",
    "error": "ERROR",
    "recovery": "RECOVERING",
}


class Dashboard:
    """Live state window driven by event-bus events."""

    def __init__(self, enabled: bool = True, title: str = "ATLAS AI - MPF Data Entry") -> None:
        self._enabled = enabled
        self._title = title
        self._queue: queue.Queue[Event] = queue.Queue(maxsize=2000)
        self._root: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._unsub: Any = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if not self._enabled:
            return
        try:
            import tkinter as tk  # noqa: F401
        except ImportError:
            logger.warning("tkinter unavailable - dashboard disabled")
            return
        self._stop.clear()
        self._unsub = get_event_bus().subscribe_all(self._enqueue)
        self._thread = threading.Thread(target=self._run, name="atlas-dashboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._unsub is not None:
            try:
                self._unsub()
            except Exception:
                pass
            self._unsub = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- event plumbing -------------------------------------------------------

    def _enqueue(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            pass

    # -- tkinter --------------------------------------------------------------

    def _run(self) -> None:
        try:
            import tkinter as tk
        except ImportError:
            return
        try:
            root = tk.Tk()
            root.title(self._title)
            root.geometry("520x560")
            root.configure(bg="#0d1117")
            root.attributes("-topmost", True)
            self._root = root
            self._build_ui(tk)
            self._poll()
            root.mainloop()
        except Exception as exc:
            logger.debug("dashboard stopped: {}", exc)
        finally:
            # Release every tkinter object on the thread that created it so the
            # Tcl interpreter is not touched from the main thread at GC/shutdown.
            self._root = None
            for name in list(vars(self)):
                if name.startswith(
                    ("state_var", "record_var", "field_var", "expected_var", "observed_var",
                     "confidence_var", "verify_var", "upload_var", "completed_var", "missing_var",
                     "progress_var", "elapsed_var")
                ) or name == "log":
                    try:
                        setattr(self, name, None)
                    except Exception:
                        pass

    def _build_ui(self, tk: Any) -> None:
        fg, bg, accent = "#c9d1d9", "#0d1117", "#7dd3fc"
        root = self._root

        header = tk.Label(root, text=self._title, fg=accent, bg=bg, font=("Consolas", 13, "bold"), anchor="w")
        header.pack(fill="x", padx=10, pady=(10, 2))

        self.state_var = tk.StringVar(value="idle")
        state = tk.Label(root, textvariable=self.state_var, fg="#e6edf3", bg=bg, font=("Consolas", 11, "bold"), anchor="w")
        state.pack(fill="x", padx=10)

        self.record_var = tk.StringVar(value="record: -")
        tk.Label(root, textvariable=self.record_var, fg=fg, bg=bg, font=("Consolas", 10), anchor="w").pack(fill="x", padx=10)

        self.field_var = tk.StringVar(value="field: -")
        tk.Label(root, textvariable=self.field_var, fg=fg, bg=bg, font=("Consolas", 10), anchor="w").pack(fill="x", padx=10)

        self.expected_var = tk.StringVar(value="expected: -")
        tk.Label(root, textvariable=self.expected_var, fg="#58a6ff", bg=bg, font=("Consolas", 10), anchor="w").pack(fill="x", padx=10)

        self.observed_var = tk.StringVar(value="observed: -")
        tk.Label(root, textvariable=self.observed_var, fg="#3fb950", bg=bg, font=("Consolas", 10), anchor="w").pack(fill="x", padx=10)

        self.confidence_var = tk.StringVar(value="confidence: -")
        tk.Label(root, textvariable=self.confidence_var, fg="#d2a8ff", bg=bg, font=("Consolas", 10), anchor="w").pack(fill="x", padx=10)

        self.verify_var = tk.StringVar(value="verify: -")
        tk.Label(root, textvariable=self.verify_var, fg=fg, bg=bg, font=("Consolas", 10), anchor="w").pack(fill="x", padx=10)

        self.upload_var = tk.StringVar(value="upload: waiting")
        tk.Label(root, textvariable=self.upload_var, fg=fg, bg=bg, font=("Consolas", 10), anchor="w").pack(fill="x", padx=10)

        self.progress_var = tk.StringVar(value="progress: 0 records / 0 failed")
        tk.Label(root, textvariable=self.progress_var, fg="#3fb950", bg=bg, font=("Consolas", 10), anchor="w").pack(fill="x", padx=10)

        self.elapsed_var = tk.StringVar(value="elapsed: 0.0s")
        tk.Label(root, textvariable=self.elapsed_var, fg="#d2a8ff", bg=bg, font=("Consolas", 10), anchor="w").pack(fill="x", padx=10)

        sep = tk.Frame(root, bg="#21262d", height=1)
        sep.pack(fill="x", padx=10, pady=6)

        tk.Label(root, text="completed fields", fg=accent, bg=bg, font=("Consolas", 9, "bold"), anchor="w").pack(fill="x", padx=10)
        self.completed_var = tk.StringVar(value="-")
        tk.Label(root, textvariable=self.completed_var, fg=fg, bg=bg, font=("Consolas", 9), justify="left", anchor="nw").pack(
            fill="x", padx=10
        )

        tk.Label(root, text="missing fields", fg="#f85149", bg=bg, font=("Consolas", 9, "bold"), anchor="w").pack(fill="x", padx=10)
        self.missing_var = tk.StringVar(value="-")
        tk.Label(root, textvariable=self.missing_var, fg=fg, bg=bg, font=("Consolas", 9), justify="left", anchor="nw").pack(
            fill="x", padx=10
        )

        tk.Label(root, text="log", fg=accent, bg=bg, font=("Consolas", 9, "bold"), anchor="w").pack(fill="x", padx=10, pady=(6, 0))
        self.log = tk.Text(root, bg="#161b22", fg="#8b949e", font=("Consolas", 9), height=10, relief="flat", state="disabled")
        self.log.pack(fill="both", expand=True, padx=10, pady=(2, 10))

    # -- update loop ----------------------------------------------------------

    def _poll(self) -> None:
        if self._stop.is_set():
            try:
                self._root.after(50, self._root.destroy)
            except Exception:
                pass
            return
        drained = 0
        while drained < 200:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            self._handle(event)
            drained += 1
        self._refresh_elapsed()
        try:
            self._root.after(150, self._poll)
        except Exception:
            pass

    def _handle(self, event: Event) -> None:
        if event.type == EventType.AGENT_STARTED:
            self._started = event.data.get("started") or __import__("time").time()
            self._completed = 0
            self._failed = 0
            self._refresh_progress()
        elif event.type == EventType.STATE_CHANGED:
            state = event.data.get("state", "")
            self.state_var.set(f"state: {_STATE_LABELS.get(state, state)}")
        elif event.type == EventType.RECORD_STARTED:
            index = event.data.get("index", "?")
            key = (event.data.get("record") or {}).get("record_key", "")
            self.record_var.set(f"record {index}  key={key or '?'}")
            self.field_var.set("field: -")
            self.verify_var.set("verify: -")
        elif event.type == EventType.ACTION_STARTED:
            a = event.data
            action_type = a.get("type", "")
            reason = a.get("reason", "")
            confidence = a.get("confidence", 1.0)
            self.field_var.set(f"field: [{action_type}] {reason}")
            self.confidence_var.set(f"confidence: {confidence:.0%}")
        elif event.type == EventType.VERIFICATION:
            data = event.data
            ok = data.get("ok")
            expected = data.get("expected") or ""
            observed = data.get("observed") or ""
            attempt = data.get("attempt", 0)
            marker = "OK " if ok else "RETRY"
            self.expected_var.set(f"expected: {expected!r}")
            self.observed_var.set(f"observed: {observed!r}")
            self.verify_var.set(
                f"verify: {marker}  attempt {attempt}"
            )
        elif event.type == EventType.SCREEN_STATE:
            data = event.data
            completed = data.get("completed_fields") or []
            missing = data.get("missing_fields") or []
            self.completed_var.set(", ".join(completed) if completed else "-")
            missing_text = ", ".join(m.get("field", "") for m in missing) if missing else "none"
            self.missing_var.set(missing_text)
        elif event.type == EventType.UPLOADING:
            self.upload_var.set("upload: clicking Upload Details ...")
        elif event.type == EventType.UPLOAD_COMPLETED:
            self.upload_var.set("upload: completed")
            self._log("upload completed")
        elif event.type == EventType.NEXT_RECORD_WAITING:
            self.upload_var.set("upload: waiting for next record ...")
        elif event.type == EventType.RECORD_COMPLETED:
            self._completed = getattr(self, "_completed", 0) + 1
            self._refresh_progress()
            self._log(f"record finished OK (index {event.data.get('index', '?')})")
        elif event.type == EventType.RECORD_FAILED:
            self._failed = getattr(self, "_failed", 0) + 1
            self._refresh_progress()
            self._log(f"record FAILED (index {event.data.get('index', '?')})")
        elif event.type == EventType.NO_RECORD:
            self._log(f"no record detected: {event.data.get('reason', '')}")
        elif event.type == EventType.WORKFLOW_COMPLETE:
            self._log(f"workflow complete: {event.data.get('stopped_reason', '')}")
        elif event.type == EventType.LOG:
            self._log(str(event.data.get("message", "")))
        elif event.type == EventType.ERROR:
            self._log(f"ERROR: {event.data.get('message', '')}")

    def _refresh_progress(self) -> None:
        done = getattr(self, "_completed", 0)
        failed = getattr(self, "_failed", 0)
        self.progress_var.set(f"progress: {done} records / {failed} failed")

    def _refresh_elapsed(self) -> None:
        import time

        started = getattr(self, "_started", None)
        if started is None:
            return
        elapsed = time.time() - started
        self.elapsed_var.set(f"elapsed: {elapsed:.1f}s")

    def _log(self, message: str) -> None:
        try:
            self.log.configure(state="normal")
            self.log.insert("end", message + "\n")
            self.log.see("end")
            lines = int(self.log.index("end-1c").split(".")[0])
            if lines > 40:
                self.log.delete("1.0", f"{lines - 40}.0")
            self.log.configure(state="disabled")
        except Exception:
            pass


__all__ = ["Dashboard"]
