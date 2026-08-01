#!/usr/bin/env python3
"""ATLAS AI - command line entry point.

Commands
--------
  run     attach to a target and run the data-entry loop once
  serve   start the localhost JSON command server
  doctor  print an environment / dependency report
  version print the version and exit

Examples
--------
  python main.py run --web --url http://localhost:5173 --max-records 3
  python main.py run --title "Customer Entry"
  python main.py serve --port 19768
  python main.py doctor
"""

from __future__ import annotations

import argparse
import json
import sys

from atlas import APP_NAME, __version__
from atlas.config import load_config
from atlas.core.logging import logger, setup_logging


def _config(args: argparse.Namespace) -> object:
    return load_config()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas", description=f"{APP_NAME} v{__version__}")
    parser.add_argument("--log-level", default="", help="override LOG_LEVEL (DEBUG/INFO/WARNING/ERROR)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="attach and run the loop")
    target = run.add_mutually_exclusive_group(required=True)
    target.add_argument("--web", action="store_true", help="attach to a web page (Playwright)")
    target.add_argument("--title", metavar="TITLE", help="attach to a desktop window by title")
    run.add_argument("--url", default="http://localhost:5173", help="web URL (default http://localhost:5173)")
    run.add_argument("--browser", choices=["chromium", "firefox", "webkit"], default="chromium")
    run.add_argument("--headless", action="store_true", help="run the browser headless")
    run.add_argument("--max-records", type=int, default=0, help="stop after N records (0 = unlimited)")
    run.add_argument("--no-overlay", action="store_true", help="disable the floating overlay")
    run.add_argument("--json", action="store_true", help="print the summary as JSON")
    run.add_argument("--anchor", action="store_true", help="desktop: wait for your click on the first form field, build a UIA field map, then run")
    run.add_argument("--out", default="debug/auto", help="debug/session output directory (default debug/auto; with --anchor default debug/mpf)")

    serve = sub.add_parser("serve", help="start the JSON command server")
    serve.add_argument("--port", type=int, default=0, help="port (default: CONTROLLER_COMMAND_PORT)")

    sub.add_parser("doctor", help="environment report")
    sub.add_parser("version", help="print version")

    diag = sub.add_parser("diagnose", help="dump a diagnostic snapshot of a target window")
    diag.add_argument("--title", default="MPF", help="window title to attach to (substring match)")
    diag.add_argument("--out", default="debug/mpf", help="output directory")

    return parser


def _setup(args: argparse.Namespace) -> object:
    config = _config(args)
    level = (args.log_level or config.log.level) if getattr(config, "log", None) else args.log_level
    setup_logging(level.upper(), config.log.folder)
    return config


def cmd_run(args: argparse.Namespace) -> int:
    from atlas.assistant import Assistant
    from atlas.dashboard import Dashboard

    config = _setup(args)
    dashboard = Dashboard(enabled=not args.no_overlay)
    with Assistant(config) as assistant:
        if args.web:
            assistant.attach_web(url=args.url, browser=args.browser, headless=args.headless)
        else:
            assistant.attach_desktop(title=args.title)
        dashboard.start()
        print(f"attached: {assistant.target.info.to_dict()}")
        if args.anchor:
            if args.web:
                print("--anchor is only valid for desktop targets; ignoring", file=sys.stderr)
                summary = assistant.run(max_records=args.max_records, out_dir=args.out)
            else:
                print("waiting for you to click the first editable field in the form's RIGHT panel ...")
                summary = assistant.run_anchored(max_records=args.max_records, out_dir=args.out)
        else:
            summary = assistant.run(max_records=args.max_records, out_dir=args.out)
        dashboard.stop()
        result = summary.to_dict()
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"records: {len(result['records'])}  completed: {result['completed']}  "
                  f"failed: {result['failed']}  duration: {result['total_duration']:.1f}s")
            if result.get("stopped_reason"):
                print(f"stopped: {result['stopped_reason']}")
    return 0 if result["failed"] == 0 else 1


def cmd_serve(args: argparse.Namespace) -> int:
    from atlas.assistant import Assistant, CommandServer, Controller

    config = _setup(args)
    with Assistant(config) as assistant:
        controller = Controller(assistant)
        port = args.port or config.controller.command_port
        server = CommandServer(controller, host="127.0.0.1", port=port)
        server.start()
        print(f"ATLAS AI command server: {server.url}  (Ctrl+C to stop)")
        try:
            import time

            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    _setup(args)
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    check("python >= 3.10", sys.version_info[:2] >= (3, 10), sys.version.split()[0])

    for module in ("numpy", "PIL", "mss", "loguru", "rapidfuzz", "requests", "dotenv", "win32gui", "pyautogui", "pyperclip"):
        try:
            __import__({"PIL": "PIL", "dotenv": "dotenv"}.get(module, module))
            check(module, True, "ok")
        except Exception as exc:
            check(module, False, str(exc))

    for optional in ("cv2", "paddleocr", "playwright"):
        try:
            __import__(optional)
            check(optional, True, "ok")
        except Exception as exc:
            check(optional, False, str(exc))

    try:
        import pyautogui

        check("screen size", True, str(pyautogui.size()))
    except Exception as exc:
        check("screen size", False, str(exc))

    ok = True
    for name, passed, detail in checks:
        flag = "ok " if passed else "MISS"
        print(f"[{flag}] {name:16s} {detail}")
        ok = ok and passed
    print(f"version: {APP_NAME} {__version__}")
    return 0 if ok else 1


def cmd_version(args: argparse.Namespace) -> int:
    print(f"{APP_NAME} {__version__}")
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    from atlas.diagnostics import Diagnostics
    from atlas.observe.window import AttachError

    _setup(args)
    diag = Diagnostics()
    try:
        folder = diag.run(out_dir=args.out, title=args.title)
    except AttachError as exc:
        print(
            f"error: {exc}\n"
            f"Open the MPF (Download and Upload Form) window first, then re-run:\n"
            f"  python main.py diagnose --title \"{args.title}\"",
            file=sys.stderr,
        )
        return 1
    finally:
        diag.close()
    print(f"diagnostics written to {folder}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers = {
        "run": cmd_run,
        "serve": cmd_serve,
        "doctor": cmd_doctor,
        "version": cmd_version,
        "diagnose": cmd_diagnose,
    }
    handler = handlers.get(args.command)
    if handler is None:
        print(f"unknown command: {args.command}", file=sys.stderr)
        return 2
    try:
        return handler(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        logger.exception("command {} failed", args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())
