#!/usr/bin/env python3
"""MPF test workflow script.

Usage:
    python run_mpf_test.py [--title "MPF"] [--records N] [--diagnose] [--no-dashboard] [--json]

Runs ATLAS AI against the MPF (Download and Upload Form) application.

Examples:
    # Run 3 records with live dashboard
    python run_mpf_test.py --records 3

    # Run a diagnostic snapshot first
    python run_mpf_test.py --diagnose

    # Run without the dashboard overlay
    python run_mpf_test.py --records 5 --no-dashboard --json

    # Run until STOP (Ctrl+C) or no more records
    python run_mpf_test.py
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
import time

from atlas.diagnostics import Diagnostics
from atlas.assistant import Assistant
from atlas.config import load_config
from atlas.dashboard import Dashboard
from atlas.core.logging import logger, setup_logging
from atlas.observe.window import AttachError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ATLAS AI - MPF Data Entry Test Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--title", default="MPF", help="MPF window title (substring match)")
    parser.add_argument("--records", type=int, default=0, help="max records to process (0 = unlimited)")
    parser.add_argument("--diagnose", action="store_true", help="run diagnostic mode instead of data entry")
    parser.add_argument("--out", default="debug/mpf", help="diagnostic output directory")
    parser.add_argument("--no-dashboard", action="store_true", help="disable the live debug dashboard")
    parser.add_argument("--json", action="store_true", help="output summary as JSON")
    parser.add_argument("--log-level", default="INFO", help="log level (DEBUG/INFO/WARNING/ERROR)")
    return parser


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\nSTOP signal received. Waiting for current action to complete...")
    raise KeyboardInterrupt()


def main() -> int:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = _build_parser().parse_args()
    setup_logging(args.log_level, Path("logs"))
    config = load_config()

    if args.diagnose:
        print(f"ATLAS AI - MPF Diagnostic Mode")
        print(f"Window: {args.title!r}")
        print(f"Output: {args.out}")
        print("-" * 50)
        diag = Diagnostics(config)
        try:
            folder = diag.run(out_dir=args.out, title=args.title)
        except AttachError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            print(f"Open the MPF (Download and Upload Form) window first, then re-run.", file=sys.stderr)
            return 1
        finally:
            diag.close()
        print(f"\nDiagnostics saved to: {folder}")
        print(f"  screen.png      - full monitor screenshot")
        print(f"  window.png      - attached window client area")
        print(f"  ui_tree.json    - native Win32 control hierarchy")
        print(f"  scene.json      - agent's structured perception")
        print(f"  controls.json   - editable form controls")
        print(f"  mapping.json    - source-to-form field mapping")
        print(f"  summary.json    - human-readable diagnosis")
        return 0

    print(f"ATLAS AI - MPF Data Entry")
    print(f"Window: {args.title!r}")
    print(f"Max records: {args.records if args.records > 0 else 'unlimited'}")
    print(f"Dashboard: {'disabled' if args.no_dashboard else 'enabled'}")
    print("-" * 50)
    print("NEXT: click the FIRST editable field in the MPF form's RIGHT panel")
    print("      (a text box, dropdown or date field) to anchor the form.")
    print("Commands during execution:")
    print("  Ctrl+C  - Stop safely after current field")
    print("-" * 50)

    dashboard = Dashboard(enabled=not args.no_dashboard)
    try:
        with Assistant(config) as assistant:
            try:
                assistant.attach_desktop(title=args.title)
            except AttachError as exc:
                print(f"\nERROR: {exc}", file=sys.stderr)
                print(f"Open the MPF (Download and Upload Form) window first, then re-run.", file=sys.stderr)
                return 1
            dashboard.start()

            target_info = assistant.target.info.to_dict() if assistant.target else {}
            print(f"Attached: {target_info.get('title', '?')} (handle={target_info.get('handle', '?')})")
            print(f"Window rect: {target_info.get('rect', '?')}")

            summary = assistant.run_anchored(max_records=args.records, out_dir=args.out)

            dashboard.stop()

            result = summary.to_dict()
            print("-" * 50)
            print(f"WORKFLOW COMPLETE")
            print(f"  Records processed: {len(result['records'])}")
            print(f"  Completed: {result['completed']}")
            print(f"  Failed: {result['failed']}")
            print(f"  Duration: {result['total_duration']:.1f}s")
            print(f"  Fields filled: {summary.fields_filled}")
            if result.get("stopped_reason"):
                print(f"  Stop reason: {result['stopped_reason']}")

            if result["completed"] > 0:
                avg_time = result["total_duration"] / result["completed"]
                print(f"  Avg time per record: {avg_time:.1f}s")

            if result["failed"] > 0:
                print(f"\n  Failed records:")
                for rec in result["records"]:
                    if not rec["success"]:
                        print(f"    Record {rec['index']}: {rec.get('message', 'unknown error')}")

            print(f"\n  Debug artifacts in {args.out}:")
            for name in ("start_control.json", "uia_tree.json", "field_map.json",
                         "left_panel.png", "right_panel.png", "planner.json",
                         "execution.json", "verification.json", "failure.json"):
                if (Path(args.out) / name).exists():
                    print(f"    {name}")

            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

            return 0 if result["failed"] == 0 else 1

    except KeyboardInterrupt:
        print("\nExecution stopped by user.")
        try:
            dashboard.stop()
        except Exception:
            pass
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        logger.exception("MPF test workflow failed")
        try:
            dashboard.stop()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())