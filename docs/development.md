# Development

## Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-optional.txt
python -m playwright install chromium
```

## Tests

```powershell
python -m pytest tests -q
```

66 tests: state machine, events, config, models, memory, mapping, planner,
executor, workflow loop, controller, and a Playwright E2E that drives a real
browser with a mock vision provider.

## Static analysis

```powershell
python -m ruff check atlas main.py tests
python -m mypy --config-file pyproject.toml atlas
```

Configuration lives in `pyproject.toml`:

- **ruff** selects `E F W I N B UP SIM`; line length 110; `B024`/`B027` are
  ignored because several interface classes deliberately declare optional
  no-op hooks (e.g. `close`, `on_event`).
- **mypy** targets Python 3.10 and checks `atlas/` with
  `ignore_missing_imports` (PaddleOCR/OpenCV/Playwright ship no stubs).

Run all three as a single gate:

```powershell
python -m pytest tests -q
python -m ruff check atlas main.py tests
python -m mypy --config-file pyproject.toml atlas
```

## Conventions

- Python 3.10+; `from __future__ import annotations` in every module.
- loguru (`logger.info/...`) for all logging; no `print` in library code.
- Type hints everywhere; run mypy before pushing changes.
- Tests live in `tests/` and use `tests/conftest.py` for path setup.
