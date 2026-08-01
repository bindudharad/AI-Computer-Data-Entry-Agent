# ATLAS AI - MPF (Download and Upload Form) Data Entry Agent

An AI-first Windows computer agent that automates MPF form data entry. Atlas behaves like a real human data entry operator: it reads source data from the LEFT panel, fills editable form fields on the RIGHT panel, clicks **Upload Details**, waits for the next record, and repeats until stopped.

```
Observe -> Understand -> Reason -> Plan -> Execute -> Verify -> Observe
```

## Quick Start

### Prerequisites

- Windows 10/11, Python 3.10+
- Install dependencies:
  ```powershell
  pip install -r requirements.txt
  pip install -r requirements-optional.txt   # optional: vision/OCR extras
  python -m playwright install chromium      # only if using --web
  python main.py doctor                      # verify environment
  ```

### Run MPF Data Entry

```powershell
python run_mpf_test.py --records 3            # recommended test workflow
python main.py run --title "MPF" --max-records 3
```

`run_mpf_test.py` is the dedicated MPF test workflow. It:
1. Attaches to the MPF window by title
2. Opens the live debug dashboard
3. Reads the LEFT source panel
4. Fills the RIGHT form fields
5. Clicks **Upload Details**
6. Waits for the next record
7. Repeats until `--records` is reached or STOP (Ctrl+C) is pressed

Other options:
```powershell
python run_mpf_test.py --records 5 --no-dashboard   # headless run
python run_mpf_test.py --json                        # JSON summary output
python run_mpf_test.py --diagnose                    # run diagnostics first
```

### Diagnostic Mode

```powershell
python run_mpf_test.py --diagnose --title "MPF"
python main.py diagnose --title "MPF" --out debug/mpf
```

Captures the complete window state for debugging into `debug/mpf/diag-<timestamp>/`:
- `screen.png` - full monitor screenshot
- `window.png` - the attached window's client area
- `ui_tree.json` - native Win32 control hierarchy
- `scene.json` - the agent's structured perception
- `controls.json` - editable form controls
- `mapping.json` - source-to-form field mapping
- `summary.json` - human-readable diagnosis

If no matching MPF window is open, Atlas prints a friendly message and exits
with code 1 instead of a raw traceback.

### Live Debug Dashboard

The dashboard shows in real-time:

```
ATLAS AI - MPF Data Entry
state: OBSERVING
record 1  key=MPF-001
field: [type] Full Name = KRISHNA
expected: 'KRISHNA'
observed: 'KRISHNA'
confidence: 95%
verify: OK  attempt 0
upload: clicking Upload Details ...
completed fields: Full Name, Gender, DOB, Mobile
missing fields: none
```

## Architecture

```
atlas/
  assistant/     Assistant facade + wiring
  act/           executor, controls, keyboard/mouse/clipboard, verification
  core/          events, logging, state machine, settings
  mapping/       source -> target field mapping
  memory/        SQLite alias learning
  observe/       capture, window attach, screen state
  overlay/       floating status overlay
  plugins/       plugin manager + MPF plugin
  reason/        planner, recovery planner, LLM advisor
  target/        DesktopTarget, WebTarget adapters
  understanding/ source record extraction, field discovery
  vision/        VLM providers, scene analyzer, OCR, debug rendering
  workflow/      AgentLoop + WorkflowSummary

plugins/mpf/
  plugin.py          MPF plugin entry point
  field_mapping.json Field definitions, types, and aliases
  mpf_detector.py    Window detector, panel splitter, upload button finder
  mpf_workflow.py    Record bookkeeping and session tracking
```

## How It Works

### 1. Window Attachment
Atlas finds the MPF window by title (substring match like "MPF"). It captures the window's client area using the Win32 API.

### 2. Scene Understanding
A Vision Language Model (VLM) converts the screenshot into a structured scene description with:
- **Elements** (labels, textboxes, comboboxes, date pickers, buttons)
- **Sections** (source panel, form panel, actions area)
- **Bounding boxes** for each element

### 3. Semantic Field Mapping
The MPF plugin tags elements by position:
- **LEFT panel** → source data (labels + values)
- **RIGHT panel** → form fields (editable controls)
- **BOTTOM** → Upload Details button

Labels from the source panel are mapped to form fields using:
- **Exact match** (same label text)
- **Alias match** (e.g., "DOB" → "Date Of Birth", "Mobile" → "Mobile Number")
- **Fuzzy match** (token overlap, containment)
- **Persistent memory** (learned aliases saved to SQLite)

### 4. Action Planning
The planner generates a deterministic sequence of actions:
- **CLICK** → focus the field
- **CLEAR** → remove existing value
- **TYPE** → type the value (human-like speed)
- **VERIFY** → read back and compare
- **SELECT** → choose from dropdown options
- **CHOOSE_DATE** → type date in the correct format
- **CLICK** Upload Details → submit

### 5. Human-like Execution
- **Mouse** → bezier curves, jitter, random delays, natural pauses
- **Keyboard** → human typing speed, tab navigation, clipboard fast-path for long values
- **Verification** → every value is read back via clipboard, OCR, or target API

### 6. Verification
After typing, Atlas never assumes success:
- **Reads back** the field value (clipboard select-all+copy, OCR region read, or DOM value)
- **Compares** with expected value (normalized: case, whitespace, boolean synonyms)
- **Retries** up to 3 times with corrective actions (re-click, scroll, re-observe)
- **Recovery planner** decides next steps if all retries fail

### 7. Record Lifecycle
```
Observe → SourceRecord → FieldMapping → FillPlan → Execute → Verify → Upload
→ Wait for next record → Repeat
→ STOP button ends execution safely at any point
```

## CLI Commands

```powershell
# Run MPF data entry (test workflow with dashboard + diagnostics)
python run_mpf_test.py --records 3

# Run MPF data entry (CLI entry point)
python main.py run --title "MPF" [--max-records N] [--no-overlay] [--json]

# Diagnostic snapshot
python main.py diagnose --title "MPF" [--out debug/mpf]

# Serve JSON command endpoint
python main.py serve [--port PORT]

# Environment check
python main.py doctor

# Full test suite
python -m pytest tests/ -q --no-header
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VISION_PROVIDER` | `auto` | `openai`, `gemini`, `local`, `auto` |
| `VISION_API_KEY` | `` | API key for VLM |
| `VISION_API_BASE` | `` | Custom API base URL |
| `WORKFLOW_VERIFY_AFTER_ACTION` | `true` | Verify every value action |
| `WORKFLOW_MAX_RETRIES_PER_ACTION` | `3` | Max retry attempts |
| `MOUSE_SPEED` | `0.35` | Mouse movement speed (0-1) |
| `TYPING_MIN_DELAY` | `0.05` | Min delay between keystrokes |
| `TYPING_MAX_DELAY` | `0.25` | Max delay between keystrokes |
| `OCR_ENGINE` | `paddle` | OCR engine: `paddle`, `tesseract`, `none` |
| `LOG_LEVEL` | `DEBUG` | Logging level |
| `PLUGINS_ENABLED` | `true` | Enable plugin system |

## MPF Field Mapping

The `plugins/mpf/field_mapping.json` file defines:
- **window_keywords** - window title patterns to match
- **upload_button_labels** - button text patterns to detect
- **fields** - form field definitions with types and requirements
- **aliases** - vocabulary mappings (source label → canonical form field)

Extend this file to add new fields or aliases without changing code.

## Testing

```powershell
# All tests
python -m pytest tests/ -q --no-header

# MPF plugin + workflow tests
python -m pytest tests/test_mpf_diagnostic.py -v --no-header

# End-to-end MPF integration (real plugin wired into the AgentLoop, 3 records)
python -m pytest tests/test_mpf_integration.py -v --no-header

# State machine tests
python -m pytest tests/test_states.py -v --no-header
```

The integration suite proves the complete record lifecycle without needing a
live MPF window: read LEFT data, fill RIGHT form (text + dropdown + date),
click **Upload Details**, wait for the next record, repeat 3 times, and stop
safely via the STOP flag.

## Project Status

- ✅ MPF window detection
- ✅ Source panel reading
- ✅ Form field discovery
- ✅ Semantic field mapping (exact + alias + fuzzy)
- ✅ Action planning with verification
- ✅ Human-like mouse and keyboard
- ✅ Verification with retry (up to 3 attempts)
- ✅ Upload button detection
- ✅ Record lifecycle (auto-repeat until STOP)
- ✅ Live debug dashboard
- ✅ Diagnostic mode (friendly error when MPF window is not open)
- ✅ MPF plugin system
- ✅ OBSERVING / UNDERSTANDING states
- ✅ Upload events fire when the plugin-located Upload Details button is clicked
- ✅ Vision pipeline degrades gracefully when optional OCR/VLM modules fail
- ✅ Web target support (Playwright)
- ✅ 79/79 tests passing (incl. end-to-end MPF integration)
#   A I - C o m p u t e r - D a t a - E n t r y - A g e n t  
 