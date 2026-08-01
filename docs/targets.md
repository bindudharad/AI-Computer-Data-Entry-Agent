# Targets

Both targets implement `TargetAdapter` (`atlas/target/base.py`) and are driven
by the same `AgentLoop`. The assistant picks the right `ControlInterface` and
`FieldVerifier` per target, so mapping/planning/loop code never changes.

## Desktop target (`atlas/target/desktop.py`)

- `attach(title)` finds and brings a window to the foreground
  (`atlas/observe/window.py`), verifies focus, and attaches a window capture.
- `observe()` screenshots the window client area and runs the VLM scene
  analyser.
- Control is physical: `ControlEngine` moves the real mouse along a bezier
  curve, clicks field bboxes, types with human-like delays, and uses
  clipboard paste for long values.
- Verification reads back via the focused-control clipboard or OCR on the
  field region.

Attach:

```powershell
python main.py run --title "Customer Entry"
```

## Web target (`atlas/target/web.py`)

Perception stays vision-first: `observe()` screenshots the page viewport and
runs the same VLM analyser. On top of that, a page-side script (`_INSPECT_JS`)
builds a DOM index that maps each form control's label to a CSS selector
(`id`, `name`, `data-testid`, or nth-of-type). `DomControlEngine` uses those
selectors for click/fill/select/check and for reading back element values, so
execution is independent of pixel coordinates.

- Supported element types map to DOM control types (textbox, checkbox, radio,
  combobox/listbox via `select_option`, date pickers via `fill`, buttons).
- `read_field_value` returns the DOM `value` (checkbox/radio return
  `checked`/`unchecked`).
- Checkbox/radio verification collapses boolean synonyms
  (Yes/on/checked -> "1").

Attach:

```powershell
python main.py run --web --url http://localhost:5173 --browser chromium
python main.py run --web --url http://localhost:5173 --headless
```

Requires `python -m playwright install <browser>` first.

## Adding a new target

Implement `TargetAdapter`, provide a `controls: ControlInterface` and a read
function for verification, and hand them to the executor via the assistant's
`_build_executor` (`atlas/assistant/assistant.py`). The loop and mapping layers
need no changes.
