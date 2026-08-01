# Architecture

ATLAS AI is a pipelined agent. The `Assistant` facade wires components once
per target; the `AgentLoop` drives the pipeline until records run out.

## The loop

```
observe -> understand -> reason -> plan -> execute -> verify
    ^________________________________________|
```

1. **Observe** - `atlas/observe/` captures the attached window/screen and the
   `SceneAnalyzer` (`atlas/vision/scene.py`) turns it into a
   `SceneDescription`. Cached by content hash; the change watcher
   (`watcher.py`) fires only on real layout changes.
2. **Understand** - `SourceReader` (`atlas/understanding/source.py`) extracts a
   `SourceRecord` from the scene's label/value pairs; `discover_fields`
   (`atlas/understanding/fields.py`) finds the target's editable fields.
3. **Reason** - the planner (`atlas/reason/planner.py`) builds a `FillPlan` for
   the record; the recovery planner (`recovery.py`) repairs failures.
   Optionally an LLM advisor reviews the plan.
4. **Execute** - `ActionExecutor` (`atlas/act/executor.py`) runs actions
   through a `ControlInterface`. Desktop uses `ControlEngine`
   (mouse/keyboard/clipboard); web uses `DomControlEngine` (Playwright).
5. **Verify** - every value action is read back by a `FieldVerifier`
   (`atlas/act/verify.py`): target/DOM value, clipboard, or OCR region.
   Failures retry (`WORKFLOW_MAX_RETRIES_PER_ACTION`) then go to recovery.

## Key interfaces

Loose coupling is what lets desktop and web share one loop:

- `TargetAdapter` (`atlas/target/base.py`) - attach/detach/observe/control.
- `ControlInterface` (`atlas/act/controls.py`) - `type_text`, `click_field`,
  `select_option`, `check_field`, `submit`, `press`, `scroll`.
- `FieldVerifier` (`atlas/act/verify.py`) - `verify(bbox, expected, field_id)`.
- `VisionProvider` (`atlas/vision/providers.py`) - `describe`/`read_text`;
  `OcrReader` (`atlas/vision/ocr.py`) is the explicit text fallback.

`AgentLoop` depends only on these interfaces, so a new target adapter (e.g. a
remote desktop client) slots in without touching the workflow.

## Mapping

`SemanticMapper` (`atlas/mapping/mapper.py`) runs two passes:

1. exact / alias matches claim target fields immediately (prevents two sources
   fighting over one field),
2. fuzzy matches (rapidfuzz token/ratio/containment) fill the rest.

A fuzzy match between two *distinct known concepts* is rejected, and verified
fuzzy mappings are persisted via `AliasStore` into `MemoryStore`
(`atlas/memory/store.py`, SQLite) so they seed future runs.

## State machine & events

`StateMachine` (`atlas/core/states.py`) models the lifecycle
(`idle -> waiting_attach -> attaching -> watching -> analyzing -> planning ->
acting -> verifying`, with pause/resume/stop/error). Every transition and
action publishes to the singleton event bus (`atlas/core/events.py`), which the
overlay and controller consume.

## Web target specifics

`WebTarget` (`atlas/target/web.py`) screenshots the page through the same VLM
analyser (perception stays vision-first), but execution and verification use a
DOM index built by a page-side script (`_INSPECT_JS`) that maps labels to CSS
selectors. `observe()` keeps scene bboxes in sync with DOM `field_id`s so the
mapping and verification code paths are identical to desktop.

## Controller / remote control

`Controller` (`atlas/assistant/controller.py`) exposes commands as JSON over a
localhost HTTP server: attach/detach, run/stop/pause/resume, learn_alias,
config, close. `python main.py serve` starts it.

## Plugins

`PluginManager` (`atlas/plugins/manager.py`) loads plugins from a folder and
calls optional hooks (`on_register`, `on_event`, `on_record`, `close`).
