# Configuration

All settings load from environment variables, optionally from a `.env` file in
the working directory (see `.env.example`). Values are read when
`load_config()` is called, so overrides take effect at startup, not import
time. Paths for screenshots and logs are created eagerly.

## Vision (`VisionConfig`) - the primary perception channel

| Variable | Default | Meaning |
| --- | --- | --- |
| `VISION_PROVIDER` | `auto` | `auto`, `openai`, `gemini`, `local`, `mock` |
| `VISION_MODEL` | *(empty)* | model name (e.g. `gpt-4o-mini`) |
| `VISION_API_KEY` | *(empty)* | API key for the VLM endpoint |
| `VISION_API_BASE` | *(empty)* | OpenAI-compatible base URL |
| `VISION_TIMEOUT` | `60` | request timeout in seconds (1-600) |
| `VISION_CONFIDENCE_THRESHOLD` | `0.4` | minimum element confidence (0-1) |

`auto` uses a configured VLM endpoint if present, otherwise falls back to the
rule-based heuristic provider.

## Reasoning (`ReasoningConfig`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `REASONING_PROVIDER` | `auto` | `auto`, `openai`, `gemini` |
| `REASONING_MODEL` | *(empty)* | model name |
| `REASONING_API_KEY` | *(empty)* | API key |
| `REASONING_API_BASE` | *(empty)* | OpenAI-compatible base URL |
| `REASONING_TIMEOUT` | `60` | request timeout in seconds (1-600) |
| `REASONING_CONFIDENCE_THRESHOLD` | `0.5` | plan confidence floor (0-1) |

## OCR (`OcrConfig`) - explicit fallback only

| Variable | Default | Meaning |
| --- | --- | --- |
| `OCR_ENGINE` | `paddle` | `paddle` or `tesseract` |
| `OCR_LANG` | `en` | language |
| `OCR_CONFIDENCE_THRESHOLD` | `0.4` | minimum line confidence (0-1) |

## Mouse (`MouseConfig`) - human-like behaviour

| Variable | Default | Meaning |
| --- | --- | --- |
| `MOUSE_BEZIER_STEPS` | `35` | points in the bezier path |
| `MOUSE_SPEED` | `0.35` | movement speed factor |
| `MOUSE_MIN_DELAY` / `MOUSE_MAX_DELAY` | `0.05` / `0.25` | random move delay range (s) |
| `MOUSE_PAUSE_BEFORE_CLICK` / `MOUSE_PAUSE_AFTER_CLICK` | `0.08` / `0.10` | click pauses (s) |
| `MOUSE_JITTER_PX` | `3` | click position jitter (px) |
| `MOUSE_DOUBLE_CLICK_INTERVAL` | `0.30` | max gap for double-click (s) |

## Typing (`TypingConfig`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `TYPING_MIN_DELAY` / `TYPING_MAX_DELAY` | `0.05` / `0.25` | per-keystroke delay range (s) |
| `TYPING_PAUSE_AFTER` | `0.15` | pause after typing (s) |
| `TYPING_USE_CLIPBOARD_FOR_LONG` | `true` | paste long values instead of typing |
| `TYPING_CLIPBOARD_MIN_LENGTH` | `25` | minimum length to use clipboard |
| `TYPING_SIMULATE_TYPOS` | `true` | introduce and fix rare typos |
| `TYPING_TYPO_RATE` | `0.02` | typo probability (0-1) |

## Observe (`ObserveConfig`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `OBSERVE_POLL_INTERVAL` | `0.8` | capture interval in seconds (0.05-60) |
| `OBSERVE_CAPTURE_FORMAT` | `png` | screenshot format |
| `OBSERVE_SCREENSHOT_DIR` | `screenshots` | debug screenshot folder |

## Workflow (`WorkflowConfig`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `WORKFLOW_VERIFY_AFTER_ACTION` | `true` | verify every value action |
| `WORKFLOW_MAX_RETRIES_PER_ACTION` | `3` | retries before recovery |
| `WORKFLOW_RETRY_DELAY` | `0.8` | delay between retries (s) |
| `WORKFLOW_NEXT_RECORD_TIMEOUT` | `120` | wait for next record (s) |
| `WORKFLOW_NEXT_RECORD_POLL` | `1.5` | poll interval for next record (s) |
| `WORKFLOW_MAX_RECORDS` | `0` | stop after N records (0 = unlimited) |
| `WORKFLOW_LOG_SCREENSHOTS` | `true` | save annotated screenshots |

## Overlay (`OverlayConfig`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `OVERLAY_ENABLED` | `true` | show the floating status overlay |
| `OVERLAY_ANIMATION_FPS` | `30` | overlay refresh rate |
| `OVERLAY_COMMAND_PORT` | `19765` | overlay control port |

## Controller (`ControllerConfig`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONTROLLER_COMMAND_PORT` | `19768` | JSON command server port |

## Memory (`MemoryConfig`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `MEMORY_DB_PATH` | `memory.db` | SQLite database file |
| `MEMORY_ALIAS_LEARNING` | `true` | persist verified fuzzy mappings |

## Logging & debug (`LogConfig`, `AppConfig`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `LOG_LEVEL` | `DEBUG` | logging level |
| `LOG_FOLDER` | `logs` | log output folder |
| `DEBUG_MODE` | `false` | global debug flag |
