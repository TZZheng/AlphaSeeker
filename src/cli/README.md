# `src/cli`

`src/cli` is AlphaSeeker's Textual terminal UI.

In frontend terms:
- a "screen" is one full terminal page, such as model selection or the live dashboard
- a "widget" is one UI component on a screen, such as `OptionList`, `TextArea`, or `DataTable`
- the local "backend" is the small adapter that runs the harness in a background thread and gives the UI read-only snapshots

## Current File Map

The CLI entry point starts in `main.py`, then hands off to `src/cli`.

```text
main.py
└── typer main()                 # Loads .env and starts run_tui()

src/cli/
├── __init__.py
├── app.py                       # AlphaSeekerApp: screen registry and shared app state
├── theme.py                     # Theme colors, global CSS, ASCII banners, status colors
├── llm_hook.py                  # Optional in-process LLM observer; not used by harness subprocesses
├── backends/
│   ├── __init__.py
│   └── harness_backend.py       # Background-thread wrapper around the harness runtime
└── screens/
    ├── __init__.py
    ├── model_select.py          # Provider/model picker
    ├── effort.py                # Quick / Medium / Deep budget selector
    ├── prompt.py                # Research question input and skill-pack toggles
    ├── dashboard.py             # Live run dashboard
    ├── done.py                  # Final report viewer
    └── config.py                # Legacy screen; not registered in AlphaSeekerApp
```

## App Flow

1. `main.py` loads `.env` and calls `run_tui()`.
2. `run_tui()` creates `AlphaSeekerApp`.
3. `AlphaSeekerApp` registers five live screens:
   `model_select -> effort -> prompt -> dashboard -> done`
4. `PromptScreen` builds a `HarnessRequest`, creates `HarnessBackend`, and pushes a fresh `DashboardScreen()` instance.
5. `DashboardScreen` polls the harness run directory once per second until the root agent finishes.
6. `DoneScreen` renders the best available report and handles copy/save/reset actions.

If `run_tui(prefill_query=...)` is used, the app skips `model_select` and opens the prompt screen directly with prefilled text.

## Shared App State

`AlphaSeekerApp` keeps a small amount of cross-screen state:

- `_selected_model`: provider/model chosen in `model_select`
- `_reasoning_effort`: `Quick`, `Medium`, or `Deep`
- `_wall_clock_budget`: run-wide budget selected in `effort`
- `_per_agent_budget`: per-agent budget selected in `effort`
- `_skill_packs`: enabled packs chosen in `prompt`
- `_query`: submitted research question
- `_backend`: live `HarnessBackend` instance
- `_run_root`: completed run directory path
- `_final_report`: final markdown shown on `done`
- `_dashboard_snapshot`: final dashboard snapshot used for summary stats
- `_backend_stop_requested`: currently present but unused

## Screen Details

### `model_select.py`

- Provider availability is driven by `_PROVIDER_CONFIG` and the matching API-key environment variable.
- Providers without a key are disabled and rendered with `(no key)`.
- Interaction is keyboard-only and uses two-tap Enter:
  first Enter selects, second Enter confirms.
- `Tab` switches between the provider list and model list.
- On confirm, the chosen model is written into `ALPHASEEKER_MODEL_*_*` environment variables for all current agent/role combinations, then the app advances to `effort`.

### `effort.py`

- Shows three presets:
  `Quick`, `Medium`, `Deep`
- These map to:
  `Quick -> 120s run / 60s per agent`
  `Medium -> 600s run / 300s per agent`
  `Deep -> 1800s run / 900s per agent`
- Uses the same two-tap Enter pattern as `model_select`.

### `prompt.py`

- Main input is a `TextArea`.
- `ctrl+j` starts research.
- `a` toggles the advanced skill-pack section.
- `core` is always enabled; `equity`, `macro`, and `commodity` are optional switches.
- On submit, the screen builds `HarnessRequest(user_prompt=..., wall_clock_budget_seconds=..., per_agent_wall_clock_seconds=..., available_skill_packs=...)`.
- The selected model is not written into `HarnessRequest`; model choice is currently passed indirectly through environment variables set in `model_select`.
- The screen pushes `DashboardScreen()` directly instead of using the registered string key so Textual does not reuse a stale dashboard instance.

### `dashboard.py`

Layout:

- left pane: agent table plus partial results
- right pane: tabbed logs (`LLM Logs`, `Thinking`, `Results`)

Data sources:

- agent status comes from `HarnessBackend.poll()`
- partial results come from `<run_root>/progress.md`
- LLM and thinking logs come from each agent's `scratch/transcript.jsonl`
- commenter notes are merged into the LLM log stream from `scratch/commenter/comments.jsonl`
- results are pulled from each agent's `publish/summary.md` and `publish/final.md`

Interaction:

- arrow keys move through the agent table
- highlighting a row sets `_selected_agent` and filters the LLM/thinking panes to that agent
- `l`, `t`, `r` switch the right-side tab
- `v` cycles layout mode: `split -> focus -> status -> split`
- `q` requests a graceful stop by touching `<run_root>/stop_requested`

Implementation details that matter:

- the table is updated in place and never cleared, so row selection stays stable while polling
- transcript entries are buffered in memory by agent, so filter changes rewrite the visible log instead of rereading all files every time
- the results tab appends updates when file modification time changes

### `done.py`

- Renders markdown from `self.app._final_report`
- summary strip shows final status, elapsed time, evidence count, and agent count
- `c` copies report text to the clipboard
- `s` saves the report into `reports/`
- `n` clears app state and restarts at `model_select`
- `q` exits the app

## `HarnessBackend`

`backends/harness_backend.py` is the CLI-to-runtime adapter.

- `start()` launches the harness supervisor in a daemon thread
- `run_root` is precomputed from `build_run_root(request)` so the dashboard can start polling immediately
- `poll(run_root)` reads the filesystem and returns `DashboardSnapshot`
- `request_soft_stop()` creates `<run_root>/stop_requested`
- `wait()` blocks for the final `HarnessResponse`

`DashboardSnapshot` currently contains:

- `agents`
- `elapsed_seconds`
- `evidence_count`
- `status`
- `final_report`
- `run_root`

`_best_available_report()` uses this fallback order:

1. `agent_root/publish/final.md`
2. `agent_root/publish/summary.md`
3. concatenated child-agent `final.md` or `summary.md`

`evidence_count` is read from `<run_root>/evidence_ledger.jsonl` if that file exists.

## Theme And Optional Hook

- `theme.py` registers the `alphaseeker` Textual theme, ASCII banners, global CSS, and the agent-status color map.
- Some CSS rules still style widgets such as `Button` and `Select`, even though the main flow is currently keyboard-first and centered on `OptionList`, `TextArea`, `DataTable`, `RichLog`, and `Markdown`.
- `llm_hook.py` is only useful for in-process LLM observation. The live harness dashboard does not use it because harness agents run as subprocesses and write transcripts to disk instead.
