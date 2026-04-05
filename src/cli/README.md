# src/cli — AlphaSeeker TUI

Textual-based terminal UI. Entry point is `run_tui()` in `app.py`, called from `main.py`.

## File map

```
src/cli/
├── app.py                        # AlphaSeekerApp — registers screens, holds cross-screen state
├── theme.py                      # CSS constants, ASCII banners, STATUS_COLORS dict
├── llm_hook.py                   # LLMObserver class + install/uninstall helpers (unused by harness)
├── backends/
│   └── harness_backend.py        # HarnessBackend — runs _supervise_async in a daemon thread;
│                                 #   exposes run_root property and poll() for the dashboard
└── screens/
    ├── model_select.py           # Screen 1: provider + model selection
    ├── effort.py                 # Screen 2: reasoning effort (Quick / Medium / Deep)
    ├── prompt.py                 # Screen 3: research question input + optional skill-pack toggle
    ├── dashboard.py              # Screen 4: live run dashboard (agent table, logs, results)
    ├── done.py                   # Screen 5: final report display
    └── config.py                 # Legacy combined config screen — NOT registered in app.py
```

## Screen flow

```
model_select  →  effort  →  prompt  →  dashboard  →  done
     ↑                                                  |
     └──────────────── n (new research) ───────────────┘
```

`AlphaSeekerApp.SCREENS` maps string keys to screen classes. Screens are pushed/popped with
`self.app.push_screen("key")` / `self.app.pop_screen()`. Esc goes back one screen.

## Cross-screen state (stored on AlphaSeekerApp)

| Attribute | Set by | Read by |
|---|---|---|
| `_selected_model` | `model_select` | — |
| `_reasoning_effort` | `effort` | — |
| `_wall_clock_budget` | `effort` | `prompt`, `dashboard` |
| `_per_agent_budget` | `effort` | `prompt` |
| `_skill_packs` | `prompt` | — |
| `_query` | `prompt` | — |
| `_backend` | `prompt` | `dashboard` |
| `_run_root` | `dashboard._on_complete` | `done` |
| `_final_report` | `dashboard._on_complete` | `done` |
| `_dashboard_snapshot` | `dashboard._on_complete` | `done` |

## Interaction model (keyboard-only)

All selection screens use a two-tap Enter pattern:
- **First Enter** on an item → selects it (filled `●` icon, others show `○`)
- **Enter again on the same item** → confirms and advances to the next screen
- **Enter on a different item** → selects that item instead (deselects the previous)
- **Esc** → go back one screen

No `Button` widgets appear on any screen. All actions are keyboard bindings shown in the `Footer`.

### model_select

Two `OptionList` widgets stacked vertically: providers (top) and models (bottom).
- Navigate providers with ↑↓, Tab to switch focus to model list.
- First Enter on provider → selects it (green `●`), populates model list, focus stays on providers.
- Second Enter on same provider → focus moves to model list.
- First Enter on model → selects it. Second Enter on same model → confirms and advances.
- Providers without an API key env var are shown as `○ (no key)` and are `disabled`.

### effort

Single `OptionList` with three items: Quick / Medium / Deep.
- ↑↓ navigate, Enter two-tap to confirm.
- Description line above the list updates as cursor moves.

### prompt

`TextArea` for the research question. Press **ctrl+j** to submit.
- Press **a** to toggle the advanced skill-pack section (Switch widgets for equity/macro/commodity).
- core pack is always enabled and non-togglable.

### dashboard

Split layout: left pane (agent status + partial results), right pane (tabbed logs).

**Left pane**
- `DataTable` (`#agent-table`): one row per agent. Columns: Agent, Preset, Status, Elapsed.
  - Table is never `clear()`-ed — rows are added for new agents, existing cells updated in place.
    This keeps `cursor_coordinate` stable so `RowHighlighted` only fires from user ↑↓ navigation.
  - Navigating to a row sets `_selected_agent` and triggers a log rewrite filtered to that agent.
- `RichLog` (`#partial-results`): refreshed each poll from `progress.md` in the run directory.

**Right pane — tabbed**
- `tab-llm` (`#log-llm`): agent text blocks from transcripts. Press **l** to switch here.
- `tab-thinking` (`#tab-thinking`): thinking blocks. Press **t** to switch here.
- `tab-results` (`#tab-results`): agent `publish/summary.md` and `publish/final.md` as they appear.

**Key bindings**: a (toggle agent pane), l/t/r (switch tabs), v (cycle verbosity), q (stop).

**Transcript reading**: agents run as subprocesses; their LLM calls are written to
`<run_root>/agents/<id>/scratch/transcript.jsonl`. The dashboard reads these files directly.
Each `assistant_response` entry has `message.content[]` blocks with `type: "text"` or
`type: "thinking"`. All entries are buffered in `_llm_entries` / `_thinking_entries` (keyed by
agent_id) so filter rewrites don't require re-reading files.

**Agent filter**: `_selected_agent` (set by `DataTable.RowHighlighted`). When it changes,
`_filter_dirty = True` → next poll calls `_rewrite_llm_log()` / `_rewrite_thinking_log()` which
clears and replays only the selected agent's buffered entries.

### done

Markdown report viewer. Shows the best available content via `_best_available_report()`:
1. `agent_root/publish/final.md` (complete report)
2. `agent_root/publish/summary.md` (partial root summary, with status notice)
3. Concatenated child-agent `summary.md` files (fallback when root timed out before synthesis)

**Key bindings**: c (copy), s (save to `reports/`), n (new research), q (quit).

## HarnessBackend

`backends/harness_backend.py` wraps `_supervise_async` in a `threading.Thread`.

- `__init__`: pre-computes `run_root = str(build_run_root(request))` so the dashboard can start
  polling before the background thread initialises the run directory.
- `start()`: launches the daemon thread.
- `poll(run_root)`: reads the filesystem to return a `DashboardSnapshot`. Safe to call from the
  TUI thread. Returns status, agent records, evidence count, and best-available report content.
- `run_root` property: exposes the pre-computed path.

## theme.py

- `get_banner(width)`: returns wide ASCII art (≥60 cols) or narrow box art.
- `THEME_KWARGS`: passed to `textual.theme.Theme` to register the `"alphaseeker"` theme.
- `STATUS_COLORS`: maps harness agent status strings to Rich colour names for the DataTable.
- `GLOBAL_CSS`: app-level CSS applied to all screens.

## llm_hook.py

`LLMObserver` + `install_llm_observer()` — intended for in-process LLM call capture. Not used by
the harness dashboard because agents run as subprocesses. Kept for potential non-subprocess usage.
