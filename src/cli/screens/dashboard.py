"""Dashboard screen — live agent status, tabbed output, and keystroke controls."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Footer,
    RichLog,
    Rule,
    Static,
    TabbedContent,
    TabPane,
)

from src.cli.backends.harness_backend import DashboardSnapshot, HarnessBackend
from src.harness.artifacts import read_jsonl
from src.cli.theme import STATUS_COLORS


TERMINAL_STATUSES = {"done", "failed", "blocked", "stale", "cancelled"}
_LAYOUT_MODES = ["split", "focus", "status"]
# split: both panes visible (default)
# focus: right pane full-width (left pane hidden)
# status: left pane full-width (right pane hidden)

# String column keys used with update_cell
_COL_STATUS = "status"
_COL_ELAPSED = "elapsed"


class DashboardScreen(Screen):
    """Live research dashboard with agent status, tabbed output, and key controls."""

    CSS = """
    DashboardScreen {
        layout: vertical;
    }

    #header-bar {
        height: 1;
        background: $primary 20%;
        color: $primary;
        padding: 0 2;
        text-style: bold;
        dock: top;
    }

    #main-area {
        layout: horizontal;
        height: 1fr;
    }

    #left-pane {
        width: 40%;
        min-width: 35;
        border-right: solid $primary 30%;
        padding: 0 1;
    }

    #right-pane {
        width: 60%;
        padding: 0 1;
    }

    .pane-title {
        color: $primary;
        text-style: bold;
        margin-bottom: 1;
        height: 1;
    }

    #agent-table {
        height: auto;
        max-height: 50%;
    }

    #partial-results {
        height: 1fr;
        min-height: 4;
    }

    #output-tabs {
        height: 1fr;
    }

    TabbedContent ContentSwitcher {
        background: $surface;
    }

    TabPane {
        padding: 0;
    }

    TabPane RichLog {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("l", "show_llm", "LLM", show=True),
        Binding("t", "show_thinking", "Think", show=True),
        Binding("r", "show_results", "Results", show=True),
        Binding("v", "cycle_layout", "Layout", show=True),
        Binding("q", "stop_research", "Stop", show=True),
    ]

    def __init__(self):
        super().__init__()
        self._layout_index = 0  # index into _LAYOUT_MODES
        self._poll_lock = threading.Lock()
        self._last_snapshot: DashboardSnapshot | None = None
        self._started = False

        # Agents already added to the DataTable (no clear ever)
        self._agents_in_table: set[str] = set()

        # Transcript state: per-agent entry buffers
        self._transcript_cursors: dict[str, int] = {}
        self._llm_entries: dict[str, list[str]] = {}
        self._thinking_entries: dict[str, list[str]] = {}

        # How many lines per agent have been written to the log widgets
        self._llm_written: dict[str, int] = {}
        self._thinking_written: dict[str, int] = {}

        # Agent filter — set by user row-navigation
        self._selected_agent: str | None = None
        # Tracks what filter is currently rendered in the log widgets.
        # Compared against _selected_agent each poll to decide rewrite vs append.
        self._llm_display_filter: str | None = None
        self._thinking_display_filter: str | None = None

        # Results tab: last-seen mtime per published file
        self._results_shown: dict[str, float] = {}

    def compose(self) -> ComposeResult:
        yield Static("AlphaSeeker ● Starting…", id="header-bar")
        with Horizontal(id="main-area"):
            with Vertical(id="left-pane"):
                yield Static(
                    "[b]AGENT STATUS[/b]  [dim](↑↓ filter logs)[/dim]",
                    classes="pane-title",
                )
                yield DataTable(id="agent-table", zebra_stripes=True, cursor_type="row")
                yield Rule()
                yield Static("[b]PARTIAL RESULTS[/b]", classes="pane-title", id="results-title")
                yield RichLog(id="partial-results", max_lines=300, markup=True, wrap=True)
            with Vertical(id="right-pane"):
                with TabbedContent(id="output-tabs"):
                    with TabPane("LLM Logs", id="tab-llm"):
                        yield RichLog(id="log-llm", max_lines=2000, markup=True, wrap=True)
                    with TabPane("Thinking", id="tab-thinking"):
                        yield RichLog(id="log-thinking", max_lines=2000, markup=True, wrap=True)
                    with TabPane("Results", id="tab-results"):
                        yield RichLog(id="log-results", max_lines=1000, markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        if self._started:
            return
        self._started = True
        backend = getattr(self.app, "_backend", None)
        if backend is None:
            return
        self._backend: HarnessBackend = backend
        self._run_root: str = self._backend.run_root

        table = self.query_one("#agent-table", DataTable)
        # add_columns(*labels) does NOT support (label, key) tuples — use add_column
        # so that update_cell(row_key, col_key) works with string keys.
        table.add_column("Agent", key="agent")
        table.add_column("Preset", key="preset")
        table.add_column("Status", key=_COL_STATUS)
        table.add_column("Elapsed", key=_COL_ELAPSED)
        table.focus()
        self._poll_timer = self.set_interval(1.0, self._poll)
        self._backend.start()

    # ── Agent row selection (user navigation only) ────────────────────────────

    @on(DataTable.RowHighlighted, "#agent-table")
    def _on_agent_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        agent_id = str(event.row_key.value) if event.row_key else None
        self._selected_agent = agent_id

    # ── Poll loop ─────────────────────────────────────────────────────────────

    async def _poll(self) -> None:
        if not hasattr(self, "_backend"):
            return
        snapshot = self._backend.poll(self._run_root)
        with self._poll_lock:
            self._last_snapshot = snapshot

        # -- Update agent table (add new rows, update existing cells) ----------
        # NEVER call table.clear() — that resets cursor_coordinate which fires
        # RowHighlighted asynchronously, clobbering the user's filter selection.
        table = self.query_one("#agent-table", DataTable)
        active_count = 0
        for agent_id, record in snapshot.agents.items():
            color = STATUS_COLORS.get(record.status, "white")
            status_text = f"[{color}]{record.status}[/{color}]"
            elapsed_text = f"{snapshot.elapsed_seconds:.0f}s"
            if record.status not in TERMINAL_STATUSES:
                active_count += 1

            if agent_id not in self._agents_in_table:
                table.add_row(
                    agent_id[:20],
                    record.preset,
                    status_text,
                    elapsed_text,
                    key=agent_id,
                )
                self._agents_in_table.add(agent_id)
            else:
                table.update_cell(agent_id, _COL_STATUS, status_text, update_width=False)
                table.update_cell(agent_id, _COL_ELAPSED, elapsed_text, update_width=False)

        # -- Header bar --------------------------------------------------------
        layout = _LAYOUT_MODES[self._layout_index]
        filter_hint = f"  ●  Filter: {self._selected_agent}" if self._selected_agent else ""
        self.query_one("#header-bar", Static).update(
            f"AlphaSeeker ● [green]{snapshot.status}[/green]  ●  "
            f"Elapsed: {snapshot.elapsed_seconds:.0f}s  ●  "
            f"Evidence: {snapshot.evidence_count}  ●  "
            f"Agents: {active_count} active  ●  "
            f"Layout: {layout}  [dim](v cycle · l/t/r tabs)[/dim]"
            f"{filter_hint}"
        )

        # -- Partial results ---------------------------------------------------
        self._update_partial_results(snapshot)

        # -- Scan transcripts → memory buffers (always) -----------------------
        self._scan_transcripts()

        # -- Apply filter if selection changed; otherwise append new entries ----
        # Compare what is currently displayed (_*_display_filter) against
        # _selected_agent.  Rewrite clears and resets the display filter tracker.
        if self._selected_agent != self._llm_display_filter:
            self._rewrite_llm_log()
        else:
            self._append_new_llm_entries()
        if self._selected_agent != self._thinking_display_filter:
            self._rewrite_thinking_log()
        else:
            self._append_new_thinking_entries()

        # -- Results tab -------------------------------------------------------
        self._update_results_tab()

        # -- Completion --------------------------------------------------------
        if snapshot.status in TERMINAL_STATUSES:
            self._on_complete(snapshot)

    # ── Transcript scanning ───────────────────────────────────────────────────

    def _scan_transcripts(self) -> None:
        """Read new transcript entries into per-agent memory buffers."""
        if not self._run_root:
            return
        agents_dir = Path(self._run_root) / "agents"
        if not agents_dir.exists():
            return
        for agent_dir in sorted(agents_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            agent_id = agent_dir.name
            transcript_path = agent_dir / "scratch" / "transcript.jsonl"
            if not transcript_path.exists():
                continue
            try:
                all_entries = read_jsonl(transcript_path)
            except Exception:
                continue
            cursor = self._transcript_cursors.get(agent_id, 0)
            new_entries = all_entries[cursor:]
            if not new_entries:
                continue
            self._transcript_cursors[agent_id] = len(all_entries)
            llm_buf = self._llm_entries.setdefault(agent_id, [])
            thinking_buf = self._thinking_entries.setdefault(agent_id, [])
            for entry in new_entries:
                if entry.get("kind") != "assistant_response":
                    continue
                turn_idx = entry.get("turn_index", "?")
                content = entry.get("message", {}).get("content") or []
                # OpenAI transport stores content as a plain string; treat it as a text block.
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}] if content.strip() else []
                added_text = False
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "text":
                        text = block.get("text", "")
                        if text and text.strip():
                            llm_buf.append(
                                f"[dim cyan]{agent_id}[/dim cyan] "
                                f"[dim]turn {turn_idx}[/dim]\n"
                                f"  {_summarize(text, 1000)}"
                            )
                            added_text = True
                    elif btype == "thinking":
                        thinking = block.get("thinking", "")
                        if thinking and thinking.strip():
                            thinking_buf.append(
                                f"[dim magenta]{agent_id}[/dim magenta] "
                                f"[dim]turn {turn_idx}[/dim]\n"
                                f"  {_summarize(thinking, 1000)}"
                            )
                # OpenAI models return content=None on tool-call turns; show tool names instead.
                if not added_text:
                    tool_calls = entry.get("decision", {}).get("tool_calls", [])
                    if tool_calls:
                        calls_str = ", ".join(tc.get("name", "?") for tc in tool_calls)
                        llm_buf.append(
                            f"[dim cyan]{agent_id}[/dim cyan] "
                            f"[dim]turn {turn_idx}[/dim]\n"
                            f"  [dim]→ {calls_str}[/dim]"
                        )

    def _agents_to_display(self, entries_map: dict[str, list[str]]) -> list[str]:
        """Return agent IDs to show given current selection.

        No selection → all agents.
        Agent selected but no entries yet → empty list (log stays blank until
        entries arrive; next append poll will add them as they come in).
        Agent selected with entries → only that agent.
        """
        if not self._selected_agent:
            return sorted(entries_map.keys())
        if self._selected_agent in entries_map:
            return [self._selected_agent]
        return []  # selected agent exists but has no transcript entries yet

    # ── LLM log ───────────────────────────────────────────────────────────────

    def _rewrite_llm_log(self) -> None:
        log = self.query_one("#log-llm", RichLog)
        log.clear()
        self._llm_written = {}
        self._llm_display_filter = self._selected_agent
        for agent_id in self._agents_to_display(self._llm_entries):
            for line in self._llm_entries[agent_id]:
                log.write(line)
            self._llm_written[agent_id] = len(self._llm_entries[agent_id])

    def _append_new_llm_entries(self) -> None:
        log = self.query_one("#log-llm", RichLog)
        for agent_id in self._agents_to_display(self._llm_entries):
            written = self._llm_written.get(agent_id, 0)
            new_lines = self._llm_entries[agent_id][written:]
            for line in new_lines:
                log.write(line)
            if new_lines:
                self._llm_written[agent_id] = len(self._llm_entries[agent_id])

    # ── Thinking log ──────────────────────────────────────────────────────────

    def _rewrite_thinking_log(self) -> None:
        log = self.query_one("#log-thinking", RichLog)
        log.clear()
        self._thinking_written = {}
        self._thinking_display_filter = self._selected_agent
        for agent_id in self._agents_to_display(self._thinking_entries):
            for line in self._thinking_entries[agent_id]:
                log.write(line)
            self._thinking_written[agent_id] = len(self._thinking_entries[agent_id])

    def _append_new_thinking_entries(self) -> None:
        log = self.query_one("#log-thinking", RichLog)
        for agent_id in self._agents_to_display(self._thinking_entries):
            written = self._thinking_written.get(agent_id, 0)
            new_lines = self._thinking_entries[agent_id][written:]
            for line in new_lines:
                log.write(line)
            if new_lines:
                self._thinking_written[agent_id] = len(self._thinking_entries[agent_id])

    # ── Partial results ───────────────────────────────────────────────────────

    def _update_partial_results(self, snapshot: DashboardSnapshot) -> None:
        log = self.query_one("#partial-results", RichLog)
        log.clear()
        progress_path = Path(self._run_root) / "progress.md"
        if progress_path.exists():
            try:
                content = progress_path.read_text(encoding="utf-8").strip()
                if content:
                    log.write(content)
                    return
            except OSError:
                pass
        log.write(
            f"[dim]Evidence: {snapshot.evidence_count}  |  Status: {snapshot.status}[/dim]"
        )

    # ── Results tab ───────────────────────────────────────────────────────────

    def _update_results_tab(self) -> None:
        if not self._run_root:
            return
        agents_dir = Path(self._run_root) / "agents"
        if not agents_dir.exists():
            return
        log = self.query_one("#log-results", RichLog)
        for agent_dir in sorted(agents_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            agent_id = agent_dir.name
            for fname in ("summary.md", "final.md"):
                pub_path = agent_dir / "publish" / fname
                if not pub_path.exists():
                    continue
                try:
                    mtime = pub_path.stat().st_mtime
                except OSError:
                    continue
                key = f"{agent_id}/{fname}"
                if self._results_shown.get(key) == mtime:
                    continue
                self._results_shown[key] = mtime
                try:
                    content = pub_path.read_text(encoding="utf-8").strip()
                except OSError:
                    continue
                if content:
                    log.write(
                        f"\n[bold cyan]{agent_id} / {fname}[/bold cyan]\n"
                        + content[:1000]
                        + ("…" if len(content) > 1000 else "")
                        + "\n"
                    )

    # ── Completion ────────────────────────────────────────────────────────────

    def _on_complete(self, snapshot: DashboardSnapshot) -> None:
        if hasattr(self, "_poll_timer"):
            self._poll_timer.stop()
        self._scan_transcripts()
        self._append_new_llm_entries()
        self._append_new_thinking_entries()
        self._update_results_tab()
        self.app._final_report = snapshot.final_report
        self.app._run_root = snapshot.run_root
        self.app._dashboard_snapshot = snapshot
        self.app.pop_screen()
        self.app.push_screen("done")

    # ── Keybinding actions ────────────────────────────────────────────────────

    def _set_layout(self, layout: str) -> None:
        """Apply a named layout mode by showing/hiding panes."""
        left = self.query_one("#left-pane", Vertical)
        right = self.query_one("#right-pane", Vertical)
        if layout == "split":
            left.display = True
            right.display = True
        elif layout == "focus":
            left.display = False
            right.display = True
        elif layout == "status":
            left.display = True
            right.display = False
        self._layout_index = _LAYOUT_MODES.index(layout)

    def action_show_llm(self) -> None:
        """Switch to LLM tab; ensure right pane is visible."""
        if not self.query_one("#right-pane", Vertical).display:
            self._set_layout("split")
        self.query_one("#output-tabs", TabbedContent).active = "tab-llm"

    def action_show_thinking(self) -> None:
        """Switch to Thinking tab; ensure right pane is visible."""
        if not self.query_one("#right-pane", Vertical).display:
            self._set_layout("split")
        self.query_one("#output-tabs", TabbedContent).active = "tab-thinking"

    def action_show_results(self) -> None:
        """Switch to Results tab; ensure right pane is visible."""
        if not self.query_one("#right-pane", Vertical).display:
            self._set_layout("split")
        self.query_one("#output-tabs", TabbedContent).active = "tab-results"

    def action_cycle_layout(self) -> None:
        """Cycle: split → focus → status → split."""
        self._layout_index = (self._layout_index + 1) % len(_LAYOUT_MODES)
        layout = _LAYOUT_MODES[self._layout_index]
        self._set_layout(layout)
        self.app.notify(f"Layout: {layout}", timeout=1)

    def action_stop_research(self) -> None:
        if hasattr(self, "_backend"):
            self._backend.request_soft_stop()
            self.app.notify(
                "Stop requested — giving agents 60s to finalize…", severity="warning"
            )
        else:
            self.app.notify("Stopping research…", severity="warning")
        # Do NOT stop the poll timer — _on_complete must fire after graceful shutdown


def _summarize(value: Any, max_len: int = 200) -> str:
    try:
        if isinstance(value, (list, tuple)):
            parts = []
            for item in value:
                s = str(item)
                if len(s) > 80:
                    s = s[:80] + "…"
                parts.append(s)
            s = " | ".join(parts)
        else:
            s = str(value)
        return s[:max_len] + ("…" if len(s) > max_len else "")
    except Exception:
        return "<unrenderable>"
