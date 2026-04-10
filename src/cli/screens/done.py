"""Done screen — final report display with Markdown rendering."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Markdown, Static


def _copyable_report_text(report: str, selected_text: str | None) -> str:
    return selected_text or report


class DoneScreen(Screen):
    """Screen shown when research completes, displaying the final report."""

    CSS = """
    DoneScreen {
        align: center middle;
    }

    #summary-stats {
        width: 100%;
        height: auto;
        padding: 0 2;
        background: $success 10%;
        color: $text;
        margin-bottom: 1;
        dock: top;
    }

    #report-panel {
        width: 90%;
        height: 1fr;
        border: round $success 30%;
        padding: 1 2;
    }

    #hint {
        width: 90%;
        height: 1;
        margin-top: 1;
        color: $text-muted;
        content-align: center middle;
    }
    """

    BINDINGS = [
        Binding("c", "copy_report", "Copy", show=True),
        Binding("s", "save_report", "Save", show=True),
        Binding("n", "new_research", "New", show=True),
        Binding("q", "quit_app", "Quit", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("", id="summary-stats")
        with VerticalScroll(id="report-panel"):
            yield Markdown("", id="report-md")
        yield Static("[dim]c copy  ·  s save  ·  n new research  ·  q quit[/dim]", id="hint")
        yield Footer()

    def on_mount(self) -> None:
        report = getattr(self.app, "_final_report", None) or ""
        self._report = report
        self._run_root = getattr(self.app, "_run_root", None)
        snapshot = getattr(self.app, "_dashboard_snapshot", None)

        # Summary stats line
        parts: list[str] = []
        if snapshot:
            status = getattr(snapshot, "status", "unknown")
            elapsed = getattr(snapshot, "elapsed_seconds", 0)
            evidence = getattr(snapshot, "evidence_count", 0)
            agents = getattr(snapshot, "agents", {})
            color = "green" if status == "done" else "yellow"
            parts.append(f"[{color}]● {status.upper()}[/{color}]")
            parts.append(f"{elapsed:.0f}s elapsed")
            parts.append(f"{evidence} evidence items")
            parts.append(f"{len(agents)} agents")
        self.query_one("#summary-stats", Static).update("  ·  ".join(parts))

        # Report content
        md = self.query_one("#report-md", Markdown)
        if report:
            md.update(report)
        else:
            md.update(
                "*No report content available.*\n\n"
                "The run may have ended before any output was produced. "
                "Check the run directory under `data/harness_runs/` for partial artifacts."
            )

    def action_copy_report(self) -> None:
        report_text = _copyable_report_text(self._report, self.get_selected_text())
        if not report_text:
            self.app.notify("Nothing to copy.", severity="warning")
            return
        try:
            self.app.copy_to_clipboard(report_text)
            self.app.notify("Copied report text.", severity="information")
        except Exception:
            self.app.notify("Failed to copy to clipboard.", severity="error")

    def action_save_report(self) -> None:
        if not self._report:
            self.app.notify("Nothing to save.", severity="warning")
            return
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        from src.shared.report_filename import build_prompt_report_filename
        filename = build_prompt_report_filename(
            prompt_text=self._report[:50],
            fallback_stem="alphaseeker_report",
        )
        path = reports_dir / filename
        try:
            path.write_text(self._report, encoding="utf-8")
            self.app.notify(f"Saved to {path}", severity="information")
        except Exception as exc:
            self.app.notify(f"Failed to save: {exc}", severity="error")

    def action_new_research(self) -> None:
        self.app._selected_model = None
        self.app._reasoning_effort = None
        self.app._skill_packs = None
        self.app._query = None
        self.app._run_root = None
        self.app._final_report = None
        self.app._dashboard_snapshot = None
        self.app.pop_screen()
        self.app.push_screen("model_select")

    def action_quit_app(self) -> None:
        self.app.exit()
