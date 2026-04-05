"""AlphaSeeker CLI — Textual application entry point."""

from __future__ import annotations

from textual.app import App
from textual.theme import Theme

from src.cli.screens.model_select import ModelSelectScreen
from src.cli.screens.effort import EffortScreen
from src.cli.screens.prompt import PromptScreen
from src.cli.screens.dashboard import DashboardScreen
from src.cli.screens.done import DoneScreen
from src.cli.theme import GLOBAL_CSS, THEME_KWARGS


class AlphaSeekerApp(App):
    """Main Textual application for AlphaSeeker."""

    CSS = GLOBAL_CSS

    SCREENS = {
        "model_select": ModelSelectScreen,
        "effort": EffortScreen,
        "prompt": PromptScreen,
        "dashboard": DashboardScreen,
        "done": DoneScreen,
    }

    def __init__(self, prefill_query: str | None = None):
        super().__init__()
        self._prefill_query = prefill_query

        # App-level state shared across screens
        self._selected_model: dict | None = None
        self._reasoning_effort: str | None = None
        self._wall_clock_budget: int | None = None
        self._per_agent_budget: int | None = None
        self._skill_packs: list[str] | None = None
        self._query: str | None = None
        self._run_root: str | None = None
        self._final_report: str | None = None
        self._dashboard_snapshot: object | None = None
        self._backend_stop_requested: bool = False
        self._backend: object | None = None

        # Register custom theme
        self.register_theme(Theme(**THEME_KWARGS))
        self.theme = "alphaseeker"

    def on_mount(self) -> None:
        """Called when the app is first mounted."""
        if self._prefill_query:
            # Skip to prompt screen with pre-filled query
            self.push_screen(PromptScreen(prefill_query=self._prefill_query))
        else:
            self.push_screen("model_select")


def run_tui(prefill_query: str | None = None) -> None:
    """Entry point for the AlphaSeeker Textual TUI."""
    app = AlphaSeekerApp(prefill_query=prefill_query)
    app.run()
