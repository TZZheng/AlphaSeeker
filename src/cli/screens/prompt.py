"""Research question screen — prompt input with optional advanced skill-pack toggle."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Footer,
    Header,
    Rule,
    Static,
    Switch,
    TextArea,
)

from src.cli.backends.harness_backend import HarnessBackend
from src.cli.screens.dashboard import DashboardScreen
from src.cli.screens.effort import REASONING_LEVELS
from src.harness.types import HarnessRequest


class PromptScreen(Screen):
    """Research question input with advanced skill-pack toggle (press [a])."""

    CSS = """
    PromptScreen {
        align: center middle;
    }

    #main-panel {
        width: 74;
        height: auto;
        max-height: 85%;
        border: round $primary 30%;
        padding: 1 2;
    }

    .section-heading {
        color: $primary;
        text-style: bold;
        margin-top: 1;
    }

    #query-input {
        height: 6;
        min-height: 4;
        max-height: 12;
        margin: 1 0;
    }

    #error-msg {
        color: $error;
        margin: 1 0;
    }

    /* Advanced skill-pack section — hidden by default */
    #advanced-section {
        height: 0;
        display: none;
    }

    #advanced-section.visible {
        height: auto;
        display: block;
    }

    #advanced-toggle-hint {
        color: $text-muted;
        margin-top: 1;
    }

    #skill-pack-row {
        height: 3;
        width: 100%;
        margin: 1 0;
    }

    .pack-toggle {
        width: 1fr;
        height: 3;
        align: left middle;
    }

    .pack-label {
        margin-left: 1;
        content-align: left middle;
    }

    #hint {
        height: 1;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("ctrl+j", "start_research", "Start  ctrl+j", show=True),
        Binding("a", "toggle_advanced", "Advanced", show=False),
    ]

    def __init__(self, prefill_query: str | None = None):
        super().__init__()
        self._prefill_query = prefill_query or ""
        self._show_advanced = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with VerticalScroll(id="main-panel"):
            yield Static("Research Question", classes="section-heading")
            yield Rule()
            yield TextArea(
                self._prefill_query,
                id="query-input",
                show_line_numbers=False,
            )
            yield Static("", id="error-msg")

            # Advanced section (toggled by 'a')
            with VerticalScroll(id="advanced-section"):
                yield Rule()
                yield Static("Advanced: Skill Packs", classes="section-heading")
                yield Static("[dim]Toggle which domains to research (core is always enabled)[/dim]")
                with Horizontal(id="skill-pack-row"):
                    with Horizontal(classes="pack-toggle"):
                        yield Switch(value=True, id="sw-core", disabled=True)
                        yield Static("core", classes="pack-label")
                    with Horizontal(classes="pack-toggle"):
                        yield Switch(value=True, id="sw-equity")
                        yield Static("equity", classes="pack-label")
                    with Horizontal(classes="pack-toggle"):
                        yield Switch(value=True, id="sw-macro")
                        yield Static("macro", classes="pack-label")
                    with Horizontal(classes="pack-toggle"):
                        yield Switch(value=True, id="sw-commodity")
                        yield Static("commodity", classes="pack-label")

            yield Static("[dim]Advanced [a]  ·  Esc back[/dim]", id="advanced-toggle-hint")
            yield Static("[dim]ctrl+j to start research[/dim]", id="hint")

        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#query-input", TextArea).focus()

    def action_toggle_advanced(self) -> None:
        self._show_advanced = not self._show_advanced
        section = self.query_one("#advanced-section", VerticalScroll)
        section.set_class(self._show_advanced, "visible")

    def action_start_research(self) -> None:
        self._do_start()

    def _do_start(self) -> None:
        query = self.query_one("#query-input", TextArea).text.strip()
        if not query:
            self.query_one("#error-msg", Static).update(
                "[red]Please enter a research question.[/red]"
            )
            return

        packs = ["core"]
        for pack in ["equity", "macro", "commodity"]:
            switch = self.query_one(f"#sw-{pack}", Switch)
            if switch.value:
                packs.append(pack)

        wall_clock = getattr(self.app, "_wall_clock_budget", None) or 600
        per_agent = getattr(self.app, "_per_agent_budget", None) or 300

        request = HarnessRequest(
            user_prompt=query,
            wall_clock_budget_seconds=wall_clock,
            per_agent_wall_clock_seconds=per_agent,
            available_skill_packs=packs,
        )

        backend = HarnessBackend(request)
        self.app._backend = backend
        self.app._wall_clock_budget = wall_clock
        self.app._skill_packs = packs
        self.app._query = query

        # Push DashboardScreen directly (not by string name) to avoid Textual
        # reusing a prior instance that has _started=True — which would skip
        # on_mount init and leave the new backend unpolled.
        self.app.push_screen(DashboardScreen())
