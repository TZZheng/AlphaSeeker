"""Config screen — reasoning effort, skill packs, and research question."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    RadioButton,
    RadioSet,
    Rule,
    Static,
    Switch,
    TextArea,
)

from src.cli.backends.harness_backend import HarnessBackend
from src.cli.screens.dashboard import DashboardScreen
from src.harness.types import HarnessRequest


REASONING_LEVELS = [
    # (label, wall_clock_seconds, per_agent_seconds)
    ("Quick", 120, 60),
    ("Medium", 600, 300),
    ("Deep", 1800, 900),
]


class ConfigScreen(Screen):
    """Configure reasoning effort, skill packs, and the research query."""

    CSS = """
    ConfigScreen {
        align: center middle;
    }

    #config-panel {
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

    #effort-radio {
        width: 100%;
        height: auto;
        margin: 1 0;
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

    #query-input {
        height: 6;
        min-height: 4;
        max-height: 12;
        margin: 1 0;
    }

    #config-error {
        color: $error;
        margin: 1 0;
    }

    #button-row {
        width: 74;
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("ctrl+j", "start_research", "Start Research", show=True),
    ]

    def __init__(self, prefill_query: str | None = None):
        super().__init__()
        self._prefill_query = prefill_query or ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with VerticalScroll(id="config-panel"):
            # Reasoning effort
            yield Static("Reasoning Effort", classes="section-heading")
            yield Rule()
            with RadioSet(id="effort-radio"):
                yield RadioButton("Quick   (2 min wall clock)", id="effort-quick")
                yield RadioButton("Medium  (10 min wall clock)", id="effort-medium", value=True)
                yield RadioButton("Deep    (30 min wall clock)", id="effort-deep")

            # Skill packs
            yield Static("Skill Packs", classes="section-heading")
            yield Rule()
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

            # Research question (TextArea — the key fix)
            yield Static("Research Question", classes="section-heading")
            yield Rule()
            yield TextArea(
                self._prefill_query,
                id="query-input",
                show_line_numbers=False,
            )

            yield Static("", id="config-error")

        # Buttons
        with Horizontal(id="button-row"):
            yield Button("← Back", id="back-btn")
            yield Button("Start Research →", id="start-btn", variant="success")
        yield Footer()

    def action_start_research(self) -> None:
        """Triggered by ctrl+enter keybinding."""
        self._do_start()

    @on(Button.Pressed, "#start-btn")
    def handle_start(self) -> None:
        self._do_start()

    @on(Button.Pressed, "#back-btn")
    def handle_back(self) -> None:
        self.app.pop_screen()

    def _do_start(self) -> None:
        """Collect config and launch the harness backend."""
        # Effort → budget
        radio = self.query_one("#effort-radio", RadioSet)
        effort_index = radio.pressed_index
        if effort_index < 0:
            effort_index = 1  # default to Medium
        _label, wall_clock, per_agent = REASONING_LEVELS[effort_index]

        # Skill packs
        packs = ["core"]
        for pack in ["equity", "macro", "commodity"]:
            switch = self.query_one(f"#sw-{pack}", Switch)
            if switch.value:
                packs.append(pack)

        # Query
        query = self.query_one("#query-input", TextArea).text.strip()
        if not query:
            self.query_one("#config-error", Static).update(
                "[red]Please enter a research question.[/red]"
            )
            return

        # Build HarnessRequest
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
