"""Effort selection screen — keyboard-navigable list with two-tap confirmation."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, OptionList, Rule, Static
from textual.widgets.option_list import Option

from src.cli.theme import get_banner


REASONING_LEVELS = [
    # (label, description, wall_clock_seconds, per_agent_seconds)
    ("Quick", "Fast overview, ~2 min", 120, 60),
    ("Medium", "Balanced analysis, ~10 min", 600, 300),
    ("Deep", "Thorough investigation, ~30 min", 1800, 900),
]

_ICON_SELECTED = "[bold green]●[/bold green]"
_ICON_UNSELECTED = "[dim]○[/dim]"


class EffortScreen(Screen):
    """Select reasoning effort level — keyboard only, two-tap confirmation."""

    CSS = """
    EffortScreen {
        align: center middle;
    }

    #main-panel {
        width: 74;
        height: auto;
        max-height: 85%;
        border: round $primary 30%;
        padding: 1 2;
    }

    #ascii-banner {
        color: $primary;
        content-align: center middle;
        width: 100%;
        height: 7;
    }

    #subtitle {
        color: $secondary;
        content-align: center middle;
        width: 100%;
        height: 1;
        text-style: italic;
        margin-bottom: 1;
    }

    #effort-desc {
        margin: 1 0;
        height: 1;
    }

    #effort-list {
        height: auto;
        margin: 1 0;
        border: none;
        padding: 0;
    }

    #hint {
        margin-top: 1;
        height: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, prefill_query: str | None = None):
        super().__init__()
        self._prefill_query = prefill_query or ""
        self._selected_idx: int | None = None  # first-Enter selection

    def _build_options(self) -> list[Option]:
        opts = []
        for i, (label, desc, _, _) in enumerate(REASONING_LEVELS):
            icon = _ICON_SELECTED if i == self._selected_idx else _ICON_UNSELECTED
            opts.append(Option(f"{icon}  {label}  [dim]{desc}[/dim]", id=f"effort-{i}"))
        return opts

    def _rebuild_list(self) -> None:
        ol = self.query_one("#effort-list", OptionList)
        cursor = ol.highlighted
        ol.clear_options()
        for opt in self._build_options():
            ol.add_option(opt)
        if cursor is not None:
            ol.highlighted = cursor

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="main-panel"):
            yield Static("", id="ascii-banner")
            yield Static("Reasoning Effort", id="subtitle")
            yield Rule()
            yield Static("", id="effort-desc")
            yield OptionList(*self._build_options(), id="effort-list")
            yield Static(
                "[dim]↑↓ navigate  ·  Enter select · Enter again confirm  ·  Esc back[/dim]",
                id="hint",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#ascii-banner", Static).update(get_banner(self.app.size.width))
        ol = self.query_one("#effort-list", OptionList)
        ol.highlighted = 1  # cursor starts on Medium
        ol.focus()

    def _update_desc(self, idx: int) -> None:
        _, desc, _, _ = REASONING_LEVELS[idx]
        self.query_one("#effort-desc", Static).update(f"[dim]{desc}[/dim]")

    @on(OptionList.OptionHighlighted, "#effort-list")
    def _on_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._update_desc(event.option_index)

    @on(OptionList.OptionSelected, "#effort-list")
    def _on_selected(self, event: OptionList.OptionSelected) -> None:
        idx = event.option_index
        if self._selected_idx == idx:
            # Second Enter on same item → confirm and advance
            self._do_continue()
        else:
            # First Enter → mark as selected, update visual
            self._selected_idx = idx
            self._rebuild_list()

    def _do_continue(self) -> None:
        idx = self._selected_idx if self._selected_idx is not None else 1
        label, _, wall_clock, per_agent = REASONING_LEVELS[idx]
        self.app._reasoning_effort = label
        self.app._wall_clock_budget = wall_clock
        self.app._per_agent_budget = per_agent
        self.app.push_screen("prompt")
