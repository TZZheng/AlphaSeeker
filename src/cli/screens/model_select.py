"""Model selection screen — keyboard-navigable provider and model lists, two-tap confirmation."""

from __future__ import annotations

import os
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, OptionList, Rule, Static
from textual.widgets.option_list import Option

from src.cli.theme import get_banner


_PROVIDER_CONFIG: dict[str, dict[str, Any]] = {
    "MINIMAX_API_KEY": {
        "label": "MiniMax",
        "default_model": "minimax/MiniMax-M2.7",
        "models": ["minimax/MiniMax-M2.7", "minimax/MiniMax-M2.1"],
    },
    "OPENAI_API_KEY": {
        "label": "OpenAI",
        "default_model": "gpt-4o",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    },
    "ANTHROPIC_API_KEY": {
        "label": "Anthropic",
        "default_model": "claude-sonnet-4-20250514",
        "models": [
            "claude-sonnet-4-20250514",
            "claude-opus-4-6-20251120",
            "claude-3-5-sonnet-latest",
        ],
    },
    "SILICONFLOW_API_KEY": {
        "label": "SiliconFlow",
        "default_model": "sf/Qwen/Qwen3-14B",
        "models": ["sf/Qwen/Qwen3-14B", "sf/Qwen/Qwen3-8B", "sf/Qwen/Qwen3.5-4B"],
    },
    "KIMI_API_KEY": {
        "label": "Kimi (Moonshot)",
        "default_model": "kimi-k2.5",
        "models": ["kimi-k2.5", "kimi-k2"],
    },
    "GOOGLE_API_KEY": {
        "label": "Google Gemini",
        "default_model": "gemini-2.5-flash",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro"],
    },
}

_PROVIDER_KEYS = list(_PROVIDER_CONFIG.keys())

_ICON_SEL = "[bold green]●[/bold green]"
_ICON_ON = "[green]○[/green]"
_ICON_OFF = "[dim]○[/dim]"


class ModelSelectScreen(Screen):
    """Keyboard-navigable provider and model selection — two-tap Enter to confirm each."""

    CSS = """
    ModelSelectScreen {
        align: center middle;
    }

    #main-panel {
        width: 70;
        height: auto;
        max-height: 90%;
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

    .section-title {
        color: $primary;
        text-style: bold;
        height: 1;
        margin: 1 0 0 0;
    }

    #provider-list {
        height: auto;
        margin: 1 0;
        border: none;
        padding: 0;
    }

    #model-list {
        height: auto;
        margin: 1 0;
        border: none;
        padding: 0;
    }

    #error-msg {
        color: $error;
        height: 1;
        margin: 1 0 0 0;
    }

    #hint {
        height: 1;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "app.exit", "Quit", show=False),
    ]

    def __init__(self, prefill_query: str | None = None):
        super().__init__()
        self._prefill_query = prefill_query
        self._selected_provider: str | None = None  # first-Enter provider selection
        self._selected_model: str | None = None  # first-Enter model selection

    # ── Provider list ──────────────────────────────────────────────────────

    def _build_provider_options(self) -> list[Option]:
        opts = []
        for env_var, config in _PROVIDER_CONFIG.items():
            available = bool(os.getenv(env_var))
            if not available:
                icon = _ICON_OFF
            elif env_var == self._selected_provider:
                icon = _ICON_SEL
            else:
                icon = _ICON_ON
            key_hint = "" if available else "  [dim red](no key)[/dim red]"
            label = f"{icon}  {config['label']}{key_hint}"
            opts.append(Option(label, id=env_var, disabled=not available))
        return opts

    def _rebuild_provider_list(self) -> None:
        ol = self.query_one("#provider-list", OptionList)
        cursor = ol.highlighted
        ol.clear_options()
        for opt in self._build_provider_options():
            ol.add_option(opt)
        if cursor is not None:
            ol.highlighted = cursor

    # ── Model list ─────────────────────────────────────────────────────────

    def _build_model_options(self, env_var: str) -> list[Option]:
        config = _PROVIDER_CONFIG[env_var]
        opts = []
        for m in config["models"]:
            icon = _ICON_SEL if m == self._selected_model else _ICON_ON
            opts.append(Option(f"{icon}  {m}", id=m))
        return opts

    def _rebuild_model_list(self) -> None:
        if not self._selected_provider:
            return
        ol = self.query_one("#model-list", OptionList)
        cursor = ol.highlighted
        ol.clear_options()
        for opt in self._build_model_options(self._selected_provider):
            ol.add_option(opt)
        if cursor is not None:
            ol.highlighted = cursor

    def _populate_model_list(self, env_var: str) -> None:
        """Load model list for a provider, resetting selection state."""
        config = _PROVIDER_CONFIG[env_var]
        ol = self.query_one("#model-list", OptionList)
        ol.clear_options()
        for m in config["models"]:
            ol.add_option(Option(f"{_ICON_ON}  {m}", id=m))
        ol.highlighted = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="main-panel"):
            yield Static("", id="ascii-banner")
            yield Static("Multi-Agent Research Terminal", id="subtitle")
            yield Rule()
            yield Static(
                "Provider  [dim]↑↓ navigate · Enter select · Enter again confirm[/dim]",
                classes="section-title",
            )
            yield OptionList(*self._build_provider_options(), id="provider-list")

            yield Static(
                "Model  [dim]↑↓ navigate · Enter select · Enter again confirm[/dim]",
                classes="section-title",
            )
            yield OptionList(id="model-list")

            yield Static("", id="error-msg")
            yield Static("[dim]Tab switch focus  ·  Esc back[/dim]", id="hint")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#ascii-banner", Static).update(get_banner(self.app.size.width))
        ol = self.query_one("#provider-list", OptionList)
        for i, env_var in enumerate(_PROVIDER_KEYS):
            if os.getenv(env_var):
                ol.highlighted = i
                self._populate_model_list(env_var)
                break
        ol.focus()

    # ── Provider events ────────────────────────────────────────────────────

    @on(OptionList.OptionHighlighted, "#provider-list")
    def _on_provider_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        env_var = event.option.id
        # Preview model list while browsing (only when no provider selected yet)
        if env_var and os.getenv(env_var) and self._selected_provider is None:
            self._populate_model_list(env_var)

    @on(OptionList.OptionSelected, "#provider-list")
    def _on_provider_selected(self, event: OptionList.OptionSelected) -> None:
        env_var = event.option.id
        if not env_var or not os.getenv(env_var):
            self.query_one("#error-msg", Static).update(
                "[red]Select a provider with an API key.[/red]"
            )
            return

        if self._selected_provider == env_var:
            # Second Enter on same provider → move focus to model list
            self.query_one("#model-list", OptionList).focus()
        else:
            # First Enter → select this provider, reset model selection
            self._selected_provider = env_var
            self._selected_model = None
            self._rebuild_provider_list()
            self._populate_model_list(env_var)
            self.query_one("#error-msg", Static).update("")

    # ── Model events ───────────────────────────────────────────────────────

    @on(OptionList.OptionSelected, "#model-list")
    def _on_model_selected(self, event: OptionList.OptionSelected) -> None:
        if not self._selected_provider:
            self.query_one("#error-msg", Static).update(
                "[red]Select a provider first (Tab back to provider list).[/red]"
            )
            return

        model_id = event.option.id
        if self._selected_model == model_id:
            # Second Enter on same model → confirm and advance
            self._do_continue()
        else:
            # First Enter → select this model
            self._selected_model = model_id
            self._rebuild_model_list()

    # ── Advance ───────────────────────────────────────────────────────────

    def _do_continue(self) -> None:
        env_var = self._selected_provider
        model = self._selected_model
        if not env_var or not model:
            self.query_one("#error-msg", Static).update(
                "[red]Select a provider and model first.[/red]"
            )
            return

        all_agents = ["supervisor", "harness", "equity", "macro", "commodity"]
        all_roles = [
            "classify", "synthesize", "agent", "condense", "plan",
            "followup", "map", "reduce", "section", "summary",
        ]
        for agent in all_agents:
            for role in all_roles:
                os.environ[f"ALPHASEEKER_MODEL_{agent.upper()}_{role.upper()}"] = model

        self.app._selected_model = {env_var: model}
        self.app.push_screen("effort")
