"""AlphaSeeker CLI — theme constants, ASCII art, and global CSS."""

from __future__ import annotations


# ---------------------------------------------------------------------------
# ASCII Banners
# ---------------------------------------------------------------------------

ASCII_BANNER_WIDE = r"""
    _    _       _           ____            _
   / \  | |_ __ | |__   __ / ___|  ___  ___| | _____ _ __
  / _ \ | | '_ \| '_ \ / _` \___ \ / _ \/ _ \ |/ / _ \ '__|
 / ___ \| | |_) | | | | (_| |___) |  __/  __/   <  __/ |
/_/   \_\_| .__/|_| |_|\__,_|____/ \___|\___|_|\_\___|_|
          |_|
"""

ASCII_BANNER_NARROW = """\
╔═══════════════════════════════════╗
║   A L P H A S E E K E R          ║
║   Multi-Agent Research Terminal   ║
╚═══════════════════════════════════╝"""


def get_banner(terminal_width: int) -> str:
    """Return the appropriate ASCII banner for the given terminal width."""
    if terminal_width >= 60:
        return ASCII_BANNER_WIDE
    return ASCII_BANNER_NARROW


# ---------------------------------------------------------------------------
# Theme registration kwargs  (passed to textual.theme.Theme)
# ---------------------------------------------------------------------------

THEME_KWARGS = dict(
    name="alphaseeker",
    primary="#00d4aa",
    secondary="#39bae6",
    background="#0a0e14",
    surface="#1a1e28",
    success="#7fd962",
    warning="#ffb454",
    error="#ff3333",
    accent="#f07178",
    dark=True,
)


# ---------------------------------------------------------------------------
# Agent status → Rich colour mapping
# ---------------------------------------------------------------------------

STATUS_COLORS: dict[str, str] = {
    "running": "green",
    "queued": "dim",
    "waiting": "yellow",
    "done": "cyan",
    "failed": "red",
    "blocked": "red",
    "stale": "dark_orange",
    "cancelled": "dim red",
    "refining": "magenta",
}


# ---------------------------------------------------------------------------
# Global CSS — applied at the App level, inherited by all screens
# ---------------------------------------------------------------------------

GLOBAL_CSS = """
Screen {
    background: $background;
}

Header {
    background: $surface;
    color: $primary;
    dock: top;
    height: 1;
}

Footer {
    background: $surface;
    color: $text-muted;
}

/* ── Buttons ── */
Button {
    background: $surface;
    color: $text;
    border: tall $primary 50%;
    min-width: 16;
    margin: 0 1;
}

Button:hover {
    background: $primary 15%;
    border: tall $primary;
}

Button.-primary {
    background: $primary 15%;
    color: $primary;
    border: tall $primary;
}

Button.-success {
    background: $success 15%;
    color: $success;
    border: tall $success;
}

Button.-error {
    background: $error 15%;
    color: $error;
    border: tall $error;
}

/* ── Select widgets ── */
Select {
    background: $surface;
    border: tall $primary 50%;
}

Select:focus {
    border: tall $primary;
}

/* ── Switch widgets ── */
Switch {
    background: $surface;
}

/* ── DataTable ── */
DataTable {
    background: $surface;
}

DataTable > .datatable--header {
    background: $primary 20%;
    color: $primary;
}

DataTable > .datatable--cursor {
    background: $primary 15%;
}

/* ── RichLog ── */
RichLog {
    background: $surface;
    border: round $primary 30%;
    scrollbar-color: $primary 30%;
    scrollbar-color-hover: $primary;
}

/* ── Input / TextArea ── */
Input {
    background: $surface;
    border: tall $primary 50%;
    color: $text;
}

Input:focus {
    border: tall $primary;
}

TextArea {
    background: $surface;
    border: tall $primary 50%;
    color: $text;
}

TextArea:focus {
    border: tall $primary;
}

/* ── RadioSet ── */
RadioSet {
    background: $surface;
    border: none;
}

RadioButton {
    background: $surface;
}

/* ── Markdown ── */
Markdown {
    margin: 1 0;
}

/* ── Scrollbars ── */
* {
    scrollbar-background: $surface;
    scrollbar-color: $primary 30%;
    scrollbar-color-hover: $primary 60%;
    scrollbar-color-active: $primary;
}

/* ── Rule ── */
Rule {
    color: $primary 30%;
    margin: 1 0;
}
"""
