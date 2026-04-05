"""AlphaSeeker CLI entry point — thin Typer wrapper around the Textual TUI."""

import os
import sys

# Add project root to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env before any other imports
from dotenv import load_dotenv
load_dotenv()

import typer
from src.cli.app import run_tui


def main(
    query: str | None = typer.Argument(
        None,
        help="Research question. If omitted, launches the interactive TUI."
    ),
) -> None:
    """Run the AlphaSeeker interactive research CLI."""
    run_tui(prefill_query=query)


if __name__ == "__main__":
    typer.run(main)
