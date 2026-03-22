from __future__ import annotations

import types

import pytest

import main
from src.harness.types import HarnessResponse

pytestmark = pytest.mark.component


def test_main_uses_harness_runtime(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(main, "get_missing_provider_env_vars", lambda: {})
    monkeypatch.setattr("builtins.input", lambda _prompt: "Analyze AAPL")
    monkeypatch.setattr(
        main,
        "run_harness",
        lambda _request: HarnessResponse(
            final_response="Harness output",
            status="completed",
            report_path="/tmp/report.md",
            trace_path="/tmp/trace.json",
            enabled_packs=["core", "equity"],
            skills_used=["search_and_read"],
        ),
    )

    main.main(["--runtime", "harness"])
    captured = capsys.readouterr().out

    assert "Runtime: harness" in captured
    assert "Harness output" in captured
    assert "search_and_read" in captured


def test_main_uses_legacy_runtime(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(main, "get_missing_provider_env_vars", lambda: {})
    monkeypatch.setattr("builtins.input", lambda _prompt: "Analyze AAPL")
    monkeypatch.setattr(
        main,
        "app",
        types.SimpleNamespace(
            invoke=lambda _state: {
                "final_response": "Legacy output",
                "agent_results": {"equity": "Equity report"},
            }
        ),
    )

    main.main(["--runtime", "legacy"])
    captured = capsys.readouterr().out

    assert "Runtime: legacy" in captured
    assert "Legacy output" in captured
    assert "Sub-agents that ran: equity" in captured
