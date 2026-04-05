from __future__ import annotations

import io

import pytest

import main
from src.harness.types import HarnessResponse

pytestmark = pytest.mark.component


def test_main_runs_harness_with_prompt(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(main, "get_missing_provider_env_vars", lambda: {})
    monkeypatch.setattr(
        main,
        "run_harness",
        lambda _request: HarnessResponse(
            status="completed",
            stop_reason="done",
            run_root="/tmp/harness-run",
            root_agent_path="/tmp/harness-run/agents/agent_root",
            final_report_path="/tmp/harness-run/agents/agent_root/publish/final.md",
        ),
    )
    monkeypatch.setattr(
        "os.path.exists",
        lambda path: path == "/tmp/harness-run/agents/agent_root/publish/final.md",
    )
    monkeypatch.setattr(
        "builtins.open",
        lambda _path, _mode="r", encoding=None: io.StringIO("Kernel output"),
    )

    main.main(["Analyze AAPL"])
    captured = capsys.readouterr().out

    assert "Success!" in captured
    assert "Kernel output" in captured
    assert "Run root: /tmp/harness-run" in captured
