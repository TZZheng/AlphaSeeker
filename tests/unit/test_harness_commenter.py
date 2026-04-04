from __future__ import annotations

from pathlib import Path

import pytest

from src.harness import agent_worker as agent_worker_module
from src.harness.agent_worker import run_agent_worker
from src.harness.artifacts import (
    agent_workspace_paths,
    append_commenter_comments,
    create_agent_workspace,
    initialize_run_root,
    load_commenter_comments,
    mark_commenter_comments_read,
    unread_commenter_comments,
    write_text_atomic,
)
from src.harness.commenter import (
    _strip_provider_thinking,
    build_comment_feed_message,
    compute_commenter_observation_fingerprint,
    refresh_commenter_for_agent,
)
from src.harness.presets import default_tool_allowlist, render_root_task_markdown, render_tools_markdown
from src.harness.transport import ModelToolCall, ModelTurnResult
from src.harness.types import HarnessRequest


def _create_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, HarnessRequest]:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Analyze XOM", run_id="commenter-test")
    run_root, agent_id = initialize_run_root(request)
    create_agent_workspace(
        run_root,
        agent_id=agent_id,
        parent_id="",
        preset="orchestrator",
        task_name="Root Task",
        description="Analyze XOM",
        task_markdown=render_root_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="orchestrator",
            available_tools=default_tool_allowlist("orchestrator"),
            available_skills=[],
        ),
    )
    return Path(run_root), agent_id, request


def test_workspace_initializes_commenter_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, _request = _create_workspace(tmp_path, monkeypatch)
    paths = agent_workspace_paths(run_root, agent_id)

    assert paths["commenter_root"].exists()
    assert paths["commenter_notes_root"].exists()
    assert paths["commenter_turns_root"].exists()
    assert paths["commenter_comments"].exists()
    assert paths["commenter_latest"].exists()


def test_comment_feed_caps_unread_comments_and_marks_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, _request = _create_workspace(tmp_path, monkeypatch)
    append_commenter_comments(
        run_root,
        agent_id,
        ["one", "two", "three", "four"],
        generated_at="2026-04-03T22:51:13Z",
    )

    feed, injected_count = build_comment_feed_message(run_root, agent_id)

    assert feed is not None
    assert injected_count == 3
    assert "...and 1 older unread comments not shown." in feed

    marked = mark_commenter_comments_read(run_root, agent_id, injected_count)
    unread, total_unread = unread_commenter_comments(run_root, agent_id)

    assert marked == 3
    assert total_unread == 1
    assert unread[0]["content"] == "four"
    assert unread[0]["read"] is False


def test_commenter_fingerprint_ignores_commenter_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, _request = _create_workspace(tmp_path, monkeypatch)
    paths = agent_workspace_paths(run_root, agent_id)
    write_text_atomic(paths["scratch_root"] / "analysis.md", "first draft\n")

    baseline = compute_commenter_observation_fingerprint(str(run_root), agent_id)

    append_commenter_comments(run_root, agent_id, ["outside angle"])
    write_text_atomic(paths["commenter_latest"], "# Latest Comments\n\n- note\n")
    after_commenter_write = compute_commenter_observation_fingerprint(str(run_root), agent_id)

    write_text_atomic(paths["scratch_root"] / "analysis.md", "second draft\n")
    after_scratch_write = compute_commenter_observation_fingerprint(str(run_root), agent_id)

    assert after_commenter_write == baseline
    assert after_scratch_write != baseline


def test_refresh_commenter_records_raw_response_and_turn_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, request = _create_workspace(tmp_path, monkeypatch)
    paths = agent_workspace_paths(run_root, agent_id)
    write_text_atomic(paths["scratch_root"] / "draft.md", "only child summaries were read\n")

    raw_response = (
        "Read the underlying research note before concluding.\n"
        "Then check whether the recommendation still holds when the current strip replaces the headline spot move."
    )
    monkeypatch.setattr(
        "src.harness.commenter._run_commenter_dialog",
        lambda **kwargs: (raw_response, [{"step": 1, "tool_calls": [], "text_blocks": [raw_response]}]),
    )

    written = refresh_commenter_for_agent(
        str(run_root),
        agent_id,
        request,
        model_name="gpt-4o",
        transport_name="text_json",
    )
    rows = load_commenter_comments(run_root, agent_id)
    turns_root = paths["commenter_turns_root"]

    assert written == 1
    assert [row["content"] for row in rows] == [raw_response]
    assert all(Path(str(row["note_path"])).exists() for row in rows)
    assert (turns_root / "0001_system_prompt.md").exists()
    input_text = (turns_root / "0001_input.md").read_text(encoding="utf-8")
    assert "Available Files" in input_text
    assert "- scratch/draft.md [scratch]" in input_text
    assert "Suggested Files To Inspect First" in input_text
    assert "only child summaries were read" not in input_text
    assert (turns_root / "0001_response.md").read_text(encoding="utf-8") == raw_response
    assert (turns_root / "0001_trace.json").exists()


def test_multiline_comment_is_preserved_in_latest_and_feed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, _request = _create_workspace(tmp_path, monkeypatch)
    raw_comment = (
        "Read the actual filing before concluding.\n"
        "Then reconcile the recommendation with the current futures strip."
    )
    append_commenter_comments(
        run_root,
        agent_id,
        [raw_comment],
        generated_at="2026-04-03T22:51:13Z",
    )

    latest_text = agent_workspace_paths(run_root, agent_id)["commenter_latest"].read_text(encoding="utf-8")
    feed, injected_count = build_comment_feed_message(run_root, agent_id)

    assert injected_count == 1
    assert raw_comment in latest_text
    assert feed is not None
    assert "[2026-04-03T22:51:13Z]" in feed
    assert raw_comment in feed


def test_strip_provider_thinking_removes_only_thinking_blocks() -> None:
    raw_text = (
        "<think>compare child drafts before responding</think>\n"
        "Reconcile the recommendation with the current futures strip.\n"
        "<thinking>ignore this hidden reasoning too</thinking>"
    )

    assert _strip_provider_thinking(raw_text) == "Reconcile the recommendation with the current futures strip."


def test_marking_comments_read_updates_note_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, _request = _create_workspace(tmp_path, monkeypatch)
    append_commenter_comments(run_root, agent_id, ["Check the newer source before concluding."])
    rows = load_commenter_comments(run_root, agent_id)

    note_path = Path(str(rows[0]["note_path"]))
    assert "status: unread" in note_path.read_text(encoding="utf-8")

    marked = mark_commenter_comments_read(run_root, agent_id, 1)
    rows = load_commenter_comments(run_root, agent_id)

    assert marked == 1
    assert rows[0]["read"] is True
    assert "status: read" in note_path.read_text(encoding="utf-8")


class _SuccessTransport:
    def __init__(self) -> None:
        self.user_messages: list[str] = []

    def ensure_initialized(self, _initial_user_prompt: str) -> None:
        return None

    def update_system_prompt(self, _system_prompt: str) -> None:
        return None

    def append_user_text(self, text: str) -> None:
        self.user_messages.append(text)

    def execute_turn(self, _tool_specs: list[dict[str, object]]) -> ModelTurnResult:
        assert any(message.startswith("Comment Feed") for message in self.user_messages)
        return ModelTurnResult(
            tool_calls=[
                ModelToolCall(
                    call_id="call_1",
                    name="write_publish_file",
                    arguments={"file_name": "summary.md", "content": "# Summary\n"},
                ),
                ModelToolCall(
                    call_id="call_2",
                    name="write_publish_file",
                    arguments={"file_name": "artifact_index.md", "content": "# Artifact Index\n"},
                ),
                ModelToolCall(
                    call_id="call_3",
                    name="write_publish_file",
                    arguments={"file_name": "final.md", "content": "# Final\n\nDone.\n"},
                ),
                ModelToolCall(call_id="call_4", name="set_status", arguments={"status": "done"}),
            ],
            text_blocks=["finish now"],
            stop_reason="tool_use",
        )

    def append_tool_results(self, _tool_results: list[dict[str, object]]) -> None:
        return None


class _FailingTransport(_SuccessTransport):
    def execute_turn(self, _tool_specs: list[dict[str, object]]) -> ModelTurnResult:
        raise RuntimeError("commenter delivery failed")


def test_worker_injects_comment_feed_and_marks_comments_read_after_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, _request = _create_workspace(tmp_path, monkeypatch)
    append_commenter_comments(run_root, agent_id, ["Check the underlying file before concluding."])
    fake_transport = _SuccessTransport()

    monkeypatch.setattr(agent_worker_module, "get_model", lambda *_args, **_kwargs: "minimax/MiniMax-M2.7")
    monkeypatch.setattr(agent_worker_module, "resolve_agent_transport", lambda *_args, **_kwargs: "minimax_anthropic")
    monkeypatch.setattr(agent_worker_module, "create_transport", lambda **_kwargs: fake_transport)

    result = run_agent_worker(str(run_root), agent_id)
    rows = load_commenter_comments(run_root, agent_id)

    assert result == 0
    assert rows[0]["read"] is True
    assert any(message.startswith("Comment Feed") for message in fake_transport.user_messages)


def test_worker_leaves_comments_unread_when_model_call_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, _request = _create_workspace(tmp_path, monkeypatch)
    append_commenter_comments(run_root, agent_id, ["Do not stop after reading only summaries."])
    fake_transport = _FailingTransport()

    monkeypatch.setattr(agent_worker_module, "MAX_CONSECUTIVE_ERRORS", 1)
    monkeypatch.setattr(agent_worker_module, "get_model", lambda *_args, **_kwargs: "minimax/MiniMax-M2.7")
    monkeypatch.setattr(agent_worker_module, "resolve_agent_transport", lambda *_args, **_kwargs: "minimax_anthropic")
    monkeypatch.setattr(agent_worker_module, "create_transport", lambda **_kwargs: fake_transport)

    result = run_agent_worker(str(run_root), agent_id)
    rows = load_commenter_comments(run_root, agent_id)

    assert result == 1
    assert rows[0]["read"] is False
