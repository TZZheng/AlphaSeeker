from __future__ import annotations

from datetime import datetime, timezone
import os
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
    load_commenter_state,
    mark_commenter_comments_read,
    read_status,
    unread_commenter_comments,
    write_status,
    write_text_atomic,
)
from src.harness.commenter import (
    _strip_provider_thinking,
    build_comment_feed_message,
    build_commenter_observation_manifest,
    build_commenter_observation_snapshot,
    compute_commenter_observation_fingerprint,
    refresh_commenter_for_agent,
)
from src.harness.presets import default_tool_allowlist
from src.harness.prompt_builder import render_task_markdown, render_tools_markdown
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
        task_markdown=render_task_markdown(request.user_prompt),
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


def test_refresh_commenter_records_frozen_incremental_prompt_and_review_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, request = _create_workspace(tmp_path, monkeypatch)
    paths = agent_workspace_paths(run_root, agent_id)
    write_text_atomic(paths["scratch_root"] / "draft.md", "draft v1\n")
    write_text_atomic(paths["publish_root"] / "old_summary.md", "# Old Summary\n")
    base_manifest = build_commenter_observation_manifest(str(run_root), agent_id)

    write_text_atomic(paths["scratch_root"] / "draft.md", "draft v2\n")
    write_text_atomic(paths["publish_root"] / "final.md", "# Final\n\nNew version.\n")
    write_text_atomic(paths["scratch_root"] / "journal.jsonl", '{"tool":"write_file"}\n')
    llm_turns_root = paths["scratch_root"] / "llm_turns"
    llm_turns_root.mkdir(parents=True, exist_ok=True)
    write_text_atomic(llm_turns_root / "0001_request.json", '{"messages":[]}\n')
    (paths["publish_root"] / "old_summary.md").unlink()
    snapshot = build_commenter_observation_snapshot(
        str(run_root),
        agent_id,
        base_manifest=base_manifest,
        fired_at="2026-04-11T12:00:00Z",
    )

    write_text_atomic(paths["scratch_root"] / "draft.md", "draft v3\n")
    assert compute_commenter_observation_fingerprint(str(run_root), agent_id) != snapshot["target_fingerprint"]

    raw_response = (
        "Read the underlying research note before concluding.\n"
        "Then check whether the recommendation still holds when the current strip replaces the headline spot move."
    )
    captured: dict[str, object] = {}

    def _fake_dialog(**kwargs):
        captured["manifest"] = kwargs["manifest"]
        return raw_response, [{"step": 1, "tool_calls": [], "text_blocks": [raw_response]}]

    monkeypatch.setattr(
        "src.harness.commenter._run_commenter_dialog",
        _fake_dialog,
    )

    written = refresh_commenter_for_agent(
        str(run_root),
        agent_id,
        request,
        model_name="gpt-4o",
        transport_name="text_json",
        observation_snapshot=snapshot,
    )
    rows = load_commenter_comments(run_root, agent_id)
    state = load_commenter_state(run_root, agent_id) or {}
    turns_root = paths["commenter_turns_root"]

    assert written == 1
    assert [row["content"] for row in rows] == [raw_response]
    assert all(Path(str(row["note_path"])).exists() for row in rows)
    assert (turns_root / "0001_system_prompt.md").exists()
    input_text = (turns_root / "0001_input.md").read_text(encoding="utf-8")
    assert "Current Task" in input_text
    assert "Changed Since Last Review" in input_text
    assert "Inspectable Scope" in input_text
    assert "Available Files" not in input_text
    assert "Suggested Files To Inspect First" not in input_text
    assert "- modified: scratch/draft.md [scratch]" in input_text
    assert "- modified: scratch/journal.jsonl [operating_log]" in input_text
    assert "- added: scratch/llm_turns/0001_request.json [llm_trace]" in input_text
    assert "- added: publish/final.md [publish]" in input_text
    assert "- deleted: publish/old_summary.md [publish]" in input_text
    assert "- task.md" in input_text
    assert "- tools.md" in input_text
    assert "- scratch/" in input_text
    assert "- publish/" in input_text
    assert "draft v2" not in input_text
    assert "draft v3" not in input_text
    assert captured["manifest"] == snapshot["target_manifest"]
    assert (turns_root / "0001_response.md").read_text(encoding="utf-8") == raw_response
    assert (turns_root / "0001_trace.json").exists()
    assert state["last_commented_manifest"] == snapshot["target_manifest"]
    assert state["last_commented_fingerprint"] == snapshot["target_fingerprint"]

    follow_up_snapshot = build_commenter_observation_snapshot(
        str(run_root),
        agent_id,
        base_manifest=state["last_commented_manifest"],
        fired_at="2026-04-11T12:01:00Z",
    )
    assert [
        (item["change_type"], item["display_path"], item["category"])
        for item in follow_up_snapshot["changed_entries"]
    ] == [("modified", "scratch/draft.md", "scratch")]


def test_commenter_manifest_classifies_runtime_files_by_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, _request = _create_workspace(tmp_path, monkeypatch)
    paths = agent_workspace_paths(run_root, agent_id)
    llm_turns_root = paths["scratch_root"] / "llm_turns"
    llm_turns_root.mkdir(parents=True, exist_ok=True)
    write_text_atomic(paths["scratch_root"] / "journal.jsonl", '{"kind":"tool"}\n')
    write_text_atomic(paths["scratch_root"] / "transcript.jsonl", '{"kind":"assistant"}\n')
    write_text_atomic(paths["scratch_root"] / "tool_history.jsonl", '{"tool":"read_file"}\n')
    write_text_atomic(llm_turns_root / "0001_request.json", '{"messages":[]}\n')
    skills_root = paths["scratch_root"] / "skills" / "001_read_file"
    skills_root.mkdir(parents=True, exist_ok=True)
    write_text_atomic(skills_root / "output.md", "tool output\n")
    write_text_atomic(paths["scratch_root"] / "notes.md", "working notes\n")

    manifest = build_commenter_observation_manifest(str(run_root), agent_id)
    categories = {item["display_path"]: item["category"] for item in manifest}

    assert categories["scratch/journal.jsonl"] == "operating_log"
    assert categories["scratch/transcript.jsonl"] == "operating_log"
    assert categories["scratch/tool_history.jsonl"] == "operating_log"
    assert categories["scratch/llm_turns/0001_request.json"] == "llm_trace"
    assert categories["scratch/skills/001_read_file/output.md"] == "tool_artifact"
    assert categories["scratch/notes.md"] == "scratch"


def test_refresh_commenter_skips_terminal_agents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, request = _create_workspace(tmp_path, monkeypatch)
    write_status(run_root, agent_id, "done")
    monkeypatch.setattr(
        "src.harness.commenter._run_commenter_dialog",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("commenter should not run for terminal agents")),
    )

    written = refresh_commenter_for_agent(
        str(run_root),
        agent_id,
        request,
        model_name="gpt-4o",
        transport_name="text_json",
    )

    assert written == 0
    assert load_commenter_comments(run_root, agent_id) == []


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
        assert any("Comment Feed" in message for message in self.user_messages)
        return ModelTurnResult(
            tool_calls=[
                ModelToolCall(
                    call_id="call_1",
                    name="write_file",
                    arguments={"path": "publish/summary.md", "content": "# Summary\n"},
                ),
                ModelToolCall(
                    call_id="call_2",
                    name="write_file",
                    arguments={"path": "publish/artifact_index.md", "content": "# Artifact Index\n"},
                ),
                ModelToolCall(
                    call_id="call_3",
                    name="write_file",
                    arguments={"path": "publish/final.md", "content": "# Final\n\nDone.\n"},
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
    assert any("Comment Feed" in message for message in fake_transport.user_messages)


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


def _iso_from_epoch(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


class _FakeClock:
    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.now = start
        self._actions: list[object] = []

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        for action in list(self._actions):
            action(self.now)

    def add_action(self, action) -> None:
        self._actions.append(action)


class _SequencedTransport:
    def __init__(self, turns: list[ModelTurnResult], clock: _FakeClock) -> None:
        self._turns = list(turns)
        self._clock = clock
        self.call_times: list[float] = []
        self.user_messages: list[str] = []

    def ensure_initialized(self, _initial_user_prompt: str) -> None:
        return None

    def update_system_prompt(self, _system_prompt: str) -> None:
        return None

    def append_user_text(self, text: str) -> None:
        self.user_messages.append(text)

    def execute_turn(self, _tool_specs: list[dict[str, object]]) -> ModelTurnResult:
        self.call_times.append(self._clock.monotonic())
        if not self._turns:
            raise AssertionError("No more fake turns configured.")
        return self._turns.pop(0)

    def append_tool_results(self, _tool_results: list[dict[str, object]]) -> None:
        return None


def _scratch_write_turn() -> ModelTurnResult:
    return ModelTurnResult(
        tool_calls=[
            ModelToolCall(
                call_id="call_work",
                name="write_file",
                arguments={"path": "scratch/notes.md", "content": "working\n"},
            )
        ],
        text_blocks=["work locally"],
        stop_reason="tool_use",
    )


def _finish_turn() -> ModelTurnResult:
    return ModelTurnResult(
        tool_calls=[
            ModelToolCall(
                call_id="call_summary",
                name="write_file",
                arguments={"path": "publish/summary.md", "content": "# Summary\n"},
            ),
            ModelToolCall(
                call_id="call_index",
                name="write_file",
                arguments={"path": "publish/artifact_index.md", "content": "# Artifact Index\n"},
            ),
            ModelToolCall(
                call_id="call_final",
                name="write_file",
                arguments={"path": "publish/final.md", "content": "# Final\n\nDone.\n"},
            ),
            ModelToolCall(call_id="call_done", name="set_status", arguments={"status": "done"}),
        ],
        text_blocks=["finish now"],
        stop_reason="tool_use",
    )


def _sleep_turn(seconds: str) -> ModelTurnResult:
    return ModelTurnResult(
        tool_calls=[
            ModelToolCall(
                call_id="call_sleep",
                name="bash",
                arguments={"argv": ["sleep", seconds]},
            )
        ],
        text_blocks=[f"sleep {seconds}"],
        stop_reason="tool_use",
    )


def _install_fake_worker_timing(monkeypatch: pytest.MonkeyPatch, clock: _FakeClock) -> None:
    monkeypatch.setattr(agent_worker_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(agent_worker_module.time, "time", clock.time)
    monkeypatch.setattr(agent_worker_module.time, "sleep", clock.sleep)


def _install_fake_transport(monkeypatch: pytest.MonkeyPatch, transport: _SequencedTransport) -> None:
    monkeypatch.setattr(agent_worker_module, "get_model", lambda *_args, **_kwargs: "minimax/MiniMax-M2.7")
    monkeypatch.setattr(agent_worker_module, "resolve_agent_transport", lambda *_args, **_kwargs: "minimax_anthropic")
    monkeypatch.setattr(agent_worker_module, "create_transport", lambda **_kwargs: transport)


def test_worker_first_turn_is_immediate_and_self_writes_do_not_wake_early(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, _request = _create_workspace(tmp_path, monkeypatch)
    clock = _FakeClock()
    start = clock.monotonic()
    transport = _SequencedTransport([_scratch_write_turn(), _finish_turn()], clock)

    _install_fake_worker_timing(monkeypatch, clock)
    _install_fake_transport(monkeypatch, transport)
    monkeypatch.setattr(agent_worker_module, "TURN_MAX_PROMPT_GAP_SECONDS", 6.0)
    monkeypatch.setattr(agent_worker_module, "TURN_PACING_POLL_SECONDS", 1.0)
    monkeypatch.setattr(agent_worker_module, "remaining_agent_seconds", lambda *_args, **_kwargs: 9999)

    result = run_agent_worker(str(run_root), agent_id)

    assert result == 0
    assert transport.call_times == [start, start + 6.0]


def test_worker_wakes_early_when_direct_child_status_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, request = _create_workspace(tmp_path, monkeypatch)
    child_id = "agent_child"
    create_agent_workspace(
        run_root,
        agent_id=child_id,
        parent_id=agent_id,
        preset="research",
        task_name="Child Task",
        description="Wait for child progress",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=[],
        ),
    )
    child_paths = agent_workspace_paths(run_root, child_id)
    clock = _FakeClock()
    transport = _SequencedTransport([_scratch_write_turn(), _finish_turn()], clock)

    def _complete_child(now: float) -> None:
        if not transport.call_times or now < transport.call_times[0] + 4.0 or read_status(run_root, child_id) == "done":
            return
        write_status(run_root, child_id, "done")
        os.utime(child_paths["status"], (now, now))

    _install_fake_worker_timing(monkeypatch, clock)
    _install_fake_transport(monkeypatch, transport)
    monkeypatch.setattr(agent_worker_module, "TURN_MAX_PROMPT_GAP_SECONDS", 10.0)
    monkeypatch.setattr(agent_worker_module, "TURN_PACING_POLL_SECONDS", 1.0)
    monkeypatch.setattr(agent_worker_module, "remaining_agent_seconds", lambda *_args, **_kwargs: 9999)
    clock.add_action(_complete_child)

    result = run_agent_worker(str(run_root), agent_id)

    assert result == 0
    assert transport.call_times[1] == transport.call_times[0] + 4.0


def test_worker_wakes_early_when_commenter_adds_new_feedback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, _request = _create_workspace(tmp_path, monkeypatch)
    clock = _FakeClock()
    transport = _SequencedTransport([_scratch_write_turn(), _finish_turn()], clock)

    def _add_comment(now: float) -> None:
        if not transport.call_times or now < transport.call_times[0] + 4.0:
            return
        unread, total = unread_commenter_comments(run_root, agent_id)
        if total:
            return
        append_commenter_comments(
            run_root,
            agent_id,
            ["New outside-angle comment."],
            generated_at=_iso_from_epoch(now),
        )

    _install_fake_worker_timing(monkeypatch, clock)
    _install_fake_transport(monkeypatch, transport)
    monkeypatch.setattr(agent_worker_module, "TURN_MAX_PROMPT_GAP_SECONDS", 10.0)
    monkeypatch.setattr(agent_worker_module, "TURN_PACING_POLL_SECONDS", 1.0)
    monkeypatch.setattr(agent_worker_module, "remaining_agent_seconds", lambda *_args, **_kwargs: 9999)
    clock.add_action(_add_comment)

    result = run_agent_worker(str(run_root), agent_id)

    assert result == 0
    assert transport.call_times[1] == transport.call_times[0] + 4.0


def test_worker_wakes_early_when_soft_time_limit_activates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, _request = _create_workspace(tmp_path, monkeypatch)
    clock = _FakeClock()
    transport = _SequencedTransport([_scratch_write_turn(), _finish_turn()], clock)

    _install_fake_worker_timing(monkeypatch, clock)
    _install_fake_transport(monkeypatch, transport)
    monkeypatch.setattr(agent_worker_module, "TURN_MAX_PROMPT_GAP_SECONDS", 10.0)
    monkeypatch.setattr(agent_worker_module, "TURN_PACING_POLL_SECONDS", 1.0)
    monkeypatch.setattr(
        agent_worker_module,
        "remaining_agent_seconds",
        lambda *_args, **_kwargs: (
            0
            if transport.call_times and clock.monotonic() >= transport.call_times[0] + 4.0
            else 9999
        ),
    )

    result = run_agent_worker(str(run_root), agent_id)

    assert result == 0
    assert transport.call_times[1] == transport.call_times[0] + 4.0


def test_worker_soft_stop_second_prompt_contains_finalization_guidance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, _request = _create_workspace(tmp_path, monkeypatch)
    clock = _FakeClock()
    transport = _SequencedTransport([_scratch_write_turn(), _finish_turn()], clock)

    _install_fake_worker_timing(monkeypatch, clock)
    _install_fake_transport(monkeypatch, transport)
    monkeypatch.setattr(agent_worker_module, "TURN_MAX_PROMPT_GAP_SECONDS", 10.0)
    monkeypatch.setattr(agent_worker_module, "TURN_PACING_POLL_SECONDS", 1.0)
    monkeypatch.setattr(
        agent_worker_module,
        "remaining_agent_seconds",
        lambda *_args, **_kwargs: (
            0
            if transport.call_times and clock.monotonic() >= transport.call_times[0] + 4.0
            else 9999
        ),
    )

    result = run_agent_worker(str(run_root), agent_id)

    shared_text = (
        "Soft-stop mode is active. Stop exploration and do not open new workstreams unless strictly necessary."
    )
    root_text = (
        "Spend the remaining turns improving publish/final.md, publish/summary.md, "
        "and publish/artifact_index.md so they stay readable if execution stops at any time."
    )

    assert result == 0
    assert len(transport.user_messages) == 2
    assert shared_text not in transport.user_messages[0]
    assert root_text not in transport.user_messages[0]
    assert shared_text in transport.user_messages[1]
    assert root_text in transport.user_messages[1]


def test_worker_calls_immediately_when_previous_turn_already_used_full_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, _request = _create_workspace(tmp_path, monkeypatch)
    clock = _FakeClock()
    start = clock.monotonic()
    transport = _SequencedTransport([_sleep_turn("7"), _finish_turn()], clock)

    _install_fake_worker_timing(monkeypatch, clock)
    _install_fake_transport(monkeypatch, transport)
    monkeypatch.setattr(agent_worker_module, "TURN_MAX_PROMPT_GAP_SECONDS", 6.0)
    monkeypatch.setattr(agent_worker_module, "TURN_PACING_POLL_SECONDS", 1.0)
    monkeypatch.setattr(agent_worker_module, "remaining_agent_seconds", lambda *_args, **_kwargs: 9999)

    result = run_agent_worker(str(run_root), agent_id)

    assert result == 0
    assert transport.call_times == [start, start + 7.0]
