from __future__ import annotations

from pathlib import Path

from src.research_platform.memo_artifacts import (
    FINAL_BYTE_FLOOR,
    collect_memo_artifacts,
    copy_product_artifacts,
)


def _root_publish(run_root: Path, *, final_text: str | None = None, source_use_table: str | None = None) -> Path:
    publish = run_root / "agents" / "agent_root" / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    if final_text is not None:
        (publish / "final.md").write_text(final_text, encoding="utf-8")
    if source_use_table is not None:
        (publish / "source_use_table.md").write_text(source_use_table, encoding="utf-8")
    return publish


def test_collect_artifacts_requires_root_final_and_source_use_table(tmp_path):
    run_root = tmp_path / "run"
    final_text = "# Final memo\n\n" + ("Evidence-backed paragraph citing S1 and S2. " * 30)
    _root_publish(
        run_root,
        final_text=final_text,
        source_use_table="| Claim | Source |\n| --- | --- |\n| Revenue grew | S1 |\n",
    )
    tool_log = run_root / "agents" / "agent_root" / "_harness" / "logs" / "tool_calls.jsonl"
    tool_log.parent.mkdir(parents=True, exist_ok=True)
    tool_log.write_text('{"tool":"search"}\n', encoding="utf-8")

    artifacts = collect_memo_artifacts(run_root=run_root)

    assert artifacts.has_usable_final is True
    assert artifacts.has_required_artifacts is True
    assert artifacts.paths.root_final_path is not None
    assert artifacts.paths.root_source_use_table_path is not None
    assert artifacts.paths.harness_tool_calls_path == str(tool_log)
    assert artifacts.missing_required == []


def test_collect_artifacts_treats_tiny_final_as_missing(tmp_path):
    run_root = tmp_path / "run"
    _root_publish(run_root, final_text="too small", source_use_table="| Claim | Source |\n| --- | --- |\n| A | S1 |\n")

    artifacts = collect_memo_artifacts(run_root=run_root)

    assert artifacts.has_usable_final is False
    assert "final.md" in artifacts.missing_required
    assert any(f"<{FINAL_BYTE_FLOOR}" in reason.message for reason in artifacts.reasons)


def test_copy_product_artifacts_is_the_final_copier(tmp_path):
    run_root = tmp_path / "run"
    memo_dir = tmp_path / "memo"
    final_text = "# Final memo\n\n" + ("Durable final content tied to source table. " * 30)
    _root_publish(
        run_root,
        final_text=final_text,
        source_use_table="| Claim | Source |\n| --- | --- |\n| Margin claim | S2 |\n",
    )

    artifacts = collect_memo_artifacts(run_root=run_root)
    copied = copy_product_artifacts(artifacts=artifacts, memo_dir=memo_dir)

    assert copied.paths.product_final_path == str(memo_dir / "final.md")
    assert copied.paths.product_source_use_table_path == str(memo_dir / "source_use_table.md")
    assert (memo_dir / "final.md").read_text(encoding="utf-8") == final_text
    assert "Margin claim" in (memo_dir / "source_use_table.md").read_text(encoding="utf-8")
