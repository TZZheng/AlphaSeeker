from __future__ import annotations

from pathlib import Path


def test_memo_prompt_requires_source_use_table_but_not_evaluator_child():
    text = Path("src/research_platform/prompts/memo_user.md").read_text(encoding="utf-8")

    assert "publish/source_use_table.md" in text
    assert "system-owned post-harness evaluator" in text
    assert "Do not spawn" in text and "evaluator child" in text
    assert "absolute URL" in text
    assert "vault-relative" in text
