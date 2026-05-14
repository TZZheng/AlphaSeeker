"""Validation gate for v3.3 staged direct-edit research state updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from pydantic import ValidationError

from src.research_platform.state.contracts import ConflictsFile, EvidenceIndex, OpenQuestionsFile, StateIndex, ValuationSnapshot, utc_now_iso
from src.research_platform.state.lint import number_without_cite_warnings
from src.research_platform.state.render import render_conflicts, render_open_questions, render_valuation_snapshot
from src.research_platform.state.staging import StageDiff, StagePaths, compute_stage_diff
from src.research_platform.state.storage import (
    MarkdownSection,
    build_state_index,
    parse_markdown_sections,
    read_json_model,
    write_json_model,
)

DERIVED_SECTIONS = {
    "open_questions": render_open_questions,
    "valuation_snapshot": render_valuation_snapshot,
    "conflicts_uncertainty": render_conflicts,
}
SECTION_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
SECTION_KEY_LINE_RE = re.compile(r"^<!--\s*key:\s*([a-z][a-z0-9_]*)\s*-->\s*$", re.MULTILINE)
CITE_TAG_RE = re.compile(r"\[(S[1-9][0-9]*)\]")


@dataclass
class StageValidationReport:
    """Validation report for one staged StateOwner edit."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diff: StageDiff | None = None
    validated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.diff is not None:
            payload["diff"] = self.diff.to_dict()
        return payload


def _load_json(path: Path, model_type, default):
    try:
        return read_json_model(path, model_type, default), None
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        return default, f"invalid {path.name}: {exc}"


def _section_block(heading: str, section_key: str, body: str) -> str:
    body = body.strip("\n")
    return f"## {heading}\n<!-- key: {section_key} -->\n\n{body}\n"


def _find_section(sections: list[MarkdownSection], section_key: str) -> MarkdownSection | None:
    return next((section for section in sections if section.section_key == section_key), None)


def _replace_section(markdown: str, section: MarkdownSection, new_body: str) -> str:
    block = _section_block(section.heading, section.section_key, new_body)
    return markdown[: section.start] + block + "\n" + markdown[section.end :].lstrip("\n")


def _replace_or_create_section(markdown: str, section_key: str, heading: str, body: str) -> str:
    sections = parse_markdown_sections(markdown)
    section = _find_section(sections, section_key)
    if section:
        return _replace_section(markdown, section, body)
    suffix = "" if markdown.endswith("\n") else "\n"
    return markdown + suffix + "\n" + _section_block(heading, section_key, body)


def _rerender_derived_sections(stage_paths: StagePaths, ticker: str) -> None:
    markdown = stage_paths.stage.research_state.read_text(encoding="utf-8")
    open_questions = read_json_model(stage_paths.stage.open_questions, OpenQuestionsFile, OpenQuestionsFile(ticker=ticker))
    conflicts = read_json_model(stage_paths.stage.conflicts, ConflictsFile, ConflictsFile(ticker=ticker))
    valuation = read_json_model(stage_paths.stage.valuation_snapshot, ValuationSnapshot, ValuationSnapshot(ticker=ticker))
    replacements = {
        "open_questions": ("Open questions", render_open_questions(open_questions)),
        "valuation_snapshot": ("Valuation snapshot", render_valuation_snapshot(valuation)),
        "conflicts_uncertainty": ("Conflicts / uncertainty", render_conflicts(conflicts)),
    }
    for key, (heading, body) in replacements.items():
        markdown = _replace_or_create_section(markdown, key, heading, body)
    stage_paths.stage.research_state.write_text(markdown, encoding="utf-8")


def _anchor_errors(markdown: str) -> list[str]:
    errors: list[str] = []
    headings = list(SECTION_HEADING_RE.finditer(markdown))
    keys: list[str] = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        block = markdown[heading.start() : end]
        key_match = SECTION_KEY_LINE_RE.search(block)
        if not key_match:
            errors.append(f"section heading lacks key anchor: {heading.group(1).strip()}")
            continue
        keys.append(key_match.group(1))
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    for key in duplicate_keys:
        errors.append(f"duplicate section key: {key}")
    return errors


def _readonly_file_errors(stage_paths: StagePaths) -> list[str]:
    errors: list[str] = []
    for name in ["evidence_index.json", "state_index.json"]:
        baseline = stage_paths.baseline.root / name
        staged = stage_paths.stage.root / name
        if baseline.exists() and staged.exists() and baseline.read_text(encoding="utf-8") != staged.read_text(encoding="utf-8"):
            errors.append(f"read-only state file was modified: {name}")
        elif baseline.exists() != staged.exists():
            errors.append(f"read-only state file presence changed: {name}")
    return errors


def _diff_volume_errors(diff: StageDiff, *, max_removed_sections: int, max_negative_bytes: int) -> list[str]:
    errors: list[str] = []
    if len(diff.sections_removed) > max_removed_sections:
        errors.append(f"stage removed too many sections: {len(diff.sections_removed)} > {max_removed_sections}")
    if diff.bytes_delta < -max_negative_bytes:
        errors.append(f"stage removed too many bytes: {diff.bytes_delta} < -{max_negative_bytes}")
    return errors


def validate_stage(
    stage_paths: StagePaths,
    *,
    ticker: str,
    run_id: str | None = None,
    hard_fail_uncited_numbers: bool = True,
    max_removed_sections: int = 2,
    max_negative_bytes: int = 50_000,
) -> StageValidationReport:
    """Validate and normalize a staged direct edit before committing it live.

    This function mutates the stage in deterministic ways only: derived markdown
    sections are re-rendered from sidecars and ``state_index.json`` is rebuilt
    from the staged markdown.
    """

    errors: list[str] = []
    warnings: list[str] = []
    ticker = ticker.upper()

    if not stage_paths.stage.research_state.exists():
        return StageValidationReport(ok=False, errors=[f"missing {stage_paths.stage.research_state}"])

    evidence_index, error = _load_json(stage_paths.stage.evidence_index, EvidenceIndex, EvidenceIndex(ticker=ticker))
    if error:
        errors.append(error)
    for path, model_type, default in [
        (stage_paths.stage.open_questions, OpenQuestionsFile, OpenQuestionsFile(ticker=ticker)),
        (stage_paths.stage.conflicts, ConflictsFile, ConflictsFile(ticker=ticker)),
        (stage_paths.stage.valuation_snapshot, ValuationSnapshot, ValuationSnapshot(ticker=ticker)),
    ]:
        _, error = _load_json(path, model_type, default)
        if error:
            errors.append(error)

    errors.extend(_readonly_file_errors(stage_paths))
    if errors:
        diff = compute_stage_diff(stage_paths)
        return StageValidationReport(ok=False, errors=errors, warnings=warnings, diff=diff)

    _rerender_derived_sections(stage_paths, ticker)
    markdown = stage_paths.stage.research_state.read_text(encoding="utf-8")
    errors.extend(_anchor_errors(markdown))

    unresolved = sorted(set(CITE_TAG_RE.findall(markdown)) - set(evidence_index.entries))
    for key in unresolved:
        errors.append(f"unresolved cite key: {key}")

    uncited_numbers = number_without_cite_warnings(markdown)
    if uncited_numbers:
        messages = [f"quantitative-looking sentence lacks same-sentence cite: {item}" for item in uncited_numbers]
        if hard_fail_uncited_numbers:
            errors.extend(messages)
        else:
            warnings.extend(messages)

    state_index = build_state_index(markdown, ticker=ticker, run_id=run_id)
    state_index.last_updated_at = utc_now_iso()
    write_json_model(stage_paths.stage.state_index, state_index)

    diff = compute_stage_diff(stage_paths)
    errors.extend(_diff_volume_errors(diff, max_removed_sections=max_removed_sections, max_negative_bytes=max_negative_bytes))

    return StageValidationReport(ok=not errors, errors=errors, warnings=warnings, diff=diff)
