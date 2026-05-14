"""Storage helpers for markdown-first company research state."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Iterable, TypeVar

from pydantic import BaseModel

from src.research_platform.state.contracts import (
    ConflictsFile,
    EvidenceIndex,
    OpenQuestionsFile,
    ProposalRecord,
    SectionMetadata,
    StateIndex,
    ValuationSnapshot,
    utc_now_iso,
)

SECTION_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
SECTION_KEY_RE = re.compile(r"^<!--\s*key:\s*([a-z][a-z0-9_]*)\s*-->\s*$", re.MULTILINE)
CITE_TAG_RE = re.compile(r"\[(S[1-9][0-9]*)\]")
T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class MarkdownSection:
    section_key: str
    heading: str
    start: int
    end: int
    body_start: int
    body: str
    cite_keys: list[str]


@dataclass(frozen=True)
class StatePaths:
    root: Path

    @property
    def research_state(self) -> Path:
        return self.root / "research_state.md"

    @property
    def state_index(self) -> Path:
        return self.root / "state_index.json"

    @property
    def evidence_index(self) -> Path:
        return self.root / "evidence_index.json"

    @property
    def open_questions(self) -> Path:
        return self.root / "open_questions.json"

    @property
    def conflicts(self) -> Path:
        return self.root / "conflicts.json"

    @property
    def valuation_snapshot(self) -> Path:
        return self.root / "valuation_snapshot.json"

    @property
    def diff_log(self) -> Path:
        return self.root / "diff_log.jsonl"


def state_root(vault_root: str | Path, ticker: str) -> Path:
    return Path(vault_root) / "companies" / ticker.upper() / "research" / "state"


def state_paths(vault_root: str | Path, ticker: str) -> StatePaths:
    return StatePaths(state_root(vault_root, ticker))


def default_research_state_markdown(ticker: str) -> str:
    ticker = ticker.upper()
    return (
        f"# {ticker} Research State\n\n"
        "<!-- Headings and key anchors are managed by StateOwner. -->\n"
        "<!-- Cite tags like [S1] resolve via evidence_index.json. -->\n\n"
        "## Current research summary\n"
        "<!-- key: current_research_summary -->\n\n"
        "No durable summary recorded yet.\n\n"
        "## Open questions\n"
        "<!-- key: open_questions -->\n"
        "<!-- derived: rendered from open_questions.json -->\n\n"
        "No open questions recorded.\n\n"
        "## Conflicts / uncertainty\n"
        "<!-- key: conflicts_uncertainty -->\n"
        "<!-- derived: rendered from conflicts.json -->\n\n"
        "No conflicts recorded.\n"
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def write_json_model(path: Path, model: BaseModel) -> None:
    _atomic_write_text(path, model.model_dump_json(indent=2) + "\n")


def read_json_model(path: Path, model_type: type[T], default: T) -> T:
    if not path.exists():
        return default
    return model_type.model_validate(json.loads(path.read_text(encoding="utf-8")))


def initialize_state_folder(vault_root: str | Path, ticker: str) -> StatePaths:
    paths = state_paths(vault_root, ticker)
    paths.root.mkdir(parents=True, exist_ok=True)
    if not paths.research_state.exists():
        _atomic_write_text(paths.research_state, default_research_state_markdown(ticker))
    write_json_model(paths.state_index, build_state_index(paths.research_state.read_text(encoding="utf-8"), ticker=ticker))
    if not paths.evidence_index.exists():
        write_json_model(paths.evidence_index, EvidenceIndex(ticker=ticker))
    if not paths.open_questions.exists():
        write_json_model(paths.open_questions, OpenQuestionsFile(ticker=ticker))
    if not paths.conflicts.exists():
        write_json_model(paths.conflicts, ConflictsFile(ticker=ticker))
    if not paths.valuation_snapshot.exists():
        write_json_model(paths.valuation_snapshot, ValuationSnapshot(ticker=ticker))
    if not paths.diff_log.exists():
        paths.diff_log.write_text("", encoding="utf-8")
    return paths


def parse_markdown_sections(markdown: str) -> list[MarkdownSection]:
    headings = list(SECTION_HEADING_RE.finditer(markdown))
    sections: list[MarkdownSection] = []
    for index, match in enumerate(headings):
        start = match.start()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        block = markdown[start:end]
        heading = match.group(1).strip()
        key_match = SECTION_KEY_RE.search(block)
        if not key_match:
            continue
        section_key = key_match.group(1)
        body_start = start + key_match.end()
        body = markdown[body_start:end].strip("\n")
        cite_keys = sorted(set(CITE_TAG_RE.findall(body)))
        sections.append(
            MarkdownSection(
                section_key=section_key,
                heading=heading,
                start=start,
                end=end,
                body_start=body_start,
                body=body,
                cite_keys=cite_keys,
            )
        )
    return sections


def build_state_index(markdown: str, *, ticker: str, run_id: str | None = None) -> StateIndex:
    sections = [
        SectionMetadata(
            section_key=section.section_key,
            heading=section.heading,
            byte_range=(section.start, section.end),
            last_updated_run_id=run_id,
            cite_keys=section.cite_keys,
        )
        for section in parse_markdown_sections(markdown)
    ]
    return StateIndex(ticker=ticker, last_run_id=run_id, sections=sections)


def read_state_sidecars(paths: StatePaths, ticker: str) -> tuple[StateIndex, EvidenceIndex, OpenQuestionsFile, ConflictsFile, ValuationSnapshot]:
    return (
        read_json_model(paths.state_index, StateIndex, StateIndex(ticker=ticker)),
        read_json_model(paths.evidence_index, EvidenceIndex, EvidenceIndex(ticker=ticker)),
        read_json_model(paths.open_questions, OpenQuestionsFile, OpenQuestionsFile(ticker=ticker)),
        read_json_model(paths.conflicts, ConflictsFile, ConflictsFile(ticker=ticker)),
        read_json_model(paths.valuation_snapshot, ValuationSnapshot, ValuationSnapshot(ticker=ticker)),
    )


def write_research_state(paths: StatePaths, markdown: str, *, ticker: str, run_id: str | None = None) -> StateIndex:
    _atomic_write_text(paths.research_state, markdown)
    index = build_state_index(markdown, ticker=ticker, run_id=run_id)
    index.last_updated_at = utc_now_iso()
    write_json_model(paths.state_index, index)
    return index


def snapshot_state(paths: StatePaths, destination: str | Path) -> Path:
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for source in [
        paths.research_state,
        paths.state_index,
        paths.evidence_index,
        paths.open_questions,
        paths.conflicts,
        paths.valuation_snapshot,
    ]:
        if source.exists():
            shutil.copy2(source, destination / source.name)
    return destination


def restore_state_snapshot(paths: StatePaths, snapshot: str | Path) -> None:
    snapshot = Path(snapshot)
    for name in [
        "research_state.md",
        "state_index.json",
        "evidence_index.json",
        "open_questions.json",
        "conflicts.json",
        "valuation_snapshot.json",
    ]:
        source = snapshot / name
        if source.exists():
            shutil.copy2(source, paths.root / name)


def read_proposals(path: str | Path) -> list[ProposalRecord]:
    path = Path(path)
    proposals: list[ProposalRecord] = []
    if not path.exists():
        return proposals
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            proposals.append(ProposalRecord.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"Invalid proposal at {path}:{line_number}: {exc}") from exc
    return proposals


def write_jsonl(path: str | Path, records: Iterable[BaseModel | dict[str, object]]) -> None:
    lines: list[str] = []
    for record in records:
        if isinstance(record, BaseModel):
            payload = record.model_dump(mode="json")
        else:
            payload = record
        lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    _atomic_write_text(Path(path), "\n".join(lines) + ("\n" if lines else ""))


def append_jsonl(path: str | Path, record: BaseModel | dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(record, BaseModel):
        payload = record.model_dump(mode="json")
    else:
        payload = record
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
