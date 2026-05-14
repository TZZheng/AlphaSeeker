"""V1 research-platform memo orchestration.

This module stitches together deterministic source acquisition, vault ingestion,
context-package rendering, and the existing AlphaSeeker harness.  It is a thin
vertical slice: durable source documents go through the vault, while v1 run
lifecycle artifacts remain JSON/Markdown files under the company research tree.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Any, Callable
import uuid

from pydantic import BaseModel, ConfigDict, Field

from src.harness.context_types import Caveat, Citation, MemoContextPackage, OutputStatus
from src.harness.runtime import run_harness
from src.harness.types import HarnessRequest, HarnessResponse
from src.retrieval.types import RetrievalBatch, RetrievalRequest, ResearchTask, SourceRecord
from src.tools.equity.company_profile import fetch_company_profile
from src.tools.equity.financials import fetch_financial_metrics
from src.tools.equity.market_data import fetch_historical_data
from src.tools.equity.sec_filings import search_and_read_filings
from src.vault.contracts import DocumentRef, EvidenceRef, QuestionRecord
from src.vault.ingest import ingest_file, ingest_text
from src.research_platform.state.apply import apply_proposals
from src.research_platform.state.contracts import EvidenceIndex
from src.research_platform.state.lint import validate_state_integrity
from src.research_platform.state.storage import initialize_state_folder, read_json_model, restore_state_snapshot, snapshot_state, write_json_model
from src.vault.paths import default_vault_paths
from src.vault.store import VaultStore

PROMPT_TEMPLATE_PATH = Path(__file__).with_name("prompts") / "memo_user.md"
MANUAL_GRADE_RATIONALE = "manual file, provenance not machine-verified"
SEC_GRADE_RATIONALE = "SEC EDGAR filing, issuer-filed primary source"
VENDOR_GRADE_RATIONALE = "vendor/API snapshot, useful but not primary-source verified"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return slug.strip("-._") or uuid.uuid4().hex[:8]


class ResearchPlatformRunResult(BaseModel):
    """Public result for one research-platform memo run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    ticker: str
    status: OutputStatus
    memo_dir: str
    manifest_path: str
    status_path: str
    source_index_path: str | None = None
    question_list_path: str | None = None
    context_package_path: str | None = None
    final_path: str | None = None
    research_state_path: str | None = None
    state_snapshot_path: str | None = None
    proposals_path: str | None = None
    accepted_changes_path: str | None = None
    harness_response: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class SourceAdapters:
    """Injectable source-provider functions for live runs and mocked tests."""

    sec_filings: Callable[..., list[dict[str, Any]]]
    company_profile: Callable[..., tuple[str, dict[str, Any]]]
    financial_metrics: Callable[..., tuple[str, dict[str, Any]]]
    historical_data: Callable[..., str]


def default_source_adapters() -> SourceAdapters:
    return SourceAdapters(
        sec_filings=search_and_read_filings,
        company_profile=fetch_company_profile,
        financial_metrics=fetch_financial_metrics,
        historical_data=fetch_historical_data,
    )


def strict_render_template(template: str, values: dict[str, Any]) -> str:
    """Render ``{{name}}`` placeholders and fail if any remain unresolved."""

    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    leftovers = sorted(set(re.findall(r"{{\s*[^{}]+\s*}}", rendered)))
    if leftovers:
        raise ValueError(f"Unresolved prompt template placeholders: {', '.join(leftovers)}")
    return rendered


def _read_json_metadata(document: dict[str, Any]) -> dict[str, Any]:
    raw = document.get("metadata_json") or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _vault_relative_path(path_value: str | Path, vault_root: Path) -> str:
    path = Path(path_value)
    try:
        if path.is_absolute():
            return path.resolve().relative_to(vault_root.resolve()).as_posix()
        return path.relative_to(vault_root).as_posix()
    except Exception:
        return path.as_posix()


def _document_ref_from_ingest(
    result: dict[str, Any],
    *,
    source_id: str,
    source_type: str,
    vault_root: Path,
) -> DocumentRef:
    document = result["document"]
    metadata = _read_json_metadata(document)
    extracted_path = document.get("path") or result.get("extracted_path")
    display_title = document.get("title") or source_id
    return DocumentRef(
        document_id=document["doc_id"],
        source_id=source_id,
        source_type=source_type,
        title=display_title,
        display_title=display_title,
        vault_relative_path=_vault_relative_path(extracted_path, vault_root),
        extracted_text_path=_vault_relative_path(extracted_path, vault_root),
        source_grade=document.get("source_grade") or "unknown",
        source_grade_rationale=metadata.get("source_grade_rationale", ""),
        checksum=document.get("checksum"),
        url=document.get("url"),
        published_at=document.get("published_at"),
        ingested_at=document.get("ingested_at") or _utc_now_iso(),
        metadata=metadata,
    )


def _snippet_for_document(document: DocumentRef, vault_root: Path, limit: int = 360) -> str:
    path = vault_root / document.extracted_text_path if document.extracted_text_path else vault_root / document.vault_relative_path
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        text = ""
    text = re.sub(r"\s+", " ", text)
    return text[:limit] or document.display_title


def _citation_for_document(
    index: int,
    document: DocumentRef,
    vault_root: Path,
    *,
    harness_context_file: str | None = None,
) -> Citation:
    metadata = {"source_grade_rationale": document.source_grade_rationale}
    if harness_context_file:
        metadata["harness_context_file"] = harness_context_file
    return Citation(
        citation_key=f"S{index}",
        document_id=document.document_id,
        vault_relative_path=document.vault_relative_path,
        display_title=document.display_title,
        snippet=_snippet_for_document(document, vault_root),
        source_grade=document.source_grade,
        metadata=metadata,
    )


def _write_harness_source_attachments(
    *,
    documents: list[DocumentRef],
    vault_root: Path,
    memo_dir: Path,
) -> dict[str, Path]:
    """Copy vault source bodies to uniquely named files visible to harness agents."""

    attachment_dir = memo_dir / "context_docs"
    attachment_dir.mkdir(parents=True, exist_ok=True)
    attachments: dict[str, Path] = {}
    for index, document in enumerate(documents, start=1):
        relative_path = document.extracted_text_path or document.vault_relative_path
        source_path = vault_root / relative_path
        suffix = source_path.suffix or ".md"
        filename = f"S{index}_{_slug(document.source_type)}_{_slug(document.document_id)}{suffix}"
        destination = attachment_dir / filename
        shutil.copy2(source_path, destination)
        attachments[document.document_id] = destination
    return attachments


def _select_sec_filings(raw_filings: list[dict[str, Any]], warnings: list[str]) -> list[dict[str, Any]]:
    def form(filing: dict[str, Any]) -> str:
        return str(filing.get("form_type") or filing.get("form") or "").upper()

    def filed_at(filing: dict[str, Any]) -> str:
        return str(filing.get("filing_date") or filing.get("filed_at") or filing.get("published_at") or "")

    def key(filing: dict[str, Any]) -> str:
        return filed_at(filing)[:10]

    tens_k = sorted([f for f in raw_filings if "10-K" in form(f)], key=key, reverse=True)
    tens_q = sorted([f for f in raw_filings if "10-Q" in form(f)], key=key, reverse=True)
    selected: list[dict[str, Any]] = []
    latest_k = tens_k[0] if tens_k else None
    latest_q = tens_q[0] if tens_q else None
    if latest_k:
        selected.append(latest_k)
    if latest_q:
        if latest_k and key(latest_k) and key(latest_q):
            if key(latest_q) > key(latest_k):
                selected.append(latest_q)
        elif latest_k:
            warnings.append("SEC 10-Q filing date missing; included latest 10-K only.")
        else:
            selected.append(latest_q)
    return selected


def _build_source_record(
    *,
    source_id: str,
    source_type: str,
    title: str,
    ticker: str,
    local_path: str | None = None,
    url: str | None = None,
    published_at: str | None = None,
    grade: str,
    rationale: str,
    method: str = "deterministic",
    document: DocumentRef | None = None,
    metadata: dict[str, Any] | None = None,
) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        source_type=source_type,  # type: ignore[arg-type]
        title=title,
        ticker=ticker,
        local_path=local_path,
        url=url,
        published_at=published_at,
        source_grade=grade,  # type: ignore[arg-type]
        source_grade_rationale=rationale,
        retrieval_method=method,  # type: ignore[arg-type]
        checksum=document.checksum if document else None,
        vault_relative_path=document.vault_relative_path if document else None,
        display_title=document.display_title if document else title,
        metadata=metadata or {},
    )


def _research_paths(vault_root: Path, ticker: str, run_id: str) -> dict[str, Path]:
    research_dir = vault_root / "companies" / ticker / "research"
    memo_dir = research_dir / "memos" / run_id
    return {
        "research_dir": research_dir,
        "memo_dir": memo_dir,
        "context_dir": research_dir / "context_packages",
        "source_index": research_dir / "source_index.md",
        "question_list": research_dir / "question_list.md",
        "context_package": research_dir / "context_packages" / f"{run_id}.json",
        "manifest": memo_dir / "manifest.json",
        "status": memo_dir / "status.json",
        "source_snapshot": memo_dir / "source_index.md",
        "question_snapshot": memo_dir / "question_list.md",
        "final": memo_dir / "final.md",
        "state_snapshot": memo_dir / "state_snapshot",
        "proposal_protocol": memo_dir / "proposal_protocol.md",
        "proposals": memo_dir / "proposals.jsonl",
        "accepted_changes": memo_dir / "accepted_changes.jsonl",
    }


def render_source_index(
    ticker: str,
    documents: list[DocumentRef],
    citations: list[Citation],
    attachment_paths: dict[str, Path] | None = None,
) -> str:
    lines = [f"# {ticker} Source Index", "", f"Last updated: {_utc_now_iso()}", ""]
    if not documents:
        lines.append("No sources ingested for this run.")
        return "\n".join(lines).strip() + "\n"
    attachment_paths = attachment_paths or {}
    by_doc = {citation.document_id: citation for citation in citations}
    for doc in documents:
        citation = by_doc.get(doc.document_id)
        lines.extend(
            [
                f"## {citation.citation_key if citation else doc.document_id} — {doc.display_title}",
                f"- Document ID: `{doc.document_id}`",
                f"- Source ID: `{doc.source_id or ''}`",
                f"- Type: `{doc.source_type}`",
                f"- Grade: `{doc.source_grade}` — {doc.source_grade_rationale or 'no rationale recorded'}",
                f"- Published: {doc.published_at or 'unknown'}",
                f"- Vault path: [[{doc.vault_relative_path}]]",
                f"- Attached context file: `{attachment_paths[doc.document_id].name}`"
                if doc.document_id in attachment_paths
                else "- Attached context file: n/a",
                f"- URL: {doc.url or 'n/a'}",
                f"- Checksum: `{doc.checksum or 'n/a'}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def render_question_list(ticker: str, caveats: list[Caveat]) -> str:
    lines = [f"# {ticker} Open Questions", "", f"Last updated: {_utc_now_iso()}", ""]
    if not caveats:
        lines.append("No open v1 caveats were generated by deterministic retrieval.")
    else:
        for caveat in caveats:
            lines.append(f"- [ ] {caveat.message} (`{caveat.caveat_id}`, severity={caveat.severity})")
    return "\n".join(lines).strip() + "\n"


def _summarize_sources(documents: list[DocumentRef]) -> str:
    if not documents:
        return "No curated documents were attached."
    lines = []
    for index, doc in enumerate(documents, start=1):
        lines.append(
            f"- S{index}: {doc.display_title} ({doc.source_type}, grade {doc.source_grade}; "
            f"path `{doc.vault_relative_path}`)"
        )
    return "\n".join(lines)


def assemble_memo_prompt(
    *,
    user_prompt: str,
    ticker: str,
    company_name: str | None,
    run_id: str,
    source_index_path: str,
    question_list_path: str,
    context_package_path: str,
    documents: list[DocumentRef],
    missing_sources: list[str],
    caveats: list[Caveat],
    warnings: list[str],
    template_path: str | Path = PROMPT_TEMPLATE_PATH,
    research_state_block: str | None = None,
) -> str:
    template = Path(template_path).read_text(encoding="utf-8")
    company_suffix = f" ({company_name})" if company_name else ""
    missing_block = "None." if not missing_sources else "\n".join(f"- {item}" for item in missing_sources)
    caveat_lines = [c.message for c in caveats] + warnings
    caveats_block = "None." if not caveat_lines else "\n".join(f"- {item}" for item in caveat_lines)
    values = {
        "user_prompt": user_prompt.strip(),
        "ticker": ticker,
        "company_name": company_suffix,
        "run_id": run_id,
        "source_index_path": source_index_path,
        "question_list_path": question_list_path,
        "context_package_path": context_package_path,
        "required_sources_summary": _summarize_sources(documents),
        "missing_sources_block": missing_block,
        "caveats_block": caveats_block,
        "freshness_block": "Use publication dates in the source index; treat vendor snapshots as point-in-time B-grade context.",
        "citation_usage_instructions": "Use citation keys S1, S2, ... from the source index and preserve their mapping to vault paths.",
    }
    rendered = strict_render_template(template, values)
    if research_state_block:
        rendered = rendered.rstrip() + "\n\n" + research_state_block.strip() + "\n"
    if len(rendered) > 32_000:
        raise ValueError("Assembled memo prompt exceeds v1 32KB inline prompt cap")
    return rendered


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _status_payload(
    run_id: str,
    status: OutputStatus,
    *,
    warnings: list[str],
    errors: list[str],
    research_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "run_id": run_id,
        "status": status,
        "finished_at": _utc_now_iso(),
        "warnings": warnings,
        "errors": errors,
    }
    if research_state is not None:
        payload["research_state"] = research_state
    return payload


def _manifest_payload(
    *,
    run_id: str,
    ticker: str,
    user_prompt: str,
    status: OutputStatus,
    documents: list[DocumentRef],
    citations: list[Citation],
    context_package_path: Path | None,
    source_index_path: Path | None,
    question_list_path: Path | None,
    final_path: Path | None,
    warnings: list[str],
    errors: list[str],
    research_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "run_id": run_id,
        "ticker": ticker,
        "user_prompt": user_prompt,
        "status": status,
        "created_at": _utc_now_iso(),
        "source_documents": [doc.model_dump(mode="json") for doc in documents],
        "citations": [citation.model_dump(mode="json") for citation in citations],
        "context_package_path": str(context_package_path) if context_package_path else None,
        "source_index_path": str(source_index_path) if source_index_path else None,
        "question_list_path": str(question_list_path) if question_list_path else None,
        "final_path": str(final_path) if final_path else None,
        "warnings": warnings,
        "errors": errors,
    }
    if research_state is not None:
        payload["research_state"] = research_state
    return payload


def _ingest_vendor_file(
    *,
    source_id: str,
    source_type: str,
    path: str,
    title: str,
    ticker: str,
    root: Path,
    metadata: dict[str, Any] | None = None,
) -> tuple[DocumentRef, SourceRecord]:
    result = ingest_file(
        path,
        ticker=ticker,
        source_type=source_type,
        title=title,
        source_grade="B",
        source_grade_rationale=VENDOR_GRADE_RATIONALE,
        metadata=metadata or {},
        root=root,
    )
    doc = _document_ref_from_ingest(result, source_id=source_id, source_type=source_type, vault_root=root)
    record = _build_source_record(
        source_id=source_id,
        source_type=source_type,
        title=title,
        ticker=ticker,
        local_path=path,
        grade="B",
        rationale=VENDOR_GRADE_RATIONALE,
        document=doc,
        metadata=metadata or {},
    )
    return doc, record


def _copy_final(response: HarnessResponse, destination: Path) -> str | None:
    if not response.final_report_path:
        return None
    source = Path(response.final_report_path)
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def _evidence_index_from_citations(
    ticker: str,
    citations: list[Citation],
    *,
    existing: EvidenceIndex | None = None,
) -> EvidenceIndex:
    entries: dict[str, EvidenceRef] = dict(existing.entries) if existing else {}
    by_evidence_id = {evidence.evidence_id: key for key, evidence in entries.items()}

    def next_key() -> str:
        max_index = 0
        for key in entries:
            if key.startswith("S") and key[1:].isdigit():
                max_index = max(max_index, int(key[1:]))
        return f"S{max_index + 1}"

    for citation in citations:
        evidence_id = citation.evidence_id or f"evidence-{citation.citation_key.lower()}-{citation.document_id}"
        if evidence_id in by_evidence_id:
            continue
        evidence = EvidenceRef(
            evidence_id=evidence_id,
            document_id=citation.document_id,
            vault_relative_path=citation.vault_relative_path,
            display_title=citation.display_title,
            heading_path=list(citation.heading_path),
            quoted_snippet=citation.snippet,
            source_grade=citation.source_grade,  # type: ignore[arg-type]
            metadata=dict(citation.metadata),
        )
        key = citation.citation_key if citation.citation_key not in entries else next_key()
        entries[key] = evidence
        by_evidence_id[evidence_id] = key
    return EvidenceIndex(ticker=ticker, entries=entries)


def _research_state_protocol_text(*, memo_dir: Path, proposals_path: Path) -> str:
    return f"""
# Durable CompanyResearchState update protocol (v3)

Read the attached `research_state.md` and state sidecars before raw source attachments. The markdown state is durable company research memory; `final.md` is this run's memo output.

## Required output

Write the final memo to `publish/final.md` as usual. Separately, write durable state proposals as JSON Lines to:

`{proposals_path}`

The proposal file must be valid JSONL: one JSON object per line, no markdown fences, no trailing comments. If there are no durable state updates, write an empty file at that exact path.

## When to propose durable state updates

Propose an update only for information that should carry into future company research runs: durable thesis points, recurring risks, changed/answered open questions, conflicts between evidence, or a point-in-time valuation snapshot. Do not propose memo-only wording, style edits, or uncited quantitative claims.

## Allowed proposal envelopes

Use only these JSON object shapes:

```jsonl
{{"proposal_id":"p1","type":"propose_section_update","section_key":"guyana_growth_engine","action":"append","body_markdown":"Durable insight with [S1] cite.","evidence_keys":["S1"],"rationale":"Why this belongs in durable state."}}
{{"proposal_id":"p2","type":"propose_question","question_id":"q-new-topic","text":"Question to carry forward?","priority":"normal","related_section_key":"guyana_growth_engine"}}
{{"proposal_id":"p3","type":"propose_close_question","question_id":"q-...","evidence_keys":["S2"],"proposed_answer":"Cited answer.","rationale":"Why this closes the question."}}
{{"proposal_id":"p4","type":"propose_conflict","summary":"Evidence conflict summary.","left_evidence_key":"S1","right_evidence_key":"S2","severity":"medium","rationale":"Why this is a real conflict."}}
{{"proposal_id":"p5","type":"propose_valuation_snapshot","as_of":"YYYY-MM-DD","fields":{{"share_price":100.0,"ev_ebitda":7.5}},"source_keys":["S3"],"assumptions_markdown":"Point-in-time valuation assumptions with [S3]."}}
{{"proposal_id":"p6","type":"propose_no_op","rationale":"No durable state updates found."}}
```

## StateOwner acceptance rules

- Use existing `[S]` citation keys from `evidence_index.json`; do not invent evidence keys.
- `body_markdown` must be section body content only: do not include `##` H2 headings or `<!-- key: ... -->` anchors.
- Direct edits to derived sections are forbidden. Use `propose_question`, `propose_close_question`, `propose_conflict`, or `propose_valuation_snapshot` for those sidecars.
- Every section update, close-question, conflict, or valuation snapshot must include resolved evidence keys and a concrete rationale.
- Quantitative-looking sentences need a same-sentence `[S]` cite.
- Prefer appending to an existing section when the state already has a matching `section_key`; create a new section only for genuinely new topics.
- Keep proposals concise. Soft cap: 20 proposals. Hard cap: 50 proposals.

The proposal file lives under this run directory: `{memo_dir}`.
""".strip() + "\n"


def _research_state_prompt_block(*, memo_dir: Path, proposals_path: Path) -> str:
    return _research_state_protocol_text(memo_dir=memo_dir, proposals_path=proposals_path)


def _research_state_manifest(
    *,
    enabled: bool,
    state_paths: dict[str, str] | None = None,
    apply_counts: dict[str, int] | None = None,
    integrity_errors: list[str] | None = None,
    rolled_back: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"enabled": enabled, "rolled_back": rolled_back}
    if state_paths:
        payload.update(state_paths)
    if apply_counts:
        payload["apply_counts"] = apply_counts
    if integrity_errors is not None:
        payload["integrity_errors"] = integrity_errors
    return payload


def run_research_memo(
    *,
    user_prompt: str,
    ticker: str,
    company_name: str | None = None,
    manual_files: list[str] | None = None,
    vault_root: str | Path | None = None,
    run_id: str | None = None,
    wall_clock_budget_seconds: int = 1200,
    adapters: SourceAdapters | None = None,
    run_harness_fn: Callable[[HarnessRequest], HarnessResponse] | None = None,
    enable_research_state: bool = False,
) -> ResearchPlatformRunResult:
    """Run the research-platform memo pipeline.

    v1 behavior is the default.  When ``enable_research_state`` is true, v3
    markdown-first state files are attached and post-run proposals are applied.
    """

    ticker_norm = ticker.strip().upper()
    if not ticker_norm:
        raise ValueError("ticker is required")
    run_id = _slug(run_id or f"memo-{ticker_norm.lower()}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    vault_paths = default_vault_paths(vault_root).ensure()
    root = vault_paths.root
    paths = _research_paths(root, ticker_norm, run_id)
    if paths["memo_dir"].exists():
        raise FileExistsError(f"Research memo run already exists: {paths['memo_dir']}")
    for path in (paths["research_dir"], paths["memo_dir"], paths["context_dir"]):
        path.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now_iso()
    warnings: list[str] = []
    errors: list[str] = []
    missing_sources: list[str] = []
    documents: list[DocumentRef] = []
    source_records: list[SourceRecord] = []
    caveats: list[Caveat] = []
    adapters = adapters or default_source_adapters()

    task = ResearchTask(
        task_id=run_id,
        ticker=ticker_norm,
        company_name=company_name,
        user_prompt=user_prompt,
        vault_root=str(root),
        manual_source_paths=list(manual_files or []),
        required_source_types=["manual_file", "sec_filing"],
        optional_source_types=["company_profile", "financial_snapshot", "market_snapshot"],
    )
    request = RetrievalRequest(
        request_id=f"retrieval-{run_id}",
        task=task,
        source_types=["manual_file", "sec_filing", "company_profile", "financial_snapshot", "market_snapshot"],
        allow_llm_discovery=False,
        allow_llm_grading=False,
        max_sources_per_type=2,
    )
    store = VaultStore(root)
    store.upsert_company(ticker_norm, name=company_name)

    for manual_path in manual_files or []:
        source_id = f"manual-{_slug(Path(manual_path).stem)}"
        result = ingest_file(
            manual_path,
            ticker=ticker_norm,
            source_type="manual_file",
            source_grade="B",
            source_grade_rationale=MANUAL_GRADE_RATIONALE,
            metadata={"run_id": run_id},
            root=root,
        )
        doc = _document_ref_from_ingest(result, source_id=source_id, source_type="manual_file", vault_root=root)
        documents.append(doc)
        source_records.append(
            _build_source_record(
                source_id=source_id,
                source_type="manual_file",
                title=doc.display_title,
                ticker=ticker_norm,
                local_path=manual_path,
                grade="B",
                rationale=MANUAL_GRADE_RATIONALE,
                method="manual",
                document=doc,
                metadata={"run_id": run_id},
            )
        )

    sec_filings: list[dict[str, Any]] = []
    try:
        raw_filings = adapters.sec_filings(
            company_name or ticker_norm,
            ticker=ticker_norm,
            form_types=["10-K", "10-Q"],
            max_filings=6,
            max_chars_per_filing=20000,
        )
        sec_filings = _select_sec_filings(list(raw_filings or []), warnings)
    except Exception as exc:
        message = f"SEC filing retrieval failed: {exc}"
        warnings.append(message)
        missing_sources.append("sec_filing")
        caveats.append(Caveat(caveat_id="caveat-sec-retrieval", message=message, severity="medium"))
    if not sec_filings and "sec_filing" not in missing_sources:
        missing_sources.append("sec_filing")

    for index, filing in enumerate(sec_filings, start=1):
        form_type = str(filing.get("form_type") or "SEC filing")
        filing_date = str(filing.get("filing_date") or "") or None
        title = f"{ticker_norm} {form_type} {filing_date or ''}".strip()
        source_id = f"sec-{_slug(form_type.lower())}-{filing_date or index}"
        result = ingest_text(
            str(filing.get("text") or ""),
            ticker=ticker_norm,
            source_type="sec_filing",
            title=title,
            source_grade="A",
            source_grade_rationale=SEC_GRADE_RATIONALE,
            url=filing.get("url"),
            published_at=filing_date,
            metadata={"run_id": run_id, "form_type": form_type, "company": filing.get("company")},
            root=root,
        )
        doc = _document_ref_from_ingest(result, source_id=source_id, source_type="sec_filing", vault_root=root)
        documents.append(doc)
        source_records.append(
            _build_source_record(
                source_id=source_id,
                source_type="sec_filing",
                title=title,
                ticker=ticker_norm,
                url=filing.get("url"),
                published_at=filing_date,
                grade="A",
                rationale=SEC_GRADE_RATIONALE,
                document=doc,
                metadata={"form_type": form_type},
            )
        )

    optional_output_dir = paths["memo_dir"] / "retrieved_sources"
    optional_output_dir.mkdir(parents=True, exist_ok=True)
    optional_specs = [
        ("company_profile", "company-profile", "Company profile", adapters.company_profile),
        ("financial_snapshot", "financial-snapshot", "Financial snapshot", adapters.financial_metrics),
    ]
    for source_type, source_id, title, provider in optional_specs:
        try:
            file_path, metadata = provider(ticker_norm, output_dir=optional_output_dir)
            doc, record = _ingest_vendor_file(
                source_id=source_id,
                source_type=source_type,
                path=file_path,
                title=f"{ticker_norm} {title}",
                ticker=ticker_norm,
                root=root,
                metadata={"run_id": run_id, **(metadata or {})},
            )
            documents.append(doc)
            source_records.append(record)
        except Exception as exc:
            message = f"{title} retrieval failed: {exc}"
            warnings.append(message)
            missing_sources.append(source_type)
            caveats.append(Caveat(caveat_id=f"caveat-{source_type}", message=message, severity="low"))
    try:
        market_path = adapters.historical_data(ticker_norm, period="1y", output_dir=optional_output_dir)
        doc, record = _ingest_vendor_file(
            source_id="market-snapshot",
            source_type="market_snapshot",
            path=market_path,
            title=f"{ticker_norm} 1y market snapshot",
            ticker=ticker_norm,
            root=root,
            metadata={"run_id": run_id, "period": "1y"},
        )
        documents.append(doc)
        source_records.append(record)
    except Exception as exc:
        message = f"Market snapshot retrieval failed: {exc}"
        warnings.append(message)
        missing_sources.append("market_snapshot")
        caveats.append(Caveat(caveat_id="caveat-market_snapshot", message=message, severity="low"))

    has_required = any(doc.source_type in {"manual_file", "sec_filing"} for doc in documents)
    retrieval_batch = RetrievalBatch(
        batch_id=f"batch-{run_id}",
        request_id=request.request_id,
        task_id=task.task_id,
        ticker=ticker_norm,
        sources=source_records,
        missing_source_types=[item for item in missing_sources if item in {"manual_file", "sec_filing", "company_profile", "financial_snapshot", "market_snapshot"}],
        warnings=warnings,
        metadata={"started_at": started_at},
    )
    attachment_paths = _write_harness_source_attachments(documents=documents, vault_root=root, memo_dir=paths["memo_dir"])
    citations = [
        _citation_for_document(
            index,
            document,
            root,
            harness_context_file=attachment_paths[document.document_id].name,
        )
        for index, document in enumerate(documents, start=1)
    ]
    documents = [
        document.model_copy(
            update={
                "metadata": {
                    **document.metadata,
                    "harness_context_file": attachment_paths[document.document_id].name,
                }
            }
        )
        for document in documents
    ]

    research_state_context_files: list[str] = []
    research_state_meta: dict[str, Any] | None = None
    research_state_paths = None
    if enable_research_state:
        research_state_paths = initialize_state_folder(root, ticker_norm)
        existing_evidence_index = read_json_model(
            research_state_paths.evidence_index,
            EvidenceIndex,
            EvidenceIndex(ticker=ticker_norm),
        )
        write_json_model(
            research_state_paths.evidence_index,
            _evidence_index_from_citations(ticker_norm, citations, existing=existing_evidence_index),
        )
        snapshot_state(research_state_paths, paths["state_snapshot"])
        paths["proposal_protocol"].write_text(
            _research_state_protocol_text(memo_dir=paths["memo_dir"], proposals_path=paths["proposals"]),
            encoding="utf-8",
        )
        research_state_context_files = [
            str(research_state_paths.research_state),
            str(research_state_paths.state_index),
            str(research_state_paths.evidence_index),
            str(research_state_paths.open_questions),
            str(research_state_paths.conflicts),
            str(research_state_paths.valuation_snapshot),
            str(paths["proposal_protocol"]),
        ]
        research_state_meta = _research_state_manifest(
            enabled=True,
            state_paths={
                "research_state_path": str(research_state_paths.research_state),
                "state_snapshot_path": str(paths["state_snapshot"]),
                "proposal_protocol_path": str(paths["proposal_protocol"]),
                "proposals_path": str(paths["proposals"]),
                "accepted_changes_path": str(paths["accepted_changes"]),
            },
        )
    package = MemoContextPackage(
        package_id=f"pkg-{run_id}",
        task_id=task.task_id,
        ticker=ticker_norm,
        user_prompt=user_prompt,
        documents=documents,
        citations=citations,
        open_questions=[
            QuestionRecord(
                question_id=f"q-{idx}",
                ticker=ticker_norm,
                question=caveat.message,
                priority="normal" if caveat.severity in {"low", "info"} else "high",
                metadata={"caveat_id": caveat.caveat_id},
            )
            for idx, caveat in enumerate(caveats, start=1)
        ],
        caveats=caveats,
        source_index_path=str(paths["source_index"]),
        question_list_path=str(paths["question_list"]),
        metadata={
            "run_id": run_id,
            "company_name": company_name,
            "vault_root": str(root),
            "retrieval_request": request.model_dump(mode="json"),
            "retrieval_batch": retrieval_batch.model_dump(mode="json"),
            "missing_sources": missing_sources,
            "freshness_notes": ["SEC: latest 10-K plus latest newer 10-Q when available", "Vendor snapshots are point-in-time API pulls"],
            "research_state": research_state_meta,
        },
    )

    paths["source_index"].write_text(
        render_source_index(ticker_norm, documents, citations, attachment_paths),
        encoding="utf-8",
    )
    paths["question_list"].write_text(render_question_list(ticker_norm, caveats), encoding="utf-8")
    shutil.copy2(paths["source_index"], paths["source_snapshot"])
    shutil.copy2(paths["question_list"], paths["question_snapshot"])
    _write_json(paths["context_package"], package.model_dump(mode="json"))

    if not has_required:
        errors.append("No required source was available: provide a manual file or retrieve at least one SEC filing.")
        status: OutputStatus = "failed"
        _write_json(paths["status"], _status_payload(run_id, status, warnings=warnings, errors=errors, research_state=research_state_meta))
        _write_json(
            paths["manifest"],
            _manifest_payload(
                run_id=run_id,
                ticker=ticker_norm,
                user_prompt=user_prompt,
                status=status,
                documents=documents,
                citations=citations,
                context_package_path=paths["context_package"],
                source_index_path=paths["source_index"],
                question_list_path=paths["question_list"],
                final_path=None,
                warnings=warnings,
                errors=errors,
                research_state=research_state_meta,
            ),
        )
        return ResearchPlatformRunResult(
            run_id=run_id,
            ticker=ticker_norm,
            status=status,
            memo_dir=str(paths["memo_dir"]),
            manifest_path=str(paths["manifest"]),
            status_path=str(paths["status"]),
            source_index_path=str(paths["source_index"]),
            question_list_path=str(paths["question_list"]),
            context_package_path=str(paths["context_package"]),
            research_state_path=str(research_state_paths.research_state) if research_state_paths else None,
            state_snapshot_path=str(paths["state_snapshot"]) if enable_research_state else None,
            proposals_path=str(paths["proposals"]) if enable_research_state else None,
            accepted_changes_path=str(paths["accepted_changes"]) if enable_research_state else None,
            warnings=warnings,
            errors=errors,
        )

    research_state_block = None
    if enable_research_state:
        research_state_block = _research_state_prompt_block(memo_dir=paths["memo_dir"], proposals_path=paths["proposals"])

    prompt = assemble_memo_prompt(
        user_prompt=user_prompt,
        ticker=ticker_norm,
        company_name=company_name,
        run_id=run_id,
        source_index_path=str(paths["source_index"]),
        question_list_path=str(paths["question_list"]),
        context_package_path=str(paths["context_package"]),
        documents=documents,
        missing_sources=missing_sources,
        caveats=caveats,
        warnings=warnings,
        research_state_block=research_state_block,
    )
    harness_request = HarnessRequest(
        user_prompt=prompt,
        run_id=run_id,
        wall_clock_budget_seconds=wall_clock_budget_seconds,
        available_skill_packs=["core", "equity"],
        context_files=[
            *research_state_context_files,
            str(paths["context_package"]),
            str(paths["source_index"]),
            str(paths["question_list"]),
            *[str(path) for path in attachment_paths.values()],
        ],
        external_writable_files=[str(paths["proposals"])] if enable_research_state else [],
    )
    harness_response = (run_harness_fn or run_harness)(harness_request)
    final_path = _copy_final(harness_response, paths["final"])
    if harness_response.status == "completed" and final_path:
        status = "succeeded"
    elif final_path:
        status = "partial"
        warnings.append(f"Harness ended with status {harness_response.status}; final was preserved.")
    else:
        status = "failed"
        errors.append(harness_response.error or f"Harness ended with status {harness_response.status} without final.md")

    if enable_research_state and research_state_paths is not None:
        if not paths["proposals"].exists():
            paths["proposals"].write_text("", encoding="utf-8")
        try:
            apply_result = apply_proposals(
                vault_root=root,
                ticker=ticker_norm,
                run_id=run_id,
                proposals_path=paths["proposals"],
                accepted_changes_path=paths["accepted_changes"],
            )
            proposal_apply_errors: list[str] = []
        except Exception as exc:
            apply_result = None
            proposal_apply_errors = [f"Research state proposal apply failed: {exc}"]
            errors.extend(proposal_apply_errors)
            status = "failed" if status == "succeeded" else status
        integrity_errors = validate_state_integrity(root, ticker_norm)
        rolled_back = False
        if proposal_apply_errors:
            restore_state_snapshot(research_state_paths, paths["state_snapshot"])
            rolled_back = True
            errors.append("Research state rolled back to pre-run snapshot after proposal apply failure.")
        elif integrity_errors:
            errors.extend(f"Research state integrity: {item}" for item in integrity_errors)
            restore_state_snapshot(research_state_paths, paths["state_snapshot"])
            rolled_back = True
            errors.append("Research state rolled back to pre-run snapshot after integrity failure.")
            status = "failed" if status == "succeeded" else status
        research_state_meta = _research_state_manifest(
            enabled=True,
            state_paths={
                "research_state_path": str(research_state_paths.research_state),
                "state_snapshot_path": str(paths["state_snapshot"]),
                "proposal_protocol_path": str(paths["proposal_protocol"]),
                "proposals_path": str(paths["proposals"]),
                "accepted_changes_path": str(paths["accepted_changes"]),
            },
            apply_counts={
                "applied": apply_result.applied_count if apply_result else 0,
                "rejected": apply_result.rejected_count if apply_result else 0,
                "revised": apply_result.revised_count if apply_result else 0,
            },
            integrity_errors=integrity_errors,
            rolled_back=rolled_back,
        )

    _write_json(paths["status"], _status_payload(run_id, status, warnings=warnings, errors=errors, research_state=research_state_meta))
    _write_json(
        paths["manifest"],
        _manifest_payload(
            run_id=run_id,
            ticker=ticker_norm,
            user_prompt=user_prompt,
            status=status,
            documents=documents,
            citations=citations,
            context_package_path=paths["context_package"],
            source_index_path=paths["source_index"],
            question_list_path=paths["question_list"],
            final_path=Path(final_path) if final_path else None,
            warnings=warnings,
            errors=errors,
            research_state=research_state_meta,
        ),
    )
    return ResearchPlatformRunResult(
        run_id=run_id,
        ticker=ticker_norm,
        status=status,
        memo_dir=str(paths["memo_dir"]),
        manifest_path=str(paths["manifest"]),
        status_path=str(paths["status"]),
        source_index_path=str(paths["source_index"]),
        question_list_path=str(paths["question_list"]),
        context_package_path=str(paths["context_package"]),
        final_path=final_path,
        research_state_path=str(research_state_paths.research_state) if research_state_paths else None,
        state_snapshot_path=str(paths["state_snapshot"]) if enable_research_state else None,
        proposals_path=str(paths["proposals"]) if enable_research_state else None,
        accepted_changes_path=str(paths["accepted_changes"]) if enable_research_state else None,
        harness_response=harness_response.model_dump(mode="json"),
        warnings=warnings,
        errors=errors,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a v1 AlphaSeeker research-platform memo.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--company-name", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--manual-file", action="append", default=[])
    parser.add_argument("--vault-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--wall-clock-budget-seconds", type=int, default=1200)
    parser.add_argument("--enable-research-state", action="store_true", help="Enable v3 markdown-first research state artifacts.")
    args = parser.parse_args()
    result = run_research_memo(
        user_prompt=args.prompt,
        ticker=args.ticker,
        company_name=args.company_name,
        manual_files=args.manual_file,
        vault_root=args.vault_root,
        run_id=args.run_id,
        wall_clock_budget_seconds=args.wall_clock_budget_seconds,
        enable_research_state=args.enable_research_state,
    )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
