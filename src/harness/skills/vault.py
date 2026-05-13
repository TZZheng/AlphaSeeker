"""Harness skill adapters for the persistent research vault."""

from __future__ import annotations

from typing import Any

from src.harness.skills.common import artifact_evidence, json_preview, make_result, safe_read
from src.harness.types import HarnessState, SkillMetrics, SkillResult, SkillSpec
from src.vault.extract import extract_company_records
from src.vault.ingest import ingest_file
from src.vault.onboard import onboard_company
from src.vault.store import VaultStore
from src.vault.synthesis import synthesize_company_research_state
from src.vault.wiki import render_company_wiki


def vault_ingest_document_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    path = str(arguments.get("path") or arguments.get("path_or_url") or "").strip()
    ticker = str(arguments.get("ticker") or "").strip().upper() or None
    if not path:
        return make_result(
            "vault_ingest_document",
            arguments,
            status="failed",
            summary="vault_ingest_document requires a local path.",
            error="Missing path.",
        )
    result = ingest_file(
        path,
        ticker=ticker,
        source_type=arguments.get("source_type"),
        title=arguments.get("title"),
        source_grade=str(arguments.get("source_grade") or "B"),
        url=arguments.get("url"),
        published_at=arguments.get("published_at"),
    )
    return make_result(
        "vault_ingest_document",
        arguments,
        status="ok",
        summary=f"Ingested document {result['doc_id']} into the research vault.",
        details=result,
        output_text=json_preview({"doc_id": result["doc_id"], "extracted_path": result["extracted_path"], "text_chars": result["text_chars"]}),
        artifacts=[result["extracted_path"], result["metadata_path"]],
        evidence=[artifact_evidence("vault_ingest_document", f"Vault document {result['doc_id']}.", result["extracted_path"])],
    )


def vault_get_company_context_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    ticker = str(arguments.get("ticker") or "").strip().upper()
    if not ticker:
        return make_result(
            "vault_get_company_context",
            arguments,
            status="failed",
            summary="vault_get_company_context requires a ticker.",
            error="Missing ticker.",
        )
    limit = int(arguments.get("limit") or 20)
    context = VaultStore().company_context(ticker, limit=limit)
    return make_result(
        "vault_get_company_context",
        arguments,
        status="ok" if context.get("company") else "partial",
        summary=f"Loaded research-vault context for {ticker}.",
        details=context,
        metrics=SkillMetrics(evidence_count=len(context["documents"]), artifact_count=0),
        output_text=json_preview(context),
    )


def vault_extract_company_records_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    ticker = str(arguments.get("ticker") or "").strip().upper()
    if not ticker:
        return make_result(
            "vault_extract_company_records",
            arguments,
            status="failed",
            summary="vault_extract_company_records requires a ticker.",
            error="Missing ticker.",
        )
    result = extract_company_records(ticker)
    return make_result(
        "vault_extract_company_records",
        arguments,
        status="ok",
        summary=(
            f"Extracted deterministic vault records for {ticker}: "
            f"{result['counts']['metrics']} metrics, {result['counts']['facts']} facts, "
            f"{result['counts']['questions']} questions."
        ),
        details=result,
        metrics=SkillMetrics(evidence_count=result["counts"]["facts"], artifact_count=0),
        output_text=json_preview({"ticker": ticker, "counts": result["counts"]}),
    )


def vault_update_company_wiki_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    ticker = str(arguments.get("ticker") or "").strip().upper()
    if not ticker:
        return make_result(
            "vault_update_company_wiki",
            arguments,
            status="failed",
            summary="vault_update_company_wiki requires a ticker.",
            error="Missing ticker.",
        )
    path = render_company_wiki(ticker, run_id=_state.run_id or None)
    preview = safe_read(str(path), max_chars=4000)
    return make_result(
        "vault_update_company_wiki",
        arguments,
        status="ok",
        summary=f"Rendered Obsidian-compatible company wiki for {ticker}.",
        details={"ticker": ticker, "wiki_path": str(path)},
        metrics=SkillMetrics(artifact_count=1),
        output_text=preview,
        artifacts=[str(path)],
        evidence=[artifact_evidence("vault_update_company_wiki", f"Company wiki for {ticker}.", str(path), content=preview)],
    )


def vault_synthesize_research_state_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    ticker = str(arguments.get("ticker") or "").strip().upper()
    if not ticker:
        return make_result(
            "vault_synthesize_research_state",
            arguments,
            status="failed",
            summary="vault_synthesize_research_state requires a ticker.",
            error="Missing ticker.",
        )
    try:
        result = synthesize_company_research_state(
            ticker=ticker,
            company_name=arguments.get("company_name"),
            model_name=arguments.get("model_name"),
            limit=int(arguments.get("limit") or 5),
            doc_char_limit=int(arguments.get("doc_char_limit") or 6000),
            bundle_char_limit=int(arguments.get("bundle_char_limit") or 24000),
        )
    except Exception as exc:
        return make_result(
            "vault_synthesize_research_state",
            arguments,
            status="failed",
            summary=f"Failed to synthesize LLM research state for {ticker}.",
            error=str(exc),
        )
    preview = safe_read(str(result["wiki_path"]), max_chars=5000)
    return make_result(
        "vault_synthesize_research_state",
        arguments,
        status="ok",
        summary=(
            f"Synthesized LLM research state for {ticker}: "
            f"{result['source_count']} sources, {result['citation_count']} citations, "
            f"{result['question_count']} open questions."
        ),
        details=result,
        metrics=SkillMetrics(
            evidence_count=result["citation_count"],
            artifact_count=2,
            sections_touched=["business_summary", "key_metrics", "guidance", "risks", "thesis", "open_questions"],
        ),
        output_text=preview,
        artifacts=[str(result["wiki_path"]), str(result["raw_response_path"])],
        evidence=[artifact_evidence("vault_synthesize_research_state", f"LLM-generated research state for {ticker}.", str(result["wiki_path"]), content=preview)],
    )


def vault_onboard_company_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    ticker = str(arguments.get("ticker") or "").strip().upper()
    if not ticker:
        return make_result(
            "vault_onboard_company",
            arguments,
            status="failed",
            summary="vault_onboard_company requires a ticker.",
            error="Missing ticker.",
        )
    forms = arguments.get("forms") or arguments.get("form_types") or ["10-K", "10-Q", "8-K"]
    if not isinstance(forms, list):
        forms = [str(forms)]
    result = onboard_company(
        ticker=ticker,
        company_name=arguments.get("company_name"),
        forms=[str(form) for form in forms],
        max_filings=int(arguments.get("max_filings") or 5),
        include_market_support=bool(arguments.get("include_market_support", True)),
        inbox_dir=arguments.get("inbox_dir"),
    )
    artifacts = [result["wiki_path"]]
    for group in ("sec_documents", "support_documents", "manual_documents"):
        for item in result.get(group, []):
            if isinstance(item, dict) and item.get("extracted_path"):
                artifacts.append(str(item["extracted_path"]))
    return make_result(
        "vault_onboard_company",
        arguments,
        status="ok",
        summary=f"Onboarded {ticker} into the research vault with {len(result['sec_documents'])} SEC documents.",
        details=result,
        metrics=SkillMetrics(evidence_count=len(result["sec_documents"]), artifact_count=len(artifacts)),
        output_text=json_preview({"ticker": ticker, "wiki_path": result["wiki_path"], "database_path": result["database_path"]}),
        artifacts=artifacts,
        evidence=[artifact_evidence("vault_onboard_company", f"Vault wiki for {ticker}.", result["wiki_path"])],
    )


VAULT_SKILLS: list[SkillSpec] = [
    SkillSpec(
        name="vault_ingest_document",
        description="Ingest a local document into the persistent research vault and optionally link it to a ticker.",
        pack="vault",
        produces_artifacts=True,
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local file path to ingest."},
                "ticker": {"type": "string"},
                "source_type": {"type": "string"},
                "title": {"type": "string"},
                "source_grade": {"type": "string", "default": "B"},
            },
            "required": ["path"],
        },
        executor=vault_ingest_document_skill,
    ),
    SkillSpec(
        name="vault_get_company_context",
        description="Load company documents, active facts/metrics, open questions, and conflicts from the persistent research vault.",
        pack="vault",
        input_schema={
            "type": "object",
            "properties": {"ticker": {"type": "string"}, "limit": {"type": "integer", "default": 20}},
            "required": ["ticker"],
        },
        executor=vault_get_company_context_skill,
    ),

    SkillSpec(
        name="vault_extract_company_records",
        description="Populate deterministic first-pass facts, metrics, and questions from a company's vault documents.",
        pack="vault",
        input_schema={"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
        executor=vault_extract_company_records_skill,
    ),
    SkillSpec(
        name="vault_update_company_wiki",
        description="Render or refresh the Obsidian-compatible company wiki page from vault records.",
        pack="vault",
        produces_artifacts=True,
        input_schema={"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
        executor=vault_update_company_wiki_skill,
    ),
    SkillSpec(
        name="vault_synthesize_research_state",
        description="Use an LLM to synthesize a cited company research-state wiki and open questions from vault source documents.",
        pack="vault",
        produces_artifacts=True,
        timeout_budget_seconds=120,
        input_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "company_name": {"type": "string"},
                "model_name": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
                "doc_char_limit": {"type": "integer", "default": 6000},
                "bundle_char_limit": {"type": "integer", "default": 24000},
            },
            "required": ["ticker"],
        },
        executor=vault_synthesize_research_state_skill,
    ),
    SkillSpec(
        name="vault_onboard_company",
        description="Auto-fetch A-grade SEC/company-primary sources and render a company database homepage in the persistent research vault.",
        pack="vault",
        produces_artifacts=True,
        timeout_budget_seconds=90,
        input_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "company_name": {"type": "string"},
                "forms": {"type": "array", "items": {"type": "string"}, "default": ["10-K", "10-Q", "8-K"]},
                "max_filings": {"type": "integer", "default": 5},
                "include_market_support": {"type": "boolean", "default": True},
                "inbox_dir": {"type": "string"},
            },
            "required": ["ticker"],
        },
        executor=vault_onboard_company_skill,
    ),
]
