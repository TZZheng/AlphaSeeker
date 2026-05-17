"""LLM-assisted company wiki synthesis from vault source bundles."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model
from src.vault.paths import default_vault_paths
from src.vault.store import VaultStore, new_id, utc_now_iso
from src.vault.wiki import render_support_pages, render_source_index

DEFAULT_BUNDLE_CHAR_LIMIT = 24000
DEFAULT_DOC_CHAR_LIMIT = 6000
REQUIRED_SECTIONS = (
    "business_summary",
    "key_metrics",
    "guidance",
    "risks",
    "thesis",
    "open_questions",
)


@dataclass(frozen=True)
class SourceExcerpt:
    """Compact source excerpt supplied to the synthesis model."""

    doc_id: str
    title: str
    path: str
    source_type: str
    source_grade: str
    published_at: str | None
    excerpt: str

    def to_prompt_block(self) -> str:
        published = self.published_at or "unknown date"
        return "\n".join(
            [
                f"### SOURCE {self.doc_id}",
                f"Title: {self.title}",
                f"Type/grade/date: {self.source_type} / {self.source_grade} / {published}",
                f"Path: {self.path}",
                "Excerpt:",
                self.excerpt,
            ]
        )


def _read_excerpt(path: str | Path, *, max_chars: int) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def build_source_bundle(
    ticker: str,
    *,
    root: str | Path | None = None,
    store: VaultStore | None = None,
    limit: int = 5,
    doc_char_limit: int = DEFAULT_DOC_CHAR_LIMIT,
    bundle_char_limit: int = DEFAULT_BUNDLE_CHAR_LIMIT,
) -> list[SourceExcerpt]:
    """Load a compact, inspectable bundle of vault source excerpts for one ticker."""

    ticker_norm = ticker.strip().upper()
    if not ticker_norm:
        raise ValueError("ticker is required")
    active_store = store or VaultStore(root)
    documents = active_store.list_documents(ticker=ticker_norm, limit=limit)
    excerpts: list[SourceExcerpt] = []
    used_chars = 0
    for document in documents:
        path = str(document.get("path") or "")
        if not path:
            continue
        try:
            excerpt = _read_excerpt(path, max_chars=doc_char_limit)
        except OSError:
            continue
        remaining = bundle_char_limit - used_chars
        if remaining <= 0:
            break
        if len(excerpt) > remaining:
            excerpt = excerpt[:remaining].rstrip() + "\n...[bundle truncated]"
        used_chars += len(excerpt)
        excerpts.append(
            SourceExcerpt(
                doc_id=str(document.get("doc_id") or ""),
                title=str(document.get("title") or document.get("doc_id") or "Untitled source"),
                path=path,
                source_type=str(document.get("source_type") or "unknown"),
                source_grade=str(document.get("source_grade") or ""),
                published_at=document.get("published_at"),
                excerpt=excerpt,
            )
        )
    return excerpts


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def parse_synthesis_json(text: str) -> dict[str, Any]:
    """Parse and normalize the model's strict JSON research-state contract."""

    payload = json.loads(_strip_json_fence(text))
    if not isinstance(payload, dict):
        raise ValueError("synthesis response must be a JSON object")
    normalized: dict[str, Any] = {}
    for key in REQUIRED_SECTIONS:
        value = payload.get(key)
        if key == "open_questions":
            if value is None:
                normalized[key] = []
            elif isinstance(value, list):
                normalized[key] = [str(item).strip() for item in value if str(item).strip()]
            else:
                raise ValueError("open_questions must be a list")
        else:
            normalized[key] = str(value or "").strip()
    citations = payload.get("citations") or []
    if not isinstance(citations, list):
        raise ValueError("citations must be a list")
    normalized_citations: list[dict[str, str]] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        normalized_citations.append(
            {
                "claim": str(citation.get("claim") or "").strip(),
                "doc_id": str(citation.get("doc_id") or "").strip(),
                "quote": str(citation.get("quote") or "").strip(),
            }
        )
    normalized["citations"] = [item for item in normalized_citations if item["claim"] or item["quote"]]
    return normalized


def build_synthesis_prompt(ticker: str, company_name: str | None, excerpts: list[SourceExcerpt]) -> list[Any]:
    """Build the source-grounded synthesis prompt."""

    source_blocks = "\n\n".join(excerpt.to_prompt_block() for excerpt in excerpts)
    system = SystemMessage(
        content=(
            "You are an equity research analyst. Synthesize only from the provided source excerpts. "
            "Do not invent facts. If sources do not mention a topic, say it is not found in the provided sources."
        )
    )
    human = HumanMessage(
        content=f"""
Company/ticker: {ticker.upper()} {f'({company_name})' if company_name else ''}

Create a minimum viable company research-state wiki from the source bundle below.

Return ONLY valid JSON with this exact shape:
{{
  "business_summary": "paragraph or bullets grounded in the sources",
  "key_metrics": "metrics/valuation points found in sources, or 'Not found in provided sources.'",
  "guidance": "management guidance/commentary found in sources, or 'Not found in provided sources.'",
  "risks": "risks/counterevidence found in sources, or 'Not found in provided sources.'",
  "thesis": "key takeaways / starting thesis grounded in the sources",
  "open_questions": ["question 1", "question 2"],
  "citations": [
    {{"claim": "short claim", "doc_id": "SOURCE doc_id", "quote": "short supporting quote copied from source"}}
  ]
}}

Citation rules:
- Every factual claim in the sections should have at least one citation entry.
- Use the SOURCE doc_id values exactly.
- Quotes must be copied from the provided excerpts and kept short.
- Prefer source-grade A evidence when available.

SOURCE BUNDLE:
{source_blocks}
""".strip()
    )
    return [system, human]


def _citations_markdown(citations: list[dict[str, str]], source_lookup: dict[str, SourceExcerpt]) -> str:
    if not citations:
        return "_No citations returned._\n"
    rows = ["| Claim | Source | Quote |", "| --- | --- | --- |"]
    for citation in citations:
        doc_id = citation.get("doc_id", "")
        source = source_lookup.get(doc_id)
        source_label = doc_id
        if source is not None:
            source_label = f"{source.title} (`{source.path}`)"
        claim = citation.get("claim", "").replace("\n", " ").replace("|", "\\|")
        quote = citation.get("quote", "").replace("\n", " ").replace("|", "\\|")
        rows.append(f"| {claim} | {source_label} | {quote} |")
    return "\n".join(rows) + "\n"


def render_synthesis_wiki(
    ticker: str,
    synthesis: dict[str, Any],
    excerpts: list[SourceExcerpt],
    *,
    root: str | Path | None = None,
    model_name: str | None = None,
) -> Path:
    """Render the LLM-generated research state as the company MVP wiki."""

    ticker_norm = ticker.strip().upper()
    paths = default_vault_paths(root).ensure()
    company_dir = paths.company_dir(ticker_norm)
    company_dir.mkdir(parents=True, exist_ok=True)
    source_lookup = {excerpt.doc_id: excerpt for excerpt in excerpts}
    timestamp = utc_now_iso()
    wiki_path = company_dir / "llm_research_state.md"
    source_lines = [f"- `{excerpt.doc_id}` — {excerpt.title} ({excerpt.source_grade}; `{excerpt.path}`)" for excerpt in excerpts]
    text = "\n".join(
        [
            f"# {ticker_norm} LLM Research State MVP",
            "",
            f"Last updated: {timestamp}",
            f"Model: {model_name or 'unknown'}",
            "Source policy: generated from the source bundle below; inspect citations before using in investment work.",
            "",
            "Navigation: [[wiki]] · [[source_index]] · [[question_list]] · [[conflicts]] · [[status_patrol]]",
            "",
            "## Business summary",
            synthesis.get("business_summary", "") or "_Not generated._",
            "",
            "## Key metrics / valuation points",
            synthesis.get("key_metrics", "") or "_Not generated._",
            "",
            "## Guidance / management commentary",
            synthesis.get("guidance", "") or "_Not generated._",
            "",
            "## Risks / counterevidence",
            synthesis.get("risks", "") or "_Not generated._",
            "",
            "## Thesis / key takeaways",
            synthesis.get("thesis", "") or "_Not generated._",
            "",
            "## Open questions",
            *(f"- {question}" for question in synthesis.get("open_questions", [])),
            "" if synthesis.get("open_questions") else "_No open questions returned._",
            "",
            "## Citations",
            _citations_markdown(synthesis.get("citations", []), source_lookup),
            "## Source bundle",
            *(source_lines or ["_No sources supplied._"]),
            "",
        ]
    )
    wiki_path.write_text(text, encoding="utf-8")
    return wiki_path


def synthesize_company_research_state(
    ticker: str,
    *,
    company_name: str | None = None,
    root: str | Path | None = None,
    store: VaultStore | None = None,
    model_name: str | None = None,
    limit: int = 5,
    doc_char_limit: int = DEFAULT_DOC_CHAR_LIMIT,
    bundle_char_limit: int = DEFAULT_BUNDLE_CHAR_LIMIT,
) -> dict[str, Any]:
    """Use an LLM to generate a cited company research-state wiki from vault docs."""

    ticker_norm = ticker.strip().upper()
    active_store = store or VaultStore(root)
    active_store.upsert_company(ticker_norm, name=company_name)
    excerpts = build_source_bundle(
        ticker_norm,
        root=root,
        store=active_store,
        limit=limit,
        doc_char_limit=doc_char_limit,
        bundle_char_limit=bundle_char_limit,
    )
    if not excerpts:
        raise ValueError(f"No vault documents found for {ticker_norm}; ingest sources before synthesis.")
    resolved_model = model_name or get_model("vault", "agent")
    response = get_llm(resolved_model).invoke(build_synthesis_prompt(ticker_norm, company_name, excerpts))
    raw_text = _response_text(response)
    synthesis = parse_synthesis_json(raw_text)
    wiki_path = render_synthesis_wiki(ticker_norm, synthesis, excerpts, root=root, model_name=resolved_model)
    render_source_index(ticker_norm, root=root)

    persisted_questions = []
    for question in synthesis.get("open_questions", []):
        question_digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
        persisted_questions.append(
            active_store.add_question(
                ticker_norm,
                question,
                priority="normal",
                question_id=f"question_llm_{ticker_norm.lower()}_{question_digest}",
            )
        )
    render_support_pages(ticker_norm, root=root, questions=active_store.company_context(ticker_norm)["questions"], conflicts=active_store.company_context(ticker_norm)["conflicts"])

    raw_path = default_vault_paths(root).company_dir(ticker_norm) / f"synthesis_raw_{new_id('run')}.json"
    raw_path.write_text(
        json.dumps({"model": resolved_model, "raw_response": raw_text, "parsed": synthesis}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "ticker": ticker_norm,
        "model": resolved_model,
        "wiki_path": str(wiki_path),
        "raw_response_path": str(raw_path),
        "source_count": len(excerpts),
        "question_count": len(synthesis.get("open_questions", [])),
        "citation_count": len(synthesis.get("citations", [])),
        "sections": synthesis,
        "persisted_questions": persisted_questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize an LLM-generated company research-state wiki from vault documents.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--company-name", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--doc-char-limit", type=int, default=DEFAULT_DOC_CHAR_LIMIT)
    parser.add_argument("--bundle-char-limit", type=int, default=DEFAULT_BUNDLE_CHAR_LIMIT)
    args = parser.parse_args()
    result = synthesize_company_research_state(
        args.ticker,
        company_name=args.company_name,
        root=args.root,
        model_name=args.model_name,
        limit=args.limit,
        doc_char_limit=args.doc_char_limit,
        bundle_char_limit=args.bundle_char_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
