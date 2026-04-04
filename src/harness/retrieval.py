"""Deterministic retrieval and reduction helpers for the harness.

In backend terms:
- "corpus building" means collecting and normalizing a large source set on disk.
- "reduction" means compressing that corpus in stages so later model calls read
  structured summaries instead of raw documents.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha1
import math
import re
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from src.harness.artifacts import sync_reduction_artifacts
from src.harness.types import (
    CoverageMatrix,
    CoverageMatrixEntry,
    DiscoveredSource,
    FactIndexRecord,
    HarnessState,
    ReadQueueEntry,
    ReadResultRecord,
    RetrievalQueryBucket,
    RetrievalStageOutput,
    SectionBrief,
    SkillResult,
    SourceCard,
)


COUNTER_KEYWORDS = (
    "risk",
    "bear",
    "downside",
    "pressure",
    "litigation",
    "regulation",
    "competition",
    "competitor",
    "peer",
    "margin compression",
)
DATE_RE = re.compile(r"\b(?:20\d{2}(?:-\d{2}-\d{2})?|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b")
NUMBER_RE = re.compile(r"\b(?:\$?\d[\d,]*(?:\.\d+)?%?|\d+\.\d+x)\b")
SOURCE_QUALITY = {
    "sec.gov": 1.0,
    "fred.stlouisfed.org": 1.0,
    "eia.gov": 1.0,
    "worldbank.org": 0.95,
    "imf.org": 0.95,
    "federalreserve.gov": 0.95,
    "bloomberg.com": 0.85,
    "reuters.com": 0.85,
    "wsj.com": 0.82,
    "ft.com": 0.82,
    "investor.": 0.9,
}


def _stable_id(prefix: str, text: str) -> str:
    digest = sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{2,}", (text or "").lower())


def _canonicalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        return url.strip()
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=False)))
    path = parsed.path.rstrip("/")
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            "",
            query,
            "",
        )
    )


def _domain_for_url(url: str) -> str:
    return urlparse(url).netloc.lower()


def _freshness_score(publication_date: str | None) -> float:
    if not publication_date:
        return 0.25
    trimmed = publication_date.strip()
    candidates = [trimmed[:10], trimmed[:7], trimmed[:4], trimmed]
    for fmt, candidate in (
        ("%Y-%m-%d", candidates[0]),
        ("%Y/%m/%d", candidates[0]),
        ("%Y-%m", candidates[1]),
        ("%Y", candidates[2]),
    ):
        try:
            dt = datetime.strptime(candidate, fmt).replace(tzinfo=timezone.utc)
            age_days = max(0.0, (datetime.now(timezone.utc) - dt).days)
            if age_days <= 30:
                return 1.0
            if age_days <= 90:
                return 0.85
            if age_days <= 365:
                return 0.65
            return 0.35
        except ValueError:
            continue
    if DATE_RE.search(publication_date):
        return 0.5
    return 0.25


def _source_quality_score(domain: str) -> float:
    if not domain:
        return 0.2
    for key, score in SOURCE_QUALITY.items():
        if key in domain:
            return score
    if domain.endswith(".gov"):
        return 0.92
    if domain.endswith(".edu"):
        return 0.82
    if domain.endswith(".org"):
        return 0.7
    return 0.45


def _relevance_score(prompt: str, *parts: str) -> float:
    prompt_tokens = set(_tokenize(prompt))
    if not prompt_tokens:
        return 0.0
    merged_tokens = set(_tokenize(" ".join(parts)))
    overlap = len(prompt_tokens & merged_tokens)
    if overlap == 0:
        return 0.05
    return min(1.0, overlap / max(4, len(prompt_tokens)))


def _extract_primary_subject(prompt: str) -> str:
    upper_tokens = re.findall(r"\b[A-Z]{1,5}\b", prompt)
    for token in upper_tokens:
        if token not in {"US", "ETF", "GDP", "AI"}:
            return token
    cleaned = re.sub(r"\s+", " ", prompt).strip()
    return cleaned[:80]


def build_query_buckets(
    prompt: str,
    domain_packs: list[str],
    *,
    query_target: int = 60,
) -> list[RetrievalQueryBucket]:
    """Build 50 to 80 deterministic retrieval variants grouped by intent."""

    subject = _extract_primary_subject(prompt)
    current_year = datetime.now(timezone.utc).year
    buckets: dict[str, list[str]] = {
        "core_thesis": [
            prompt,
            f"{subject} investment thesis",
            f"{subject} business overview strategy",
            f"{subject} key drivers and thesis",
            f"{subject} bull case vs bear case",
            f"{subject} key facts and current evidence",
            f"{subject} what matters most today {current_year}",
        ],
        "latest_developments": [
            f"{subject} latest developments {current_year}",
            f"{subject} recent news and catalysts {current_year}",
            f"{subject} current evidence and dated updates",
            f"{subject} quarterly results guidance {current_year}",
            f"{subject} latest management commentary",
            f"{subject} recent announcements and outlook",
            f"{subject} current narrative shift",
        ],
        "risks_counterevidence": [
            f"{subject} bear case downside risks",
            f"{subject} conflicting evidence risks",
            f"{subject} negative catalyst scenario",
            f"{subject} what could go wrong",
            f"{subject} margin pressure regulation competition",
            f"{subject} bearish thesis and rebuttal",
            f"{subject} underperformance drivers",
        ],
    }

    if "equity" in domain_packs:
        buckets["financial_performance"] = [
            f"{subject} revenue margins free cash flow",
            f"{subject} valuation multiples and scenarios",
            f"{subject} earnings estimates revisions",
            f"{subject} return on capital capital allocation",
            f"{subject} balance sheet debt buybacks",
            f"{subject} segment performance growth",
            f"{subject} valuation downside upside case",
        ]
        buckets["peers_competitors"] = [
            f"{subject} competitors market share",
            f"{subject} peer comparison valuation",
            f"{subject} competitive pressure margins",
            f"{subject} moat erosion substitute products",
            f"{subject} peer revenue growth comparison",
            f"{subject} competitor pricing pressure",
            f"{subject} peer bearish case",
        ]
        buckets["management_capital"] = [
            f"{subject} management credibility guidance",
            f"{subject} capital allocation buybacks dividends",
            f"{subject} insider ownership incentives",
            f"{subject} management execution risk",
            f"{subject} board governance capital returns",
            f"{subject} M&A strategy integration",
            f"{subject} management commentary transcript",
        ]
        buckets["regulation_litigation"] = [
            f"{subject} SEC filing latest 10-K 10-Q",
            f"{subject} litigation antitrust regulatory risk",
            f"{subject} supply chain concentration risk",
            f"{subject} product recalls disputes compliance",
            f"{subject} legal risk and penalties",
            f"{subject} filing risk factors latest",
            f"{subject} regulatory change impact",
        ]

    if "macro" in domain_packs:
        buckets["macro_transmission"] = [
            f"{subject} interest rates transmission",
            f"{subject} inflation growth and policy sensitivity",
            f"{subject} credit conditions demand elasticity",
            f"{subject} macro scenario downside case",
            f"{subject} rates FX and financing cost",
            f"{subject} policy path and economic sensitivity",
            f"{subject} macro indicator relevance",
        ]

    if "commodity" in domain_packs:
        buckets["commodity_balance"] = [
            f"{subject} inventory supply demand balance",
            f"{subject} futures curve structure",
            f"{subject} positioning COT and squeeze risk",
            f"{subject} OPEC sanctions exports production",
            f"{subject} physical balance downside upside",
            f"{subject} storage utilization and spreads",
            f"{subject} commodity bearish case",
        ]

    order = [
        "core_thesis",
        "latest_developments",
        "financial_performance",
        "peers_competitors",
        "management_capital",
        "regulation_litigation",
        "macro_transmission",
        "commodity_balance",
        "risks_counterevidence",
    ]
    bucket_defs: list[RetrievalQueryBucket] = []
    seen: set[str] = set()
    total_queries = 0
    for label in order:
        queries = buckets.get(label)
        if not queries:
            continue
        deduped: list[str] = []
        for query in queries:
            key = query.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(query)
        bucket_defs.append(
            RetrievalQueryBucket(
                label=label,
                intent=label.replace("_", " "),
                queries=deduped,
            )
        )
        total_queries += len(deduped)

    if total_queries < query_target:
        shortfall = query_target - total_queries
        variants = [
            "scenario analysis",
            "numerical support",
            "counterarguments",
            "fresh evidence",
            "peer reaction",
            "bearish risks",
            "management execution",
            "valuation update",
        ]
        extras: list[str] = []
        for variant in variants:
            extras.append(f"{subject} {variant}")
            extras.append(f"{prompt} {variant}")
            if len(extras) >= shortfall:
                break
        bucket_defs.append(
            RetrievalQueryBucket(
                label="supplemental",
                intent="supplemental coverage",
                queries=extras[:shortfall],
            )
        )

    return bucket_defs


def flatten_query_buckets(query_buckets: list[RetrievalQueryBucket]) -> list[tuple[str, str]]:
    """Return (bucket_label, query) pairs in stable order."""

    pairs: list[tuple[str, str]] = []
    for bucket in query_buckets:
        for query in bucket.queries:
            pairs.append((bucket.label, query))
    return pairs


def discover_sources(
    prompt: str,
    query_buckets: list[RetrievalQueryBucket],
    *,
    candidate_target: int = 300,
    search_web_fn: Callable[[str, int], list[dict[str, Any]]] | None = None,
    search_news_fn: Callable[[str, int], list[dict[str, Any]]] | None = None,
) -> list[DiscoveredSource]:
    """Run broad discovery and return deterministic candidate-source records."""

    from src.shared.web_search import search_news, search_web

    web_fn = search_web_fn or search_web
    news_fn = search_news_fn or search_news
    query_pairs = flatten_query_buckets(query_buckets)
    if not query_pairs:
        return []

    results_per_query = max(3, math.ceil(candidate_target / len(query_pairs)))
    candidates: list[DiscoveredSource] = []
    for bucket_label, query in query_pairs:
        use_news = bucket_label in {"latest_developments", "regulation_litigation"}
        raw_results = (news_fn if use_news else web_fn)(query, results_per_query)
        for index, item in enumerate(raw_results, start=1):
            title = _normalize_whitespace(str(item.get("title", query)))
            url = str(item.get("href") or item.get("url") or "").strip()
            if not url:
                continue
            canonical_url = _canonicalize_url(url)
            domain = _domain_for_url(canonical_url)
            publication_date = str(item.get("date") or "").strip() or None
            snippet = _normalize_whitespace(str(item.get("body") or ""))
            coverage_tags = [bucket_label]
            lowered = f"{title} {snippet}".lower()
            if any(keyword in lowered for keyword in COUNTER_KEYWORDS):
                coverage_tags.append("counterevidence")
            if "peer" in lowered or "competitor" in lowered or "competition" in lowered:
                coverage_tags.append("peer")
            if publication_date:
                coverage_tags.append("dated")
            source = DiscoveredSource(
                source_id=_stable_id("SRC", f"{bucket_label}|{canonical_url}|{title}"),
                query=query,
                query_bucket=bucket_label,
                search_type="news" if use_news else "web",
                title=title or canonical_url,
                url=url,
                canonical_url=canonical_url,
                domain=domain,
                snippet=snippet,
                publication_date=publication_date,
                discovered_rank=index,
                freshness_score=_freshness_score(publication_date),
                relevance_score=_relevance_score(prompt, query, title, snippet),
                uniqueness_score=1.0,
                source_quality_score=_source_quality_score(domain),
                composite_score=0.0,
                coverage_tags=sorted(set(coverage_tags)),
            )
            candidates.append(source)
    return candidates


def rank_discovered_sources(
    discovered_sources: list[DiscoveredSource],
    prompt: str,
) -> list[DiscoveredSource]:
    """Deduplicate and rank candidates deterministically."""

    domain_counts = Counter(source.domain for source in discovered_sources if source.domain)
    ranked: list[DiscoveredSource] = []
    seen_canonical: set[str] = set()
    seen_title_domain: set[str] = set()

    def _title_key(source: DiscoveredSource) -> str:
        return f"{source.domain}|{re.sub(r'[^a-z0-9]+', ' ', source.title.lower()).strip()}"

    for source in discovered_sources:
        source.uniqueness_score = round(1.0 / max(1, domain_counts.get(source.domain, 1)), 4)
        source.relevance_score = max(
            source.relevance_score,
            _relevance_score(prompt, source.query, source.title, source.snippet),
        )
        source.composite_score = round(
            0.42 * source.relevance_score
            + 0.23 * source.freshness_score
            + 0.2 * source.source_quality_score
            + 0.15 * source.uniqueness_score,
            6,
        )

    ordered = sorted(
        discovered_sources,
        key=lambda item: (
            -item.composite_score,
            -item.relevance_score,
            -item.freshness_score,
            item.canonical_url,
            item.title.lower(),
        ),
    )

    for source in ordered:
        canonical = source.canonical_url or source.url
        title_key = _title_key(source)
        if canonical in seen_canonical or title_key in seen_title_domain:
            continue
        seen_canonical.add(canonical)
        seen_title_domain.add(title_key)
        ranked.append(source)
    return ranked


def build_read_queue(
    ranked_sources: list[DiscoveredSource],
    *,
    queue_target: int = 120,
) -> list[ReadQueueEntry]:
    """Construct a diverse read queue from ranked candidates."""

    by_bucket: dict[str, list[DiscoveredSource]] = defaultdict(list)
    for source in ranked_sources:
        by_bucket[source.query_bucket].append(source)

    ordered_buckets = sorted(by_bucket)
    selected: list[ReadQueueEntry] = []
    seen_sources: set[str] = set()
    rank = 1
    while len(selected) < min(queue_target, len(ranked_sources)):
        added_this_round = False
        for bucket_label in ordered_buckets:
            bucket_sources = by_bucket[bucket_label]
            while bucket_sources and bucket_sources[0].source_id in seen_sources:
                bucket_sources.pop(0)
            if not bucket_sources:
                continue
            source = bucket_sources.pop(0)
            seen_sources.add(source.source_id)
            selected.append(
                ReadQueueEntry(
                    source_id=source.source_id,
                    canonical_url=source.canonical_url or source.url,
                    title=source.title,
                    domain=source.domain,
                    query_bucket=source.query_bucket,
                    priority_rank=rank,
                    priority_score=source.composite_score,
                    coverage_tags=source.coverage_tags,
                    reason=f"{source.query_bucket} score={source.composite_score:.3f}",
                )
            )
            rank += 1
            added_this_round = True
            if len(selected) >= queue_target:
                break
        if not added_this_round:
            break
    return selected


def ingest_read_queue(
    read_queue: list[ReadQueueEntry],
    discovered_sources: list[DiscoveredSource],
    existing_results: list[ReadResultRecord],
    *,
    batch_size: int = 20,
    max_chars_per_url: int = 12000,
    read_urls_parallel_fn: Callable[[list[str], int, int], dict[str, str]] | None = None,
) -> list[ReadResultRecord]:
    """Read the next batch of queued URLs into full text."""

    from src.shared.web_search import read_urls_parallel

    seen_ids = {item.source_id for item in existing_results}
    pending = [item for item in read_queue if item.source_id not in seen_ids][:batch_size]
    if not pending:
        return []

    read_fn = read_urls_parallel_fn or (
        lambda urls, workers, max_chars: read_urls_parallel(
            urls,
            max_workers=workers,
            max_chars_per_url=max_chars,
        )
    )
    source_index = {item.source_id: item for item in discovered_sources}
    urls = [entry.canonical_url for entry in pending]
    text_by_url = read_fn(urls, min(12, max(2, len(urls))), max_chars_per_url)

    results: list[ReadResultRecord] = []
    for entry in pending:
        discovered = source_index.get(entry.source_id)
        text = text_by_url.get(entry.canonical_url, "")
        results.append(
            ReadResultRecord(
                source_id=entry.source_id,
                canonical_url=entry.canonical_url,
                title=entry.title,
                status="read" if text else "failed",
                text=text,
                text_chars=len(text),
                publication_date=discovered.publication_date if discovered else None,
                error=None if text else "Full-text read failed.",
                query_bucket=entry.query_bucket,
                coverage_tags=entry.coverage_tags,
            )
        )
    return results


def _sentence_split(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", _normalize_whitespace(text)) if part.strip()]


def _infer_sections(
    required_sections: list[str],
    text_parts: list[str],
    query_bucket: str,
) -> list[str]:
    lowered = " ".join(text_parts).lower()
    matched: list[str] = []
    for section in required_sections:
        tokens = [token for token in re.findall(r"[a-z]{3,}", section.lower()) if token not in {"and"}]
        if section in {"Executive Summary", "Key Findings", "Sources"}:
            continue
        if tokens and any(token in lowered for token in tokens):
            matched.append(section)
    bucket_map = {
        "latest_developments": ["Executive Summary", "Key Findings"],
        "financial_performance": ["Valuation and Scenarios", "Key Findings"],
        "peers_competitors": ["Peer and Competitive Pressure", "Risks and Counterevidence"],
        "risks_counterevidence": ["Risks and Counterevidence"],
        "macro_transmission": ["Macro Transmission", "Scenarios"],
        "commodity_balance": ["Commodity Balance", "Curve and Positioning"],
    }
    matched.extend(bucket_map.get(query_bucket, []))
    if not matched and required_sections:
        matched.extend(required_sections[:2])
    return list(dict.fromkeys(section for section in matched if section in required_sections))


def extract_source_cards(
    read_results: list[ReadResultRecord],
    discovered_sources: list[DiscoveredSource],
    required_sections: list[str],
    *,
    existing_cards: list[SourceCard] | None = None,
) -> list[SourceCard]:
    """Reduce successful full-text reads into normalized source cards."""

    discovered_index = {item.source_id: item for item in discovered_sources}
    seen_ids = {item.source_id for item in (existing_cards or [])}
    new_cards: list[SourceCard] = []
    for result in read_results:
        if result.status != "read" or result.source_id in seen_ids:
            continue
        discovered = discovered_index.get(result.source_id)
        sentences = _sentence_split(result.text)
        summary = " ".join(sentences[:2]) or result.title
        extracted_facts = [sentence for sentence in sentences if len(sentence) >= 40][:6]
        extracted_numbers = NUMBER_RE.findall(result.text)[:12]
        extracted_dates = DATE_RE.findall(result.text)[:8]
        counterevidence = [
            sentence for sentence in extracted_facts if any(keyword in sentence.lower() for keyword in COUNTER_KEYWORDS)
        ][:4]
        supporting = [
            sentence for sentence in extracted_facts if sentence not in counterevidence
        ][:4]
        section_relevance = _infer_sections(
            required_sections,
            [result.title, summary, result.text[:1200]],
            result.query_bucket,
        )
        new_cards.append(
            SourceCard(
                source_id=result.source_id,
                title=result.title,
                canonical_url=result.canonical_url,
                domain=(discovered.domain if discovered else _domain_for_url(result.canonical_url)),
                source_kind="news" if discovered and discovered.search_type == "news" else "web",
                publication_date=result.publication_date or (discovered.publication_date if discovered else None),
                summary=summary[:600],
                extracted_facts=extracted_facts,
                extracted_numbers=extracted_numbers,
                extracted_dates=extracted_dates,
                supporting_evidence=supporting,
                counterevidence=counterevidence,
                section_relevance=section_relevance,
                freshness_label="dated" if extracted_dates or result.publication_date else "undated",
                evidence_ids=[],
            )
        )
    return new_cards


def build_fact_index(source_cards: list[SourceCard]) -> list[FactIndexRecord]:
    """Normalize extracted facts into a cross-source fact index."""

    facts: dict[str, FactIndexRecord] = {}
    counter = 1
    for card in source_cards:
        for sentence in [*card.supporting_evidence, *card.counterevidence]:
            key = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
            if not key:
                continue
            record = facts.get(key)
            if record is None:
                stance = "counterevidence" if sentence in card.counterevidence else "supporting"
                facts[key] = FactIndexRecord(
                    fact_id=f"F{counter}",
                    fact=sentence,
                    source_ids=[card.source_id],
                    evidence_ids=list(card.evidence_ids),
                    section_labels=card.section_relevance,
                    numbers=NUMBER_RE.findall(sentence),
                    dates=DATE_RE.findall(sentence),
                    stance=stance,
                )
                counter += 1
                continue
            if card.source_id not in record.source_ids:
                record.source_ids.append(card.source_id)
            for evidence_id in card.evidence_ids:
                if evidence_id and evidence_id not in record.evidence_ids:
                    record.evidence_ids.append(evidence_id)
            for section in card.section_relevance:
                if section not in record.section_labels:
                    record.section_labels.append(section)
    return list(facts.values())


def build_section_briefs(
    required_sections: list[str],
    source_cards: list[SourceCard],
    fact_index: list[FactIndexRecord],
) -> list[SectionBrief]:
    """Compress the corpus into section-level briefs for the writer."""

    briefs: list[SectionBrief] = []
    for section in required_sections:
        cards = [card for card in source_cards if section in card.section_relevance]
        if not cards and section in {"Executive Summary", "Key Findings", "Sources"}:
            cards = source_cards[:4]
        facts = [record for record in fact_index if section in record.section_labels]
        summaries = [card.summary for card in cards[:4]]
        key_facts = [record.fact for record in facts[:6]]
        counterpoints = [record.fact for record in facts if record.stance == "counterevidence"][:4]
        evidence_ids = list(
            dict.fromkeys(
                [
                    evidence_id
                    for card in cards
                    for evidence_id in card.evidence_ids
                    if evidence_id
                ]
                + [
                    evidence_id
                    for record in facts
                    for evidence_id in record.evidence_ids
                    if evidence_id
                ]
            )
        )[:12]
        if len(cards) >= 3 or len(facts) >= 4:
            status = "strong"
        elif cards or facts:
            status = "partial"
        else:
            status = "missing"
        summary = " ".join(summaries[:3]).strip()
        if not summary:
            summary = f"No reduced brief exists yet for {section}."
        briefs.append(
            SectionBrief(
                section_label=section,
                summary=summary[:1200],
                evidence_ids=evidence_ids,
                source_ids=[card.source_id for card in cards[:8]],
                key_facts=key_facts,
                counterpoints=counterpoints,
                coverage_status=status,
            )
        )
    return briefs


def _coverage_entry(
    coverage_type: str,
    label: str,
    source_ids: list[str],
    *,
    evidence_ids: list[str] | None = None,
    notes: list[str] | None = None,
) -> CoverageMatrixEntry:
    if len(source_ids) >= 3:
        status = "strong"
    elif source_ids:
        status = "partial"
    else:
        status = "missing"
    return CoverageMatrixEntry(
        coverage_type=coverage_type,  # type: ignore[arg-type]
        label=label,
        status=status,
        evidence_count=len(source_ids),
        evidence_ids=evidence_ids or [],
        source_ids=source_ids,
        notes=notes or [],
    )


def build_coverage_matrix(state: HarnessState) -> CoverageMatrix:
    """Summarize current corpus coverage for controller and critic decisions."""

    required_sections = list(getattr(state, "required_sections", []) or [])
    brief_index = {brief.section_label: brief for brief in state.section_briefs}
    section_entries: list[CoverageMatrixEntry] = []
    for section in required_sections:
        brief = brief_index.get(section)
        source_ids = brief.source_ids if brief else []
        notes = [brief.summary] if brief and brief.summary else []
        entry = _coverage_entry(
            "section",
            section,
            source_ids,
            evidence_ids=(brief.evidence_ids if brief else []),
            notes=notes[:1],
        )
        if brief is not None:
            entry.status = brief.coverage_status
        section_entries.append(entry)

    contract_entries: list[CoverageMatrixEntry] = []
    research_contract = getattr(state, "research_contract", None)
    if research_contract:
        clauses = [
            *research_contract.global_clauses,
            *research_contract.section_clauses,
            *research_contract.freshness_clauses,
            *research_contract.numeric_clauses,
            *research_contract.counterevidence_clauses,
        ]
        for clause in clauses:
            clause_sources: list[str] = []
            for section in clause.applies_to_sections:
                brief = brief_index.get(section)
                if brief:
                    clause_sources.extend(brief.source_ids)
            if not clause_sources and not clause.applies_to_sections:
                clause_sources = [card.source_id for card in state.source_cards[:6]]
            contract_entries.append(
                _coverage_entry("contract", clause.id, list(dict.fromkeys(clause_sources)))
            )

    dated_sources = [card.source_id for card in state.source_cards if card.publication_date or card.extracted_dates]
    freshness_entries = [
        _coverage_entry(
            "freshness",
            clause.id,
            dated_sources[:8],
            notes=[clause.text],
        )
        for clause in (research_contract.freshness_clauses if research_contract else [])
    ]

    counter_sources = [
        card.source_id
        for card in state.source_cards
        if card.counterevidence or any("counterevidence" in section.lower() for section in card.section_relevance)
    ]
    counter_entries = [
        _coverage_entry(
            "counterevidence",
            clause.id,
            counter_sources[:8],
            notes=[clause.text],
        )
        for clause in (research_contract.counterevidence_clauses if research_contract else [])
    ]

    evidence_types: list[CoverageMatrixEntry] = []
    peer_sources = [
        card.source_id
        for card in state.source_cards
        if "peer" in card.summary.lower()
        or "competitor" in card.summary.lower()
        or "competition" in card.summary.lower()
    ]
    domain_tool_sources = [
        card.source_id
        for card in state.source_cards
        if card.source_kind in {"artifact", "dataset", "note"}
    ]
    evidence_types.append(
        _coverage_entry(
            "evidence_type",
            "web_or_news_corpus",
            [card.source_id for card in state.source_cards if card.source_kind in {"web", "news"}][:12],
        )
    )
    evidence_types.append(_coverage_entry("evidence_type", "counterevidence", counter_sources[:12]))
    if "equity" in state.enabled_packs:
        evidence_types.append(_coverage_entry("evidence_type", "peer_or_competitor_evidence", peer_sources[:12]))
    evidence_types.append(_coverage_entry("evidence_type", "dated_current_evidence", dated_sources[:12]))
    if any(pack != "core" for pack in state.enabled_packs):
        evidence_types.append(_coverage_entry("evidence_type", "domain_tool_evidence", domain_tool_sources[:12]))

    successful_reads = sum(1 for item in state.read_results if item.status == "read")
    final_word_count = len((state.final_response or state.latest_draft or "").split())
    next_priority_labels = [
        entry.label
        for entry in [*section_entries, *counter_entries, *evidence_types]
        if entry.status != "strong"
    ][:8]
    needs_more_retrieval = bool(
        next_priority_labels
        or successful_reads < min(getattr(state.request, "successful_read_target", 12), max(8, len(state.read_queue)))
    )

    return CoverageMatrix(
        sections=section_entries,
        contract_clauses=contract_entries,
        freshness_requirements=freshness_entries,
        counterevidence_requirements=counter_entries,
        evidence_types=evidence_types,
        needs_more_retrieval=needs_more_retrieval,
        next_priority_labels=next_priority_labels,
        stats={
            "discovered_count": len(state.discovered_sources),
            "read_queue_count": len(state.read_queue),
            "full_read_count": successful_reads,
            "source_card_count": len(state.source_cards),
            "fact_count": len(state.fact_index),
            "extraction_batch_count": math.ceil(len(state.source_cards) / max(1, getattr(state.request, "read_batch_size", 10))),
            "critic_report_count": len(state.critic_reports),
            "elapsed_seconds": round(state.elapsed_seconds, 2),
            "final_word_count": final_word_count,
            "retrieval_wave_count": state.retrieval_wave_count,
        },
    )


def refresh_reduction_state(state: HarnessState) -> None:
    """Rebuild the retrieval reduction layers and persist them to disk."""

    required_sections = list(getattr(state, "required_sections", []) or [])
    state.fact_index = build_fact_index(state.source_cards)
    state.section_briefs = build_section_briefs(required_sections, state.source_cards, state.fact_index)
    state.coverage_matrix = build_coverage_matrix(state)
    sync_reduction_artifacts(state)


def merge_skill_result_into_corpus(state: HarnessState, result: SkillResult) -> None:
    """Map non-retrieval skill outputs into the same normalized source corpus."""

    if result.skill_name in {"retrieve_sources", "search_web_resources"} or not result.evidence:
        return

    existing_ids = {card.source_id for card in state.source_cards}
    required_sections = list(getattr(state, "required_sections", []) or [])
    for item in result.evidence:
        source_id = item.id or _stable_id("EVID", f"{result.skill_name}|{item.summary}")
        if source_id in existing_ids:
            continue
        content = _normalize_whitespace(item.content or item.summary)
        facts = _sentence_split(content or item.summary)[:5]
        section_relevance = _infer_sections(required_sections, [item.summary, content], result.skill_name)
        source_kind = {
            "url": "web",
            "artifact": "artifact",
            "dataset": "dataset",
            "note": "note",
        }.get(item.source_type, "note")
        state.source_cards.append(
            SourceCard(
                source_id=source_id,
                title=item.summary[:200],
                canonical_url=(item.sources[0] if item.sources else (item.artifact_paths[0] if item.artifact_paths else source_id)),
                domain=_domain_for_url(item.sources[0]) if item.sources else "",
                source_kind=source_kind,  # type: ignore[arg-type]
                publication_date=str(item.metadata.get("date") or "") or None,
                summary=item.summary,
                extracted_facts=facts,
                extracted_numbers=NUMBER_RE.findall(content)[:8],
                extracted_dates=DATE_RE.findall(content)[:6],
                supporting_evidence=facts[:4],
                counterevidence=[
                    fact for fact in facts if any(keyword in fact.lower() for keyword in COUNTER_KEYWORDS)
                ][:3],
                section_relevance=section_relevance,
                freshness_label="dated" if item.metadata.get("date") else "undated",
                evidence_ids=[item.id] if item.id else [],
            )
        )
    refresh_reduction_state(state)


def build_stage_output(state: HarnessState, stage: str, artifact_paths: list[str]) -> RetrievalStageOutput:
    """Build the typed stage summary returned by the composite skill."""

    coverage_status = "needs_more_retrieval"
    if state.coverage_matrix and not state.coverage_matrix.needs_more_retrieval:
        coverage_status = "sufficient"
    successful_reads = sum(1 for item in state.read_results if item.status == "read")
    failed_reads = sum(1 for item in state.read_results if item.status == "failed")
    return RetrievalStageOutput(
        stage=stage,  # type: ignore[arg-type]
        query_bucket_count=len(state.query_buckets),
        query_count=sum(len(bucket.queries) for bucket in state.query_buckets),
        discovered_count=len(state.discovered_sources),
        deduped_count=len({item.source_id for item in state.discovered_sources}),
        read_queue_count=len(state.read_queue),
        successful_read_count=successful_reads,
        failed_read_count=failed_reads,
        source_card_count=len(state.source_cards),
        extraction_batch_count=(
            state.coverage_matrix.stats.get("extraction_batch_count", 0)
            if state.coverage_matrix
            else 0
        ),
        coverage_status=coverage_status,
        artifact_paths=artifact_paths,
    )
