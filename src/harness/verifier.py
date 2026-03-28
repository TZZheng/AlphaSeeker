"""Contract-aware evaluator for harness drafts."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from src.harness.controller import get_unblocked_steps
from src.harness.types import HarnessState, ReportSectionFeedback, SkillCall, VerificationReport
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model


_VERDICT_VALUES = {"pass", "revise", "fail"}
_RATING_FIELDS = (
    "grounding",
    "completeness",
    "numeric_consistency",
    "citation_coverage",
    "formatting",
)
_STRING_LIST_FIELDS = (
    "improvement_instructions",
    "missing_evidence",
    "blocking_issues",
    "missing_sections",
    "missing_evidence_types",
    "missing_citations",
    "freshness_warnings",
    "numeric_inconsistencies",
    "counterevidence_gaps",
    "unresolved_gaps",
    "suggested_retrieval_queries",
)


def _parse_sections(draft: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = "__preamble__"
    buffer: list[str] = []
    for line in draft.splitlines():
        if line.startswith("## "):
            sections[current] = "\n".join(buffer).strip()
            current = line[3:].strip()
            buffer = []
            continue
        bold_match = re.fullmatch(r"\*\*([^*]+)\*\*:?", line.strip())
        if bold_match:
            sections[current] = "\n".join(buffer).strip()
            current = bold_match.group(1).strip()
            buffer = []
            continue
        buffer.append(line)
    sections[current] = "\n".join(buffer).strip()
    return sections


def _format_evidence(state: HarnessState, limit: int = 12) -> str:
    if not state.evidence_ledger:
        return "No evidence collected."

    lines = []
    for item in state.evidence_ledger[-limit:]:
        source_text = ", ".join(item.sources or item.artifact_paths) or item.source_type
        lines.append(f"[{item.id}] {item.summary} | {source_text}")
    return "\n".join(lines)


def _required_sections(state: HarnessState) -> list[str]:
    if state.research_plan and state.research_plan.required_sections:
        return state.research_plan.required_sections
    return []


def _rule_based_evaluation(state: HarnessState, draft: str) -> VerificationReport:
    if not draft.strip():
        return VerificationReport(
            decision="fail",
            summary="The draft is empty.",
            grounding="fail",
            completeness="fail",
            numeric_consistency="fail",
            citation_coverage="fail",
            formatting="fail",
            improvement_instructions=["Generate a non-empty draft before finalizing."],
            blocking_issues=["The draft is empty."],
            raw_feedback="Empty draft.",
        )

    sections = _parse_sections(draft)
    required_sections = _required_sections(state)
    missing_sections = [section for section in required_sections if section not in sections]

    report_section_feedback: list[ReportSectionFeedback] = []
    missing_citations: list[str] = []
    for section in required_sections:
        if section not in sections:
            report_section_feedback.append(
                ReportSectionFeedback(
                    section_label=section,
                    issue="Missing required section",
                    why_it_fails="The contract requires this section but it does not appear in the draft.",
                    suggested_fix=f"Add a '{section}' section grounded in collected evidence.",
                )
            )
            continue
        section_text = sections[section]
        cited_ids = re.findall(r"\[(E\d+)\]", section_text)
        if section != "Sources" and not cited_ids:
            missing_citations.append(section)
            report_section_feedback.append(
                ReportSectionFeedback(
                    section_label=section,
                    quoted_text=section_text[:200],
                    issue="Missing citations",
                    why_it_fails="The section makes claims without inline evidence ids.",
                    suggested_fix="Tie each material claim in this section to one or more evidence ids.",
                )
            )

    if not state.evidence_ledger:
        report_section_feedback.append(
            ReportSectionFeedback(
                section_label="Executive Summary",
                issue="No evidence ledger",
                why_it_fails="The report has no collected evidence behind it.",
                suggested_fix="Collect grounded evidence before finalizing.",
            )
        )

    missing_evidence_types: list[str] = []
    counterevidence_gaps: list[str] = []
    freshness_warnings: list[str] = []
    numeric_inconsistencies: list[str] = []
    blocking_issues: list[str] = []
    unresolved_gaps: list[str] = []
    suggested_retrieval_queries: list[str] = []

    if state.research_plan and state.research_plan.counterevidence_topics:
        risk_text = sections.get("Risks and Counterevidence", "")
        if not risk_text or "risk" not in risk_text.lower():
            counterevidence_gaps.append("Risks and Counterevidence section is weak or absent.")
            report_section_feedback.append(
                ReportSectionFeedback(
                    section_label="Risks and Counterevidence",
                    quoted_text=risk_text[:200],
                    issue="Weak counterevidence coverage",
                    why_it_fails="The report does not clearly discuss the bear case or conflicting evidence.",
                    suggested_fix="Add explicit counterevidence, downside scenarios, and peer challenge evidence.",
                )
            )

    if state.research_plan and state.research_plan.freshness_requirements:
        if not re.search(r"\b(?:20\d{2}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", draft):
            freshness_warnings.append("Time-sensitive report lacks an explicit date.")

    if "equity" in state.enabled_packs:
        peer_evidence = [
            item.id
            for item in state.evidence_ledger
            if item.id and ("peer" in item.summary.lower() or "competitor" in item.summary.lower())
        ]
        if not peer_evidence:
            missing_evidence_types.append("peer_or_competitor_evidence")
            counterevidence_gaps.append("No peer or competitor evidence was collected.")

    if "commodity" in state.enabled_packs:
        if "Curve and Positioning" in required_sections and "Curve and Positioning" in sections:
            if not re.findall(r"\[(E\d+)\]", sections["Curve and Positioning"]):
                missing_evidence_types.append("curve_or_positioning_evidence")

    if "macro" in state.enabled_packs:
        if "Scenarios" in required_sections and "Scenarios" in sections:
            if "scenario" not in sections["Scenarios"].lower():
                report_section_feedback.append(
                    ReportSectionFeedback(
                        section_label="Scenarios",
                        quoted_text=sections["Scenarios"][:200],
                        issue="Weak scenario analysis",
                        why_it_fails="The macro section does not clearly discuss alternative scenarios.",
                        suggested_fix="Add at least one alternative scenario or policy failure mode.",
                    )
                )

    if missing_sections:
        blocking_issues.extend(f"Missing section: {section}" for section in missing_sections)
    if missing_citations:
        blocking_issues.extend(f"Missing citations in: {section}" for section in missing_citations)
    if not state.evidence_ledger:
        blocking_issues.append("No evidence ledger entries were available.")
    unresolved_gaps.extend(blocking_issues)

    if state.request.research_profile == "deep" and state.coverage_matrix is not None:
        if state.coverage_matrix.needs_more_retrieval:
            unresolved_gaps.extend(
                f"Weak coverage: {label}" for label in state.coverage_matrix.next_priority_labels
            )
            suggested_retrieval_queries.extend(
                f"{state.request.user_prompt} {label}"
                for label in state.coverage_matrix.next_priority_labels[:4]
            )

    follow_up_calls: list[SkillCall] = []
    ready_steps = get_unblocked_steps(state)
    if missing_sections or missing_citations or counterevidence_gaps:
        for step in ready_steps[:2]:
            follow_up_calls.extend(step.recommended_skill_calls[:1])
    if (
        state.request.research_profile == "deep"
        and state.coverage_matrix is not None
        and state.coverage_matrix.needs_more_retrieval
    ):
        follow_up_calls.insert(
            0,
            SkillCall(
                name="deep_retrieval",
                arguments={
                    "stage": "run_wave",
                    "prompt": state.request.user_prompt,
                    "gap_queries": suggested_retrieval_queries[:4],
                    "ingest_batch_size": state.request.deep_read_batch_size,
                },
            ),
        )
    if not follow_up_calls and counterevidence_gaps:
        follow_up_calls.append(
            SkillCall(
                name="search_and_read",
                arguments={
                    "queries": [f"{state.request.user_prompt} risks counterevidence bearish case"],
                    "urls_per_query": 3,
                    "use_news": True,
                },
            )
        )

    if blocking_issues:
        decision = "revise"
    else:
        decision = "pass"

    if not state.evidence_ledger:
        grounding = "fail"
    else:
        grounding = "pass" if not missing_sections else "revise"
    completeness = "pass" if not missing_sections else "revise"
    if not state.evidence_ledger:
        citation_coverage = "fail"
    else:
        citation_coverage = "pass" if not missing_citations else "revise"
    formatting = "pass" if draft.startswith("# ") else "revise"
    numeric_consistency = "pass" if not numeric_inconsistencies else "revise"

    return VerificationReport(
        decision=decision,
        summary="Structured evaluator completed contract checks.",
        grounding=grounding,
        completeness=completeness,
        numeric_consistency=numeric_consistency,
        citation_coverage=citation_coverage,
        formatting=formatting,
        improvement_instructions=[
            feedback.suggested_fix for feedback in report_section_feedback[:8]
        ],
        missing_evidence=[item for item in missing_evidence_types],
        blocking_issues=blocking_issues,
        missing_sections=missing_sections,
        missing_evidence_types=missing_evidence_types,
        missing_citations=missing_citations,
        freshness_warnings=freshness_warnings,
        numeric_inconsistencies=numeric_inconsistencies,
        required_follow_up_calls=follow_up_calls,
        counterevidence_gaps=counterevidence_gaps,
        report_section_feedback=report_section_feedback,
        unresolved_gaps=list(dict.fromkeys(unresolved_gaps)),
        suggested_retrieval_queries=list(dict.fromkeys(suggested_retrieval_queries)),
        evaluator_parse_mode="rule_based_fallback",
        raw_feedback="Fallback evaluator applied structural checks.",
    )


def _extract_message_text(message: Any) -> str:
    payload = getattr(message, "content", message)
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, list):
        parts: list[str] = []
        for item in payload:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("content") or item.get("value")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    if payload is None:
        return ""
    return str(payload).strip()


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    stripped = raw_text.strip()
    if not stripped:
        return None

    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    decoder = json.JSONDecoder()
    for candidate in candidates:
        for start_char in ("{", "["):
            start = candidate.find(start_char)
            if start < 0:
                continue
            fragment = candidate[start:].strip()
            try:
                parsed, _ = decoder.raw_decode(fragment)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _dedupe_strings(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _dedupe_strings([value])
    if isinstance(value, list):
        items: list[str] = []
        for entry in value:
            if isinstance(entry, str):
                items.append(entry)
                continue
            if isinstance(entry, (int, float, bool)):
                items.append(str(entry))
                continue
            if not isinstance(entry, dict):
                continue
            text = (
                entry.get("text")
                or entry.get("value")
                or entry.get("label")
                or entry.get("name")
                or entry.get("issue")
            )
            if isinstance(text, str):
                items.append(text)
        return _dedupe_strings(items)
    if isinstance(value, dict):
        if isinstance(value.get("items"), list):
            return _coerce_string_list(value.get("items"))
        items = []
        for key, item in value.items():
            if isinstance(item, str) and item.strip():
                items.append(f"{key}: {item}")
            elif isinstance(key, str) and key.strip():
                items.append(key)
        return _dedupe_strings(items)
    return _dedupe_strings([str(value)])


def _normalize_verdict(value: Any, fallback: str = "revise") -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        aliases = {
            "ok": "pass",
            "good": "pass",
            "approved": "pass",
            "accept": "pass",
            "accepted": "pass",
            "sufficient": "pass",
            "needs_revision": "revise",
            "needs revision": "revise",
            "revision": "revise",
            "retry": "revise",
            "insufficient": "revise",
            "block": "fail",
            "blocked": "fail",
            "failed": "fail",
            "reject": "fail",
            "rejected": "fail",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in _VERDICT_VALUES:
            return normalized
    if isinstance(value, bool):
        return "pass" if value else fallback
    return fallback


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _coerce_int(value: Any, fallback: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return fallback


def _coerce_call_arguments(name: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        if name == "search_and_read":
            return {"queries": [str(item) for item in value]}
        return {"items": value}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        if name == "search_and_read":
            return {"queries": [text]}
        return {"value": text}
    if value is None:
        return {}
    return {"value": value}


def _normalize_skill_call_entries(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        entries = value
    else:
        entries = [value]

    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, str):
            name = entry.strip()
            if name:
                normalized.append({"name": name, "arguments": {}})
            continue
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("calls"), list | dict):
            normalized.extend(_normalize_skill_call_entries(entry.get("calls")))
            continue

        name = (
            entry.get("name")
            or entry.get("skill")
            or entry.get("skill_name")
            or entry.get("tool")
            or entry.get("call")
        )
        if isinstance(name, str) and name.strip():
            arguments = entry.get("arguments")
            if arguments is None:
                for alias in ("args", "params", "parameters", "kwargs"):
                    if alias not in entry:
                        continue
                    arguments = entry[alias]
                    break
            normalized.append(
                {
                    "name": name.strip(),
                    "arguments": _coerce_call_arguments(name.strip(), arguments),
                }
            )
            continue

        for key, item in entry.items():
            if not isinstance(key, str) or not key.strip():
                continue
            normalized.append(
                {
                    "name": key.strip(),
                    "arguments": _coerce_call_arguments(key.strip(), item),
                }
            )

    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in normalized:
        try:
            call = SkillCall.model_validate(item)
        except Exception:
            continue
        payload = call.model_dump(mode="json")
        fingerprint = json.dumps(payload, sort_keys=True)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        validated.append(payload)
    return validated


def _normalize_feedback_entry(entry: Any, *, section_label: str | None = None) -> list[dict[str, Any]]:
    if isinstance(entry, str):
        issue = entry.strip()
        if not issue:
            return []
        return [
            {
                "section_label": section_label or "Unknown",
                "quoted_text": "",
                "issue": issue,
                "why_it_fails": issue,
                "suggested_fix": "Revise this section to address the evaluator issue.",
                "missing_evidence_ids": [],
            }
        ]

    if not isinstance(entry, dict):
        return []

    if isinstance(entry.get("items"), list):
        rows: list[dict[str, Any]] = []
        for item in entry["items"]:
            rows.extend(_normalize_feedback_entry(item, section_label=section_label))
        return rows

    label = (
        entry.get("section_label")
        or entry.get("section")
        or entry.get("section_name")
        or entry.get("label")
        or section_label
        or "Unknown"
    )
    issue = (
        entry.get("issue")
        or entry.get("problem")
        or entry.get("feedback")
        or entry.get("title")
        or "Section needs revision"
    )
    why_it_fails = (
        entry.get("why_it_fails")
        or entry.get("reason")
        or entry.get("why")
        or entry.get("details")
        or issue
    )
    suggested_fix = (
        entry.get("suggested_fix")
        or entry.get("fix")
        or entry.get("recommendation")
        or entry.get("action")
        or "Revise this section to address the evaluator issue."
    )
    quoted_text = entry.get("quoted_text") or entry.get("quote") or entry.get("excerpt") or ""
    return [
        {
            "section_label": str(label).strip() or "Unknown",
            "quoted_text": str(quoted_text).strip(),
            "issue": str(issue).strip() or "Section needs revision",
            "why_it_fails": str(why_it_fails).strip() or str(issue).strip() or "Section needs revision",
            "suggested_fix": str(suggested_fix).strip() or "Revise this section to address the evaluator issue.",
            "missing_evidence_ids": _coerce_string_list(
                entry.get("missing_evidence_ids") or entry.get("missing_evidence")
            ),
        }
    ]


def _normalize_section_feedback(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    entries = value if isinstance(value, list) else [value]
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, dict) and not {
            "section_label",
            "section",
            "section_name",
            "label",
            "issue",
            "problem",
            "feedback",
            "why_it_fails",
            "suggested_fix",
        }.intersection(entry):
            for key, item in entry.items():
                normalized.extend(
                    _normalize_feedback_entry(item, section_label=key if isinstance(key, str) else None)
                )
            continue
        normalized.extend(_normalize_feedback_entry(entry))

    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in normalized:
        try:
            feedback = ReportSectionFeedback.model_validate(item)
        except Exception:
            continue
        payload = feedback.model_dump(mode="json")
        fingerprint = json.dumps(payload, sort_keys=True)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        validated.append(payload)
    return validated


def _unwrap_verification_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("verification_report", "report", "result", "output"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested
    return payload


def _infer_decision(payload: dict[str, Any]) -> str:
    explicit = _normalize_verdict(payload.get("decision"), fallback="")
    if explicit:
        return explicit

    status_values = [
        _normalize_verdict(payload.get(field), fallback="")
        for field in _RATING_FIELDS
        if payload.get(field) not in (None, "")
    ]
    if any(value == "fail" for value in status_values):
        return "fail"
    if any(value == "revise" for value in status_values):
        return "revise"

    issue_fields = (
        "blocking_issues",
        "missing_sections",
        "missing_evidence_types",
        "missing_citations",
        "counterevidence_gaps",
        "report_section_feedback",
        "required_follow_up_calls",
        "unresolved_gaps",
    )
    if any(payload.get(field) for field in issue_fields):
        return "revise"
    if status_values:
        return "pass"
    return "revise"


def _normalize_verification_payload(payload: dict[str, Any], raw_text: str) -> dict[str, Any] | None:
    candidate = _unwrap_verification_payload(payload)
    if not candidate:
        return None

    aliases = {
        "required_follow_up_calls": (
            "required_follow_up_calls",
            "follow_up_calls",
            "followup_calls",
        ),
        "report_section_feedback": (
            "report_section_feedback",
            "section_feedback",
            "section_reviews",
        ),
        "summary": ("summary", "overall_summary", "feedback_summary"),
    }

    def _lookup(*names: str) -> Any:
        for name in names:
            if name in candidate:
                return candidate[name]
        return None

    decision = _infer_decision(candidate)
    normalized: dict[str, Any] = {
        "decision": decision,
        "summary": str(_lookup(*aliases["summary"]) or "").strip(),
        "required_follow_up_calls": _normalize_skill_call_entries(
            _lookup(*aliases["required_follow_up_calls"])
        ),
        "report_section_feedback": _normalize_section_feedback(
            _lookup(*aliases["report_section_feedback"])
        ),
        "wall_clock_exhausted": _coerce_bool(candidate.get("wall_clock_exhausted")),
        "qa_iteration": _coerce_int(candidate.get("qa_iteration"), fallback=0),
        "raw_feedback": raw_text.strip() or str(candidate.get("raw_feedback") or "").strip(),
    }

    for field in _RATING_FIELDS:
        normalized[field] = _normalize_verdict(candidate.get(field), fallback=decision)

    for field in _STRING_LIST_FIELDS:
        normalized[field] = _coerce_string_list(candidate.get(field))

    if not normalized["summary"]:
        issue_count = sum(
            len(normalized[field])
            for field in (
                "blocking_issues",
                "missing_sections",
                "missing_citations",
                "counterevidence_gaps",
            )
        )
        if issue_count:
            normalized["summary"] = f"Evaluator returned {decision} with {issue_count} unresolved issue(s)."
        else:
            normalized["summary"] = f"Evaluator returned {decision}."

    has_signal = any(
        normalized.get(field)
        for field in (
            "summary",
            "blocking_issues",
            "missing_sections",
            "missing_citations",
            "report_section_feedback",
            "required_follow_up_calls",
        )
    )
    if not has_signal and raw_text.strip():
        return None
    return normalized


def _parse_verifier_result(result: Any) -> tuple[VerificationReport | None, dict[str, Any] | None, str]:
    raw_text = ""

    if isinstance(result, VerificationReport):
        return result, result.model_dump(mode="json"), result.raw_feedback

    if isinstance(result, dict):
        raw_message = result.get("raw")
        raw_text = _extract_message_text(raw_message)

        parsed = result.get("parsed")
        if isinstance(parsed, VerificationReport):
            return parsed, parsed.model_dump(mode="json"), raw_text
        if isinstance(parsed, dict):
            return None, parsed, raw_text

        if isinstance(raw_message, dict):
            return None, raw_message, raw_text

        payload = _extract_json_object(raw_text)
        return None, payload, raw_text

    raw_text = _extract_message_text(result)
    return None, _extract_json_object(raw_text), raw_text


def _attach_parse_metadata(
    report: VerificationReport,
    *,
    parse_mode: str,
    raw_text: str,
) -> VerificationReport:
    report.evaluator_parse_mode = parse_mode
    if raw_text.strip():
        report.raw_feedback = raw_text.strip()
    return report


def verify_draft(
    state: HarnessState,
    draft: str,
    judge_fn: Callable[[HarnessState, str], VerificationReport] | None = None,
) -> VerificationReport:
    """Run the evaluator, with optional test injection and a rule-based fallback."""

    if judge_fn is not None:
        return judge_fn(state, draft)

    system_prompt = """
You are the evaluator for a research harness draft.

Return exactly one JSON object. Do not return markdown, code fences, commentary, or prose before or after the JSON.
Every key below is required. Use [] for empty lists, false for booleans, 0 for qa_iteration, and "" for empty strings. Do not use null.

Allowed rating values for decision, grounding, completeness, numeric_consistency, citation_coverage, and formatting: pass, revise, fail.

JSON contract:
{
  "decision": "pass|revise|fail",
  "summary": "short factual summary",
  "grounding": "pass|revise|fail",
  "completeness": "pass|revise|fail",
  "numeric_consistency": "pass|revise|fail",
  "citation_coverage": "pass|revise|fail",
  "formatting": "pass|revise|fail",
  "improvement_instructions": ["string"],
  "missing_evidence": ["string"],
  "blocking_issues": ["string"],
  "missing_sections": ["string"],
  "missing_evidence_types": ["string"],
  "missing_citations": ["string"],
  "freshness_warnings": ["string"],
  "numeric_inconsistencies": ["string"],
  "required_follow_up_calls": [{"name": "skill_name", "arguments": {}}],
  "counterevidence_gaps": ["string"],
  "report_section_feedback": [{
    "section_label": "section name",
    "quoted_text": "optional excerpt",
    "issue": "what failed",
    "why_it_fails": "why the section failed",
    "suggested_fix": "specific fix",
    "missing_evidence_ids": ["E1"]
  }],
  "unresolved_gaps": ["string"],
  "suggested_retrieval_queries": ["string"],
  "wall_clock_exhausted": false,
  "qa_iteration": 0,
  "raw_feedback": ""
}

Rules:
- Inspect the supplied plan, contract, coverage, and evidence ledger rather than only the prose.
- If a required section is missing, do not pass completeness.
- If a material claim lacks support from the provided evidence list, do not pass grounding.
- If the draft does not cite evidence IDs, do not pass citation_coverage.
- required_follow_up_calls must be a JSON array of objects with keys exactly name and arguments.
- report_section_feedback must be a JSON array of objects. Never return a dictionary for this field.
- Keep summary short and factual.
"""

    user_prompt = f"""Mission:
{state.mission_text or state.request.user_prompt}

Progress:
{state.progress_text or "None"}

Evidence ledger:
{_format_evidence(state)}

Research plan:
{state.research_plan.model_dump_json(indent=2) if state.research_plan else "None"}

Research contract:
{state.research_contract.model_dump_json(indent=2) if state.research_contract else "None"}

Coverage matrix:
{state.coverage_matrix.model_dump_json(indent=2) if state.coverage_matrix else "None"}

Section briefs:
{chr(10).join(f"- {brief.section_label}: {brief.coverage_status}" for brief in state.section_briefs[:12]) or "None"}

Draft to judge:
{draft}
"""

    raw_text = ""
    try:
        model_name = get_model("harness", "evaluator")
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        base_llm = get_llm(model_name)
        try:
            llm = base_llm.with_structured_output(
                VerificationReport,
                method="json_mode",
                include_raw=True,
            )
            result = llm.invoke(messages)
        except TypeError:
            result = base_llm.invoke(messages)

        direct_report, payload, raw_text = _parse_verifier_result(result)
        if direct_report is not None:
            return _attach_parse_metadata(
                direct_report,
                parse_mode="direct_structured",
                raw_text=raw_text,
            )

        if payload is None:
            raise ValueError("Evaluator did not return a parseable JSON object.")

        try:
            report = VerificationReport.model_validate(payload)
            return _attach_parse_metadata(
                report,
                parse_mode="direct_structured",
                raw_text=raw_text,
            )
        except Exception as direct_exc:
            normalized_payload = _normalize_verification_payload(payload, raw_text)
            if normalized_payload is None:
                raise direct_exc
            report = VerificationReport.model_validate(normalized_payload)
            return _attach_parse_metadata(
                report,
                parse_mode="normalized_structured",
                raw_text=raw_text,
            )
    except Exception as exc:
        print(f"Harness verifier fallback triggered: {exc}")
        report = _rule_based_evaluation(state, draft)
        if raw_text.strip():
            report.raw_feedback = raw_text.strip()
        else:
            report.raw_feedback = (
                "Fallback evaluator applied structural checks. "
                f"Parse error: {exc}"
            )
        return report
