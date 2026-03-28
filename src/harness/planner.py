"""Planner/initializer logic for the harness runtime."""

from __future__ import annotations

import re
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from src.harness.contracts import build_default_contract
from src.harness.registry import build_skill_registry
from src.harness.selector import select_packs
from src.harness.types import (
    DOMAIN_PACKS,
    ResearchBrief,
    ResearchContract,
    ResearchPlan,
    ResearchStep,
    SkillCall,
)
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model


_LATEST_KEYWORDS = ("latest", "current", "today", "recent", "fresh")


def _extract_ticker(prompt: str) -> str | None:
    candidates = re.findall(r"\b[A-Z]{1,5}\b", prompt)
    for candidate in candidates:
        if candidate not in {"US", "AI", "GDP", "ETF"}:
            return candidate
    return None


def _extract_asset(prompt: str) -> str | None:
    lowered = prompt.lower()
    for asset in ("crude oil", "oil", "copper", "gold", "natural gas"):
        if asset in lowered:
            return asset
    return None


def _extract_macro_topic(prompt: str) -> str:
    lowered = prompt.lower()
    if "rates" in lowered or "margin" in lowered:
        return "interest rates and bank margins"
    if "inflation" in lowered:
        return "inflation and growth"
    if "macro" in lowered:
        return "macro outlook"
    return "growth, inflation, policy, and rates"


def _likely_sections(domain_packs: list[str]) -> list[str]:
    sections = ["Executive Summary", "Key Findings"]
    if "equity" in domain_packs:
        sections.append("Equity Overview")
        sections.append("Peer and Competitive Pressure")
        sections.append("Valuation and Scenarios")
    if "macro" in domain_packs:
        sections.append("Macro Transmission")
        sections.append("Scenarios")
    if "commodity" in domain_packs:
        sections.append("Commodity Balance")
        sections.append("Curve and Positioning")
    sections.append("Risks and Counterevidence")
    sections.append("Sources")
    return sections


def _fallback_brief(prompt: str) -> ResearchBrief:
    selected = select_packs(prompt)
    domain_packs = [pack for pack in selected if pack in DOMAIN_PACKS]
    if not domain_packs:
        domain_packs = ["equity"]

    sub_questions = []
    if "equity" in domain_packs:
        sub_questions.extend(
            [
                "What does the current company profile imply about the business model?",
                "What do financials, market data, filings, and peers imply about valuation and risk?",
            ]
        )
    if "macro" in domain_packs:
        sub_questions.append("What macro variables transmit into the user question?")
    if "commodity" in domain_packs:
        sub_questions.append("What do physical balances, curve structure, and positioning imply?")

    return ResearchBrief(
        primary_question=prompt.strip(),
        sub_questions=sub_questions,
        domain_packs=domain_packs,
        user_constraints=["Use current evidence when available."],
        likely_report_shape=_likely_sections(domain_packs),
        key_unknowns=["Precise numeric support depends on live data retrieval."],
        likely_risks_of_failure=[
            "Time-sensitive evidence may be incomplete.",
            "Cross-domain prompts may miss an explicit transmission mechanism without a plan.",
        ],
        rationale="Heuristic planner fallback based on the prompt and supervisor classifier.",
    )


def _brief_from_selected_packs(prompt: str, selected_packs: list[str]) -> ResearchBrief:
    domain_packs = [pack for pack in selected_packs if pack in DOMAIN_PACKS]

    sub_questions = []
    if "equity" in domain_packs:
        sub_questions.extend(
            [
                "What does the current company profile imply about the business model?",
                "What do financials, market data, filings, and peers imply about valuation and risk?",
            ]
        )
    if "macro" in domain_packs:
        sub_questions.append("What macro variables transmit into the user question?")
    if "commodity" in domain_packs:
        sub_questions.append("What do physical balances, curve structure, and positioning imply?")

    return ResearchBrief(
        primary_question=prompt.strip(),
        sub_questions=sub_questions,
        domain_packs=domain_packs,
        user_constraints=["Use current evidence when available."],
        likely_report_shape=_likely_sections(domain_packs),
        key_unknowns=["Precise numeric support depends on live data retrieval."],
        likely_risks_of_failure=[
            "Time-sensitive evidence may be incomplete.",
            "Cross-domain prompts may miss an explicit transmission mechanism without a plan.",
        ],
        rationale="Explicit pack override supplied by the caller.",
    )


def _default_search_queries(prompt: str) -> list[str]:
    queries = [
        prompt,
        f"{prompt} latest evidence and data points",
        f"{prompt} key drivers scenarios risks",
        f"{prompt} counterarguments and bear case",
    ]
    lowered = prompt.lower()
    if any(keyword in lowered for keyword in _LATEST_KEYWORDS):
        queries.append(f"{prompt} latest news analysis")

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = query.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped[:5]


def _fallback_plan(brief: ResearchBrief, *, research_profile: str = "standard") -> ResearchPlan:
    prompt = brief.primary_question
    ticker = _extract_ticker(prompt) or "AAPL"
    asset = _extract_asset(prompt) or "crude oil"
    macro_topic = _extract_macro_topic(prompt)
    company_name = ticker

    steps: list[ResearchStep] = []
    if research_profile == "deep":
        steps.append(
            ResearchStep(
                id="build_deep_corpus",
                objective="Build a staged retrieval corpus with ranked reads and reduced source cards.",
                recommended_skill_calls=[
                    SkillCall(
                        name="deep_retrieval",
                        arguments={
                            "stage": "run_wave",
                            "prompt": prompt,
                            "query_target": 60,
                            "candidate_target": 300,
                            "read_queue_target": 120,
                            "ingest_batch_size": 20,
                        },
                    )
                ],
                required_outputs=[
                    "discovered_sources.json",
                    "read_queue.json",
                    "read_results.json",
                    "source_cards.jsonl",
                    "coverage_matrix.json",
                ],
                completion_criteria=[
                    "A broad candidate source set exists.",
                    "The read queue and reduced source cards are persisted.",
                ],
                report_sections=_likely_sections(brief.domain_packs),
            )
        )
    else:
        steps.append(
            ResearchStep(
                id="collect_context",
                objective="Collect broad web and news evidence for the mission.",
                recommended_skill_calls=[
                    SkillCall(
                        name="search_and_read",
                        arguments={
                            "queries": _default_search_queries(prompt),
                            "urls_per_query": 4,
                            "use_news": any(keyword in prompt.lower() for keyword in _LATEST_KEYWORDS),
                        },
                    ),
                    SkillCall(
                        name="search_web",
                        arguments={"query": prompt, "max_results": 8},
                    ),
                    SkillCall(
                        name="search_news",
                        arguments={"query": prompt, "max_results": 8},
                    ),
                ],
                required_outputs=["broad research evidence", "recent discovery results"],
                completion_criteria=[
                    "Multiple grounded evidence items exist.",
                    "Recent evidence is collected when the prompt is time-sensitive.",
                ],
                report_sections=["Executive Summary", "Key Findings"],
            )
        )

    context_dependency = "build_deep_corpus" if research_profile == "deep" else "collect_context"

    if "equity" in brief.domain_packs:
        steps.extend(
            [
                ResearchStep(
                    id="collect_company_profile",
                    objective="Collect company profile evidence.",
                    depends_on=[context_dependency],
                    recommended_skill_calls=[
                        SkillCall(name="fetch_company_profile", arguments={"ticker": ticker})
                    ],
                    required_outputs=["company profile artifact"],
                    completion_criteria=["The company profile artifact or evidence is available."],
                    report_sections=["Equity Overview"],
                    domain_pack="equity",
                    parallel_safe=True,
                ),
                ResearchStep(
                    id="collect_financials",
                    objective="Collect financial statement and ratio evidence.",
                    depends_on=[context_dependency],
                    recommended_skill_calls=[
                        SkillCall(name="fetch_financials", arguments={"ticker": ticker})
                    ],
                    required_outputs=["financial metrics artifact"],
                    completion_criteria=["Financial metrics evidence is available."],
                    report_sections=["Valuation and Scenarios", "Key Findings"],
                    domain_pack="equity",
                    parallel_safe=True,
                ),
                ResearchStep(
                    id="collect_market_data",
                    objective="Collect market data and price history evidence.",
                    depends_on=[context_dependency],
                    recommended_skill_calls=[
                        SkillCall(name="fetch_market_data", arguments={"ticker": ticker, "period": "1y"})
                    ],
                    required_outputs=["market data artifact"],
                    completion_criteria=["Historical market data is available."],
                    report_sections=["Valuation and Scenarios"],
                    domain_pack="equity",
                    parallel_safe=True,
                ),
                ResearchStep(
                    id="collect_filings",
                    objective="Collect recent SEC filing evidence.",
                    depends_on=[context_dependency],
                    recommended_skill_calls=[
                        SkillCall(
                            name="search_sec_filings",
                            arguments={"ticker": ticker, "company_name": company_name, "max_filings": 3},
                        )
                    ],
                    required_outputs=["recent filing evidence"],
                    completion_criteria=["At least one filing or filing summary is collected."],
                    report_sections=["Equity Overview", "Valuation and Scenarios"],
                    domain_pack="equity",
                    parallel_safe=True,
                ),
                ResearchStep(
                    id="peer_challenge",
                    objective="Collect peer and competitor challenge evidence.",
                    depends_on=["collect_company_profile"],
                    recommended_skill_calls=[
                        SkillCall(name="analyze_peers", arguments={"ticker": ticker}),
                        SkillCall(
                            name="search_and_read",
                            arguments={
                                "queries": [
                                    f"{ticker} competitor market share loss",
                                    f"{ticker} peer margins valuation competition",
                                    f"{ticker} moat erosion substitute products",
                                ],
                                "urls_per_query": 3,
                            },
                        ),
                    ],
                    required_outputs=["peer evidence", "competitor counterevidence"],
                    completion_criteria=["Peer or competitor evidence is linked to the thesis."],
                    report_sections=["Peer and Competitive Pressure", "Risks and Counterevidence"],
                    domain_pack="equity",
                ),
            ]
        )

    if "macro" in brief.domain_packs:
        steps.extend(
            [
                ResearchStep(
                    id="macro_indicators",
                    objective="Collect macro indicator evidence.",
                    depends_on=[context_dependency],
                    recommended_skill_calls=[
                        SkillCall(
                            name="fetch_macro_indicators",
                            arguments={"topic": macro_topic, "countries": ["United States"]},
                        )
                    ],
                    required_outputs=["macro indicators artifact"],
                    completion_criteria=["FRED-style indicator evidence is available."],
                    report_sections=["Macro Transmission", "Scenarios"],
                    domain_pack="macro",
                    parallel_safe=True,
                ),
                ResearchStep(
                    id="macro_cross_country",
                    objective="Collect cross-country macro context when useful.",
                    depends_on=[context_dependency],
                    recommended_skill_calls=[
                        SkillCall(
                            name="fetch_world_bank_indicators",
                            arguments={"countries": ["United States"], "date_range": "2019:2025"},
                        )
                    ],
                    required_outputs=["World Bank artifact or summary"],
                    completion_criteria=["Cross-country context is available when needed."],
                    report_sections=["Macro Transmission", "Scenarios"],
                    domain_pack="macro",
                    parallel_safe=True,
                ),
            ]
        )

    if "commodity" in brief.domain_packs:
        steps.extend(
            [
                ResearchStep(
                    id="commodity_inventory",
                    objective="Collect inventory and supply evidence.",
                    depends_on=[context_dependency],
                    recommended_skill_calls=[
                        SkillCall(name="fetch_eia_inventory", arguments={"asset": asset})
                    ],
                    required_outputs=["inventory artifact"],
                    completion_criteria=["Inventory or production evidence is available."],
                    report_sections=["Commodity Balance"],
                    domain_pack="commodity",
                    parallel_safe=True,
                ),
                ResearchStep(
                    id="commodity_positioning",
                    objective="Collect positioning evidence.",
                    depends_on=[context_dependency],
                    recommended_skill_calls=[
                        SkillCall(name="fetch_cot_report", arguments={"asset": asset, "num_weeks": 12})
                    ],
                    required_outputs=["positioning artifact"],
                    completion_criteria=["CFTC positioning evidence is available."],
                    report_sections=["Curve and Positioning", "Risks and Counterevidence"],
                    domain_pack="commodity",
                    parallel_safe=True,
                ),
                ResearchStep(
                    id="commodity_curve",
                    objective="Collect futures-curve evidence.",
                    depends_on=[context_dependency],
                    recommended_skill_calls=[
                        SkillCall(name="fetch_futures_curve", arguments={"asset": asset, "num_contracts": 12})
                    ],
                    required_outputs=["futures curve artifact"],
                    completion_criteria=["Curve evidence is available."],
                    report_sections=["Curve and Positioning"],
                    domain_pack="commodity",
                    parallel_safe=True,
                ),
            ]
        )

    steps.append(
        ResearchStep(
            id="collect_counterevidence",
            objective="Collect explicit counterevidence and alternative scenarios.",
            depends_on=[context_dependency],
            recommended_skill_calls=[
                (
                    SkillCall(
                        name="deep_retrieval",
                        arguments={
                            "stage": "run_wave",
                            "prompt": prompt,
                            "ingest_batch_size": 20,
                        },
                    )
                    if research_profile == "deep"
                    else SkillCall(
                        name="search_and_read",
                        arguments={
                            "queries": [
                                f"{prompt} risks bearish case alternative scenario",
                                f"{prompt} conflicting evidence downside triggers",
                            ],
                            "urls_per_query": 3,
                            "use_news": any(keyword in prompt.lower() for keyword in _LATEST_KEYWORDS),
                        },
                    )
                )
            ],
            required_outputs=["counterevidence"],
            completion_criteria=["Explicit challenge evidence is collected."],
            report_sections=["Risks and Counterevidence"],
        )
    )

    freshness_requirements: list[str] = []
    if any(keyword in prompt.lower() for keyword in _LATEST_KEYWORDS):
        freshness_requirements.append("Time-sensitive claims should name specific dates.")

    required_numeric_checks: list[str] = []
    if "equity" in brief.domain_packs:
        required_numeric_checks.append(
            "Discuss at least one valuation or financial metric with citation support."
        )
    if "macro" in brief.domain_packs:
        required_numeric_checks.append(
            "Discuss at least one macro series or policy-sensitive metric with citation support."
        )
    if "commodity" in brief.domain_packs:
        required_numeric_checks.append(
            "Discuss at least one inventory, positioning, or curve metric with citation support."
        )

    counterevidence_topics = ["Bear case and adverse scenarios"]
    if "equity" in brief.domain_packs:
        counterevidence_topics.append("Peer and competitor pressure")
    if "macro" in brief.domain_packs:
        counterevidence_topics.append("Alternative macro scenarios and policy failure modes")
    if "commodity" in brief.domain_packs:
        counterevidence_topics.append("Supply surprise, squeeze risk, and curve-regime changes")

    return ResearchPlan(
        primary_question=brief.primary_question,
        sub_questions=brief.sub_questions,
        domain_packs=brief.domain_packs,
        required_sections=brief.likely_report_shape or _likely_sections(brief.domain_packs),
        required_evidence=["Grounded primary evidence", "Recent discovery evidence"],
        freshness_requirements=freshness_requirements,
        required_numeric_checks=required_numeric_checks,
        counterevidence_topics=counterevidence_topics,
        steps=steps,
        rationale="Heuristic plan compiled from prompt features and domain-specific step templates.",
    )


def _filter_plan_to_registry(plan: ResearchPlan, registry_map: dict[str, object]) -> ResearchPlan:
    filtered_steps: list[ResearchStep] = []
    for step in plan.steps:
        filtered_calls = [
            call for call in step.recommended_skill_calls if call.name in registry_map
        ]
        if not filtered_calls:
            continue
        filtered_steps.append(step.model_copy(update={"recommended_skill_calls": filtered_calls}))

    filtered_step_ids = {step.id for step in filtered_steps}
    normalized_steps = []
    for step in filtered_steps:
        normalized_steps.append(
            step.model_copy(
                update={
                    "depends_on": [dep for dep in step.depends_on if dep in filtered_step_ids]
                }
            )
        )
    return plan.model_copy(update={"steps": normalized_steps})


def _ensure_profile_specific_steps(
    brief: ResearchBrief,
    plan: ResearchPlan,
    *,
    research_profile: str,
) -> ResearchPlan:
    if research_profile != "deep":
        return plan
    has_deep_retrieval = any(
        call.name == "deep_retrieval"
        for step in plan.steps
        for call in step.recommended_skill_calls
    )
    if has_deep_retrieval:
        return plan
    deep_seed = _fallback_plan(brief, research_profile="deep").steps[0]
    return plan.model_copy(update={"steps": [deep_seed, *plan.steps]})


def validate_research_plan(plan: ResearchPlan, registry: dict[str, object] | None = None) -> ResearchPlan:
    """Reject illegal packs, illegal skill names, and pack/skill mismatches."""

    plan = ResearchPlan.model_validate(plan.model_dump(mode="json"))
    registry_map = registry or build_skill_registry()
    for pack in plan.domain_packs:
        if pack not in DOMAIN_PACKS:
            raise ValueError(f"Illegal domain pack in plan: {pack}")

    legal_packs = {"core", *plan.domain_packs}
    for step in plan.steps:
        if step.domain_pack and step.domain_pack not in plan.domain_packs:
            raise ValueError(
                f"Step '{step.id}' references domain pack '{step.domain_pack}' not present in the plan."
            )
        for call in step.recommended_skill_calls:
            spec = registry_map.get(call.name)
            if spec is None:
                raise ValueError(f"Illegal skill name in plan: {call.name}")
            if getattr(spec, "pack", None) not in legal_packs:
                raise ValueError(
                    f"Skill '{call.name}' belongs to pack '{getattr(spec, 'pack', None)}', "
                    f"which is not enabled by the plan."
                )
    return plan


def _llm_brief(prompt: str) -> ResearchBrief:
    model_name = get_model("harness", "planner")
    llm = get_llm(model_name).with_structured_output(ResearchBrief, method="json_mode")
    system_prompt = """
You are the research planner for a financial research harness.

Produce a ResearchBrief that:
- identifies only the needed domain packs from: equity, macro, commodity
- excludes core from domain_packs because core is added automatically by code
- keeps the brief concise and factual
- anticipates likely failure modes
"""
    return cast(
        ResearchBrief,
        llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=prompt)],
        ),
    )


def _llm_plan(brief: ResearchBrief, registry: dict[str, object]) -> ResearchPlan:
    model_name = get_model("harness", "planner")
    llm = get_llm(model_name).with_structured_output(ResearchPlan, method="json_mode")
    skill_lines = []
    for spec in registry.values():
        skill_lines.append(f"- {spec.name} ({spec.pack})")
    system_prompt = """
You are compiling a dependency-aware ResearchPlan for a research harness.

Rules:
- Return only domain packs from: equity, macro, commodity
- Use only legal skill names supplied by the user prompt
- Make the step graph acyclic
- Include explicit counterevidence collection
- Prefer compact, actionable steps
"""
    user_prompt = f"""Research brief:
{brief.model_dump_json(indent=2)}

Legal skills:
{chr(10).join(skill_lines)}
"""
    return cast(
        ResearchPlan,
        llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
        ),
    )


def _llm_contract(brief: ResearchBrief, plan: ResearchPlan) -> ResearchContract:
    model_name = get_model("harness", "planner")
    llm = get_llm(model_name).with_structured_output(ResearchContract, method="json_mode")
    system_prompt = """
You are defining the contract for a financial research harness.

Rules:
- Convert the plan into explicit acceptance clauses
- Keep time-sensitive requirements explicit
- Include counterevidence requirements
- Prefer required over optional only when truly mandatory
"""
    user_prompt = f"""Research brief:
{brief.model_dump_json(indent=2)}

Research plan:
{plan.model_dump_json(indent=2)}
"""
    return cast(
        ResearchContract,
        llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
        ),
    )


def plan_research(
    prompt: str,
    *,
    registry: dict[str, object] | None = None,
    selected_packs: list[str] | None = None,
    research_profile: str = "standard",
) -> tuple[ResearchBrief, ResearchPlan, ResearchContract]:
    """Run the multi-pass planner with deterministic fallbacks."""

    registry_map = registry or build_skill_registry()

    used_fallback_plan = False
    if selected_packs is not None:
        normalized = []
        for pack in selected_packs:
            pack_name = pack.strip().lower()
            if pack_name in DOMAIN_PACKS and pack_name not in normalized:
                normalized.append(pack_name)
        brief = _brief_from_selected_packs(prompt, normalized)
        plan = _fallback_plan(brief, research_profile=research_profile)
        used_fallback_plan = True
    else:
        try:
            brief = _llm_brief(prompt)
        except Exception as exc:
            print(f"Harness planner brief fallback triggered: {exc}")
            brief = _fallback_brief(prompt)

        try:
            plan = _llm_plan(brief, registry_map)
        except Exception as exc:
            print(f"Harness planner plan fallback triggered: {exc}")
            plan = _fallback_plan(brief, research_profile=research_profile)
            used_fallback_plan = True

    if used_fallback_plan:
        plan = _filter_plan_to_registry(plan, registry_map)

    plan = _ensure_profile_specific_steps(brief, plan, research_profile=research_profile)
    plan = validate_research_plan(plan, registry_map)

    try:
        if selected_packs is not None:
            raise RuntimeError("Explicit pack override uses deterministic contract templates.")
        contract = _llm_contract(brief, plan)
    except Exception as exc:
        print(f"Harness planner contract fallback triggered: {exc}")
        contract = build_default_contract(brief, plan)

    return brief, plan, contract
