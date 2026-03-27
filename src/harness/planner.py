"""Planner subagent for the harness runtime."""

from __future__ import annotations

from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from src.harness.types import (
    ContractClause,
    HarnessState,
    ResearchBrief,
    ResearchContract,
    ResearchPlan,
    ResearchStep,
    SkillCall,
)
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model


_DOMAIN_ORDER = ["equity", "macro", "commodity"]


def _get_harness_model(*roles: str) -> str:
    for role in roles:
        try:
            return get_model("harness", role)
        except Exception:
            continue
    return get_model("harness", "controller")


def _normalize_domain_packs(state: HarnessState) -> list[str]:
    packs = [pack for pack in state.enabled_packs if pack != "core"]
    seen: set[str] = set()
    ordered: list[str] = []
    for pack in [*packs, *_DOMAIN_ORDER]:
        if pack in packs and pack not in seen:
            ordered.append(pack)
            seen.add(pack)
    return ordered


def _available_skill_names(state: HarnessState) -> set[str]:
    return {spec.name for spec in state.available_skills}


def _prompt_needs_freshness(prompt: str) -> bool:
    lowered = prompt.lower()
    tokens = ("latest", "current", "today", "recent", "now", "this week", "this month")
    return any(token in lowered for token in tokens)


def _brief_fallback(state: HarnessState) -> ResearchBrief:
    prompt = state.request.user_prompt.strip()
    domain_packs = _normalize_domain_packs(state)
    likely_shape = ["Executive Summary"]
    if "equity" in domain_packs:
        likely_shape.extend(["Business Overview", "Financials", "Valuation", "Risks"])
    if "macro" in domain_packs:
        likely_shape.extend(["Macro Context", "Scenario Analysis"])
    if "commodity" in domain_packs:
        likely_shape.extend(["Physical Market", "Curve And Positioning"])

    key_unknowns = ["What evidence most directly answers the user request?"]
    if _prompt_needs_freshness(prompt):
        key_unknowns.append("What sources are freshest enough to support latest-state claims?")

    return ResearchBrief(
        primary_question=prompt,
        sub_questions=[prompt],
        domain_packs=domain_packs,
        user_constraints=["Ground the answer in collected evidence."],
        likely_report_shape=likely_shape,
        key_unknowns=key_unknowns,
        likely_risks_of_failure=[
            "Missing fresh evidence.",
            "Missing counter-evidence.",
            "Missing citations in the final draft.",
        ],
        rationale="Heuristic planner fallback derived report shape from selected domain packs.",
    )


def build_research_brief(state: HarnessState) -> ResearchBrief:
    """Produce a high-level brief for the current request."""

    system_prompt = """
You are planning a financial research task.
Return a structured research brief that identifies:
- the main question,
- sub-questions,
- domain packs needed (equity, macro, commodity),
- likely report sections,
- likely failure risks,
- important constraints.
Do not include the core pack; runtime adds it automatically.
"""
    user_prompt = f"""User request:
{state.request.user_prompt}

Currently enabled domain packs:
{", ".join(_normalize_domain_packs(state)) or "none"}

Available skills:
{", ".join(sorted(_available_skill_names(state)))}
"""
    try:
        llm = get_llm(_get_harness_model("planner", "controller")).with_structured_output(
            ResearchBrief,
            method="json_mode",
        )
        result = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return cast(ResearchBrief, result)
    except Exception as exc:
        print(f"Harness planner brief fallback triggered: {exc}")
        return _brief_fallback(state)


def _core_collection_calls(state: HarnessState) -> list[SkillCall]:
    skill_names = _available_skill_names(state)
    prompt = state.request.user_prompt
    calls: list[SkillCall] = []
    if "search_and_read" in skill_names:
        calls.append(
            SkillCall(
                name="search_and_read",
                arguments={
                    "queries": [prompt],
                    "urls_per_query": 2,
                    "use_news": _prompt_needs_freshness(prompt),
                },
            )
        )
    elif "search_web" in skill_names:
        calls.append(SkillCall(name="search_web", arguments={"query": prompt, "max_results": 5}))
    return calls


def _plan_fallback(state: HarnessState, brief: ResearchBrief) -> ResearchPlan:
    skill_names = _available_skill_names(state)
    steps: list[ResearchStep] = []

    core_calls = _core_collection_calls(state)
    if core_calls:
        steps.append(
            ResearchStep(
                id="core_context",
                objective="Collect broad context and recent evidence for the user request.",
                recommended_skill_calls=core_calls,
                required_outputs=["Initial grounded evidence"],
                completion_criteria=["At least one evidence item is collected."],
                can_run_parallel=False,
            )
        )

    if "equity" in brief.domain_packs:
        if "fetch_company_profile" in skill_names:
            steps.append(
                ResearchStep(
                    id="equity_profile",
                    objective="Collect company profile and business context.",
                    recommended_skill_calls=[
                        SkillCall(name="fetch_company_profile", arguments=_infer_ticker_args(state))
                    ],
                    required_outputs=["Company profile artifact"],
                    completion_criteria=["Business context is available in evidence or artifacts."],
                    can_run_parallel=True,
                )
            )
        if "fetch_financials" in skill_names:
            steps.append(
                ResearchStep(
                    id="equity_financials",
                    objective="Collect financial statement and ratio data.",
                    recommended_skill_calls=[
                        SkillCall(name="fetch_financials", arguments=_infer_ticker_args(state))
                    ],
                    required_outputs=["Financial metrics artifact"],
                    completion_criteria=["Financial evidence is available for valuation or risk analysis."],
                    can_run_parallel=True,
                )
            )
        if "fetch_market_data" in skill_names:
            steps.append(
                ResearchStep(
                    id="equity_market_data",
                    objective="Collect market data and historical price context.",
                    recommended_skill_calls=[
                        SkillCall(name="fetch_market_data", arguments={**_infer_ticker_args(state), "period": "1y"})
                    ],
                    required_outputs=["Market data artifact"],
                    completion_criteria=["Historical price evidence is available."],
                    can_run_parallel=True,
                )
            )
        if "search_sec_filings" in skill_names:
            steps.append(
                ResearchStep(
                    id="equity_filings",
                    objective="Collect recent filing evidence.",
                    recommended_skill_calls=[
                        SkillCall(name="search_sec_filings", arguments={**_infer_ticker_args(state), "max_filings": 3})
                    ],
                    required_outputs=["Recent filings evidence"],
                    completion_criteria=["At least one recent filing is collected or reviewed."],
                    can_run_parallel=True,
                )
            )
        if "analyze_peers" in skill_names:
            steps.append(
                ResearchStep(
                    id="equity_peers",
                    objective="Build a peer and competitor view.",
                    depends_on=["equity_profile", "equity_financials", "equity_filings"],
                    recommended_skill_calls=[
                        SkillCall(name="analyze_peers", arguments=_infer_ticker_args(state))
                    ],
                    required_outputs=["Peer analysis artifact"],
                    completion_criteria=["Peer evidence is available."],
                )
            )
        if "search_and_read" in skill_names:
            steps.append(
                ResearchStep(
                    id="equity_counterevidence",
                    objective="Collect counter-evidence and competitor pressure signals.",
                    depends_on=["core_context"],
                    recommended_skill_calls=[
                        SkillCall(
                            name="search_and_read",
                            arguments={
                                "queries": [
                                    f"{state.request.user_prompt} risks competition peers bearish case",
                                ],
                                "urls_per_query": 2,
                                "use_news": _prompt_needs_freshness(state.request.user_prompt),
                            },
                        )
                    ],
                    required_outputs=["Counter-evidence and competitor evidence"],
                    completion_criteria=["At least one counter-evidence source is collected."],
                    counterevidence=True,
                )
            )

    if "macro" in brief.domain_packs:
        if "fetch_macro_indicators" in skill_names:
            steps.append(
                ResearchStep(
                    id="macro_indicators",
                    objective="Collect macro indicators relevant to the topic.",
                    recommended_skill_calls=[
                        SkillCall(
                            name="fetch_macro_indicators",
                            arguments={"topic": state.request.user_prompt, "countries": ["US"]},
                        )
                    ],
                    required_outputs=["Macro indicator artifact"],
                    completion_criteria=["At least one indicator artifact exists."],
                    can_run_parallel=True,
                )
            )
        if "fetch_world_bank_indicators" in skill_names:
            steps.append(
                ResearchStep(
                    id="macro_world_bank",
                    objective="Collect cross-country indicator context when useful.",
                    recommended_skill_calls=[
                        SkillCall(
                            name="fetch_world_bank_indicators",
                            arguments={"countries": ["USA"], "date_range": "2019:2025"},
                        )
                    ],
                    required_outputs=["World Bank artifact"],
                    completion_criteria=["Cross-country indicator artifact exists."],
                    can_run_parallel=True,
                )
            )
        if "search_and_read" in skill_names:
            steps.append(
                ResearchStep(
                    id="macro_counterevidence",
                    objective="Collect macro scenario counterpoints and policy failure modes.",
                    depends_on=["core_context"],
                    recommended_skill_calls=[
                        SkillCall(
                            name="search_and_read",
                            arguments={
                                "queries": [f"{state.request.user_prompt} downside scenario risks"],
                                "urls_per_query": 2,
                                "use_news": _prompt_needs_freshness(state.request.user_prompt),
                            },
                        )
                    ],
                    required_outputs=["Macro counter-evidence"],
                    completion_criteria=["At least one downside macro scenario source is collected."],
                    counterevidence=True,
                )
            )

    if "commodity" in brief.domain_packs:
        if "fetch_eia_inventory" in skill_names:
            steps.append(
                ResearchStep(
                    id="commodity_eia",
                    objective="Collect physical market inventory or production evidence.",
                    recommended_skill_calls=[
                        SkillCall(name="fetch_eia_inventory", arguments={"asset": state.request.user_prompt})
                    ],
                    required_outputs=["EIA artifact"],
                    completion_criteria=["A physical-market artifact exists."],
                    can_run_parallel=True,
                )
            )
        if "fetch_cot_report" in skill_names:
            steps.append(
                ResearchStep(
                    id="commodity_cot",
                    objective="Collect positioning evidence.",
                    recommended_skill_calls=[
                        SkillCall(name="fetch_cot_report", arguments={"asset": state.request.user_prompt, "num_weeks": 12})
                    ],
                    required_outputs=["COT artifact"],
                    completion_criteria=["A positioning artifact exists."],
                    can_run_parallel=True,
                )
            )
        if "fetch_futures_curve" in skill_names:
            steps.append(
                ResearchStep(
                    id="commodity_curve",
                    objective="Collect futures curve structure evidence.",
                    recommended_skill_calls=[
                        SkillCall(name="fetch_futures_curve", arguments={"asset": state.request.user_prompt, "num_contracts": 12})
                    ],
                    required_outputs=["Futures curve artifact"],
                    completion_criteria=["A curve artifact exists."],
                    can_run_parallel=True,
                )
            )
        if "search_and_read" in skill_names:
            steps.append(
                ResearchStep(
                    id="commodity_counterevidence",
                    objective="Collect commodity downside scenarios and regime-change risk.",
                    depends_on=["core_context"],
                    recommended_skill_calls=[
                        SkillCall(
                            name="search_and_read",
                            arguments={
                                "queries": [f"{state.request.user_prompt} downside scenario oversupply risk"],
                                "urls_per_query": 2,
                                "use_news": _prompt_needs_freshness(state.request.user_prompt),
                            },
                        )
                    ],
                    required_outputs=["Commodity counter-evidence"],
                    completion_criteria=["At least one downside commodity scenario source is collected."],
                    counterevidence=True,
                )
            )

    required_sections = list(dict.fromkeys(brief.likely_report_shape or ["Answer", "Sources"]))
    required_evidence = ["Grounded sources", "Inline citations"]
    freshness_requirements = []
    if _prompt_needs_freshness(state.request.user_prompt):
        freshness_requirements.append("Any latest-state claim must cite explicit dates.")
    required_numeric_checks = ["Claims with numbers should map to cited evidence."]
    counterevidence_topics = []
    if "equity" in brief.domain_packs:
        counterevidence_topics.extend(["Bear case", "Competitor pressure", "Peer valuation challenge"])
    if "macro" in brief.domain_packs:
        counterevidence_topics.extend(["Downside scenario", "Policy failure mode"])
    if "commodity" in brief.domain_packs:
        counterevidence_topics.extend(["Oversupply risk", "Positioning unwind", "Curve regime change"])

    return ResearchPlan(
        primary_question=brief.primary_question,
        sub_questions=brief.sub_questions,
        domain_packs=brief.domain_packs,
        required_sections=required_sections,
        required_evidence=required_evidence,
        freshness_requirements=freshness_requirements,
        required_numeric_checks=required_numeric_checks,
        counterevidence_topics=counterevidence_topics,
        steps=steps,
        rationale="Heuristic plan fallback derived steps from enabled packs and available skills.",
    )


def compile_research_plan(state: HarnessState, brief: ResearchBrief) -> ResearchPlan:
    """Compile a step graph from a research brief."""

    system_prompt = """
You are compiling a financial research brief into an executable plan.
Return a structured plan with:
- domain packs,
- required sections,
- required evidence,
- freshness requirements,
- numeric checks,
- counterevidence topics,
- dependency-aware steps.
Use only legal skill names from the available skills list.
"""
    user_prompt = f"""Research brief:
{brief.model_dump_json(indent=2)}

Available skills:
{", ".join(sorted(_available_skill_names(state)))}
"""
    try:
        llm = get_llm(_get_harness_model("planner_compile", "planner", "controller")).with_structured_output(
            ResearchPlan,
            method="json_mode",
        )
        result = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return cast(ResearchPlan, result)
    except Exception as exc:
        print(f"Harness planner plan fallback triggered: {exc}")
        return _plan_fallback(state, brief)


def _base_contract_clauses(brief: ResearchBrief, plan: ResearchPlan) -> ResearchContract:
    global_clauses = [
        ContractClause(id="global_dates", category="freshness", text="Any latest-state claim must cite explicit dates."),
        ContractClause(id="global_citations", category="citations", text="Each major claim should cite evidence IDs."),
        ContractClause(id="global_inference", category="reasoning", text="Any inference must be labeled clearly as inference."),
    ]
    section_clauses = [
        ContractClause(
            id=f"section_{idx}",
            category="sections",
            text=f"Include section: {section}",
            applies_to_sections=[section],
        )
        for idx, section in enumerate(plan.required_sections, start=1)
    ]
    step_clauses = [
        ContractClause(
            id=f"step_{step.id}",
            category="step",
            text=f"Step '{step.objective}' must complete before finalization.",
            applies_to_steps=[step.id],
        )
        for step in plan.steps
    ]
    freshness_clauses = [
        ContractClause(
            id=f"fresh_{idx}",
            category="freshness",
            text=text,
        )
        for idx, text in enumerate(plan.freshness_requirements, start=1)
    ]
    numeric_clauses = [
        ContractClause(
            id=f"num_{idx}",
            category="numeric",
            text=text,
        )
        for idx, text in enumerate(plan.required_numeric_checks, start=1)
    ]
    counterevidence_clauses = [
        ContractClause(
            id=f"counter_{idx}",
            category="counterevidence",
            text=f"Address counter-evidence topic: {topic}",
        )
        for idx, topic in enumerate(plan.counterevidence_topics, start=1)
    ]
    if "equity" in brief.domain_packs:
        counterevidence_clauses.append(
            ContractClause(
                id="counter_equity_competitors",
                category="counterevidence",
                text="When relevant, include competitor or peer evidence that challenges the thesis.",
            )
        )
    return ResearchContract(
        global_clauses=global_clauses,
        section_clauses=section_clauses,
        step_clauses=step_clauses,
        freshness_clauses=freshness_clauses,
        numeric_clauses=numeric_clauses,
        counterevidence_clauses=counterevidence_clauses,
    )


def build_research_contract(state: HarnessState, brief: ResearchBrief, plan: ResearchPlan) -> ResearchContract:
    """Build a research contract from templates and prompt-specific needs."""

    system_prompt = """
You are building a research contract for a financial report.
Merge default contract clauses with prompt-specific clauses.
Return structured clauses only.
"""
    user_prompt = f"""Research brief:
{brief.model_dump_json(indent=2)}

Research plan:
{plan.model_dump_json(indent=2)}
"""
    try:
        llm = get_llm(_get_harness_model("contract_builder", "planner", "verify")).with_structured_output(
            ResearchContract,
            method="json_mode",
        )
        result = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return cast(ResearchContract, result)
    except Exception as exc:
        print(f"Harness planner contract fallback triggered: {exc}")
        return _base_contract_clauses(brief, plan)


def plan_research(state: HarnessState) -> tuple[ResearchBrief, ResearchPlan, ResearchContract]:
    """Run the full multi-pass planner pipeline."""

    brief = build_research_brief(state)
    plan = compile_research_plan(state, brief)
    contract = build_research_contract(state, brief, plan)
    return brief, plan, contract


def plan_research_fallback(state: HarnessState) -> tuple[ResearchBrief, ResearchPlan, ResearchContract]:
    """Run the deterministic planner fallback without any model calls."""

    brief = _brief_fallback(state)
    plan = _plan_fallback(state, brief)
    contract = _base_contract_clauses(brief, plan)
    return brief, plan, contract


def validate_research_plan(plan: ResearchPlan, available_skill_names: set[str] | None = None) -> None:
    """Validate the research plan's domain packs, steps, and dependencies."""

    valid_packs = set(_DOMAIN_ORDER)
    invalid_packs = [pack for pack in plan.domain_packs if pack not in valid_packs]
    if invalid_packs:
        raise ValueError(f"Invalid domain packs in research plan: {invalid_packs}")

    step_ids = [step.id for step in plan.steps]
    if len(step_ids) != len(set(step_ids)):
        raise ValueError("Research plan contains duplicate step IDs.")

    known_step_ids = set(step_ids)
    for step in plan.steps:
        unknown_dependencies = [dep for dep in step.depends_on if dep not in known_step_ids]
        if unknown_dependencies:
            raise ValueError(f"Step '{step.id}' depends on unknown steps: {unknown_dependencies}")
        if available_skill_names is not None:
            unknown_skills = [
                call.name for call in step.recommended_skill_calls if call.name not in available_skill_names
            ]
            if unknown_skills:
                raise ValueError(f"Step '{step.id}' uses unknown skills: {unknown_skills}")

    visiting: set[str] = set()
    visited: set[str] = set()
    dependency_map = {step.id: step.depends_on for step in plan.steps}

    def _visit(step_id: str) -> None:
        if step_id in visited:
            return
        if step_id in visiting:
            raise ValueError("Research plan contains cyclic step dependencies.")
        visiting.add(step_id)
        for dependency in dependency_map.get(step_id, []):
            _visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in step_ids:
        _visit(step_id)


def _infer_ticker_args(state: HarnessState) -> dict[str, str]:
    prompt = state.request.user_prompt.strip()
    tokens = [token.strip(",.()").upper() for token in prompt.split()]
    ticker = next((token for token in tokens if 1 <= len(token) <= 5 and token.isalpha() and token == token.upper()), "")
    return {"ticker": ticker} if ticker else {}
