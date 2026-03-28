"""Deterministic contract templates and contract-building helpers."""

from __future__ import annotations

from src.harness.types import ContractClause, ResearchBrief, ResearchContract, ResearchPlan


def _clause(
    clause_id: str,
    category: str,
    text: str,
    severity: str,
    *,
    steps: list[str] | None = None,
    sections: list[str] | None = None,
) -> ContractClause:
    return ContractClause(
        id=clause_id,
        category=category,
        text=text,
        severity=severity,  # type: ignore[arg-type]
        applies_to_steps=steps or [],
        applies_to_sections=sections or [],
    )


DEFAULT_GLOBAL_CLAUSES = [
    _clause(
        "global_dates_for_latest_claims",
        "freshness",
        "Any latest, current, or recent claim must include an explicit date or date range.",
        "required",
    ),
    _clause(
        "global_causal_inference_labeling",
        "reasoning",
        "Causal claims must distinguish observed facts from inference.",
        "required",
    ),
    _clause(
        "global_citations_for_material_claims",
        "citation",
        "Each material claim must cite at least one evidence id.",
        "required",
    ),
]


PACK_SPECIFIC_CLAUSES: dict[str, dict[str, list[ContractClause]]] = {
    "equity": {
        "section": [
            _clause(
                "equity_risk_section",
                "structure",
                "The report must include an explicit risks and counterevidence section.",
                "required",
                sections=["Risks and Counterevidence"],
            ),
            _clause(
                "equity_peer_pressure",
                "counterevidence",
                "Equity reports must include peer or competitor challenge evidence when relevant.",
                "required",
                sections=["Peer and Competitive Pressure", "Risks and Counterevidence"],
            ),
        ],
        "numeric": [
            _clause(
                "equity_valuation_support",
                "numeric",
                "Valuation claims must cite financial or market evidence.",
                "required",
                sections=["Valuation and Scenarios", "Key Findings"],
            )
        ],
        "counter": [
            _clause(
                "equity_bull_vs_bear",
                "counterevidence",
                "Bullish conclusions must discuss material downside risks such as margin pressure, regulation, balance-sheet stress, or competition.",
                "required",
                sections=["Risks and Counterevidence"],
            )
        ],
    },
    "macro": {
        "section": [
            _clause(
                "macro_scenarios",
                "structure",
                "Macro reports must include at least one alternative scenario or failure mode.",
                "required",
                sections=["Scenarios", "Risks and Counterevidence"],
            )
        ],
        "counter": [
            _clause(
                "macro_conflicting_indicators",
                "counterevidence",
                "Conflicting indicators or policy failure modes must be discussed explicitly.",
                "required",
                sections=["Risks and Counterevidence", "Scenarios"],
            )
        ],
    },
    "commodity": {
        "section": [
            _clause(
                "commodity_curve_requirement",
                "structure",
                "Commodity reports must include curve structure, inventory, or positioning evidence when those data are in scope.",
                "required",
                sections=["Commodity Balance", "Risks and Counterevidence"],
            )
        ],
        "counter": [
            _clause(
                "commodity_supply_surprise",
                "counterevidence",
                "Commodity reports must discuss supply surprises, squeeze risk, or curve-regime changes.",
                "required",
                sections=["Risks and Counterevidence"],
            )
        ],
    },
}


def build_default_contract(brief: ResearchBrief, plan: ResearchPlan) -> ResearchContract:
    """Build a deterministic contract from defaults, domain packs, and prompt cues."""

    section_clauses: list[ContractClause] = []
    numeric_clauses: list[ContractClause] = []
    counter_clauses: list[ContractClause] = []
    step_clauses: list[ContractClause] = []
    freshness_clauses: list[ContractClause] = []

    for section in plan.required_sections:
        section_clauses.append(
            _clause(
                f"section_{section.lower().replace(' ', '_')}",
                "structure",
                f"The report must include a '{section}' section.",
                "required",
                sections=[section],
            )
        )

    for pack in plan.domain_packs:
        template = PACK_SPECIFIC_CLAUSES.get(pack, {})
        section_clauses.extend(template.get("section", []))
        numeric_clauses.extend(template.get("numeric", []))
        counter_clauses.extend(template.get("counter", []))

    for step in plan.steps:
        if step.required_outputs:
            step_clauses.append(
                _clause(
                    f"step_{step.id}_outputs",
                    "execution",
                    f"Step '{step.id}' must produce: {', '.join(step.required_outputs)}.",
                    "important",
                    steps=[step.id],
                    sections=step.report_sections,
                )
            )

    prompt_text = " ".join(
        [
            brief.primary_question,
            *brief.sub_questions,
            *brief.user_constraints,
        ]
    ).lower()
    if any(keyword in prompt_text for keyword in ("latest", "current", "today", "recent")):
        freshness_clauses.append(
            _clause(
                "prompt_freshness_explicit_dates",
                "freshness",
                "Time-sensitive prompts require explicit dates in the final report.",
                "required",
            )
        )
    for requirement in plan.freshness_requirements:
        freshness_clauses.append(
            _clause(
                f"freshness_{len(freshness_clauses) + 1}",
                "freshness",
                requirement,
                "important",
            )
        )

    for numeric_check in plan.required_numeric_checks:
        numeric_clauses.append(
            _clause(
                f"numeric_{len(numeric_clauses) + 1}",
                "numeric",
                numeric_check,
                "important",
            )
        )

    for topic in plan.counterevidence_topics:
        counter_clauses.append(
            _clause(
                f"counter_{len(counter_clauses) + 1}",
                "counterevidence",
                f"The report must address counterevidence related to: {topic}.",
                "required",
                sections=["Risks and Counterevidence"],
            )
        )

    if len(plan.domain_packs) > 1:
        section_clauses.append(
            _clause(
                "cross_domain_section_coverage",
                "structure",
                "Cross-domain prompts must include at least one section per planned domain pack.",
                "required",
            )
        )

    return ResearchContract(
        global_clauses=DEFAULT_GLOBAL_CLAUSES.copy(),
        section_clauses=section_clauses,
        step_clauses=step_clauses,
        freshness_clauses=freshness_clauses,
        numeric_clauses=numeric_clauses,
        counterevidence_clauses=counter_clauses,
    )
