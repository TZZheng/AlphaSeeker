# Harness Design Task

Based on: [Anthropic, "Harness design for long-running application development"](https://www.anthropic.com/engineering/harness-design-long-running-apps), published March 24, 2026.

## Purpose

This document proposes a coherent next-step design for AlphaSeeker's `src/harness` runtime.

The goal is not to copy Anthropic's coding harness directly. Their system is built for long-running software development. Our system is built for financial research. The useful transfer is architectural:

- define the work clearly before execution,
- keep structured artifacts on disk,
- separate execution from evaluation,
- preserve progress across long runs,
- and let each layer do one job well.

The immediate objective is to build a reliable harness skeleton that does not lose the task, does not drift during long runs, and can later support more capable planning and evaluation.

## Current Harness

Today the harness is a compact loop:

`selector -> controller -> skill calls -> writer -> verifier -> finalize`

That minimal design is valuable. It gives us:

- bounded execution through `max_steps` and `max_revision_rounds`,
- a normalized evidence ledger,
- a persisted report and trace,
- simple test injection for controller, writer, and verifier.

However, the current harness is still missing several pieces that matter for research quality and runtime stability:

- there is no true planner,
- there is no durable mission or progress document that gets reread every phase,
- the verifier mainly judges the final draft instead of inspecting the whole run,
- revision behavior is too generic,
- step execution is still too shallow for complex research tasks,
- the runtime has no checkpoint-and-resume model,
- and counter-evidence is not treated as a first-class requirement.

These weaknesses are especially important for long or cross-domain prompts, where the agent can forget the central task or produce a polished but incomplete report.

## Design Principles

The proposed design is guided by six principles.

### 1. Persist the task outside the model context

The model should not rely only on conversational context to remember what it is doing.

Instead, the harness should keep a small set of explicit artifacts on disk and reread them at major phases. This is the most direct adaptation of the article's handoff-artifact idea to research.

### 2. Separate planning, execution, writing, and evaluation

These are different jobs and should not be forced into one prompt. A model can do them all, but not optimally in one shot. Each role should produce typed outputs that code can validate.

### 3. Treat execution as structured work, not as an open-ended chat

The runtime should execute explicit research steps with dependencies, outputs, and completion criteria. This is closer to a workflow engine than a free-form agent loop.

If the term "workflow engine" is unfamiliar, it means a control layer that runs a series of tasks according to state and dependency rules.

### 4. Evaluate with a skeptical external reviewer

The evaluator should not only score prose. It should inspect:

- evidence,
- artifacts,
- structured findings,
- required sections,
- numerical checks,
- freshness requirements,
- and counter-evidence coverage.

### 5. Prefer structured artifacts over long text summaries

The current trace is helpful, but too much meaning still lives in free text. Structured files let the harness preserve state more reliably, resume work later, and debug failures more easily.

### 6. Keep the harness under review as models improve

Every harness component encodes an assumption about model weakness. Those assumptions will go stale. The system should be designed so that components can be simplified or removed when they are no longer load-bearing.

## Proposed Runtime Architecture

The next harness should be organized into five roles:

1. planner subagent
2. controller/orchestrator
3. step worker subagent
4. writer
5. evaluator

This changes the current architecture in one important way: the current `selector` is too narrow a name if it starts creating plans and steps. Once it designs the work, it is no longer just a selector. It becomes a planner or initializer subagent.

## Persistent Mission And Progress Dossier

The first requirement is a durable on-disk skeleton for the run.

Each run should maintain a small dossier of artifacts:

- `mission.md`
- `progress.md`
- `research_plan.json`
- `research_contract.json`
- `qa_report.json`

These files serve different purposes.

### `mission.md`

This is the stable task brief. It should contain:

- the original user question,
- user constraints,
- current objective,
- important definitions,
- what success means at a high level.

### `progress.md`

This is the evolving work log. It should contain:

- completed steps,
- open steps,
- key findings so far,
- unresolved questions,
- next scheduled actions.

### `research_plan.json`

This is the structured plan and dependency graph.

### `research_contract.json`

This is the definition of done, expressed as explicit acceptance clauses.

### `qa_report.json`

This is the evaluator's structured report of what passed, what failed, and what must happen next.

These artifacts should be reread by the planner, controller, writer, and evaluator on every major phase. That repeated rereading is a deliberate anti-forgetting mechanism.

## Domain Packs And Core Skills

The harness currently has four packs in practice:

- `core`
- `equity`
- `macro`
- `commodity`

`core` should always be active in code and should not need to be returned by the planner.

The planner should return only the domain packs required by the user question. For example:

- `["equity"]` for a single-company research prompt
- `["equity", "macro"]` for a banking prompt driven by rates
- `["commodity", "macro"]` for an oil prompt that depends on macro catalysts

At runtime, the enabled packs should always be:

- `core`
- plus the planned domain packs

This keeps the plan focused on domain coverage while the code handles invariant behavior.

## Planner Design

Planning should be a real subagent, not a single helper prompt.

The planner's job is to convert the raw user request into three increasingly concrete artifacts:

1. `ResearchBrief`
2. `ResearchPlan`
3. `ResearchContract`

This should be done in multiple passes because these are different transformations.

### Planner Pass A: Research Brief

This pass answers: what is the user actually asking, what domains are involved, and what report shape is likely required?

Suggested output:

```python
class ResearchBrief(BaseModel):
    primary_question: str
    sub_questions: list[str]
    domain_packs: list[str]
    user_constraints: list[str]
    likely_report_shape: list[str]
    key_unknowns: list[str]
    likely_risks_of_failure: list[str]
    rationale: str
```

This pass should use a strong model because it is the main interpretation step.

### Planner Pass B: Step Graph Compiler

This pass answers: what steps are needed, in what order, with what dependencies, and which skills are relevant to each step?

Suggested output:

```python
class ResearchStep(BaseModel):
    id: str
    objective: str
    depends_on: list[str] = []
    recommended_skill_calls: list[SkillCall] = []
    required_outputs: list[str] = []
    completion_criteria: list[str] = []


class ResearchPlan(BaseModel):
    primary_question: str
    sub_questions: list[str]
    domain_packs: list[str]
    required_sections: list[str]
    required_evidence: list[str]
    freshness_requirements: list[str]
    required_numeric_checks: list[str]
    counterevidence_topics: list[str]
    steps: list[ResearchStep]
    rationale: str = ""
```

This is where the harness becomes dependency-aware.

For example, an equity prompt may produce steps like:

1. collect company profile
2. collect financial statements
3. collect market data
4. collect recent filings
5. build peer view
6. synthesize valuation and risk

Some steps are independent and can be parallelized. Some depend on earlier outputs. `analyze_peers`, for example, often depends on prior evidence from profile, filings, or financials.

### Planner Pass C: Contract Builder

This pass answers: what exactly counts as done, and what would cause evaluation to fail?

The contract should combine:

- default contract templates,
- domain-specific mandatory clauses,
- prompt-specific additional clauses.

Suggested output:

```python
class ContractClause(BaseModel):
    id: str
    category: str
    text: str
    severity: Literal["required", "important", "optional"]
    applies_to_steps: list[str] = []
    applies_to_sections: list[str] = []


class ResearchContract(BaseModel):
    global_clauses: list[ContractClause]
    section_clauses: list[ContractClause]
    step_clauses: list[ContractClause]
    freshness_clauses: list[ContractClause]
    numeric_clauses: list[ContractClause]
    counterevidence_clauses: list[ContractClause]
```

The contract should not be created from scratch every time. It should start from pre-filled templates in code or configuration, then let the planner add prompt-specific clauses.

Examples of default clauses:

- any "latest" claim must carry explicit dates,
- valuation claims must cite financial or market evidence,
- causal claims must distinguish fact from inference,
- cross-domain prompts must contain at least one section per planned domain pack,
- bullish conclusions must include material risk discussion.

Examples of prompt-specific clauses:

- if the prompt asks about AI competition, the contract may require competitor evidence related to AI product overlap,
- if the prompt asks about rates sensitivity, the contract may require explicit macro transmission logic,
- if the prompt asks about commodity curve structure, the contract may require curve evidence and positioning evidence.

## Turning Planner Output Into Actionables

The runtime should not treat planner output as free text. It should treat it as typed data.

The code path should be:

1. parse planner outputs into typed models
2. validate that packs and skill names are legal
3. add `core` automatically
4. normalize `ResearchStep` items into a dependency graph
5. persist the mission, plan, and contract artifacts
6. hand the next executable step to a worker

The key actionable unit for execution is therefore a validated `ResearchStep`, not a generic paragraph of planner advice.

## Controller Design

The controller should not invent the work. Its job is to schedule the work.

It should read:

- `mission.md`
- `progress.md`
- `research_plan.json`
- `research_contract.json`
- current step states
- evaluator feedback, if any

Then it should decide:

- which steps are currently unblocked,
- whether any steps can run in parallel,
- whether the report is ready for writing,
- whether evaluator feedback requires new steps or rework.

This is a scheduling role, not a planning role.

## Step Worker Design

Each `ResearchStep` may require more than one action. A step may need:

- multiple skill calls,
- local reasoning over returned data,
- artifact reads,
- progress writes,
- and one or more internal retries before it can be considered complete.

For that reason, step execution should be delegated to a step worker subagent with its own bounded inner loop.

Suggested output:

```python
class StepExecutionResult(BaseModel):
    step_id: str
    status: Literal["completed", "partial", "blocked", "failed"]
    summary: str
    evidence_ids: list[str] = []
    artifact_paths: list[str] = []
    findings: list[str] = []
    open_questions: list[str] = []
    suggested_next_steps: list[str] = []
```

The worker should update `progress.md` and add structured artifacts as it completes work.

If the term "bounded inner loop" is unfamiliar, it means the worker can iterate several times internally, but only up to an explicit limit.

## Writing Model

The writer should no longer draft from raw working memory alone. It should draft from:

- mission,
- plan,
- contract,
- step results,
- evidence ledger,
- claim map,
- and evaluator feedback from prior rounds.

The writer's job is to produce a report that satisfies the contract, not just a fluent answer.

## Evaluator Design

The current verifier should evolve into a stronger evaluator.

The evaluator should inspect:

- the final draft,
- evidence ledger entries,
- step results,
- artifact paths,
- required sections,
- required numeric checks,
- freshness requirements,
- counter-evidence coverage.

It should produce structured feedback rather than a single global judgment.

Suggested additions:

```python
class ReportSectionFeedback(BaseModel):
    section_label: str
    quoted_text: str = ""
    issue: str
    why_it_fails: str
    suggested_fix: str
    missing_evidence_ids: list[str] = []
```

And a richer evaluator report should include fields such as:

- `blocking_issues`
- `missing_sections`
- `missing_evidence_types`
- `missing_citations`
- `freshness_warnings`
- `numeric_inconsistencies`
- `required_follow_up_calls`
- `counterevidence_gaps`
- `report_section_feedback`

This is important for two reasons.

First, the evaluator can direct the system toward concrete next actions instead of saying only "revise."

Second, it can point to exactly which part of the report is weak and why, so the writer can repair only the failing sections instead of rewriting everything.

## Counter-Evidence And Competitor Evidence

Counter-evidence should be a required part of the harness, not an optional extra.

In finance, a report that only collects supportive evidence is often misleading even when it is well cited.

The plan and contract should therefore require explicit counter-evidence coverage.

Examples:

- equity: bearish risks, margin pressure, regulation, balance-sheet concerns, and competitor or peer evidence that weakens the thesis
- macro: alternative scenarios, policy failure modes, conflicting indicators
- commodity: supply surprises, positioning squeeze risk, and curve regime changes

For equity specifically, competitor and peer evidence should be treated as a first-class category of counter-evidence. The harness should look for:

- peers with stronger growth,
- peers with stronger margins,
- peers trading at more attractive valuation multiples,
- substitute products,
- market-share loss evidence,
- moat erosion,
- customer concentration relative to competitors.

## Structured Research Artifacts

In addition to the final report and full trace, the runtime should preserve more structured intermediate artifacts:

- `research_plan.json`
- `research_contract.json`
- `findings.json`
- `claim_map.json`
- `qa_report.json`

The `claim_map` is especially important. It should represent claims in a normalized structure.

If the term "normalized" is unfamiliar in database work, it means each important object is stored once in a clean structured schema instead of being duplicated across many free-text fragments.

A `claim_map` record should capture:

- claim text,
- whether it is fact or inference,
- supporting evidence IDs,
- complicating evidence IDs,
- freshness date,
- whether the claim appears in the final report.

This lets the writer and evaluator work with structured reasoning instead of trying to recover everything from prose.

## Checkpoints And Resume

The current harness writes its trace at the end. That is not enough for long or fragile runs.

The runtime should persist checkpoints after each major step, for example:

- `data/harness_runs/<run_id>/checkpoint_step_001.json`
- `data/harness_runs/<run_id>/checkpoint_step_002.json`

This makes it possible to add resume behavior later.

Checkpointing and context reset are not the same thing.

- checkpointing saves the state,
- context reset starts a fresh model session from saved state.

The harness should implement checkpointing first. Reset behavior can be added later if traces show that long-context execution is degrading quality.

## Parallel Execution

The harness is currently single-skill-per-loop. That is simple, but often too slow.

The new runtime should support parallel execution for independent, I/O-bound steps.

If the term "I/O-bound" is unfamiliar, it means the program is mostly waiting on network or file operations rather than CPU-heavy computation.

Examples of parallelizable collections:

- equity: `fetch_company_profile`, `fetch_financials`, `fetch_market_data`, `search_sec_filings`
- macro: `fetch_macro_indicators` and `fetch_world_bank_indicators`
- commodity: `fetch_eia_inventory`, `fetch_cot_report`, and `fetch_futures_curve`

Dependent steps should remain sequential. For example, `plot_price_history` depends on prior market data.

## Model-Evolution Review

The harness should include a recurring review process for model capability assumptions.

Questions to ask periodically:

- does the planner still need multiple passes?
- do all prompt classes still need evaluator QA?
- does batching still help, or can stronger models handle larger steps well?
- are any artifacts no longer load-bearing?

This review should be empirical. It should be based on evaluation results, not intuition.

## Validation Strategy

Every implementation phase should define not only what to build, but how to tell whether the phase actually succeeded.

The harness should be tested in at least two live-model lanes:

1. weak-model lane
Use `sf/Qwen/Qwen3-8B` for harness planning, control, writing, and evaluation roles. This lane tests whether the harness skeleton is strong enough to keep a weaker model on task.

2. strong-model lane
Use `kimi-k2.5` for harness planning, control, writing, and evaluation roles. This lane tests the higher-quality ceiling and shows whether the harness is still useful with a stronger model.

An optional third lane is useful later:

3. mixed lane
Use `kimi-k2.5` for planner and evaluator roles, and `sf/Qwen/Qwen3-8B` for controller or step-worker-style roles. This is a realistic cost-aware deployment pattern once those roles exist explicitly in code.

The benchmark prompt set should include at least:

- a single-name equity prompt,
- an equity + macro cross-domain prompt,
- a macro scenario prompt,
- a commodity prompt,
- a time-sensitive "latest" prompt.

Suggested examples:

- `Analyze AAPL valuation and risk using current evidence.`
- `How do higher rates affect JPM and bank margins?`
- `US macro outlook for the next 12 months.`
- `Crude oil supply-demand and futures curve outlook.`
- `What is the latest evidence on copper demand and supply risks?`

For each prompt, the benchmark runner should track:

- artifact creation success,
- contract satisfaction,
- citation coverage,
- missing-section count,
- freshness failures,
- numeric inconsistency count,
- counter-evidence coverage,
- runtime,
- and token cost when available.

## Implementation Tasks And Phase Tests

### Phase 0: Stable Skeleton

Tasks:

1. Add persistent run artifacts: `mission.md`, `progress.md`, `research_plan.json`, `research_contract.json`, `qa_report.json`.
2. Make planner, controller, writer, and evaluator reread those artifacts on every major phase.
3. Add stepwise checkpoint writing, even before full resume support exists.
4. Define default contract templates in code or configuration.

Success criteria:

- every run creates the dossier files plus the final report and trace,
- `mission.md` preserves the original task accurately,
- `progress.md` is updated after each major phase,
- checkpoints are written in step order,
- contract templates load deterministically.

Tests:

1. unit test: artifact writer creates all required files with valid JSON or Markdown shape.
2. component test: a stubbed harness run updates `progress.md` after planner, controller, worker, writer, and evaluator phases.
3. live weak-model test with `sf/Qwen/Qwen3-8B`: run a simple equity prompt and assert all dossier artifacts exist and contain the task.
4. live strong-model test with `kimi-k2.5`: run the same prompt and assert the same artifact guarantees hold.

### Phase 1: Planning And Step Execution

Tasks:

1. Replace the current selector role with a planner/initializer subagent.
2. Implement multi-pass planning: `ResearchBrief` -> `ResearchPlan` -> `ResearchContract`.
3. Add typed models for `ResearchBrief`, `ResearchStep`, `ResearchPlan`, `ContractClause`, and `ResearchContract`.
4. Normalize planned steps into a dependency graph.
5. Add a step worker subagent with a bounded inner loop and typed `StepExecutionResult`.
6. Update the controller so it schedules planned steps instead of inventing work ad hoc.

Success criteria:

- the planner returns valid typed objects,
- `core` is added automatically while domain packs are planner-driven,
- dependency ordering is valid and acyclic,
- the controller schedules only unblocked steps,
- a step worker can complete a step that requires multiple internal actions,
- progress artifacts reflect per-step completion.

Tests:

1. unit test: plan validator rejects illegal packs, illegal skill names, and cyclic step dependencies.
2. component test: a cross-domain prompt produces a multi-step dependency graph and the controller schedules steps in valid order.
3. component test: a step worker completes a synthetic step that requires two internal skill calls and writes back progress.
4. live weak-model test with `sf/Qwen/Qwen3-8B`: assert the planner still produces a valid `ResearchBrief`, `ResearchPlan`, and `ResearchContract`.
5. live strong-model test with `kimi-k2.5`: verify the same prompt produces a more complete plan while remaining structurally valid.

### Phase 2: Evaluation Upgrade

Tasks:

1. Replace the current verifier contract with a stronger evaluator schema.
2. Add `report_section_feedback` so the evaluator can point to weak report sections directly.
3. Add support for `required_follow_up_calls` so evaluation can trigger concrete next actions.
4. Make the writer revise sections against evaluator feedback instead of regenerating the whole report by default.

Success criteria:

- the evaluator can fail a report for specific structural reasons,
- evaluator output identifies weak sections precisely,
- evaluator follow-up calls are executable by the runtime,
- the writer can revise targeted sections without rewriting the whole document,
- revision measurably improves contract compliance.

Tests:

1. unit test: evaluator schema accepts section-level feedback and required follow-up calls.
2. component test: seed a draft with missing citations and a weak risk section, then assert the evaluator flags the exact section and proposes follow-up actions.
3. component test: feed evaluator feedback to the writer and assert only targeted sections change.
4. live weak-model test with `sf/Qwen/Qwen3-8B`: verify the evaluator still emits structured section feedback on a deliberately flawed draft.
5. live strong-model test with `kimi-k2.5`: verify evaluator-directed revision improves citation coverage and missing-section count on the same prompt.

### Phase 3: Research Quality Improvements

Tasks:

1. Add counter-evidence search as a first-class planning and contract requirement.
2. Add explicit competitor and peer pressure checks for equity.
3. Add `claim_map.json` and supporting claim-level data structures.
4. Add domain-specific evaluator rubrics for equity, macro, and commodity reports.

Success criteria:

- the plan and contract explicitly require counter-evidence,
- equity reports include peer or competitor challenge evidence when relevant,
- `claim_map.json` links claims to supporting and complicating evidence,
- domain-specific evaluator rubrics catch issues that the generic evaluator would miss.

Tests:

1. unit test: claim-map builder correctly marks fact vs inference and attaches supporting and complicating evidence IDs.
2. component test: an equity run with peer risk signals produces competitor-oriented counter-evidence in both plan and output artifacts.
3. component test: domain-specific evaluator rubrics produce different findings for equity, macro, and commodity fixtures.
4. live weak-model test with `sf/Qwen/Qwen3-8B`: verify counter-evidence is still explicitly collected on an equity prompt.
5. live strong-model test with `kimi-k2.5`: verify competitor pressure and counter-evidence improve evaluator completeness scores.

### Phase 4: Reliability And Throughput

Tasks:

1. Add resume-from-checkpoint support.
2. Add optional context-reset execution using saved artifacts.
3. Add parallel scheduling for independent steps.
4. Add compaction rules for `mission.md` and `progress.md` so they stay small and useful.

Success criteria:

- an interrupted run can resume from checkpoint without losing state,
- context reset reproduces the same mission and progress accurately,
- parallel execution reduces wall-clock time for independent steps,
- dossier files remain compact and readable over long runs.

Tests:

1. component test: interrupt a run after checkpoint N, resume, and assert final artifacts are equivalent to a non-interrupted run.
2. component test: force a context reset and verify mission, plan, contract, and progress are restored correctly.
3. benchmark test: compare sequential vs parallel execution on a prompt with multiple independent collection steps and assert lower wall-clock time without missing artifacts.
4. live weak-model test with `sf/Qwen/Qwen3-8B`: verify resume and context reset preserve task focus.
5. live strong-model test with `kimi-k2.5`: verify parallel execution improves runtime while preserving evaluator scores.

### Phase 5: Ongoing Review

Tasks:

1. Build a harness evaluation set across equity, macro, commodity, and cross-domain prompts.
2. Track metrics such as citation coverage, freshness failures, numeric inconsistencies, missing-section count, runtime, and token cost.
3. Review which harness components are still load-bearing whenever the underlying models change materially.

Success criteria:

- there is a fixed benchmark set that can be rerun across revisions,
- regression reports show metric movement over time,
- model changes trigger explicit harness review rather than ad hoc guesses,
- at least one harness component can be justified, simplified, or removed based on data.

Tests:

1. build a repeatable benchmark runner that executes the full prompt set in weak-model and strong-model lanes.
2. generate a regression summary report comparing the current branch against a saved baseline.
3. require a documented review whenever a model role changes materially in `models.yaml`.
