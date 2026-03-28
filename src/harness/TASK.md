# Harness Design Task

Based on: [Anthropic, "Harness design for long-running application development"](https://www.anthropic.com/engineering/harness-design-long-running-apps), published March 24, 2026.

## Purpose

This document defines the next-step design for AlphaSeeker's `src/harness` runtime.

The goal is not to copy Anthropic's coding harness directly. Their system is built for long-running software development. Our system is built for financial research. The useful transfer is architectural:

- define the work clearly before execution,
- keep structured artifacts on disk,
- separate execution from evaluation,
- preserve progress across long runs,
- and let each layer do one job well.

The harness should support two profiles:

- a lighter `standard` mode for shorter bounded runs,
- and a `deep` mode for long-duration, human-competitive research reports.

The deep mode is the main target of this task. It should support tens of minutes of work, hundreds of candidate sources, repeated retrieval and QA cycles, and outputs that compete with a strong human first-draft research memo rather than a short answer.

## Current Harness

Today the harness is a compact loop:

`selector -> controller -> skill calls -> writer -> verifier -> finalize`

That design is still useful for shorter bounded runs. It gives us:

- bounded execution through `max_steps` and `max_revision_rounds`,
- a normalized evidence ledger,
- a persisted report and trace,
- and simple test injection for controller, writer, and verifier.

However, the current harness is still too shallow for deep research:

- retrieval fan-out is too small,
- there is no corpus-building stage,
- there is no gradual context-reduction pipeline,
- the writer still drafts from too little structured context,
- the evaluator does not own a true pass-or-timeout loop,
- and the runtime does not yet aim for long-form, human-competitive report depth.

These weaknesses matter most for single-name equity research, cross-domain prompts, and time-sensitive prompts where a polished but lightly sourced report is not good enough.

## Design Principles

The proposed design is guided by seven principles.

### 1. Persist the task outside the model context

The model should not rely only on conversational context to remember what it is doing.

Instead, the harness should keep explicit artifacts on disk and reread them at major phases. That repeated rereading is a deliberate anti-forgetting mechanism.

### 2. Separate planning, retrieval, execution, writing, and evaluation

These are different jobs and should not be forced into one prompt. A model can do them all, but not optimally in one shot. Each role should produce typed outputs that code can validate.

### 3. Treat retrieval as staged corpus building, not a small helper call

Deep mode should discover broadly, rank candidates, read selectively, and extract normalized facts before drafting. A deep report should be built from a corpus, not from a few snippets.

### 4. Reduce context gradually

Long-form research cannot rely on one giant prompt filled with raw article text. The system should compress information in layers:

- raw documents,
- normalized source cards,
- thematic briefs,
- section briefs,
- final report.

### 5. Evaluate with a skeptical external reviewer

The evaluator should not only score prose. It should inspect:

- evidence,
- artifacts,
- structured findings,
- required sections,
- numerical checks,
- freshness requirements,
- and counterevidence coverage.

### 6. Prefer structured artifacts over long text summaries

The current trace is helpful, but too much meaning still lives in free text. Structured files let the harness preserve state more reliably, resume work later, and debug failures more easily.

### 7. Keep the harness under review as models improve

Every harness component encodes an assumption about model weakness. Those assumptions will go stale. The system should be designed so that components can be simplified or removed when they are no longer load-bearing.

## Proposed Runtime Architecture

The next harness should be organized into five roles:

1. planner subagent
2. controller or orchestrator
3. step worker subagent
4. writer
5. evaluator

For deep mode, the controller must also manage a staged retrieval pipeline:

1. query generation
2. broad discovery
3. ranking and deduplication
4. read queue construction
5. full-text ingestion
6. structured extraction
7. section synthesis
8. report drafting
9. evaluator-driven gap fill

This changes the current architecture in one important way: the current `selector` is too narrow a name if it starts creating plans and steps. Once it designs the work, it is no longer just a selector. It becomes a planner or initializer subagent.

## Deep Research Mode

The harness should support two operating profiles:

- `standard`, which preserves a shorter bounded workflow for interactive use,
- `deep`, which is an explicit long-duration mode for extreme-quality reports.

Deep mode should not be the default behavior for all runs. It is a separate profile for prompts where the user wants depth comparable to a strong human first-draft research memo.

Deep mode is designed for tens of minutes, not seconds.

The default expectation is that the agent may work for up to 20 minutes, ingest a large corpus, and iterate through retrieval, extraction, synthesis, and QA multiple times before finalizing.

The first optimization target for deep mode is single-name equity research.

## Persistent Mission And Progress Dossier

Each run should maintain a durable on-disk dossier:

- `mission.md`
- `progress.md`
- `research_plan.json`
- `research_contract.json`
- `qa_report.json`

### `mission.md`

This is the stable task brief. It should contain:

- the original user question,
- user constraints,
- current objective,
- important definitions,
- and what success means at a high level.

### `progress.md`

This is the evolving work log. It should contain:

- completed steps,
- open steps,
- key findings so far,
- unresolved questions,
- next scheduled actions,
- and current coverage gaps.

### `research_plan.json`

This is the structured plan and dependency graph.

### `research_contract.json`

This is the definition of done, expressed as explicit acceptance clauses.

### `qa_report.json`

This is the evaluator's structured report of what passed, what failed, what remains weak, and what should happen next.

These artifacts should be reread by the planner, controller, worker, writer, and evaluator on every major phase.

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

Deep retrieval should live inside `core` as a composite skill, for example `deep_retrieval`, rather than being split across ad hoc controller code.

If the term "composite skill" is unfamiliar in backend design, it means one higher-level module that coordinates several smaller primitive operations behind one typed interface.

In practice, the composite deep-retrieval skill should orchestrate atomic capabilities such as:

- `search_web`
- `search_news`
- `search_and_read`
- URL canonicalization and deduplication
- source ranking
- read-queue construction
- batched full-text ingestion
- structured extraction into `SourceCard` records

The controller should still decide when to invoke deep retrieval, with what budget, and for which coverage gaps. The composite skill should own the mechanics, state transitions, and artifact writes for each retrieval wave.

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

### Planner Pass C: Contract Builder

This pass answers: what exactly counts as done, and what would cause evaluation to fail?

The contract should combine:

- default contract templates,
- domain-specific mandatory clauses,
- and prompt-specific additional clauses.

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

## Deep Retrieval Architecture

Deep mode should replace shallow retrieval assumptions with a staged corpus-building pipeline.

### Placement In The System

Deep retrieval should be implemented as a composite skill in the `core` pack, not as scattered controller logic and not as one opaque end-to-end black box.

This design choice separates scheduling from execution:

- the planner decides that deep retrieval is needed,
- the controller decides when another retrieval wave should run,
- the step worker invokes the composite skill,
- and the composite skill coordinates the underlying atomic retrieval and extraction operations.

The composite skill should expose stage-oriented actions such as:

- `discover`
- `rank`
- `build_read_queue`
- `ingest_batch`
- `extract_batch`
- `refresh_coverage`

Each action should take typed inputs and produce typed outputs plus persisted artifacts. This keeps deep retrieval testable and resumable without forcing the controller to know the low-level retrieval details.

### Query Generation

Deep mode should generate roughly 50 to 80 query variants for one report.

These queries should be grouped by intent, for example:

- core thesis and company overview
- latest developments and dated evidence
- financial performance and valuation
- risks and counterarguments
- competitors and peer pressure
- management and capital allocation
- regulation, litigation, and supply chain
- macro transmission where relevant

For deep equity mode, competitor and bearish-case query groups are mandatory.

### Broad Discovery

Deep mode should search both web and news broadly enough to produce about 300 discovered candidate sources for one run.

This is a discovery phase, not an ingestion phase. The goal is coverage, diversity, and freshness.

### Ranking And Deduplication

The runtime should score candidate sources by:

- relevance to the primary question,
- freshness,
- uniqueness,
- source quality,
- and usefulness for closing coverage gaps.

It should deduplicate aggressively across:

- canonical URL overlap,
- mirrored articles,
- near-duplicate titles,
- repeated snippets,
- and repeated coverage from the same domain.

### Read Queue Construction

Deep mode should select about 120 read attempts from the discovered candidate set.

The queue should preserve source diversity and reserve capacity for:

- primary company evidence,
- counterevidence,
- peer evidence,
- and dated current-event evidence.

### Full-Text Ingestion

The ingestion target for one deep run is about 80 successful full-text reads.

The runtime should read these in controlled batches rather than in one flat burst. That reduces failure concentration and lets the controller respond to coverage gaps while the run is still in progress.

### Structured Extraction

Each successful full-text read should be converted into a normalized `SourceCard`.

A `SourceCard` should capture at least:

- source metadata,
- publication date or freshness indicator,
- extracted facts,
- extracted numbers and dates,
- supporting evidence,
- counterevidence,
- likely section relevance,
- and a short normalized summary.

Deep retrieval and reduction must persist these artifacts for later synthesis, QA, resume, and benchmarking:

- `discovered_sources.json`
- `read_queue.json`
- `read_results.json`
- `source_cards.jsonl`
- `fact_index.json`
- `section_briefs.json`
- `coverage_matrix.json`

## Turning Planner Output Into Actionables

The runtime should not treat planner output as free text. It should treat it as typed data.

The code path should be:

1. parse planner outputs into typed models
2. validate that packs and skill names are legal
3. add `core` automatically
4. normalize `ResearchStep` items into a dependency graph
5. persist the mission, plan, and contract artifacts
6. hand the next executable step to a worker

The key actionable unit for execution is a validated `ResearchStep`, not a generic paragraph of planner advice.

For deep mode, planner steps should be allowed to reference the composite `deep_retrieval` skill directly, for example when the plan needs:

- broad discovery,
- counterevidence expansion,
- peer refresh,
- or a follow-up retrieval wave after evaluator feedback.

## Controller Design

The controller should not invent the work. Its job is to schedule the work.

It should read:

- `mission.md`
- `progress.md`
- `research_plan.json`
- `research_contract.json`
- current step states
- evaluator feedback, if any
- coverage matrix state in deep mode

Then it should decide:

- which steps are currently unblocked,
- whether any steps can run in parallel,
- whether retrieval coverage is still too weak to allow final writing,
- whether the report is ready for writing,
- and whether evaluator feedback requires new steps or rework.

This is a scheduling role, not a planning role.

## Step Worker Design

Each `ResearchStep` may require more than one action. A step may need:

- multiple skill calls,
- local reasoning over returned data,
- artifact reads,
- progress writes,
- one or more internal retries,
- and, in deep mode, multiple retrieval or extraction waves.

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

In deep mode, a step should be able to:

- invoke the composite `deep_retrieval` skill for discovery, ranking, or ingestion,
- request additional sources for specific gaps,
- build or refresh the ranked read queue,
- ingest a new read batch,
- convert the read batch into `SourceCard` records,
- and update the coverage matrix before handing control back to the controller.

The worker should not manually reimplement deep retrieval by chaining atomic calls itself. That logic should stay inside the composite skill so it remains reusable, deterministic, and easier to benchmark.

## Gradual Context Reduction

Deep mode should not attempt to draft long reports directly from raw working memory or a flat evidence ledger once the corpus becomes large.

Instead, it should reduce the corpus hierarchically:

1. raw document text -> `SourceCard`
2. `SourceCard` batches -> thematic briefs
3. thematic briefs -> section briefs
4. section briefs -> final report

This gradual reduction is necessary because hundreds of sources produce far more context than one model call should ingest directly.

The `fact_index.json` artifact should normalize:

- claims,
- dates,
- numbers,
- evidence links,
- and counterevidence links

so the writer and evaluator can work from structured context rather than repeatedly reparsing long free-text documents.

## Writing Model

The writer should no longer draft from raw working memory alone. It should draft from:

- mission,
- plan,
- contract,
- step results,
- evidence ledger in standard mode,
- `SourceCard` summaries and section briefs in deep mode,
- claim map,
- fact index,
- coverage matrix,
- and evaluator feedback from prior rounds.

The writer's job is to produce a report that satisfies the contract, not just a fluent answer.

Deep mode writing requirements:

- always start with exactly one H1 title,
- always use proper markdown `##` section headings,
- never emit bold-only pseudo-headings as the primary structure,
- produce a full-length report rather than a memo,
- and target about 4,000 to 8,000 words for single-name equity when the corpus has been successfully built.

## Evaluator Design

The evaluator should inspect:

- the final draft,
- evidence ledger entries,
- step results,
- artifact paths,
- required sections,
- required numeric checks,
- freshness requirements,
- counterevidence coverage,
- and coverage-matrix state in deep mode.

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

A richer evaluator report should include fields such as:

- `blocking_issues`
- `missing_sections`
- `missing_evidence_types`
- `missing_citations`
- `freshness_warnings`
- `numeric_inconsistencies`
- `required_follow_up_calls`
- `counterevidence_gaps`
- `report_section_feedback`

## Deep QA Stop Rule

Deep mode should use a stronger stop rule than a small fixed revision loop.

The stop rule should be:

- continue iterating while the evaluator says `revise` and wall-clock budget remains,
- stop immediately on evaluator `pass`,
- or stop on wall-clock budget exhaustion and persist the best available draft plus unresolved gaps.

Timeout is not equivalent to a clean pass.

Evaluator feedback in deep mode should be able to trigger both:

- targeted rewrites of weak sections,
- and additional retrieval or extraction waves when coverage is still insufficient.

## Counter-Evidence And Competitor Evidence

Counterevidence should be a required part of the harness, not an optional extra.

In finance, a report that only collects supportive evidence is often misleading even when it is well cited.

Examples:

- equity: bearish risks, margin pressure, regulation, balance-sheet concerns, and competitor or peer evidence that weakens the thesis
- macro: alternative scenarios, policy failure modes, conflicting indicators
- commodity: supply surprises, positioning squeeze risk, and curve regime changes

For equity specifically, competitor and peer evidence should be treated as a first-class category of counterevidence.

## Structured Research Artifacts

In addition to the final report and full trace, the runtime should preserve structured intermediate artifacts:

- `research_plan.json`
- `research_contract.json`
- `findings.json`
- `claim_map.json`
- `qa_report.json`
- `discovered_sources.json`
- `read_queue.json`
- `read_results.json`
- `source_cards.jsonl`
- `fact_index.json`
- `section_briefs.json`
- `coverage_matrix.json`

The `claim_map` should represent claims in a normalized structure.

A `claim_map` record should capture:

- claim text,
- whether it is fact or inference,
- supporting evidence IDs,
- complicating evidence IDs,
- freshness date,
- whether the claim appears in the final report.

The `coverage_matrix` should explicitly map:

- required report sections,
- contract clauses,
- evidence categories,
- freshness requirements,
- and counterevidence requirements

to current coverage state so the controller can decide whether more retrieval is still required.

## Checkpoints And Resume

The runtime should persist checkpoints after each major step, for example:

- `data/harness_runs/<run_id>/checkpoint_step_001.json`
- `data/harness_runs/<run_id>/checkpoint_step_002.json`

Checkpointing and context reset are not the same thing:

- checkpointing saves the state,
- context reset starts a fresh model session from saved state.

The harness should support both, but checkpointing must come first.

## Parallel Execution

The harness should support parallel execution for independent, I/O-bound steps.

Examples of parallelizable collections:

- equity: `fetch_company_profile`, `fetch_financials`, `fetch_market_data`, `search_sec_filings`
- macro: `fetch_macro_indicators` and `fetch_world_bank_indicators`
- commodity: `fetch_eia_inventory`, `fetch_cot_report`, and `fetch_futures_curve`
- deep retrieval: read-batch ingestion and extraction on independent documents

Dependent steps should remain sequential.

## Deep Mode Runtime Defaults

Deep mode should be exposed through both the Python API and the CLI.

The explicit deep profile should use these defaults:

- `research_profile = "deep"`
- `wall_clock_budget_seconds = 1200`
- `max_steps = 120`
- `max_revision_rounds = 20`
- `max_worker_iterations = 6`

Standard mode should keep lighter defaults suitable for shorter runs.

## Deep Equity Requirements

The first domain target for deep mode is single-name equity research.

For deep equity runs, the harness should always include:

- company profile,
- financials,
- market data,
- SEC filings,
- peer and competitor pressure analysis,
- and explicit bearish-case or counterevidence retrieval.

These structured domain-tool outputs should be merged into the same normalized extraction pipeline as web and news sources. They should become `SourceCard` or fact-index inputs, not remain separate side channels that the writer must reconcile manually.

## Validation Strategy

Every implementation phase should define not only what to build, but how to tell whether the phase actually succeeded.

The harness should still be tested in at least two live-model lanes:

1. weak-model lane
Use `sf/Qwen/Qwen3-8B` for harness planning, control, writing, and evaluation roles.

2. strong-model lane
Use `kimi-k2.5` for harness planning, control, writing, and evaluation roles.

An optional third lane remains useful:

3. mixed lane
Use `kimi-k2.5` for planner and evaluator roles, and `sf/Qwen/Qwen3-8B` for controller or worker roles.

In addition to smoke tests, the harness should include a long soak lane for deep mode. The first soak target should be a single-name equity prompt because that is the clearest setting for evaluating whether the output is approaching human-style research depth.

The benchmark prompt set should include at least:

- a single-name equity prompt,
- an equity + macro cross-domain prompt,
- a macro scenario prompt,
- a commodity prompt,
- and a time-sensitive latest prompt.

For each prompt, the benchmark runner should track:

- artifact creation success,
- contract satisfaction,
- citation coverage,
- missing-section count,
- freshness failures,
- numeric inconsistency count,
- counterevidence coverage,
- runtime,
- token cost when available,
- discovered-source count,
- full-read count,
- extraction batch count,
- QA iteration count,
- and final word count.

## Implementation Tasks And Phase Tests
NO STOP UNTIL FINISHES THESE TASKS. FINISH THE PHASE ONE BY ONE. DETERMINE FINISH BY PASS THE SUCCESS CRITERIA AND PASS THE LIVE TESTS.

### Phase A: Deep Retrieval Architecture

Tasks:

1. Add an explicit deep research profile instead of making long-duration behavior the default for every run.
2. Implement a composite `deep_retrieval` skill in the `core` pack instead of spreading retrieval mechanics across controller code.
3. Build broad discovery, ranking, and read-queue stages for deep mode inside that composite skill.
4. Generate 50 to 80 query variants and discover about 300 candidate sources.
5. Add deterministic deduplication and source-quality scoring.
6. Persist `discovered_sources.json`, `read_queue.json`, and `read_results.json`.

Success criteria:

- the runtime can discover around 300 candidate sources for one deep run,
- the controller schedules retrieval waves without owning low-level retrieval mechanics,
- the composite `deep_retrieval` skill exposes typed stage operations and writes artifacts deterministically,
- the ranked queue and read results are persisted as structured artifacts,
- source diversity and freshness are measurable rather than implicit,
- and deep mode remains opt-in rather than silently replacing standard mode.

Tests:

1. unit test: query planner produces multiple retrieval buckets.
2. unit test: the composite `deep_retrieval` skill exposes valid stage actions and typed outputs.
3. unit test: deduplication and ranking are deterministic.
4. component test: a synthetic discovery run persists candidate sources, ranked queue, and read results.
5. live weak-model test with `sf/Qwen/Qwen3-8B`: verify deep mode can build a large candidate set without losing the task.
6. live strong-model test with `kimi-k2.5`: verify the same deep run produces broader and cleaner candidate coverage.

### Phase B: Corpus Extraction And Reduction

Tasks:

1. Implement `SourceCard` records for successful full-text reads.
2. Add `source_cards.jsonl`, `fact_index.json`, `section_briefs.json`, and `coverage_matrix.json`.
3. Convert raw document text from the composite `deep_retrieval` skill into `SourceCard` batches, thematic briefs, and section briefs.
4. Make the controller use the coverage matrix to schedule additional retrieval waves when sections or contract clauses are still weak.

Success criteria:

- the runtime can ingest about 80 successful full-text reads in batches,
- large corpora can be reduced without flattening into one giant prompt,
- missing sections and missing evidence types are machine-visible,
- and domain-tool outputs and web sources feed one normalized extraction pipeline.

Tests:

1. unit test: batch extraction creates valid `SourceCard` and `fact_index` records.
2. component test: a 300-source synthetic corpus flows through staged reduction into section briefs.
3. component test: coverage gaps trigger another retrieval or extraction wave instead of premature drafting.
4. live weak-model test with `sf/Qwen/Qwen3-8B`: verify the reduction pipeline still preserves core facts from a large corpus.
5. live strong-model test with `kimi-k2.5`: verify section briefs preserve more distinct evidence and counterevidence.

### Phase C: Long-Form Writer And Pass-Or-Timeout QA

Tasks:

1. Make deep-mode writing depend on section briefs, fact index, coverage matrix, and evaluator feedback.
2. Require the deep writer to emit one H1 title, proper markdown `##` section headings, and a full-length report.
3. Target about 4,000 to 8,000 words for deep single-name equity when corpus depth is available.
4. Replace small fixed-round QA assumptions with a pass-or-timeout loop driven by wall-clock budget.
5. Allow evaluator feedback to trigger both targeted rewrites and additional retrieval.
6. Persist unresolved gaps in `qa_report.json` when timeout occurs.

Success criteria:

- the report always has a title and correct headings,
- the report reaches long-form target length when corpus depth is available,
- QA performs multiple retrieval and rewrite cycles in one run,
- and finalization happens only on evaluator `pass` or wall-clock exhaustion.

Tests:

1. unit test: writer normalization guarantees title plus markdown section structure.
2. unit test: evaluator schema supports pass-or-timeout deep-mode behavior.
3. component test: QA loops multiple times and does not stop after the first `revise`.
4. component test: timeout persists the best draft plus unresolved gaps.
5. live strong-model test with `kimi-k2.5`: verify a deep run can iterate through retrieval and QA until pass or timeout.

### Phase D: Long Soak Testing And Benchmarking

Tasks:

1. Add a long-duration deep-mode soak lane for single-name equity.
2. Track discovered-source count, full-read count, extraction batch count, QA iteration count, elapsed time, and final word count.
3. Keep regression reporting for citation coverage, freshness failures, missing sections, numeric inconsistencies, and counterevidence gaps.
4. Require documented review when model-role assignments change materially.

Success criteria:

- the first soak target is single-name equity,
- the soak lane can discover at least 250 candidate sources,
- the soak lane can achieve at least 60 successful full-text reads,
- at least one evaluator-driven gap-fill cycle occurs,
- the final report reaches at least 4,000 words,
- and the final result is either evaluator `pass` or explicit timeout with unresolved gaps recorded.

Tests:

1. live soak test: single-name equity deep mode under a 20-minute wall-clock budget.
2. benchmark report: discovered count, full-read count, QA iterations, final word count, elapsed time, and final evaluator decision.
3. regression summary: compare deep-mode benchmark results against a saved baseline.
4. review gate: require a documented harness review whenever `models.yaml` materially changes harness role assignments.

## Assumptions And Defaults

The implementer should treat these as explicit defaults, not open questions:

- deep mode is opt-in, not the global default,
- the first deep-mode domain target is single-name equity,
- the deep stop rule is evaluator `pass` or wall-clock timeout,
- the deep discovery target is about 300 candidate sources,
- the deep ingestion target is about 80 successful full-text reads,
- the deep report target is 4,000 to 8,000 words for single-name equity,
- and the default deep runtime envelope is:
  - `research_profile = "deep"`
  - `wall_clock_budget_seconds = 1200`
  - `max_steps = 120`
  - `max_revision_rounds = 20`
  - `max_worker_iterations = 6`
