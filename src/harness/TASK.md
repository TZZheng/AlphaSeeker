# Harness Status

Document status date: March 28, 2026.

Primary target: deep single-name equity research.

## Purpose

This file is the current status memo for `src/harness`.

It records:

1. the intended end state,
2. what is implemented,
3. what has been validated offline and live,
4. the short historical baseline that has already been proven,
5. the current definition of done,
6. and the next actionable work.

## Intended End State

The harness is designed to support two runtime profiles:

- `standard`: a bounded research run for normal interactive work
- `deep`: a longer research run that builds a broad source corpus, reduces it in stages, and iterates until evaluator pass or explicit timeout

In practical terms:

- `standard` should be fast, stable, and operationally safe
- `deep` should be evidence-heavy, slower, and capable of producing a strong first-draft research memo rather than a short answer

For deep mode, the intended operating shape is:

- many query variants instead of one or two searches
- broad source discovery instead of a tiny evidence set
- staged reduction from raw reads into structured research artifacts
- multiple draft and evaluation cycles when needed
- finalization only on evaluator pass or explicit timeout

If the term "structured" is unfamiliar in backend work, it means the data is saved in an explicit machine-checked shape instead of being left as loose free text.

## Current Implementation Status

The harness refactor is substantially implemented in code.

Implemented runtime behavior:

- explicit `standard` and `deep` research profiles
- deep-mode defaults:
  - `research_profile = "deep"`
  - `wall_clock_budget_seconds = 1200`
  - `max_steps = 120`
  - `max_revision_rounds = 20`
  - `max_worker_iterations = 6`
- separate planner, controller, worker, writer, and evaluator roles
- checkpointing and resume behavior
- deep-mode pass-or-timeout QA semantics
- evaluator follow-up execution in deep mode
- per-phase latency telemetry for planner, retrieval, writer, and evaluator

Implemented persistent artifacts:

- dossier artifacts:
  - `mission.md`
  - `progress.md`
  - `research_plan.json`
  - `research_contract.json`
  - `qa_report.json`
- deep-research artifacts:
  - `discovered_sources.json`
  - `read_queue.json`
  - `read_results.json`
  - `source_cards.jsonl`
  - `fact_index.json`
  - `section_briefs.json`
  - `coverage_matrix.json`
- trace and benchmark artifacts:
  - `trace.json`
  - `benchmark_metrics.json`

Implemented deep-research mechanics:

- composite `deep_retrieval` skill in the `core` pack
- deterministic query bucketing, ranking, and deduplication
- staged ingestion and extraction into `SourceCard` records
- reduction into fact index and section briefs
- coverage-matrix hooks for follow-up scheduling
- benchmark metric extraction for corpus scale, QA iteration count, report length, evaluator parse mode, and per-phase timing

If the term "deterministic" is unfamiliar in backend work, it means the same inputs should produce the same ordered outputs. That matters for reproducibility, which means being able to rerun a test and get the same behavior again.

If the term "typed" is unfamiliar in backend work, it means the code validates data against explicit structures instead of passing around loose dictionaries.

## Validation Status

### Offline Validation

Current offline result:

- `38 passed` across the harness unit and component suites after the latency-telemetry patch

Covered areas include:

- deep query-bucket generation
- typed deep-research stage outputs
- deterministic ranking and deduplication
- artifact persistence
- `SourceCard` and `fact_index` generation
- deep timeout persistence
- controller follow-up behavior in deep mode
- benchmark metric extraction
- elapsed wall-clock consistency versus summed phase timings

### Live Validation

Live provider connectivity was confirmed on March 27, 2026:

- `sf/Qwen/Qwen3-8B -> OK`
- `kimi-k2.5 -> OK`

The official live smoke suite was run on March 27, 2026:

```bash
uv run pytest -m "live and network" tests/live/test_live_harness_smoke.py -q -s
```

Observed result from that run:

- pytest result: `passed`
- runtime: about `244` seconds
- enabled packs: `["core", "equity"]`
- executed skills: `["search_and_read"]`
- report length: about `1830` words

The March 28, 2026 completion runs proved:

- evaluator structured parsing is stable enough for normal live runs
- survivability smoke and bounded-quality smoke are separate
- deep weak and deep strong lanes both complete
- corpus scale reaches near-target depth
- evaluator-driven multi-cycle QA occurs in saved artifacts
- long-form deep behavior either reaches longer outputs or times out cleanly with unresolved gaps saved
- a deep soak baseline exists on disk
- per-phase latency telemetry is persisted and diagnostically useful

If the term "contract" is unfamiliar in backend work, it means the agreed input-output shape between two components. Here it refers to the exact `VerificationReport` shape the evaluator is expected to return.

If the term "telemetry" is unfamiliar in backend work, it means internal measurement data collected so the system can be observed and debugged.

## Historical Baseline Summary

The earlier milestone of "prove the deep harness exists and is instrumented" has already been completed. The important evidence has been kept, but the detailed milestone-by-milestone writeup is intentionally compressed here.

What has already been proven:

- evaluator structured parsing works in validated standard, deep weak, and deep strong runs without rule-based fallback
- survivability smoke and bounded-quality smoke are separated
- deep weak and deep strong lanes both complete and persist the expected artifacts
- live deep runs reached near-target corpus scale
- evaluator-driven multi-cycle QA was observed and persisted
- a deep soak baseline exists on disk
- per-phase latency telemetry is persisted and wall-clock usage is diagnosable from saved artifacts

Key historical artifacts:

- standard structured-parse validation:
  - `data/harness_runs/Analyze_AAPL_valuation_and_risk_using_current_evidence_harness_20260328_003521`
- deep weak validation:
  - `data/harness_runs/Analyze_AAPL_valuation_and_risk_using_current_evidence_including_peers_and_counterevidence_harness_20260328_004308`
- deep strong validation:
  - `data/harness_runs/Analyze_AAPL_valuation_and_risk_using_current_evidence_including_peers_and_counterevidence_harness_20260328_004723`
- quality smoke:
  - `data/harness_runs/Analyze_AAPL_valuation_and_risk_using_current_evidence_Include_peer_comparisons_at_least_one_cou_harness_20260328_005932`
- multi-cycle QA:
  - `data/harness_runs/Analyze_AAPL_valuation_and_risk_using_current_evidence_including_peers_SEC_filings_countereviden_harness_20260328_012052`
- deep soak baseline:
  - `data/harness_baselines/deep_soak_aapl_20260328`
- latency telemetry validation:
  - `data/harness_runs/Analyze_AAPL_valuation_and_risk_using_current_evidence_harness_20260328_015848`

## Definition Of Done

This document now uses the next-phase definition of done as the active one.

The current goal is to make deep runs complete the planned research path before draft or finalize unless an explicit timeout occurs.

In the current harness schema, the evaluator returns `pass`, `revise`, or `fail`. In plain language, evaluator `pass` is the harness equivalent of success.

For the current phase, "done" means all of the following are true:

1. the deep run executes the required planned steps before the first successful finalization attempt
2. evaluator follow-up work is either mapped back into planned-step completion or explicitly recorded as extra work
3. `progress.md` and `trace.json` make unfinished versus completed work obvious
4. at least one fixed deep live run reaches evaluator `pass` without rule-based fallback
5. if a run cannot finish within budget, it times out explicitly with unfinished steps and reasons saved

## Current Bottom Line

The core deep harness is already proven.

Current state:

- offline harness validation is green with `38 passed`
- standard survivability and bounded-quality live smoke coverage both exist
- deep weak, deep strong, corpus-scale, multi-cycle, soak-baseline, and latency-telemetry validations exist on disk
- live evaluator parsing is stable enough for the validated lanes
- the active gap is now process-completion reliability, not missing subsystems
- saved deep runs show that drafting or finalization can still occur while planned steps remain pending
## YOUR TASK
FINISH THE TASKS LISTED BELOW ONE BY ONE. COMPLETE THEM IN SEQUENCE. DO NOT treat a later task as complete until the earlier task's exit signal is met. DON'T STOP RUNNING UNTIL EVERY TASK EXIT CRITERIA IS MET.
## TASKS

### 1. Make The Full Deep Process Successfully Finish With Unlimited Budgets

Goal:

- prove that the deep control flow can converge when wall-clock and step limits are set high enough that the harness, not the budget, is the main thing being tested

If the term "budget" is unfamiliar in backend work, it means an explicit limit such as wall-clock seconds, `max_steps`, `max_revision_rounds`, or `max_worker_iterations`.

Required work:

1. fix one deep single-name equity validation prompt and keep it unchanged for the whole task:
   - use the deep AAPL prompt already represented in the saved March 28, 2026 runs
   - record the exact request settings used for the reference run
2. inspect recent deep `trace.json`, `progress.md`, and `qa_report.json` artifacts to identify the dominant non-convergence patterns:
   - draft generated before required steps completed
   - finalization reached while plan steps were still pending
   - evaluator follow-up calls executed outside the planned-step accounting
   - timeout or step exhaustion without clear unfinished-step diagnostics
3. make deep-mode scheduling prefer planned-step execution (`execute_step` or `execute_parallel_steps`) over direct `call_skill` whenever planned work is available
4. ensure evaluator-requested follow-up skills either:
   - update the matching planned step status, or
   - are recorded as explicit extra work that still points back to the affected planned step
5. block the first draft until:
   - the deep corpus step has completed or explicitly failed
   - the required domain collection steps have completed or explicitly failed
   - the saved artifacts make that state visible
6. block clean finalization while required deep steps remain pending, unless the run has explicitly hit wall-clock exhaustion or max-step exhaustion
7. persist step-completion diagnostics in saved artifacts so the run history clearly shows:
   - completed steps
   - partial or failed steps
   - still-pending steps
   - why the run stopped
8. rerun the fixed deep AAPL case with effectively unconstrained budgets and verify that the run reaches strict evaluator `pass` through the intended planned path
9. save that run as the unlimited-budget process reference for later comparison

Exit signal:

- one deep live run completes the required planned path before successful finalization under effectively unconstrained budgets
- that run ends with evaluator `pass`
- `evaluator_parse_mode` is not `rule_based_fallback`
- saved artifacts clearly show completed versus unfinished steps
- the run does not stop because of budget exhaustion
- the saved unlimited-budget reference artifact can be used as the baseline for later tasks

### 2. Make The Full Deep Process Successfully Finish With Reasonable Budgets

Goal:

- turn the unlimited-budget success path into a reproducible deep profile that still converges under explicit practical limits rather than diagnostic near-unlimited limits

Required work:

1. use the unlimited-budget reference run from Task 1 as the baseline
2. extract the actual resource usage from the saved artifacts:
   - wall-clock time
   - total steps
   - revision count
   - worker iterations
   - retrieval-wave count
   - per-phase timing totals
3. define a candidate "reasonable deep" envelope based on observed usage plus a small safety margin rather than arbitrary numbers
4. rerun the fixed deep AAPL case under that candidate envelope and reduce one limit at a time to find which limits actually matter:
   - wall-clock budget
   - `max_steps`
   - `max_revision_rounds`
   - `max_worker_iterations`
5. separate genuine budget failures from other failures:
   - control-flow regressions
   - missing references
   - evaluator dissatisfaction despite completed process
6. once one reasonable budget profile succeeds, rerun the same case again with the same settings to check that success is not a one-off accident
7. record the final recommended deep-mode budget profile and the expected artifact-level signs of healthy convergence

Exit signal:

- the fixed deep AAPL case reaches evaluator `pass` under an explicit reproducible reasonable-budget profile
- the passing run completes the planned path before finalization
- the same settings succeed again on a repeat run or produce equivalent artifact-level convergence
- budget failures, if they still occur in comparison runs, are clearly diagnosable from saved artifacts

### 3. Fix The Missing Reference Problem

Goal:

- remove broken, missing, phantom, or misleading report references so every material claim points to valid saved evidence

If the term "reference" is unfamiliar in backend work, it means an inline evidence pointer such as `[E12]` that should map to a real entry in the saved evidence ledger.

Required work:

1. review failing `qa_report.json` files and final drafts to classify the reference problems:
   - missing citations
   - citations to ids that do not exist
   - citations to the wrong evidence
   - sections that rely on uncited summary text
2. trace the reference path end to end:
   - skill outputs into the evidence ledger
   - deep artifacts such as source cards, fact index, and section briefs
   - writer inputs
   - final report markdown
   - verifier checks
3. make the report use one clear citation scheme for user-facing output and forbid any id that is not valid under that scheme
4. ensure the deep reduction artifacts preserve enough evidence linkage that the writer can cite real saved evidence rather than invented placeholders
5. ensure the `Sources` section and inline citations reconcile with the same saved evidence set
6. strengthen verification so a report cannot pass if it contains phantom references or material sections with unresolved missing citations
7. rerun the fixed deep case and any recent failing reference-heavy case to confirm the problem is removed in practice

Exit signal:

- passing runs contain no phantom or dangling references
- `qa_report.json` does not report missing citations or nonexistent evidence ids on the fixed deep passing case
- the `Sources` section, inline citations, and saved evidence ledger all point to the same evidence set

### 4. Improve The Evaluator `Pass` Rate On The First Completed Draft

Goal:

- increase the share of runs that reach evaluator `pass` on the first completed draft after Tasks 1 through 3 have made the process and references stable

Required work:

1. define the interim first-pass baseline using a fixed validation set:
   - start with repeated runs of the fixed deep AAPL case
   - if a broader fixed set is added later, keep it stable during comparison
2. collect the remaining first-pass non-`pass` reasons after process completion and reference integrity are already in place
3. separate the remaining causes into the main buckets:
   - incomplete section coverage
   - weak counterevidence treatment
   - missing explicit dates for time-sensitive claims
   - weak numeric grounding
   - evaluator instructions that are too vague to drive a clean first pass
4. tighten the writer and contract guidance only around the recurring first-pass failures, rather than reopening the full architecture
5. make evaluator follow-up requests more specific and machine-actionable so repeated revision loops are less ambiguous
6. rerun the fixed validation set after each meaningful change and track first-pass outcomes separately from eventual final outcomes

Exit signal:

- the first-pass evaluator `pass` rate is materially higher than the baseline measured at the start of this task
- the improvement does not depend on rule-based fallback parsing
- the improvement does not come from weakening the evaluator standard

### 5. Find The Bottleneck Of Deep-Mode Runs And Improve Latency

Goal:

- identify which phase dominates deep-run time and reduce that wall-clock cost without breaking process completion or evaluator quality

If the term "latency" is unfamiliar in backend work, it means elapsed wall-clock time spent waiting for one phase of the system to finish.

Required work:

1. use passing runs from Tasks 1 through 4 as the measurement set instead of mixing them with obviously broken runs
2. read saved phase timing data from `trace.json` and `benchmark_metrics.json` to rank the slowest phases:
   - planner
   - retrieval
   - writer
   - evaluator
3. separate structural bottlenecks from noise:
   - structural bottleneck means a phase that is consistently dominant across similar runs
   - noise means one-off provider variance or transient network delay
4. inspect the dominant slow phase for avoidable work, such as:
   - repeated retrieval waves caused by poor scheduling
   - duplicated or oversized writer context
   - evaluator loops caused by vague feedback or missing references
   - repeated processing of already-satisfied steps
5. make one latency change at a time and rerun the fixed validation case after each change
6. record before-versus-after timing deltas and confirm that the run still:
   - completes the planned path
   - reaches evaluator `pass`
   - preserves reference integrity

Exit signal:

- one dominant deep-mode bottleneck is identified with evidence from saved timing artifacts
- at least one measured latency improvement is demonstrated on the fixed validation case
- the faster run still reaches evaluator `pass` without regressing process completion or references
