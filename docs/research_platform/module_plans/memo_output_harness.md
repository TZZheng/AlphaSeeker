# Memo / Output Harness + Artifact Cleanup Plan

Last updated: 2026-05-13  
Planner: `memo_output_planner`  
Status: planning only; no implementation or commits

## 1. Module purpose

The memo/output harness is the product-facing generation layer. It turns an already prepared, source-grounded task context into a final memo/deck/briefing and a small set of auditable artifacts. It preserves AlphaSeeker's original promise — prompt to useful investment memo — while fitting the new persistent research platform.

The key boundary: this module is a **consumer of prepared context**, not the owner of retrieval, ingestion, vault truth, or lifecycle decisions. Xiaohongshu/Librarian's lesson is that by memo/deck time, source collection, source grading, cross-source QC, and conflict surfacing should already be done. The output harness should choose presentation and write the deliverable; it should not redo upstream work at the last minute.

## 2. Owned responsibilities

- Accept a validated `MemoContextPackage` plus run settings.
- Convert that package into a harness run: context files copied into the root workspace, prompt instructions, budget/settings, and final-output requirements.
- Run the existing multi-agent harness or a thin successor around it.
- Enforce product outputs: `publish/final.md`, `publish/summary.md`, `publish/artifact_index.md`, status, and output/source manifests.
- Keep final memo versions/checksums for audit.
- Apply explicit cleanup policy after run finalization.
- Emit a `MemoResultPackage` to downstream vault/lifecycle code containing final memo reference and proposed research updates.

## 3. Explicit non-responsibilities

- Broad source discovery, SEC/news/web retrieval, or daily monitoring.
- Vault ingestion, source grading, canonical doc ids, checksums for original documents, or company/source registry maintenance.
- Durable fact/metric/question/conflict mutation.
- Conflict arbitration or marking warnings resolved.
- Deciding which sources/excerpts belong in the memo package; that is context assembly.
- Treating LLM memo prose as durable truth.
- Binary deck rendering in v1, unless Ted explicitly scopes PPTX/PDF into this module later.

## 4. Inputs / upstream ports

Primary input: `MemoContextPackage` from task-specific context assembly.

Required v0/v1 fields:

```text
package_id
package_version
created_at
ticker
company_name
user_prompt
task_type: memo | deck | briefing
source_index[]
question_list[]
selected_sources[]
conflicts[]
review_warnings[]
context_files[]
manifest
```

`source_index[]` should include `doc_id`, title, source type, grade, date, vault path/URL, checksum, and short description. `selected_sources[]` should include curated excerpts or package-local files, each tied to a `doc_id`, grade, checksum, and reason selected. `question_list[]` carries upstream questions, status, priority, preliminary answer, and refs. `conflicts[]` carries unresolved left/right source refs and severity. `review_warnings[]` carries status patrol/freshness/QC warnings. `manifest` carries package checksum, package-builder version, source counts, upstream module versions, and truncation flags.

Run settings should include output type, run id, model/transport settings, wall-clock budget, cleanup mode, and whether output-time retrieval is allowed.

## 5. Outputs / downstream ports

User-facing outputs:

- root `publish/final.md`: canonical memo/deck/briefing artifact.
- root `publish/summary.md`: short run summary, ticker, prompt, status, key conclusions, and major warnings.
- root `publish/artifact_index.md`: human-readable index of final artifacts, source package, and optional debug archive.
- `publish/source_manifest.json` or `.md`: package sources used, copied/derived from the input package manifest.
- `publish/output_manifest.json`: final artifact refs, checksums, cleanup mode, status, package id/checksum.
- existing final report versions under `registry/report_versions/` and `registry/final_report_versions.jsonl` where applicable.

Downstream machine output: `MemoResultPackage`, containing run id, package id/checksum, ticker, status, final artifact refs, unresolved warnings, cleanup result, and proposed research updates. This package should feed a lifecycle/vault registration port such as `register_memo_result(result)`.

## 6. Persistent artifacts and schema needs

Keep the existing harness layout as the execution substrate in debug mode:

```text
data/harness_runs/<run_id>/
  request.json
  progress.md
  registry/
  agents/agent_root/context/
  agents/agent_root/publish/
  agents/*/scratch/
  agents/*/artifacts/
  agents/*/_harness/
```

Product-retained artifacts should be minimal: request/status, final/summary/index, package files needed for audit, source/output manifests, cleanup manifest, and final-report versions.

Schema needs:

1. `MemoContextPackage` input schema.
2. `MemoResultPackage` output schema.
3. `ArtifactRef(path, sha256, role, bytes/chars, created_at)`.
4. `CleanupManifest(mode, kept_paths, deleted_paths, archived_paths, errors)`.
5. Optional vault table such as `memo_outputs` or `research_artifacts` with ticker, run id, package id/checksum, final path, final sha256, status, prompt, created_at, and metadata.
6. Optional `research_update_proposals` table for memo-derived thesis/question/follow-up candidates pending human review.

## 7. Deterministic code vs LLM decision boundary

Deterministic code owns package validation, checksum verification, context-file copying, prompt/request construction, status mapping, required output checks, manifest generation, version/checksum records, cleanup, and vault registration of artifact references.

The LLM owns memo writing: synthesis, prioritization of package material, narrative structure, caveats, and proposed thesis/questions/follow-ups.

The LLM must not mutate durable vault truth, resolve conflicts, assign source grade, ingest new durable sources, decide cleanup policy, or silently treat its own memo as verified fact state.

Output-time retrieval should default to off. If enabled for exceptional gaps, any new source must be marked as supplemental/unregistered in the manifest and routed back to ingestion/review before it can become vault truth.

## 8. Failure modes and human escalation points

- Invalid package: missing ticker/prompt/manifest, checksum mismatch, missing context file, selected excerpt references unknown doc id. Fail before launching agents.
- Hard warnings: stale sources, missing A-grade support, unresolved high-severity conflicts. Either force caveats or block depending on Ted/orchestrator policy.
- Harness runtime failure: no final, timeout, stale child, transport error. Return failed/timeout status and keep debug artifacts.
- Partial deliverable: final exists at timeout. Return `time_out_with_deliverable`, mark partial in manifest, keep debug artifacts.
- Cleanup failure: never delete outside run root; if cleanup errors, keep full tree and record error.
- Vault registration failure: final memo still exists; write replayable `memo_result.json`; do not retry destructively.
- LLM grounding failure: unsupported claims, missing warning disclosure, fake source ids. Evaluator/validator should fail or require refinement in later versions.

## 9. Versioned/staged coding plan

### v0: interfaces/skeleton/contracts only

- Document/define `MemoContextPackage`, `MemoResultPackage`, cleanup modes, and output-time retrieval policy.
- Keep current `src/vault/memo_flow.py` behavior as transitional: it attaches `llm_research_state.md` as a context file and builds the harness prompt.
- Add prompt/template language that the root agent consumes prepared context and does not redo collection/QC.
- Build a dry-run cleanup planner that computes keep/delete sets but does not delete.
- Tests: package schema validation, context-file copying, prompt does not inline package, cleanup dry-run protects final/context files.

### v1: first useful implementation

- Add a package adapter from current vault outputs (`llm_research_state.md`, source index, questions/conflicts/status pages where available) to minimal `MemoContextPackage` JSON/Markdown.
- Add `run_package_backed_memo()` that validates package, calls `run_harness()`, writes output/source manifests, and returns `MemoResultPackage`.
- Implement guarded cleanup modes: `debug_keep_all` and `product_final_only`.
- Add minimal memo artifact registration in vault/lifecycle: final memo ref/checksum/version only; proposals remain pending review.
- Add validation gates requiring final/summary/index/manifests and warning disclosure.

### v2+: enhancements

- Deck/briefing variants and later renderer integration.
- Evaluator as mandatory quality gate for grounding/freshness/numerical discipline.
- Structured `proposed_updates.json` extracted from memo but routed to human review.
- Debug archive mode with compressed logs/scratch and retention policy.
- Governed output-time retrieval with supplemental source queue.
- Reproducible reruns from exact package checksum and memo version diffs.

## 10. Tests / validation gates

Unit tests:

- Valid/invalid `MemoContextPackage`.
- Selected source references must match `source_index`.
- Package checksum catches modified files.
- Prompt references package files without inlining large content.
- Cleanup mode enum validation.
- Cleanup planner keeps final/summary/index/context/package/manifests and marks scratch/tool outputs only in product mode.
- `MemoResultPackage` includes final sha256 and package checksum.
- Proposed updates default to `requires_human_review=True`.

Integration tests:

- Existing `test_vault_memo_flow.py` remains green.
- Fake package-backed run copies package into root `context/`, writes final/summary/index, writes manifests, and completes.
- Failed run keeps debug artifacts.
- Timeout with final returns partial deliverable.
- Product cleanup removes scratch/logs but preserves final artifacts.
- Vault registration test proves no facts/metrics/questions/conflicts are directly mutated.

Completion gates: nonempty final, summary, artifact index, source/output manifests, package checksum match, high-severity warnings disclosed or waived, and no unauthorized vault mutation.

## 11. Open questions for Ted/orchestrator

1. Should high-severity warnings block memo generation or force prominent caveats?
2. What cleanup default should product runs use after v1: final-only or keep debug until stable?
3. How much selected-source excerpt text should remain in product artifacts versus only source manifest/checksum?
4. Should memo-derived questions/thesis candidates be inserted as pending review automatically or only returned as proposals?
5. Should final memo references live in a new `memo_outputs` table, a generic `research_artifacts` table, or existing `wiki_versions`?
6. What citation style should final memos standardize on: `[doc_id]`, footnotes, links, or source table?
7. Is output-time retrieval ever allowed in normal product mode?
8. Should failed memo attempts appear in company research history?
9. Does this module own PPTX/PDF rendering, or only markdown/structured deck content?

## 12. Interactions with other modules

- **Input/retrieval:** provides source candidates upstream; output harness should not duplicate it.
- **Vault ingestion/source registry:** owns documents, source grades, checksums, paths, and company links consumed through `source_index`.
- **Research lifecycle:** owns questions, conflicts, freshness, review warnings, and promotion of memo proposals.
- **Context assembly:** creates `MemoContextPackage`; most important upstream port.
- **Existing harness runtime:** remains the execution engine in v0/v1, especially `HarnessRequest.context_files`, root workspace, publish files, status, events, and final snapshots.
- **`src/vault/memo_flow.py`:** transitional bridge from current `llm_research_state.md` flow to package-backed flow.
- **Evaluator/commenter:** later quality gate consuming final memo plus package/source manifests.
- **UI/CLI/orchestrator:** should present clean final artifacts, warnings, vault registration status, and optional debug archive link.
