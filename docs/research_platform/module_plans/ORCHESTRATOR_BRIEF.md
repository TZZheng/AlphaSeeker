# AlphaSeeker Research Platform Module Planning Brief

Last updated: 2026-05-13
Owner/orchestrator: `codex`

## Ted's latest instruction

Do **not** implement the improvements yet. First use avatars to make detailed plans for every module of the system. The orchestrator is responsible for coordinating avatars and ensuring:

1. connected ports between different modules are clear;
2. decision boundaries of modules are clear;
3. staged coding plan is clear.

"Staged" means: for the first version, define modules clearly; in every later version, add new features inside each module to enhance the whole system without muddling boundaries. If a boundary has no clear answer, ask Ted rather than guessing.

## Product direction

AlphaSeeker is shifting from a one-shot investment memo generator toward a persistent equity research operating system inspired by the Xiaohongshu/Librarian post.

The desired direction is **not** generic RAG and not a durable freeform LLM-written wiki as source of truth. Durable state should be source-grounded, inspectable, and versionable.

Target workflow hypothesis:

```text
company/ticker + research task + optional manual files
  -> input/retrieval layer automatically retrieves required source materials
       deterministic fetch first
       LLM-assisted discovery only when deterministic code cannot find/choose something
  -> deterministic vault ingestion
       documents, metadata, source grade, URL/path, checksum, company link
  -> research object / index / lifecycle maintenance
       source_index
       question_list
       optional source-grounded facts/claims/metrics/guidance/catalysts/risks
       conflict/review queue
       freshness/staleness/diff state
  -> task-specific context assembly
       for memo: selected source docs/excerpts + source_index + questions + conflicts
       for meeting prep/monitoring later: different packages
  -> output harness
       memo/deck/briefing generation
       keep final output + minimal manifest
       delete or archive temporary agent scratch files
  -> persistent vault update
       preserve sources, questions, conflicts, claim/fact state, final memo reference/version history
```

## Existing code landmarks

- `src/tools/equity/*`: low-level deterministic/semi-deterministic fetchers/parsers:
  - `sec_filings.py`
  - `company_profile.py`
  - `financials.py`
  - `earnings_calls.py`
  - `market_data.py`
  - `peers.py`
  - etc.
- `src/harness/skills/equity.py`: current agent-facing wrappers around those tools. Some of this behavior likely belongs in the new input/retrieval layer rather than inside memo agents.
- `src/vault/*`: current vault primitives:
  - `ingest.py`, `schema.py`, `paths.py`, `wiki.py`, `extract.py`, `status.py`, `synthesis.py`, `memo_flow.py`.
- `src/harness/*`: multi-agent memo harness and artifact/workspace lifecycle.
- `docs/research_platform/xiaohongshu_librarian_2026-05-12/librarian_xhs_extracted_report.md`: Xiaohongshu/Librarian source analysis.
- `docs/research_platform/implementation_map.md`: current implementation map, but note it still reflects the now-transitional `llm_research_state.md` MVP and should be revised by the final orchestrated plan.

## Important current decision

`llm_research_state.md` is now considered transitional/prototype output. Do not build plans around it as the durable source of truth. If an LLM summary is useful, treat it as a temporary context artifact or as a source-grounded extraction step whose durable outputs are structured/indexed objects.

## Required output format for each avatar plan

Each avatar should write a markdown plan under:

```text
docs/research_platform/module_plans/<module_name>.md
```

Use this structure:

1. **Module purpose**
2. **Owned responsibilities**
3. **Explicit non-responsibilities**
4. **Inputs / upstream ports**
5. **Outputs / downstream ports**
6. **Persistent artifacts and schema needs**
7. **Deterministic code vs LLM decision boundary**
8. **Failure modes and human escalation points**
9. **Versioned/staged coding plan**
   - v0: interfaces/skeleton/contracts only
   - v1: first useful implementation
   - v2+: enhancements
10. **Tests / validation gates**
11. **Open questions for Ted/orchestrator**
12. **Interactions with other modules**

Do not write implementation code. Planning docs only.

## Module planning domains

The initial avatar assignments are expected to cover:

1. Input/retrieval layer.
2. Deterministic vault ingestion + source registry.
3. Research-object/question/conflict/freshness lifecycle.
4. Memo/output harness and artifact cleanup.
5. Integration architecture, module ports, staged coding roadmap, and tests.

The orchestrator will merge these into a consolidated design and ask Ted for any unresolved decision boundaries.
