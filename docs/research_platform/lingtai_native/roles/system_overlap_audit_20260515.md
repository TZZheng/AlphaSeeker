# System-overlap audit for AlphaSeeker team role prompts

Date: 2026-05-15 15:03 CT  
Author: codex

## Why this audit exists

Ted pointed out that AlphaSeeker role prompts may be restating design already expressed at LingTai system level: covenant, principle, substrate, procedures, and system prompt composition. I re-read those layers and audited the recent reviewer-discovery patch against them.

## System-level behavior already present in every spawned avatar

The team avatars are not bare LLM prompts. They are full LingTai agents with the following system layers:

- `system/covenant.md`: act on need, master tools, learn continuously, collaborate, deposit knowledge, shed/preserve memory.
- `system/principle.md`: human input only via email; text output is private diary; reply on the channel where the message arrived.
- `system/substrate.md`: avatar/daemon/MCP distinctions, spawn discipline, life states, communication channel discipline, knowledge flow, privacy, idle/soul, proactive molt.
- `system/procedures.md`: skill creation, idle vs nap, life states, avatar escalation, molt, pad discipline, sharing knowledge, mail as time machine, addon ownership, preset tiers, web browsing, issue reporting.
- `system/system.md`: composed prompt that includes covenant, tools, substrate, procedures, skills, identity, pad, and role-specific layers.

Therefore role templates should not repeat generic LingTai behavior unless the repetition creates a domain-specific threshold or application.

## Overlap map

### Already system-level; avoid bloating role prompts with these as generic rules

- Act rather than wait: covenant §I already says if a tool/action can solve the problem, do it.
- Seek missing evidence/tools: covenant §II/III already says learn, search, install/use tools, and do not rely on guesswork.
- Ask peers for help and report outcomes: covenant §IV and procedures/avatar escalation already cover this.
- Use mail correctly: principle/substrate already enforce channel discipline.
- Maintain pad/knowledge/skills: covenant §V, substrate Knowledge Flow, procedures Tending the Pad, Write Skills As You Work.
- Do not treat caveats as an excuse for inaction: this is substantially implied by action-over-words and learning-without-cease.
- Do not block forever / practical stopping: system life-state/stamina/molt and general agent operation imply resource boundedness, but domain prompts may still need task-specific stopping criteria.

### Role-specific material that should remain in AlphaSeeker templates

These are not already expressed by LingTai system level and are legitimate in AlphaSeeker role contracts:

- The AlphaSeeker filesystem contract:
  - raw inputs: `vault/companies/<TICKER>/team/raw/`
  - final output: `vault/companies/<TICKER>/team/published/latest.md`
- The team topology and division of labor:
  - orchestrator, source maintainer, writer, reviewer.
- The investment-memo objective:
  - advance toward a full investment conclusion unless explicitly scoped as preliminary.
- Source artifact expectations:
  - when evidence is used to compare entities/periods/regions/scenarios, land it in re-queryable form with units/dates/currencies/fiscal-period basis/denominators/source links where possible.
- Writer traceability expectations:
  - important numbers/comparisons should trace to named artifacts or explicit unavailable-evidence notes.
- Reviewer discovery stance:
  - demanding plausible reader, fresh artifact expectations, proposition-to-artifact map, hedge/softener artifact-gap scan, one-more-cycle artifact question.
- Two-axis verdict:
  - The old reviewer-owned `criteria-satisfied` / `criteria-sufficient` split has been retired in favor of orchestrator-owned natural-language cold-read judgment; this remains domain/task-control logic, not generic LingTai behavior.

## Assessment of commit `26574b9 Add reviewer discovery stance`

The commit mostly adds role-specific material and should stand:

- Reviewer discovery stance is not system-level; it is an AlphaSeeker investment-review method.
- Source re-queryable comparison artifacts are not system-level; they are research-data hygiene for investment memos.
- Writer artifact traceability is not system-level; it is memo-quality control.
- Reviewer prompt design doc correctly explains the two-axis verdict as a domain mechanism.

One sentence in reviewer template overlaps with existing role/system practicality:

> Do not demand perfection. Distinguish inherent uncertainty, feasible decision-relevant missing artifacts, and work whose cost exceeds likely value. Preserve the practical rule that you should raise the smallest set of material issues and should not block forever waiting for perfect material.

The first clause and final phrase are partly redundant with existing reviewer operating rules and broad system practicality. The middle distinction is still important and domain-specific. If we want the leanest prompt, this sentence can be shortened to:

> Distinguish inherent uncertainty, feasible decision-relevant missing artifacts, and work whose cost exceeds likely value.

I have not changed the commit yet because the current wording is not harmful, but if Ted wants a minimal role prompt, that is the first trim I would make.

## Recommendation

Keep the role templates as thin adapters over LingTai system culture:

1. Let LingTai system layers carry generic behavior: act, use tools, collaborate, report, preserve knowledge, communicate correctly.
2. Let AlphaSeeker role templates carry only:
   - investment memo contract,
   - file paths,
   - role division,
   - domain artifact standards,
   - review/stopping criteria.
3. Avoid adding broad virtues like diligence, courage, collaboration, memory discipline, or tool mastery to AlphaSeeker prompts unless translated into a concrete AlphaSeeker artifact or decision threshold.

Practical next cleanup if desired: trim a few generic phrases from role templates, especially repeated “do not block forever / do not demand perfection” language, while preserving the artifact-discovery and two-axis-verdict machinery.
