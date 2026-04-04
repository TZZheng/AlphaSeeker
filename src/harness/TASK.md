# Harness Kernel Notes

## Current Design

The harness is a file-based multi-agent kernel.

- The runtime does not choose research steps.
- Agents use tools, write files, spawn children, and decide when to stop.
- The filesystem is the only handoff channel between agents.

## Stable Runtime Guarantees

- Run registry is append-only:
  - `registry/agents.jsonl`
  - `registry/events.jsonl`
- Agent state is operational only:
  - `status.txt`
  - `heartbeat.txt`
  - `pid.txt`
- Published child outputs live in `publish/`.
- Large or messy work belongs in `scratch/`.
- Deterministic skills are exposed as first-class child-agent tools.
- Agents should see high-level research tools, not hidden internal retrieval-stage controls.
- The root agent should prefer `spawn_subagent` over direct work.
- The only legal child presets are `orchestrator`, `research`, `source_triage`, `writer`, `synthesizer`, and `evaluator`.
- Published child handoff must include canonical file paths so parents can read those files explicitly.

## Engineering Quality Bar

- Root completion means `publish/final.md` exists for the root agent and root status is `done`.
- A stale heartbeat must be detectable and recoverable on resume.
- The supervisor must enforce run-wide live-agent limits and per-agent wall-clock limits.
- A crashed child must not corrupt sibling workspaces.
- Tool outputs must stay small; bulky outputs go to files.
- Resume must be possible from filesystem state without any in-memory graph state.
