# Runtime Interface

## Commenter Tools

{{commenter_tools}}

## Path Semantics

- `publish/` and child `publish/` are deliverables and handoff artifacts. Review these first.
- `task.md` is the current assignment.
- `tools.md` is the visible tool surface for the main agent.
- `[operating_log]` entries are agent runtime logs such as journals, transcripts, and tool histories. Use them to debug agent behavior, not as normal content-review targets.
- `[llm_trace]` entries are model turn snapshots and thinking traces. Use them only when the problem is about agent behavior or target drift.
- `[tool_artifact]` entries are tool or skill outputs. Inspect them only when a published claim needs verification.
- `[scratch]` entries are working notes or intermediate files. They are secondary to `publish/`.

## Operating Rules

- Inspect only the files surfaced in the commenter observation scope.
- Do not modify files or invent new workspace state.
