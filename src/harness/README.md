# Harness

`src/harness` is a file-based async multi-agent runtime.

In backend terms:
- a "kernel" means runtime code only handles lifecycle, process control, files, and recovery
- the LLM owns meaningful decisions, including delegation and stop timing
- the filesystem is the handoff protocol between agents

## Prompt Philosophy

Prompting is split into three layers:
- `prompts/system.md`: AlphaSeeker-wide operating doctrine
- `prompts/roles/*.md`: preset-specific behavior such as orchestrator vs. research
- `prompts/environment.md`: the actual runtime surface, including visible tools, skills, and child presets

The system prompt is intentionally harness-native rather than a generic "all-purpose agent civilization" prompt:
- it is organized around five duties: act on need, master your tools, learn without cease, work together, and shed the chaff
- it keeps the action-first and specialization-first spirit
- it maps durable memory to `scratch/` and `publish/` instead of imaginary memory tools
- it tells agents to use only visible tools and real runtime outputs
- it treats file publication and explicit child handoff as the durable coordination mechanism

## Architecture

- `runtime.py`: async supervisor kernel that launches agent worker subprocesses and watches heartbeats
- `agent_worker.py`: long-lived worker process shared by root agents and child agents
- `transport.py`: MiniMax Anthropic/OpenAI transport adapters with transcript replay
- `commenter.py`: paired commenter sidecars that read an agent's workspace and inject advisory comments on later turns
- `artifacts.py`: run root, agent workspace, registry, object promotion, atomic writes, and persisted skill state
- `executor.py`: file-first agent tools including `spawn_subagent`, `wait_children`, `bash`, `write_file`, `edit_file`, and artifact promotion
- `presets.py`: prompt presets and default tool allowlists
- `registry.py` and `skills/`: deterministic skill library reused as tools

## Workspace Protocol

Each run lives under `data/harness_runs/<run_id>/`:

```text
registry/
  agents.jsonl
  events.jsonl
  objects/
  objects_manifest.jsonl
request.json
progress.md
agents/
  <agent_id>/
    task.md
    tools.md
    context/
    scratch/
      commenter/
        comments.jsonl
        latest.md
        notes/
        turns/
      transcript.jsonl
      tool_history.jsonl
      llm_turns/
        0001_system_prompt.json
        0001_request.json
        0001_response.json
    publish/
      summary.md
      artifact_index.md
      final.md
    state/
      status.txt
      heartbeat.txt
      pid.txt
      parent.txt
      preset.txt
      skill_state.json
      transport_state.json
      commenter_state.json
```

Notes:
- Parents explicitly read child `publish/` files. They do not automatically ingest child `scratch/` notes.
- Each agent gets a paired commenter sidecar. Commenter output is stored under `scratch/commenter/` and later injected back into the agent as a `Comment Feed`.
- `scratch/llm_turns/` stores the exact system prompt snapshot, request envelope, and provider response for each model turn.

## Public Interface

```python
from src.harness import HarnessRequest, run_harness

response = run_harness(
    HarnessRequest(
        user_prompt="Analyze AAPL valuation and risk using current evidence.",
        run_id="live-aapl-demo",
        wall_clock_budget_seconds=1800,
    )
)

print(response.status)
print(response.run_root)
print(response.final_report_path)
```

`HarnessRequest` is now operational rather than scheduler-oriented:
- `user_prompt`
- `runtime`
- `run_id`
- `root_preset`
- `agent_transport`
- `wall_clock_budget_seconds`
- `root_wall_clock_seconds`
- `max_agents_per_run`
- `max_live_agents`
- `max_live_children_per_parent`
- `per_agent_wall_clock_seconds`
- `stale_heartbeat_seconds`
- `available_skill_packs`
- `resume_from_run_root`

`HarnessResponse` now reports:
- `status`
- `stop_reason`
- `run_root`
- `root_agent_path`
- `final_report_path`
- `error`

Default behavior:

- `root_preset="orchestrator"`
- `agent_transport="auto"`
- `max_live_agents=16`
- `available_skill_packs=["core", "equity", "macro", "commodity"]` if you do not override it
- for MiniMax models, `auto` selects the Anthropic-compatible transport

If you are not used to the backend term "transport": it means the API/message protocol the worker uses to talk to the model provider.

## Runtime Rules

- The root agent uses the `orchestrator` preset by default, but actual decomposition is still an LLM decision.
- The legal child presets are fixed: `orchestrator`, `research`, `source_triage`, `writer`, `synthesizer`, and `evaluator`.
- Unknown preset names are rejected and returned to the agent as repair feedback; the runtime does not silently remap them.
- The root agent and child agents all run the same worker implementation.
- Presets are prompt bundles, not separate runtimes.
- Child handoff is file-based, not schema-based.
- Child handoff always includes canonical published file paths so a parent can explicitly read them later.
- Agents see higher-level deterministic research tools, not internal retrieval-stage controls like `retrieve_sources(stage=...)`.
- File content tools are path-based:
  - `read_file(path=..., start_line=..., max_lines=...)` for exact or partial reads
  - `write_file(path=..., content=...)` for full writes under the current agent's `publish/` or `scratch/`
  - `edit_file(path=..., operation=..., ...)` for bounded edits under the current agent's `publish/` or `scratch/`
- `bash(argv=[...], cwd=...)` is repo-scoped and currently allows `cp`, `ls`, `mkdir`, `mv`, and `rg`
- `search_in_files(pattern=..., paths=[...])` remains the structured grep-style tool and is usually easier for agents than parsing raw shell output
- Only operational constraints are enforced: valid status files, fresh heartbeats, atomic file writes, replayable transcripts, and required published files on `done`.
- Deep delegation is allowed, but limited by run-wide agent budgets, run-wide live-agent limits, and per-parent live-child limits.
- Provider transcript replay is persisted under each agent's `scratch/transcript.jsonl`; some providers include explicit `thinking` blocks there, while others only expose tool calls and visible text.
- Each agent also writes turn-level debug artifacts under `scratch/llm_turns/`. These capture the exact system prompt snapshot, the request envelope sent to the model, and the provider response payload for each turn.
- If you are not used to the backend term "request envelope": it means the full API payload the worker sends to the model, including prompt text, replayed messages, and tool schemas.
- Agents do not see countdown-style time prompts by default.
- When the run-wide wall clock is reached, the supervisor requests a soft stop first, then gives the run a 60-second grace window before hard failure.
- Each agent also has a paired commenter sidecar. The commenter refreshes periodically from the agent's visible workspace state, writes advisory notes to `scratch/commenter/comments.jsonl`, and unread notes are injected into the next agent turn as a `Comment Feed`.

## Running A Live Test

Use `uv run python`, not the system `python`, because this repo's environment is managed through `uv`.

The exact pattern used for the live XOM memo tests was:

```bash
set -a
source .env
set +a

uv run python - <<'PY'
from src.harness import HarnessRequest, run_harness

response = run_harness(
    HarnessRequest(
        user_prompt=(
            "Write a cross-domain investment memo on XOM using current evidence. "
            "Assess valuation, balance-sheet and shareholder-return quality, crude-oil "
            "supply-demand and futures-curve drivers, and the U.S. macro backdrop. "
            "Explain the main bull and bear cases, key quantitative evidence, and the "
            "12-month risk/reward."
        ),
        run_id="live-xom-cross-domain-bash-tools-30m",
        agent_transport="auto",
        wall_clock_budget_seconds=1800,
    )
)

print("STATUS:", response.status)
print("STOP_REASON:", response.stop_reason)
print("RUN_ROOT:", response.run_root)
print("FINAL_REPORT:", response.final_report_path)
print("ERROR:", response.error)
PY
```

Notes:
- `set -a; source .env; set +a` exports `MINIMAX_API_KEY` and related provider variables into the current shell.
- `wall_clock_budget_seconds=1800` gives the run a 30-minute budget.
- If you want a different prompt or preset, change `user_prompt` or `root_preset` in the `HarnessRequest`.
