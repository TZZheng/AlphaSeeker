# Harness

`src/harness` is now a file-based async agent kernel with MiniMax-native delegation.

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
- `artifacts.py`: run root, agent workspace, registry, object promotion, atomic writes, and persisted skill state
- `executor.py`: file-first agent tools including `spawn_subagent`, `wait_children`, direct deterministic skill tools, file reads/writes, and artifact promotion
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
    scratch/
      transcript.jsonl
      tool_history.jsonl
      llm_turns/
        0001_system_prompt.json
        0001_request.json
        0001_response.json
```

The parent agent reads child `publish/` files explicitly. It does not automatically ingest child scratch notes.

## Public Interface

```python
from src.harness import HarnessRequest, run_harness

response = run_harness(
    HarnessRequest(
        user_prompt="Analyze AAPL valuation and risk using current evidence.",
    )
)

print(response.status)
print(response.run_root)
print(response.final_report_path)
```

`HarnessRequest` is now operational rather than scheduler-oriented:
- `user_prompt`
- `run_id`
- `root_preset`
- `agent_transport`
- `wall_clock_budget_seconds`
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
- for MiniMax models, `auto` selects the Anthropic-compatible transport

If you are not used to the backend term "transport": it means the API/message protocol the worker uses to talk to the model provider.

## Runtime Rules

- The root agent is delegation-first and should mostly use `spawn_subagent`, `wait_children`, and file tools.
- The legal child presets are fixed: `orchestrator`, `research`, `source_triage`, `writer`, `synthesizer`, and `evaluator`.
- Unknown preset names are rejected and returned to the agent as repair feedback; the runtime does not silently remap them.
- The root agent and child agents all run the same worker implementation.
- Presets are prompt bundles, not separate runtimes.
- Child handoff is file-based, not schema-based.
- Child handoff always includes canonical published file paths so a parent can explicitly read them later.
- Agents see higher-level deterministic research tools, not internal retrieval-stage controls like `retrieve_sources(stage=...)`.
- Only operational constraints are enforced: valid status files, fresh heartbeats, atomic file writes, replayable transcripts, and required published files on `done`.
- Deep delegation is allowed, but limited by run-wide agent budgets, run-wide live-agent limits, per-parent live-child limits, and soft per-agent wall-clock guidance.
- Provider transcript replay is persisted under each agent's `scratch/transcript.jsonl`; some providers include explicit `thinking` blocks there, while others only expose tool calls and visible text.
- Each agent also writes turn-level debug artifacts under `scratch/llm_turns/`. These capture the exact system prompt snapshot, the request envelope sent to the model, and the provider response payload for each turn.
- If you are not used to the backend term "request envelope": it means the full API payload the worker sends to the model, including prompt text, replayed messages, and tool schemas.
- Agents do not see countdown-style time prompts by default. When the soft per-agent limit is reached, the worker injects a runtime control message telling the agent to stop exploring and finalize on its next turn.
