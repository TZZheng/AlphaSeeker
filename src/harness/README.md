# `src/harness`

`src/harness` is AlphaSeeker's file-based multi-agent runtime.

In backend terms:
- a "supervisor" is the control loop that launches workers, watches heartbeats, and decides when a run stops
- a "transport" is the provider-specific API protocol used to talk to a model
- a "sidecar" is a helper that watches another worker and adds feedback; here the commenter is a supervisor-managed reviewer that runs after worker turns
- a "registry" is append-only run metadata stored under `registry/*.jsonl`

## Current File Map

```text
src/harness/
├── __init__.py              # Public import surface: HarnessRequest, HarnessResponse, run_harness
├── runtime.py               # Async supervisor; launches, monitors, stops, resumes runs
├── agent_worker.py          # Per-agent worker loop: prompt build, model call, tool execution
├── artifacts.py             # Filesystem layout, atomic writes, JSON/JSONL helpers, progress refresh
├── executor.py              # Model-visible tools and deterministic skill execution bridge
├── transport.py             # Anthropic/OpenAI/MiniMax adapters plus text_json fallback
├── commenter.py             # Commenter refresh logic and comment-feed generation
├── prompt_builder.py        # Prompt bundle assembly for agents and commenters
├── presets.py               # Preset tool allowlists and visible-skill policy
├── registry.py              # Skill-pack registry builder
├── retrieval.py             # Deterministic retrieval and reduction helpers
├── benchmark.py             # Benchmark cases, lanes, and metrics extraction
├── types.py                 # Pydantic contracts for requests, state, evidence, events
├── TASK.md                  # Local package notes
├── prompts/
│   ├── system.md
│   ├── task.md
│   ├── tools.md
│   ├── commenter_interface.md
│   ├── actors/
│   ├── response_modes/
│   ├── roles/
│   └── internal/
└── skills/
    ├── __init__.py
    ├── core.py
    ├── equity.py
    ├── macro.py
    └── commodity.py
```

## Runtime Flow

1. `run_harness()` in `runtime.py` is the blocking public wrapper around `_supervise_async()`.
2. `artifacts.initialize_run_root()` creates `data/harness_runs/<run_id>/`, writes `request.json`, and initializes the append-only registry files.
3. The supervisor launches `python -m src.harness.agent_worker` subprocesses for queued agents.
4. Each worker rebuilds prompt context from its on-disk workspace, then calls the active transport in `transport.py`.
5. Model tool calls are executed through `executor.py`, which can spawn children, write/edit files, run a small shell allowlist, or invoke deterministic skills from enabled packs.
6. After each successful worker turn, `commenter.py` reviews changed workspace state after a configurable delay and injects a short `Comment Feed` before that worker's next model call.
7. The run finishes when the root agent reaches a terminal status. A successful run is one where the root reaches `done` and `publish/final.md` exists.

## Public API

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
print(response.stop_reason)
print(response.run_root)
print(response.final_report_path)
```

Important `HarnessRequest` fields:

- `user_prompt`: root research task.
- `run_id`: optional stable folder name under `data/harness_runs/`.
- `root_preset`: starting role. Default is `orchestrator`.
- `agent_transport`: provider protocol. `auto` resolves from the selected model family.
- `wall_clock_budget_seconds`: run-wide time budget enforced by the supervisor.
- `root_wall_clock_seconds`: root-agent time cap used when computing remaining root time.
- `max_agents_per_run`, `max_live_agents`, `max_live_children_per_parent`: run concurrency and fan-out caps.
- `per_agent_wall_clock_seconds`: non-root worker time budget.
- `stale_heartbeat_seconds`: heartbeat timeout before an agent is marked `stale`.
- `available_skill_packs`: enabled packs from `core`, `equity`, `macro`, `commodity`. If omitted, the runtime enables all four.
- `continuous_refinement`: after a `done` root pass, wait for fresh commenter feedback and rerun the root agent.
- `commenter_interval_seconds`: post-turn commenter delay. `None` uses 5 seconds; `0` runs as soon as the supervisor observes the completed turn.
- `resume_from_run_root`: reopen an existing run directory and relaunch unfinished work.

`HarnessResponse` returns:

- `status`: `completed`, `failed`, `time_out`, or `time_out_with_deliverable`
- `stop_reason`: root terminal state such as `done`, `failed`, `stale`, `cancelled`, or `wall_clock_budget_exhausted`
- `wall_clock_budget_exhausted` maps to `time_out` when no root `publish/final.md` exists, and to `time_out_with_deliverable` when a readable root deliverable exists but the root did not finish cleanly
- `run_root`: absolute run directory
- `root_agent_path`: root workspace path
- `final_report_path`: root `publish/final.md` if it exists
- `error`: final failure text, if any

## Validation Runner

For live patch-edit validation, use the direct Python runner so `.env` is loaded before the harness imports:

```bash
uv run python scripts/xom_patch_validation.py run --run-id xom-edit-patch-v1
uv run python scripts/xom_patch_validation.py compare --before-run-root data/harness_runs/xom-edit-baseline-prefix-snapshot --after-run-id xom-edit-patch-v1
```

The script writes `validation_summary.json` or `validation_comparison.json` into the run directory and prints the same JSON to stdout.

Current defaults:

- `root_preset="orchestrator"`
- `agent_transport="auto"`
- `wall_clock_budget_seconds=1200`
- `max_agents_per_run=64`
- `max_live_agents=16`
- `max_live_children_per_parent=8`
- `per_agent_wall_clock_seconds=1800`
- `stale_heartbeat_seconds=45`
- `commenter_interval_seconds=None`, which means a 5-second post-turn commenter delay
- `available_skill_packs=["core", "equity", "macro", "commodity"]` when omitted

## Tool Surface

Base model-visible tools come from `executor.py`:

- `delegate`
- `agents`
- `files`
- `promote` for `research` and `source_triage`
- `bash`
- `write`
- `edit`
- `patch`
- `status`

`bash` is intentionally small. The current allowlist is:

- `cp`
- `mv`
- `mkdir`
- `ls`
- `rg`
- `sleep`

Skill packs are registered in `registry.py` and `skills/`:

- `core`: local file reads/search, web search/read, context condensation, composite retrieval
- `equity`: market data, company profile, financials, SEC filings, insiders, peers
- `macro`: macro indicator helpers
- `commodity`: EIA inventory, COT, futures curve helpers

Visibility is preset-dependent:

- `research` sees the broadest visible skill surface from the enabled packs; internal helper skills can still stay hidden
- `source_triage` only sees `core`
- other presets get a reduced core read/search surface plus the file/status tools above

## On-Disk Run Layout

Each run lives under `data/harness_runs/<run_id>/`.

```text
data/harness_runs/<run_id>/
├── request.json
├── progress.md
├── stop_requested                 # Optional sentinel written by the TUI for graceful stop
├── registry/
│   ├── agents.jsonl
│   ├── events.jsonl
│   ├── final_report_versions.jsonl
│   ├── objects/
│   ├── objects_manifest.jsonl
│   └── report_versions/
│       └── agent_root/
│           ├── v0001.md
│           └── v0002.md
└── agents/
    └── <agent_id>/
        ├── task.md
        ├── tools.md
        ├── context/
        ├── publish/
        │   ├── summary.md
        │   ├── final.md
        │   └── artifact_index.md
        ├── scratch/               # Agent-authored working files only
        ├── artifacts/             # Tool outputs, readable by exact path
        │   ├── skills/
        │   ├── search/
        │   └── reduction/
        └── _harness/              # Harness-private logs and state
            ├── state/
            │   ├── status.txt
            │   ├── heartbeat.txt
            │   ├── pid.txt
            │   ├── parent.txt
            │   ├── preset.txt
            │   ├── prompt_memory.md
            │   ├── history_summary.md
            │   ├── skill_state.json
            │   ├── transport_state.json
            │   └── commenter_state.json
            ├── logs/
            │   ├── transcript.jsonl
            │   ├── tool_calls.jsonl
            │   ├── worker.log
            │   └── events_queue.jsonl
            ├── llm_turns/
            └── commenter/
                ├── comments.jsonl
                ├── latest.md
                ├── notes/
                └── turns/
```

Important runtime conventions:

- `tools.md` is the canonical runtime-interface artifact for normal agents and is loaded into the system prompt.
- `task.md` is the inspectable task contract and is loaded into the per-turn user prompt, not the system prompt.
- parents explicitly read child `publish/` files; child `scratch/` data is not auto-ingested
- `scratch/` is only for agent-authored working files; harness logs and state live under `_harness/`, which is not LLM-readable
- `artifacts/` stores deterministic tool outputs; artifact files are inspectable only when addressed by exact file path
- `progress.md` is the supervisor-maintained human-readable summary used by the TUI
- root `publish/final.md` is the latest report, while changed versions are copied to `registry/report_versions/agent_root/` and logged in `registry/final_report_versions.jsonl`
- `_harness/logs/transcript.jsonl` stores replayable model messages
- `_harness/logs/tool_calls.jsonl` is the canonical tool-call log
- `_harness/llm_turns/` stores the exact system prompt snapshot, request payload, response payload, and any extracted thinking text for each turn
- `_harness/state/prompt_memory.md` is the optional carried-forward self-summary loaded into the agent system prompt
- `_harness/state/history_summary.md` is the compacted semantic memory for older transcript history once raw replay is trimmed
- `_harness/logs/events_queue.jsonl` is the parent-visible completion queue used by `agents`

## Stop, Resume, And Refinement

- When wall-clock budget is exhausted, or the TUI writes `stop_requested`, the supervisor first requests a soft stop.
- The runtime then gives agents a 60-second grace window to finish publishing before forcing failure.
- `resume_from_run_root` reloads the original request from disk and only relaunches unfinished agents.
- `continuous_refinement=True` keeps the root in `refining` after `done` and relaunches it when the post-turn commenter produces fresh feedback.

## Benchmark Support

`benchmark.py` provides:

- `run_benchmark_suite()`
- default benchmark cases for equity, macro, and commodity prompts
- benchmark lanes such as `default` and `wide`
- metrics extraction from the on-disk run registry
