# Codex Subscription Integration Report for AlphaSeeker

Date: 2026-05-12 PT
Request: Ted wants AlphaSeeker's driving LLM / harness agents to use his Codex subscription, similar to the Lingtai parent agent currently running on Codex (`gpt-5.5`).

## Short answer

AlphaSeeker **cannot currently use the Codex/ChatGPT subscription directly**. It can use OpenAI API models via `OPENAI_API_KEY`, but that is a different authentication and billing path from Ted's local Codex subscription.

The local machine **does have Codex CLI installed and authenticated** (`codex-cli 0.130.0`), and a quick non-interactive structured-output probe succeeded through `codex exec`. Therefore the feasible integration path is a new **`codex_cli` harness transport** that shells out to Codex CLI, asks it for strict JSON decisions, and lets AlphaSeeker continue executing deterministic tools itself.

Do **not** try to make `gpt-5.5` work by simply putting it in `config/models.yaml` as an OpenAI model. Current OpenAI paths require `OPENAI_API_KEY`, not the Codex subscription.

## Current AlphaSeeker model/provider architecture

### Model config

`config/models.yaml:8-15` documents the currently supported prefixes:

- `sf/*` -> SiliconFlow
- `kimi-*` -> Moonshot Kimi
- `minimax/*` -> MiniMax via OpenAI-compatible API
- `gemini-*` -> Google Gemini
- `gpt-*` -> OpenAI API
- `claude-*` -> Anthropic API

There is no `codex/*` or `codex-cli/*` model family.

`config/models.yaml:22-24` currently sets:

```yaml
harness:
  agent: "minimax/Minimax-M2.7"
  condense: "minimax/Minimax-M2.7"
```

### Model env-var resolution

`src/shared/model_config.py:102-142` maps model names to provider labels and required env vars. OpenAI-like names (`gpt-*`, `o1`, `o3`, `o4`) require `OPENAI_API_KEY` (`src/shared/model_config.py:116-140`). There is no mapping for Codex CLI or subscription auth.

### LangChain LLM manager

`src/shared/llm_manager.py:283-355` builds models for:

- Kimi via `KIMI_API_KEY`
- OpenAI API via `OPENAI_API_KEY`
- Anthropic via `ANTHROPIC_API_KEY`
- SiliconFlow via `SILICONFLOW_API_KEY`
- MiniMax via `MINIMAX_API_KEY`
- Gemini via `GOOGLE_API_KEY`

There is no Codex CLI branch. The OpenAI branch at `src/shared/llm_manager.py:305-312` uses `ChatOpenAI(... api_key=_secret_from_env("OPENAI_API_KEY"))`.

### Harness native transport layer

`src/harness/transport.py:121-130` resolves transport types:

- MiniMax -> `minimax_anthropic`
- Anthropic -> `anthropic`
- OpenAI API -> `openai`
- unknown -> `text_json`

`src/harness/types.py:191` and `src/harness/types.py:257` validate `agent_transport` as only:

```text
auto, minimax_anthropic, minimax_openai, anthropic, openai, text_json
```

`src/harness/transport.py:2031-2038` implements native OpenAI as:

```python
self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

So the current OpenAI transport is definitely API-key based, not subscription based.

### Worker integration points

`src/harness/agent_worker.py:390-414` resolves the harness model and transport. If transport is not `text_json`, it calls `create_transport(...)` and uses native tool-call flow.

`src/harness/agent_worker.py:824-859` executes native transports by:

1. appending pending user text;
2. calling `runtime.transport.execute_turn(prepared.tool_specs)`;
3. executing returned tool calls through AlphaSeeker;
4. appending tool results back to the transport.

A `codex_cli` transport can fit this native transport shape if it implements:

- `append_user_text()`
- `execute_turn()`
- `append_tool_results()`

## Local Codex CLI status

Observed locally:

```text
$ command -v codex
/Users/tianzhezheng/.nvm/versions/node/v24.14.1/bin/codex

$ codex --version
codex-cli 0.130.0
```

`~/.codex/config.toml` exists and contains `model = "gpt-5.5"`; the AlphaSeeker project is trusted. Credentials were not printed.

`codex exec --help` shows useful non-interactive options:

- `--json` prints JSONL events
- `--output-last-message <FILE>` writes final answer to a file
- `--output-schema <FILE>` constrains final response schema
- `--cd <DIR>` sets working root
- `--sandbox read-only|workspace-write|danger-full-access`
- `--ephemeral`
- `--model <MODEL>`

## Probe result

A minimal structured-output probe succeeded:

```bash
codex exec \
  --ephemeral \
  --skip-git-repo-check \
  --cd <AlphaSeeker repo> \
  --sandbox read-only \
  --output-schema scratch/codex_cli_probe/schema.json \
  --output-last-message scratch/codex_cli_probe/last.json \
  --json < scratch/codex_cli_probe/prompt.txt \
  > scratch/codex_cli_probe/events.jsonl
```

With a strict schema:

```json
{"text":"probe-ok","tool_calls":[]}
```

The output file contained:

```json
{"text":"probe-ok","tool_calls":[]}
```

The events JSONL included a normal `turn.completed` usage object.

Important schema gotcha: the Responses API rejects arbitrary object fields unless nested objects also specify `additionalProperties: false`. For tool arguments, use `arguments_json: string` instead of a free-form object, then parse that string in AlphaSeeker.

## Integration options

### Option A — OpenAI API key only

Set `OPENAI_API_KEY` and use `gpt-*`/`o*` names. This already mostly exists.

Pros:

- least code change;
- native OpenAI tool-calling path already implemented.

Cons:

- does **not** use Ted's Codex/ChatGPT subscription;
- requires API billing/key;
- not what was requested.

Verdict: keep as separate path; do not call this “Codex subscription.”

### Option B — Codex CLI subprocess transport

Add a new `codex_cli` transport that calls `codex exec` non-interactively.

Pros:

- uses local Codex subscription/auth;
- proof-of-feasibility succeeded;
- keeps AlphaSeeker deterministic tool execution rather than letting Codex mutate files/shell directly;
- can be isolated to harness transport first.

Cons:

- slower: one Codex CLI process per model turn;
- no native OpenAI-style function calling over SDK; must ask for structured JSON decisions;
- must handle CLI errors, auth expiry, schema failures, and stdout/stderr logs;
- must be careful not to let Codex CLI run arbitrary tools/shell.

Verdict: best match for Ted's request.

### Option C — Reuse LingTai Codex backend directly

The parent Lingtai agent runs on provider `codex`, endpoint `https://chatgpt.com/backend-api/codex`, but AlphaSeeker does not currently expose that backend as an importable Python client. Reusing it would couple AlphaSeeker to Lingtai internals and likely credential/session assumptions.

Verdict: not recommended as first implementation. Use Codex CLI as the stable boundary.

### Option D — unsupported without new adapter

True today: without `OPENAI_API_KEY` or a new Codex CLI adapter, AlphaSeeker cannot use Codex subscription.

## Recommended architecture: `codex_cli` harness transport

### Config shape

Add a distinct model prefix and transport:

```yaml
harness:
  agent: "codex-cli/gpt-5.5"
  condense: "minimax/Minimax-M2.7"   # keep condense on API model initially
```

or use explicit request transport:

```python
HarnessRequest(..., agent_transport="codex_cli")
```

### Transport behavior

`CodexCLITransport(BaseAgentTransport)` should:

1. Store conversation in AlphaSeeker's existing transcript/conversation files, like other native transports.
2. On `execute_turn(tool_specs)`, build a prompt containing:
   - system prompt;
   - replayed conversation messages;
   - available tool schemas;
   - strict instruction: return only JSON matching schema.
3. Write a temporary JSON schema file requiring:

```json
{
  "text": "string",
  "tool_calls": [
    {"name": "string", "arguments_json": "string"}
  ]
}
```

4. Run:

```bash
codex exec \
  --ephemeral \
  --cd <repo root> \
  --sandbox read-only \
  --output-schema <schema_file> \
  --output-last-message <last_message_file> \
  --json
```

5. Parse `last_message_file`.
6. Convert each item in `tool_calls` to `ModelToolCall(call_id="codex_cli_<turn>_<n>", name=name, arguments=json.loads(arguments_json))`.
7. Record request/response artifacts using `_record_model_request()` / `_record_model_response()`.
8. Return `ModelTurnResult` to the existing worker native-tool execution path.

### Security posture

Use `--sandbox read-only` and instruct Codex CLI not to run shell commands. The harness already executes deterministic tools after the model returns tool calls, so the CLI itself should not need workspace writes.

Do **not** use `--dangerously-bypass-approvals-and-sandbox`.

Do **not** run Codex CLI with `workspace-write` in the harness loop unless there is a separate, explicit code-editing workflow.

### Why not let Codex CLI use its own tools?

AlphaSeeker already has a controlled tool surface, trace logging, artifacts, and status management. If Codex CLI runs shell edits internally, AlphaSeeker loses traceability and safety. For harness driving, Codex should only decide the next tool call / status; AlphaSeeker should execute it.

## Implementation plan

### Patch 1 — model/transport config recognition

Files:

- `config/models.yaml`
- `src/shared/model_config.py`
- `src/harness/transport.py`
- `src/harness/types.py`

Changes:

- Document `codex-cli/*` prefix.
- Add provider label/env handling where appropriate. This provider should not require `OPENAI_API_KEY`; instead it requires local `codex` executable/auth.
- Add `codex_cli` to `AGENT_TRANSPORTS` and `HarnessRequest.agent_transport` Literal.
- Add `is_codex_cli_model()` and update `resolve_agent_transport()` to return `codex_cli` for `codex-cli/*`.

### Patch 2 — Codex CLI helper functions

In `src/harness/transport.py` or a new `src/harness/codex_cli_transport.py`:

- locate `codex` with `shutil.which("codex")`;
- build strict output schema;
- build prompt;
- call `subprocess.run(..., timeout=...)`;
- capture stdout JSONL, stderr, final message file;
- parse structured result.

Prefer a separate file if `transport.py` is already too large.

### Patch 3 — `CodexCLITransport`

Implement class with same native transport API as OpenAI/Anthropic transports.

Add to `create_transport()`.

### Patch 4 — tests

Extend `tests/unit/test_harness_transport.py`.

Mock `subprocess.run` and `shutil.which`; do not call real Codex in offline tests.

Test cases:

1. `resolve_agent_transport("auto", "codex-cli/gpt-5.5") == "codex_cli"`.
2. `HarnessRequest(agent_transport="codex_cli", ...)` validates.
3. Codex CLI response with `tool_calls=[{"name":"status","arguments_json":"{\"state\":\"done\"}"}]` becomes a `ModelToolCall`.
4. Malformed `arguments_json` becomes an error or empty dict with a recorded failure policy.
5. Missing `codex` executable gives clear error.
6. Nonzero CLI exit stores stdout/stderr in response artifact and raises clear error.
7. Schema file is strict enough for nested objects.

Optional live/manual test:

```bash
uv run pytest tests/unit/test_harness_transport.py -k codex_cli
uv run python -m src.harness...   # tiny run with agent_transport=codex_cli
```

### Patch 5 — documentation

Update:

- `README.md` model provider table
- `config/models.yaml` comments
- maybe `src/harness/README.md` transport section

Document clearly:

- `codex-cli/*` uses local Codex CLI subscription, not OpenAI API;
- requires `codex login` outside AlphaSeeker;
- no API key required;
- slower than API transports;
- recommended first for local trusted development, not CI.

## Key risks

1. **Subscription vs API ambiguity** — must label clearly. OpenAI API path is not subscription path.
2. **CLI stability** — Codex CLI output/events may change across versions; keep parser conservative and tested.
3. **Latency/cost perception** — subprocess per turn is slower than SDK API calls.
4. **Auth expiry** — if `codex login` expires, harness should fail with clear remediation: run `codex login`.
5. **Tool execution safety** — use read-only sandbox and prevent Codex CLI from doing its own file edits/shell commands in the harness loop.
6. **Structured-output schema limitations** — use `arguments_json` string; avoid arbitrary nested object schema.
7. **Context size** — Codex CLI accepted a small probe, but full harness prompts/tools are large; first integration should test a minimal run before production use.

## Recommended next step

Build a small prototype branch/patch for `codex_cli` transport with mocked unit tests and one manual live smoke test.

The first live smoke should not run a full XOM memo. Use a tiny harness task with a short budget and force the model to call `status(done)` or write a tiny file. Confirm:

- Codex CLI is invoked;
- the final structured JSON is parseable;
- AlphaSeeker executes the returned tool call;
- run artifacts record request/response/stdout/stderr;
- no Codex CLI file edits occur outside AlphaSeeker tool execution.

Only after this smoke passes should we switch normal AlphaSeeker harness agent config to `codex-cli/gpt-5.5`.
