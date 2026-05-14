# LingTai Codex Backend Integration Report for AlphaSeeker

Date: 2026-05-12 PT
Request: Ted noted that this LingTai agent is itself driven by a Codex subscription/backend and asked to inspect that code path before choosing how AlphaSeeker should use Codex.

## Short answer

LingTai does **not** drive this agent through the Codex CLI. It has a Python `provider="codex"` LLM adapter that uses OpenAI OAuth tokens from `~/.lingtai-tui/codex-auth.json`, refreshes them automatically, and calls ChatGPT's Codex backend at:

```text
https://chatgpt.com/backend-api/codex/responses
```

This is closer and better than my first CLI-only recommendation. For AlphaSeeker, the best long-term path is to reuse/copy the **LingTai Codex backend pattern** rather than shelling out to `codex exec` for every harness turn.

The Codex CLI path is still useful as a fallback/prototype, but the cleaner AlphaSeeker integration is a native Python transport based on LingTai's `CodexTokenManager` + `CodexOpenAIAdapter` / `CodexResponsesSession` behavior.

## Evidence from this agent's config

This running agent's config (`.lingtai/codex/.agent.json`) says:

```json
"llm": {
  "provider": "codex",
  "model": "gpt-5.5",
  "base_url": "https://chatgpt.com/backend-api/codex",
  "context_limit": 200000
}
```

The saved preset (`~/.lingtai-tui/presets/saved/codex.json`) likewise sets:

```json
"llm": {
  "api_key": null,
  "api_key_env": "",
  "base_url": "https://chatgpt.com/backend-api/codex",
  "model": "gpt-5.5",
  "provider": "codex"
}
```

So Ted is right: this agent is not just using generic OpenAI API. It is using LingTai's Codex provider.

## LingTai auth path

File: `/Users/tianzhezheng/Documents/lingtai-dev/lingtai-kernel/src/lingtai/auth/codex.py`

Key points:

- `CodexTokenManager` reads tokens from `~/.lingtai-tui/codex-auth.json` by default (`codex.py:30-37`).
- It checks whether a refresh token exists (`codex.py:46-52`).
- `get_access_token()` returns a valid access token and refreshes when within five minutes of expiry (`codex.py:54-66`).
- Refresh uses `https://auth.openai.com/oauth/token` with `grant_type=refresh_token`, the stored `refresh_token`, and client id `app_EMoamEEZ73f0CkXaXp7hrann` (`codex.py:17-19`, `codex.py:110-118`).
- Refresh writes the updated token file with mode `0600` via a temp file and replace (`codex.py:137-141`).
- Concurrent refresh is guarded with `filelock.FileLock` (`codex.py:99-104`).
- On 401/403, it raises a user-facing `CodexAuthError` telling the user to re-authenticate in the TUI (`codex.py:121-125`).

No secrets are hardcoded except the public OAuth client id. The private user tokens remain in `~/.lingtai-tui/codex-auth.json`.

## LingTai provider registration

File: `/Users/tianzhezheng/Documents/lingtai-dev/lingtai-kernel/src/lingtai/llm/_register.py`

The `codex` provider factory is at `_register.py:54-82`:

- Imports `CodexOpenAIAdapter`.
- Imports `CodexTokenManager`.
- Ignores any env-resolved `api_key` and `base_url` (`kw.pop("api_key", None)`, `kw.pop("base_url", None)`).
- Creates `CodexTokenManager()`.
- Builds `CodexOpenAIAdapter` with:

```python
api_key=mgr.get_access_token(),
base_url="https://chatgpt.com/backend-api/codex",
use_responses=True,
force_responses=True,
```

- Stores the token manager on the adapter and wraps `create_chat` / `generate` so `adapter._client.api_key` is refreshed before each API call (`_register.py:67-79`).
- Registers provider name `codex` with `LLMService.register_adapter("codex", _codex)` (`_register.py:82`).

This is the core path AlphaSeeker should mimic.

## LingTai Codex adapter path

File: `/Users/tianzhezheng/Documents/lingtai-dev/lingtai-kernel/src/lingtai/llm/openai/adapter.py`

### Responses tool schema

LingTai has separate builders for Chat Completions tools and Responses API tools:

- `_build_tools()` creates nested Chat Completions shape (`adapter.py:59-73`).
- `_build_responses_tools()` creates flat Responses shape with `type=function`, `name`, `description`, `parameters` (`adapter.py:83-107`).
- It scrubs top-level `allOf`, `oneOf`, `anyOf`, `not`, and `enum`, because the Responses API rejects those at the root of tool parameters (`adapter.py:76-107`).

### Responses parsing

`_parse_responses_api_response()` parses raw Responses output into LingTai's provider-neutral `LLMResponse` (`adapter.py:177-219`):

- `message/output_text` blocks become text.
- `function_call` items become tool calls after JSON parsing `item.arguments`.
- `reasoning` summaries become thoughts.
- usage fields are read from Responses-style usage metadata.

### Codex stateless session

`CodexResponsesSession` is defined at `adapter.py:1185-1341`.

Important behavior:

- Codex backend is stateless: no `previous_response_id` is sent (`adapter.py:1188-1192`, `adapter.py:1263`).
- It forces `store=False` (`adapter.py:1191-1192`, `adapter.py:1253-1255`).
- It forces streaming; `send()` delegates to `send_stream()` because non-streaming Codex responses cannot be unmarshaled reliably (`adapter.py:1193-1199`).
- It replays the full canonical interface each request via `to_responses_input(self._interface)` (`adapter.py:1201-1239`).
- It enforces tool pairing before building wire input (`adapter.py:1236-1238`).
- It calls `self._client.responses.create(**kwargs)` (`adapter.py:1273`).
- It handles stream events:
  - `response.output_text.delta`
  - `response.function_call_arguments.delta`
  - `response.output_item.added` for function calls
  - `response.output_item.done`
  - `response.completed` for usage (`adapter.py:1273-1305`).
- It records assistant text/tool-call blocks back into the interface so the next stateless request includes prior assistant turns (`adapter.py:1314-1335`).

### Codex adapter

`CodexOpenAIAdapter` is defined at `adapter.py:1344-1398`.

It subclasses the OpenAI adapter, but creates `CodexResponsesSession` instead of the normal stateful Responses session. It documents the required configuration:

```text
provider=codex only
use_responses=True
force_responses=True
base_url='https://chatgpt.com/backend-api/codex'
```

It also deliberately leaves `compact_threshold=None` because Codex's backend does not accept `context_management` compaction (`adapter.py:1386-1397`).

## Design note from LingTai docs/discussion

The patch discussion `/Users/tianzhezheng/Documents/lingtai-dev/lingtai-kernel/discussions/codex-oauth-stateless-patch.md` explains the origin:

- ChatGPT backend serves Codex at `/backend-api/codex/responses`, not `/backend-api/responses`.
- Responses API uses `reasoning: { effort: ... }`, not Chat Completions' `reasoning_effort`.
- Codex backend is stateless: every request must carry full input; do not use `previous_response_id`; use `store=false`, `stream=true`.
- Responses tools need flat function schema and top-level combinator scrubbing.

That discussion directly confirms the implementation details above.

## Comparison to AlphaSeeker today

AlphaSeeker currently lacks this path.

Current AlphaSeeker:

- `config/models.yaml` documents only API-key providers; no `codex` provider.
- `src/shared/model_config.py` maps `gpt-*`/`o*` to `OPENAI_API_KEY`, not OAuth tokens.
- `src/shared/llm_manager.py` creates LangChain `ChatOpenAI` using `OPENAI_API_KEY`.
- `src/harness/transport.py` has native OpenAI transport using `OpenAI(api_key=os.environ["OPENAI_API_KEY"])`.
- `src/harness/types.py` allows only `auto`, `minimax_anthropic`, `minimax_openai`, `anthropic`, `openai`, `text_json`.

Therefore AlphaSeeker cannot use the Codex subscription until it gains either:

1. a native LingTai-style `codex` transport, or
2. a subprocess Codex CLI transport.

## Updated recommendation

Prefer **native LingTai-style Codex backend integration** over Codex CLI.

Why:

- It is how this agent is actually driven.
- It avoids one CLI process per turn.
- It supports real function/tool calling through Responses API events.
- It reuses the same OAuth token file and refresh logic.
- It preserves AlphaSeeker's existing deterministic tool execution model.

Codex CLI remains a fallback because it is easy to test, but it is less clean and slower.

## Proposed AlphaSeeker implementation path

### Patch 1 — minimal Codex token manager

Add one of:

- copy a small self-contained `CodexTokenManager` into AlphaSeeker, e.g. `src/shared/codex_auth.py`; or
- add `lingtai-kernel` as a local/dev dependency and import `lingtai.auth.codex.CodexTokenManager`.

Recommendation: copy the small manager first to avoid coupling AlphaSeeker packaging to LingTai internals. The file is only about 145 lines and depends on `httpx` + `filelock`; AlphaSeeker already indirectly has `httpx`, but may need explicit `filelock` in `pyproject.toml`.

### Patch 2 — native Codex transport

Add `CodexNativeTransport` to AlphaSeeker harness.

It should use OpenAI SDK with:

```python
OpenAI(
    api_key=CodexTokenManager().get_access_token(),
    base_url="https://chatgpt.com/backend-api/codex",
)
```

Then call:

```python
client.responses.create(
    model="gpt-5.5",
    input=<full conversation as Responses input>,
    instructions=<system prompt>,
    tools=<flat responses tools>,
    stream=True,
    store=False,
)
```

Do not send `previous_response_id`.
Do not send `context_management`.
Refresh `client.api_key` before each call.

### Patch 3 — interface conversion

AlphaSeeker currently stores OpenAI/Anthropic-style conversation messages in `src/harness/transport.py`. It needs a converter from the harness conversation to Responses `input` items.

Simplest first version:

- system prompt -> `instructions`
- user messages -> `{"role":"user","content":[{"type":"input_text","text": ...}]}`
- assistant text -> `{"role":"assistant","content":[{"type":"output_text","text": ...}]}` if accepted; otherwise use Responses-compatible message shape from OpenAI SDK expectations
- assistant tool calls -> function_call items
- tool results -> function_call_output items

This is the riskiest piece. LingTai already solved it in `to_responses_input(self._interface)`. If feasible, copy/adapt that converter rather than rediscovering the protocol.

### Patch 4 — config/validation

- Add `codex/*` or `codex/gpt-5.5` model naming.
- Add `codex` or `codex_responses` transport.
- Update `resolve_agent_transport()`.
- Update `HarnessRequest.agent_transport` Literal.
- Document that this uses `~/.lingtai-tui/codex-auth.json`, not `OPENAI_API_KEY`.

### Patch 5 — tests

Mock all network calls.

Tests should cover:

1. `CodexTokenManager` reads token file and refreshes near expiry.
2. 401/403 refresh maps to clear re-login error.
3. `resolve_agent_transport("auto", "codex/gpt-5.5") == "codex"`.
4. Responses tools use flat shape and scrub disallowed top-level schema keys.
5. Request payload uses `stream=True`, `store=False`, no `previous_response_id`.
6. Streaming function-call events become AlphaSeeker `ModelToolCall`s.
7. Tool results are replayed as function-call outputs on the next request.

### Patch 6 — live smoke

Only after mocked tests pass:

- Run a tiny harness task with `harness.agent = "codex/gpt-5.5"`.
- Ask it to call `status(done)` or write a tiny file.
- Verify artifacts and no hidden direct file mutation.

## Security / product risks

- `~/.lingtai-tui/codex-auth.json` is sensitive; never commit it or copy it into the repo.
- Need to handle expired login gracefully: tell Ted to re-run LingTai/Codex OAuth login.
- Codex backend is not a public stable API in the same way OpenAI API is; CLI or backend behavior may change.
- Native Responses protocol is more complex than Chat Completions; copying LingTai's known-good converter is safer than inventing a new one.
- If AlphaSeeker imports LingTai internals directly, dependency boundaries get messy. Copying the small auth/session logic may be cleaner for AlphaSeeker.

## Final recommendation

Revise the previous plan:

1. Do **not** make Codex CLI the primary path.
2. Implement a LingTai-style native Codex Responses transport in AlphaSeeker.
3. Reuse the same token file and refresh semantics as LingTai.
4. Use Codex CLI only as a fallback/probe.
5. Build mocked tests before any real run.

This is the path most consistent with how the current LingTai agent is actually driven by Ted's Codex subscription.
