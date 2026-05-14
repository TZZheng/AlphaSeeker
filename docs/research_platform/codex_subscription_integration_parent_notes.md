# Parent Notes — Codex Subscription Integration

These are interim notes from the parent agent while daemon `em-1` investigates.

## Current AlphaSeeker model config

- `config/models.yaml:8-15` documents supported prefixes: `sf/*`, `kimi-*`, `minimax/*`, `gemini-*`, `gpt-*`, `claude-*`. No `codex/*` or subscription provider.
- `config/models.yaml:22-24` currently sets harness agent/condense to `minimax/Minimax-M2.7`.

## Current provider resolution

- `src/shared/model_config.py:102-142` maps model strings to provider labels/env vars. OpenAI models (`gpt-*`, `o1`, `o3`, `o4`) require `OPENAI_API_KEY`; there is no Codex subscription auth path.
- `src/shared/llm_manager.py:283-355` builds LangChain models for Kimi, OpenAI API, Anthropic API, SiliconFlow, MiniMax, Gemini. No Codex CLI/subscription branch.
- `src/harness/transport.py:121-130` resolves MiniMax/Anthropic/OpenAI/text_json only.
- `src/harness/types.py:191,257` validates `agent_transport` Literal only as `auto`, `minimax_anthropic`, `minimax_openai`, `anthropic`, `openai`, `text_json`.
- `src/harness/transport.py:2031-2038` OpenAI native transport uses `OpenAI(api_key=os.environ["OPENAI_API_KEY"])`, i.e. API-key OpenAI, not ChatGPT/Codex subscription.

## Local Codex CLI observed

- `codex --version` -> `codex-cli 0.130.0`.
- `~/.codex/config.toml` exists and configures `model = "gpt-5.5"`; AlphaSeeker project is trusted.
- `codex exec --help` supports non-interactive mode, `--json`, `--output-last-message`, `--output-schema`, `--cd`, `--sandbox`, `--model`, `--ephemeral`.

## Interim conclusion

AlphaSeeker cannot currently use Ted's Codex subscription directly. It can use OpenAI API models if `OPENAI_API_KEY` exists, but that is not the same as the Codex/ChatGPT subscription used by this Lingtai agent. The feasible subscription path is likely a new Codex CLI subprocess transport or a narrower Codex CLI wrapper for harness turns.

Recommended architecture to evaluate:

1. Add a `codex_cli` transport for harness only, not LangChain tools initially.
2. Use `codex exec --json --output-last-message <file> --cd <repo> --sandbox read-only/never ...` with a constructed single-turn prompt containing system prompt, conversation replay, available tools schema, and strict JSON output schema.
3. Because Codex CLI does not expose OpenAI-style function calling through a simple SDK interface, make the harness ask for JSON commands/text in the final message, then parse into `ModelTurnResult`. This resembles a custom `text_json` transport rather than native OpenAI tool calls.
4. Keep deterministic tool execution in AlphaSeeker; do not let Codex CLI run arbitrary shell tools for harness agents. Use read-only sandbox and no approvals when possible.
5. First prototype should be a small standalone proof: given a prompt and one fake tool schema, Codex CLI returns parseable JSON with either text or tool_calls. Only after proof wire into `transport.py`.
