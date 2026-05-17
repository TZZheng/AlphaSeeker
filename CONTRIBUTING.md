# Contributing

Thanks for contributing to AlphaSeeker.

## Local setup

1. Install dependencies:

```bash
uv sync
```

2. Create a local env file if you need live provider credentials:

```bash
cp .env.example .env
```

3. Run the local validation gates:

```bash
uv run python -m compileall -q src scripts
uv run pytest -q
uv run python -m py_compile scripts/lingtai_ticker_harness.py
```

## Pull request expectations

- Keep changes scoped and well-described.
- Add or update docs when behavior changes.
- Do not commit secrets (`.env`, API keys, OAuth tokens, or live agent credentials).
- Keep runtime-generated and local-only artifacts out of git (`tmp/`, `data/`, `reports/`, `charts/`, `.trash/`, `.venv/`, ticker `.lingtai/` directories, vault raw/draft/published/wiki artifacts unless explicitly requested).
- Preserve the ticker-native role templates under `templates/lingtai_native/` when changing the bridge script.

## Coding standards

- Use type hints for public functions.
- Prefer deterministic vault/source transformations before model-assisted synthesis.
- Treat `vault/companies/<TICKER>/.lingtai/` as live runtime state, not normal source code.
- Handle external API failures with clear error paths and logging.
