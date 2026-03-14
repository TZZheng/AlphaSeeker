# Contributing

Thanks for contributing to AlphaSeeker.

## Local setup

1. Install dependencies:
```bash
uv sync
```

2. Create local env file:
```bash
cp .env.example .env
```

3. Run compile gate:
```bash
uv run python -m compileall -q src main.py
uv run pytest -m "not live"
```

## Pull request expectations

- Keep changes scoped and well-described.
- Add or update docs when behavior changes.
- Do not commit secrets (`.env`, API keys, credentials).
- Keep runtime-generated artifacts out of git (`data/`, `reports/`, `charts/`).

## Coding standards

- Use type hints for public functions.
- Prefer explicit state contracts between graph nodes.
- Handle external API failures with clear error paths and logging.
