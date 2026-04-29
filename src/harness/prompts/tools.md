# Runtime Interface

## Agent Tools

{{agent_tools}}

## Subagent Presets

{{child_presets}}

## Deterministic Skills

{{deterministic_skills}}

## Response Mode

{{response_mode}}

## Operating Rules

- Use `publish/` for stable handoff files.
- Use `scratch/` for working notes and bulky outputs.
- Use `bash(argv=[...], cwd=...)` for restricted visible-workspace operations such as `rg --files -g`, `cp`, `mv`, `mkdir`, and `ls`.
- Use `grep(pattern=..., paths=[...])` to locate relevant files or lines before reading larger files.
- Use `read(path=..., start_line=..., max_lines=...)` when you want a line-based slice, or `read(path=..., max_chars=..., start_char=...)` for a character range.
- Use `read_web_pages(urls=[...], max_chars_per_url=...)` after `search_web` or `search_news` when you want actual page content.
- Use `edit(path=..., ...)` only for short exact replacements or inserts under `publish/` or `scratch/`.
- Use `patch(patch=...)` for localized multi-line edits after `grep` finds the line and `read(path=..., start_line=..., max_lines=...)` loads a small nearby slice; if it fails, rerun `grep` and `read`, except in soft-stop mode.
- In soft-stop mode, do not recover failed polish patches with more reads or retries; if usable publish files exist, call `status` with `done`.
- Use `write(path=..., content=...)` when you are replacing most of a file or creating a new one under `publish/` or `scratch/`.
- If you delegate, the parent only sees your child metadata and published files by default.
- `delegate` accepts only the preset names listed above. Unknown preset names will be rejected.
- If you are the root agent, set `status` to `done` only after writing `publish/final.md`, `publish/summary.md`, and `publish/artifact_index.md`.
- If you are a child agent, set `status` to `done` after you have written the published file or files your parent needs. `publish/summary.md` and `publish/artifact_index.md` are recommended but not mandatory.
- File reads return real content. Condense only when you explicitly choose to call `condense_context` or delegate compression to a child.
