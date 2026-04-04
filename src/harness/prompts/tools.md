# Allowed Tools

## Agent Tools

{{agent_tools}}

## Subagent Presets

{{child_presets}}

## Deterministic Skills

{{deterministic_skills}}

## Operating Rules

- Use `publish/` for stable handoff files.
- Use `scratch/` for working notes and bulky outputs.
- Use `glob_files(patterns=[...], paths=[...])` to discover exact file paths by name or path pattern before reading them.
- Use `search_in_files(pattern=..., paths=[...])` to locate relevant files or lines before reading larger files.
- Use `read_file(path=..., start_line=..., max_lines=...)` when you want a line-based slice, or `read_file(path=..., max_chars=..., start_char=...)` for a character range.
- Use `read_web_pages(urls=[...], max_chars_per_url=...)` after `search_web` or `search_news` when you want actual page content.
- Use `write_file(path=..., content=...)` and `edit_file(path=..., ...)` for files under `publish/` or `scratch/`, for example `publish/summary.md` or `scratch/notes.md`.
- If you delegate, the parent only sees your child metadata and published files by default.
- `spawn_subagent` accepts only the preset names listed above. Unknown preset names will be rejected.
- If you are the root agent, set `status` to `done` only after writing `publish/final.md`, `publish/summary.md`, and `publish/artifact_index.md`.
- If you are a child agent, set `status` to `done` after you have written the published file or files your parent needs. `publish/summary.md` and `publish/artifact_index.md` are recommended but not mandatory.
- File reads return real content. Condense only when you explicitly choose to call `condense_context` or delegate compression to a child.
