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
- Use `search_in_files(pattern=..., paths=[...])` to locate relevant files or lines before reading larger files.
- Use `read_file(path=..., max_chars=..., start_char=...)` for any exact file path, including publish files, copied context files, and tool-returned artifact paths.
- Use `read_web_pages(urls=[...], max_chars_per_url=...)` after `search_web` or `search_news` when you want actual page content.
- If you delegate, the parent only sees your child metadata and published files by default.
- `spawn_subagent` accepts only the preset names listed above. Unknown preset names will be rejected.
- If you are the root agent, set `status` to `done` only after writing `publish/final.md`, `publish/summary.md`, and `publish/artifact_index.md`.
- If you are a child agent, set `status` to `done` after you have written the published file or files your parent needs. `publish/summary.md` and `publish/artifact_index.md` are recommended but not mandatory.
- File reads return real content. Condense only when you explicitly choose to call `condense_context` or delegate compression to a child.
