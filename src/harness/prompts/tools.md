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
- Use `bash(argv=[...], cwd=...)` for repo-scoped filesystem and Unix operations such as `rg --files -g`, `cp`, `mv`, `mkdir`, and `ls`.
- Use `search_in_files(pattern=..., paths=[...])` to locate relevant files or lines before reading larger files.
- Use `read_file(path=..., start_line=..., max_lines=...)` when you want a line-based slice, or `read_file(path=..., max_chars=..., start_char=...)` for a character range.
- Use `read_web_pages(urls=[...], max_chars_per_url=...)` after `search_web` or `search_news` when you want actual page content.
- Use `edit_file(path=..., ...)` only for short exact replacements or inserts under `publish/` or `scratch/`.
- Use `apply_patch(patch=...)` for localized multi-line edits after reading the relevant file slice with `read_file(path=...)`.
- The `apply_patch` payload must be formatted exactly like:
  `*** Begin Patch`
  `*** Update File: publish/example.md`
  `@@`
  `-old line`
  `+new line`
  `*** End Patch`
- In each `apply_patch` hunk, use a leading space for unchanged context lines, `-` for removed lines, and `+` for added lines. Context matching is exact; if the context is missing or duplicated, the patch fails.
- If `apply_patch` fails because the context is missing or ambiguous, call `read_file(path=...)` on that file again before retrying and rebuild the patch from the exact current lines.
- Use `write_file(path=..., content=...)` when you are replacing most of a file or creating a new one under `publish/` or `scratch/`.
- If you delegate, the parent only sees your child metadata and published files by default.
- `spawn_subagent` accepts only the preset names listed above. Unknown preset names will be rejected.
- If you are the root agent, set `status` to `done` only after writing `publish/final.md`, `publish/summary.md`, and `publish/artifact_index.md`.
- If you are a child agent, set `status` to `done` after you have written the published file or files your parent needs. `publish/summary.md` and `publish/artifact_index.md` are recommended but not mandatory.
- File reads return real content. Condense only when you explicitly choose to call `condense_context` or delegate compression to a child.
