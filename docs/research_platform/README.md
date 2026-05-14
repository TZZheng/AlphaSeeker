# Research Platform Expansion Materials

This folder collects materials for expanding AlphaSeeker from a one-shot multi-agent memo generator into a persistent equity research platform / research OS.

## Files

- `AlphaSeeker_expansion_plan_2026-05-12.md` — repo-specific implementation plan: persistent vault, SQLite schema, `src/vault/`, `src/harness/skills/vault.py`, and implementation order.
- `xiaohongshu_librarian_2026-05-12/librarian_xhs_extracted_report.md` — reconstructed content and analysis of the Xiaohongshu post “Librarian 升级：从记忆系统到主动投研助手”.
- `xiaohongshu_librarian_2026-05-12/images/` — downloaded images from the post used for OCR/reconstruction.
- `xiaohongshu_librarian_2026-05-12/metadata.json` — source URL/title/image URL metadata from extraction.
- `xiaohongshu_librarian_2026-05-12/page.html` — fetched SSR HTML snapshot if available.
- `xiaohongshu-post-extraction-skill/` — reusable extraction workflow/script for future Xiaohongshu image-heavy posts.

## Product direction

Current AlphaSeeker shape:

```text
prompt -> multi-agent run -> final memo
```

Target shape:

```text
source docs + live data -> persistent company vault -> multi-agent run -> wiki updates + conflicts + memo
```

First implementation target: add a local `data/research_vault/` SQLite-backed company/ticker vault, then expose it through harness skills.
