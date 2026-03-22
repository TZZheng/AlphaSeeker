"""Skill registry for the harness runtime."""

from __future__ import annotations

from src.harness.skills import COMMODITY_SKILLS, CORE_SKILLS, EQUITY_SKILLS, MACRO_SKILLS
from src.harness.types import SkillSpec


def build_skill_registry() -> dict[str, SkillSpec]:
    """Build the full harness skill registry."""

    registry: dict[str, SkillSpec] = {}
    for spec in [*CORE_SKILLS, *EQUITY_SKILLS, *MACRO_SKILLS, *COMMODITY_SKILLS]:
        registry[spec.name] = spec
    return registry


def get_skills_for_packs(
    registry: dict[str, SkillSpec],
    enabled_packs: list[str],
) -> list[SkillSpec]:
    """Return available skills for the enabled packs, always including core."""

    packs = set(enabled_packs)
    packs.add("core")
    return [spec for spec in registry.values() if spec.pack in packs]
