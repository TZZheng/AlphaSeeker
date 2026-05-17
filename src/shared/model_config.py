"""
Shared model configuration for AlphaSeeker.

Model assignments are code defaults plus environment-variable overrides.  The
old ``config/models.yaml`` file was removed with the legacy harness cleanup; use
``ALPHASEEKER_MODEL_<AREA>_<ROLE>`` to override a default for local runs.

Examples:
    from src.shared.model_config import get_model
    VAULT_MODEL = get_model("vault", "agent")       # -> "codex/gpt-5.5"
    EQUITY_CONDENSE = get_model("equity", "condense")
"""

import os
from typing import Dict, Set, Tuple


# ---------------------------------------------------------------------------
# Built-in defaults
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Dict[str, str]] = {
    "vault": {
        "agent": "codex/gpt-5.5",
    },
    "equity": {
        "condense": "codex/gpt-5.5",
    },
}


def _load_config() -> Dict[str, Dict[str, str]]:
    """Return built-in model defaults. Kept as a helper for tests."""
    return {agent: dict(roles) for agent, roles in _DEFAULTS.items()}


def get_model(agent: str, role: str) -> str:
    """
    Returns the model string for a given agent and role.

    Resolution order (highest priority first):
      1. Environment variable: ALPHASEEKER_MODEL_<AGENT>_<ROLE>
      2. Built-in defaults

    Args:
        agent: Area name, e.g. "vault" or "equity".
        role: Model role within the area, e.g. "agent" or "condense".

    Returns:
        Model string, e.g. "sf/Qwen/Qwen3-14B" or "kimi-k2.5".

    Raises:
        ValueError: If neither environment nor defaults contain the requested agent/role.
    """
    # 1. Check env var override
    env_key = f"ALPHASEEKER_MODEL_{agent.upper()}_{role.upper()}"
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val

    # 2. Check built-in defaults
    default_config = _load_config().get(agent, {})
    if role in default_config:
        return default_config[role]

    raise ValueError(
        f"No model configured for agent='{agent}', role='{role}'. "
        f"Add a default in src.shared.model_config or set env var {env_key}."
    )


def _provider_label(model_name: str) -> str | None:
    normalized = model_name.lower()
    if model_name.startswith("sf/"):
        return "sf/*"
    if model_name.startswith("gemini-"):
        return "gemini-*"
    if model_name.startswith("kimi-"):
        return "kimi-*"
    if normalized.startswith("codex/"):
        return "codex/*"
    if (
        normalized.startswith("minimax/")
        or normalized.startswith("minimax-")
        or normalized.startswith("codex-minimax-")
    ):
        return "minimax/*"
    if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4"):
        return "openai"
    if model_name.startswith("claude-"):
        return "anthropic"
    return None


def _provider_env_candidates(model_name: str) -> Tuple[str, ...] | None:
    """Map model naming convention to one-or-more acceptable env vars."""
    normalized = model_name.lower()
    if model_name.startswith("sf/"):
        return ("SILICONFLOW_API_KEY",)
    if model_name.startswith("gemini-"):
        return ("GOOGLE_API_KEY",)
    if model_name.startswith("kimi-"):
        return ("KIMI_API_KEY",)
    if normalized.startswith("codex/"):
        return None
    if (
        normalized.startswith("minimax/")
        or normalized.startswith("minimax-")
        or normalized.startswith("codex-minimax-")
    ):
        return ("MINIMAX_API_KEY",)
    if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4"):
        return ("OPENAI_API_KEY",)
    if model_name.startswith("claude-"):
        return ("ANTHROPIC_API_KEY",)
    return None


def _collect_required_provider_env_candidates() -> Dict[str, Tuple[str, ...]]:
    """Collect required env-var candidates per active provider family."""
    config = _load_config()
    required: Dict[str, Tuple[str, ...]] = {}

    for agent, agent_config in config.items():
        roles = set(agent_config)
        for role in roles:
            try:
                model_name = get_model(agent, role)
            except ValueError:
                continue
            label = _provider_label(model_name)
            candidates = _provider_env_candidates(model_name)
            if label and candidates:
                required[label] = candidates
    return required


def get_required_provider_env_vars() -> Set[str]:
    """
    Return env vars required by currently configured model providers.

    This function evaluates model assignments with environment-variable overrides
    applied (via get_model), then derives the provider key requirements.
    """
    required_candidates = _collect_required_provider_env_candidates()
    # Return canonical key (first entry) for each provider family.
    return {candidates[0] for candidates in required_candidates.values()}


def get_missing_provider_env_vars() -> Dict[str, str]:
    """
    Return missing provider env requirements for the active model configuration.

    Output format:
      {
        "kimi-*": "KIMI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY"
      }
    """
    missing: Dict[str, str] = {}
    for label, candidates in sorted(_collect_required_provider_env_candidates().items()):
        if any(os.getenv(candidate) for candidate in candidates):
            continue
        if len(candidates) == 1:
            missing[label] = candidates[0]
        else:
            missing[label] = " or ".join(candidates)
    return missing
