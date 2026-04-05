"""
Shared Model Config — Loads model assignments from config/models.yaml.

Each agent calls `get_model(agent, role)` to get its configured model string.
Users can override any model via:
  1. Editing config/models.yaml
  2. Setting env var ALPHASEEKER_MODEL_<AGENT>_<ROLE> (highest priority)

Example:
    from src.shared.model_config import get_model
    MODEL_AGENT = get_model("harness", "agent")       # → "minimax/MiniMax-M2.7"
    MODEL_CONDENSE = get_model("harness", "condense")  # → "kimi-k2.5"
"""

import os
from typing import Dict, Set, Tuple
from functools import lru_cache

import yaml


# ---------------------------------------------------------------------------
# Defaults (fallback if YAML is missing or a key is absent)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Dict[str, str]] = {
    "harness": {
        "agent": "kimi-k2.5",
        "condense": "sf/Qwen/Qwen3-8B",
    },
}


@lru_cache(maxsize=1)
def _load_config() -> Dict[str, Dict[str, str]]:
    """
    Loads model config from config/models.yaml.
    Falls back to _DEFAULTS if the file is missing or malformed.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "models.yaml"
    )
    if not os.path.exists(config_path):
        print(f"Warning: {config_path} not found, using default model config.")
        return _DEFAULTS.copy()

    try:
        with open(config_path, "r") as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            print(f"Warning: models.yaml is not a valid dict, using defaults.")
            return _DEFAULTS.copy()
        return loaded
    except Exception as e:
        print(f"Warning: Failed to load models.yaml ({e}), using defaults.")
        return _DEFAULTS.copy()


def get_model(agent: str, role: str) -> str:
    """
    Returns the model string for a given agent and role.

    Resolution order (highest priority first):
      1. Environment variable: ALPHASEEKER_MODEL_<AGENT>_<ROLE>
      2. config/models.yaml
      3. Hardcoded defaults

    Args:
        agent: Agent name, e.g. "equity", "macro", "commodity", "supervisor".
        role: Model role within the agent, e.g. "plan", "section", "summary".

    Returns:
        Model string, e.g. "sf/Qwen/Qwen3-14B" or "kimi-k2.5".

    Raises:
        ValueError: If neither config nor defaults contain the requested agent/role.
    """
    # 1. Check env var override
    env_key = f"ALPHASEEKER_MODEL_{agent.upper()}_{role.upper()}"
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val

    # 2. Check config file
    config = _load_config()
    agent_config = config.get(agent, {})
    if role in agent_config:
        return agent_config[role]

    # 3. Check hardcoded defaults
    default_config = _DEFAULTS.get(agent, {})
    if role in default_config:
        return default_config[role]

    raise ValueError(
        f"No model configured for agent='{agent}', role='{role}'. "
        f"Add it to config/models.yaml or set env var {env_key}."
    )


def _provider_label(model_name: str) -> str | None:
    normalized = model_name.lower()
    if model_name.startswith("sf/"):
        return "sf/*"
    if model_name.startswith("gemini-"):
        return "gemini-*"
    if model_name.startswith("kimi-"):
        return "kimi-*"
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
    all_agents = set(_DEFAULTS) | set(config)

    for agent in all_agents:
        roles = set(_DEFAULTS.get(agent, {})) | set(config.get(agent, {}))
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
