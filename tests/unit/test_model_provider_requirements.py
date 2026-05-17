from __future__ import annotations

import pytest

from src.shared import model_config

pytestmark = pytest.mark.unit


def test_required_provider_env_vars_from_defaults() -> None:
    required = model_config.get_required_provider_env_vars()

    # The vault agent and equity condense roles default to native Codex OAuth,
    # so the model layer itself requires no API-key env var.
    assert required == set()


def test_defaults_use_native_codex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPHASEEKER_MODEL_VAULT_AGENT", raising=False)
    monkeypatch.delenv("ALPHASEEKER_MODEL_EQUITY_CONDENSE", raising=False)

    assert model_config.get_model("vault", "agent") == "codex/gpt-5.5"
    assert model_config.get_model("equity", "condense") == "codex/gpt-5.5"


def test_missing_provider_env_vars_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    missing = model_config.get_missing_provider_env_vars()

    assert missing == {}


def test_minimax_requires_dedicated_key_no_openai_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHASEEKER_MODEL_VAULT_AGENT", "minimax/MiniMax-M2.5")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-present")
    monkeypatch.setenv("KIMI_API_KEY", "kimi-key-present")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sf-key-present")
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    missing = model_config.get_missing_provider_env_vars()

    assert missing == {"minimax/*": "MINIMAX_API_KEY"}


def test_all_required_keys_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key-present")

    missing = model_config.get_missing_provider_env_vars()

    assert missing == {}


def test_minimax_requires_its_own_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHASEEKER_MODEL_VAULT_AGENT", "minimax/MiniMax-M2.5")
    monkeypatch.setenv("KIMI_API_KEY", "kimi-key-present")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sf-key-present")
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    missing = model_config.get_missing_provider_env_vars()

    assert missing["minimax/*"] == "MINIMAX_API_KEY"


def test_minimax_provider_detection_accepts_raw_model_names() -> None:
    assert model_config._provider_label("MiniMax-M2.5") == "minimax/*"
    assert model_config._provider_label("codex-MiniMax-M2.5") == "minimax/*"
    assert model_config._provider_env_candidates("MiniMax-M2.5") == ("MINIMAX_API_KEY",)


def test_codex_provider_uses_oauth_token_file_not_api_key_env() -> None:
    assert model_config._provider_label("codex/gpt-5.5") == "codex/*"
    assert model_config._provider_env_candidates("codex/gpt-5.5") is None
