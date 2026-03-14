from __future__ import annotations

import pytest

from src.shared import model_config

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_config_cache() -> None:
    model_config._load_config.cache_clear()


def test_required_provider_env_vars_from_default_config() -> None:
    required = model_config.get_required_provider_env_vars()

    assert "KIMI_API_KEY" in required
    assert "SILICONFLOW_API_KEY" in required


def test_missing_provider_env_vars_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    missing = model_config.get_missing_provider_env_vars()

    assert missing["kimi-*"] == "KIMI_API_KEY"
    assert missing["sf/*"] == "SILICONFLOW_API_KEY"


def test_kimi_requires_dedicated_key_no_openai_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-present")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sf-key-present")
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    missing = model_config.get_missing_provider_env_vars()

    assert missing["kimi-*"] == "KIMI_API_KEY"
    assert "sf/*" not in missing


def test_all_required_keys_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIMI_API_KEY", "kimi-key-present")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sf-key-present")

    missing = model_config.get_missing_provider_env_vars()

    assert missing == {}
