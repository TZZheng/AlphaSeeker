from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.shared import llm_manager

pytestmark = pytest.mark.unit


class _FakeCodexResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> list[object]:
        self.calls.append(dict(kwargs))
        return [
            SimpleNamespace(type="response.output_text.delta", delta="Condensed "),
            SimpleNamespace(type="response.output_text.delta", delta="text."),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(id="resp_1")),
        ]


class _FakeCodexClient:
    def __init__(self) -> None:
        self.api_key = "initial-token"
        self.responses = _FakeCodexResponses()


def test_codex_llm_manager_uses_native_responses_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeCodexClient()
    monkeypatch.setattr(
        "src.shared.llm_manager.CodexTokenManager",
        lambda: SimpleNamespace(get_access_token=lambda: "fresh-token"),
    )
    monkeypatch.setattr("src.shared.llm_manager.OpenAI", lambda **kwargs: fake_client)

    model = llm_manager.CodexNativeChatModel("codex/gpt-5.5")
    response = model.invoke("Condense this text")

    assert response.content == "Condensed text."
    call = fake_client.responses.calls[0]
    assert call["model"] == "gpt-5.5"
    assert call["input"] == [{"role": "user", "content": "Condense this text"}]
    assert call["instructions"] == "You are a helpful text-processing assistant."
    assert call["stream"] is True
    assert call["store"] is False
    assert "previous_response_id" not in call
    assert "context_management" not in call
    assert fake_client.api_key == "fresh-token"


def test_get_llm_routes_codex_to_native_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.shared.llm_manager.CodexNativeChatModel",
        lambda model_name: SimpleNamespace(model_name=model_name),
    )
    llm_manager._registry.clear()

    model = llm_manager.get_llm("codex/gpt-5.5")

    assert model.model.model_name == "codex/gpt-5.5"
