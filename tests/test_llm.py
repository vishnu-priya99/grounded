"""LLM client wiring — provider/model resolution and the structured-output
shape example. No network calls."""

from __future__ import annotations

import pytest

from grounded import llm
from grounded.agents.state import RouterDecision
from grounded.config import settings


@pytest.fixture
def anthropic_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")


def test_resolve_model_passes_claude_ids_through(anthropic_provider) -> None:
    assert llm._resolve_model("claude-sonnet-5") == "claude-sonnet-5"


def test_resolve_model_swaps_stale_ollama_tag(anthropic_provider, capsys) -> None:
    assert llm._resolve_model("qwen2.5:7b") == llm._DEFAULT_ANTHROPIC_MODEL
    assert "not a Claude model" in capsys.readouterr().out


def test_resolve_model_noop_for_ollama(monkeypatch) -> None:
    # Explicit, not relying on the ambient default — a real .env (e.g. this repo's
    # own, once a paid key is configured) can set LLM_PROVIDER=anthropic globally.
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    assert llm._resolve_model("qwen2.5:7b") == "qwen2.5:7b"


def test_example_uses_null_for_optional_fields() -> None:
    shape = llm._example(RouterDecision)
    assert shape["ordinals"] == []           # list
    assert shape["sub_queries"] == []        # list
    assert shape["needs_tables"] is False


def test_router_decision_coerces_blank_ordinals() -> None:
    d = RouterDecision.model_validate(
        {"intent": "doc_qa", "rewritten_question": "q", "needs_tables": False,
         "ordinals": ""}
    )
    assert d.ordinals == []
    d2 = RouterDecision.model_validate(
        {"intent": "doc_qa", "rewritten_question": "q", "needs_tables": False,
         "ordinals": None}
    )
    assert d2.ordinals == []
