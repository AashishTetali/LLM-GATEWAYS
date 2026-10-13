"""Tests for the gateway API. All LLM calls are mocked, so no API keys or
network access are needed:  pytest -q
"""

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gateway  # noqa: E402
from main import app  # noqa: E402

client = TestClient(app)


def fake_response(text="ok", prompt_tokens=20, completion_tokens=30):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@pytest.fixture
def calls(monkeypatch):
    """Patch litellm's completion; record which models were called."""
    called = []

    def fake_completion(model, **kwargs):
        called.append(model)
        return fake_response(f"answer from {model}")

    monkeypatch.setattr(gateway, "completion", fake_completion)
    return called


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize(
    "query,task",
    [
        ("Write a Python function to reverse a string.", "code"),
        ("Summarize what REST APIs are and how they work.", "summary"),
        ("Why is the sky blue?", "general"),
    ],
)
def test_classifier(query, task):
    assert gateway.classify_task(query) == task


def test_chat_routes_code_to_20b(calls):
    r = client.post("/chat", json={"query": "Write a Python function to reverse a string."})
    assert r.status_code == 200
    data = r.json()
    assert data["task"] == "code"
    assert data["model"] == "groq/openai/gpt-oss-20b"
    assert data["fallback_used"] is False
    assert calls == ["groq/openai/gpt-oss-20b"]
    # (20 * 0.075 + 30 * 0.30) / 1e6
    assert data["cost"] == pytest.approx(1.05e-5)
    assert data["usage"]["total_tokens"] == 50


def test_chat_routes_summary_to_120b(calls):
    r = client.post("/chat", json={"query": "Summarize supervised learning in 3 bullet points."})
    assert r.json()["model"] == "groq/openai/gpt-oss-120b"


def test_fallback_when_primary_fails(monkeypatch):
    called = []

    def flaky(model, **kwargs):
        called.append(model)
        if model == "groq/openai/gpt-oss-20b":
            raise RuntimeError("provider down")
        return fake_response("from fallback")

    monkeypatch.setattr(gateway, "completion", flaky)

    r = client.post("/chat", json={"query": "Why is the sky blue?"})
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "groq/openai/gpt-oss-120b"
    assert data["fallback_used"] is True
    assert called == ["groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b"]
    assert [a["status"] for a in data["attempts"]] == ["failed", "success"]


def test_all_models_fail_returns_502(monkeypatch):
    def always_fail(model, **kwargs):
        raise RuntimeError("down")

    monkeypatch.setattr(gateway, "completion", always_fail)

    r = client.post("/chat", json={"query": "Why is the sky blue?"})
    assert r.status_code == 502
    assert len(r.json()["detail"]["attempts"]) == 3


def test_empty_query_rejected():
    assert client.post("/chat", json={"query": ""}).status_code == 422


def test_compare_returns_both(calls):
    r = client.post("/compare", json={"query": "Why is the sky blue?"})
    assert r.status_code == 200
    data = r.json()
    assert data["baseline"]["model"] == gateway.BASELINE_MODEL
    assert data["gateway"]["task"] == "general"


def test_benchmark_summary(calls):
    r = client.post("/benchmark?limit=3")
    assert r.status_code == 200
    summary = r.json()["summary"]
    assert summary["queries_run"] == 3
    assert "cost_reduction_pct" in summary
    assert "latency_improvement_pct" in summary


def test_routing_endpoint():
    data = client.get("/routing").json()
    assert set(data["routing"]) == {"code", "summary", "general"}
