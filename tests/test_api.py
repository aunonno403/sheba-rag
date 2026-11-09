from fastapi.testclient import TestClient

from api import main

client = TestClient(main.app)


def test_ask_returns_answer_and_citations(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(main.ans, "answer", lambda q: {
        "answer": "৬৩০ টাকা [1]", "abstained": False, "hits": [],
        "citations": [{"n": 1, "title": "ভূমিসেবার ফি", "section": None, "page": None, "url": "u", "doc_id": "land_fees"}],
    })
    r = client.post("/ask", json={"question": "মৌজা ম্যাপ কত টাকা?"})
    assert r.status_code == 200
    assert r.json()["citations"][0]["doc_id"] == "land_fees"
    assert "hits" not in r.json()


def test_missing_key_gives_setup_message(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(main.config, "LLM_MODEL", "gemini/gemini-3.5-flash")
    r = client.post("/ask", json={"question": "মৌজা ম্যাপ কত টাকা?"})
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["detail"]


def test_llm_failure_is_503(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")

    def boom(q):
        raise RuntimeError("quota exceeded")
    monkeypatch.setattr(main.ans, "answer", boom)
    r = client.post("/ask", json={"question": "hello there"})
    assert r.status_code == 503 and "quota exceeded" in r.json()["detail"]


def test_health_reports_missing_index(monkeypatch):
    monkeypatch.setattr(main.store, "index_size", lambda: 0)
    assert client.get("/health").json()["status"].startswith("no index")
    monkeypatch.setattr(main.store, "index_size", lambda: 379)
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["chunks"] == 379


def test_rejects_empty_question():
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_rate_limit_is_friendly_429(monkeypatch):
    import litellm
    monkeypatch.setenv("GEMINI_API_KEY", "test")

    def limited(q):
        raise litellm.RateLimitError("429 quota", llm_provider="gemini", model="gemini/gemini-3.5-flash")
    monkeypatch.setattr(main.ans, "answer", limited)
    r = client.post("/ask", json={"question": "hello there"})
    assert r.status_code == 429 and "try again in a minute" in r.json()["detail"]
