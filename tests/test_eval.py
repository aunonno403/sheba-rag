import litellm

from eval import run_eval


def test_paced_spaces_calls_and_retries_rate_limits(monkeypatch):
    clock = {"t": 0.0}
    sleeps = []
    monkeypatch.setattr(run_eval.time, "time", lambda: clock["t"])
    monkeypatch.setattr(run_eval.time, "sleep", lambda s: sleeps.append(s) or clock.__setitem__("t", clock["t"] + s))

    calls = []

    def fake_complete(prompt, model):
        calls.append(model)
        if len(calls) == 2:  # second call hits the quota once
            raise litellm.RateLimitError("429", llm_provider="gemini", model=model)
        return "ok"

    paced = run_eval.paced(fake_complete, rpm=5)
    assert paced("p", "m") == "ok"
    assert paced("p", "m") == "ok"
    # 12s spacing before the 2nd call, 60s backoff after the 429, then spacing is already satisfied
    assert sleeps == [12.0, 60]
    assert calls == ["m", "m", "m"]


def test_generate_resumes_without_repeating_llm_calls(monkeypatch, tmp_path):
    import json
    items = [
        {"id": "a1", "question": "q1", "answerable": True, "expected_answer": "x", "expected_doc": "d", "evidence": "x"},
        {"id": "u1", "question": "q2", "answerable": False, "expected_answer": "", "expected_doc": None, "evidence": None},
    ]
    (tmp_path / "testset.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    monkeypatch.setattr(run_eval, "HERE", tmp_path)
    calls = []

    def fake_answer(q, chunker, model):
        calls.append(q)
        abstained = q == "q2"
        return {"answer": "x [1]", "abstained": abstained, "hits": [],
                "citations": [] if abstained else [{"doc_id": "d"}]}
    monkeypatch.setattr(run_eval.ans, "answer", fake_answer)
    monkeypatch.setattr(run_eval, "judge", lambda it, r: (1, 1))

    first = run_eval.generate(["sentence:m"], rpm=1000)
    assert calls == ["q1", "q2"]
    second = run_eval.generate(["sentence:m"], rpm=1000)
    assert calls == ["q1", "q2"]  # everything reused from results.jsonl
    assert first == second and "| 1.00 | 1.00 | 1.00 | 0.00 |" in first
