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
    index = {"fp": "idx1"}
    monkeypatch.setattr(run_eval.store, "index_fingerprint", lambda chunker, model: index["fp"])
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

    items[0]["expected_answer"] = "corrected answer"  # reviewing the test set must re-run that item only
    (tmp_path / "testset.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    run_eval.generate(["sentence:m"], rpm=1000)
    assert calls == ["q1", "q2", "q1"]

    edited = run_eval.inspect.getsource(run_eval.ans) + "\n# edited citation rule\n"  # any answer.py edit re-runs all
    monkeypatch.setattr(run_eval.inspect, "getsource", lambda module: edited)
    run_eval.generate(["sentence:m"], rpm=1000)
    assert calls == ["q1", "q2", "q1", "q1", "q2"]

    index["fp"] = "idx2"  # re-ingesting with different chunks re-runs everything too
    run_eval.generate(["sentence:m"], rpm=1000)
    assert calls == ["q1", "q2", "q1", "q1", "q2", "q1", "q2"]


def test_judge_reply_parsing(monkeypatch):
    import pytest
    item = {"id": "a1", "question": "q", "expected_answer": "x"}
    for reply in ('{"correct": 1, "grounded": 1} (based on {context})',
                  'Sure: {"correct": 1, "grounded": 1, "notes": {"why": "ok"}}'):
        monkeypatch.setattr(run_eval.ans, "complete", lambda p, m, reply=reply: reply)
        assert run_eval.judge(item, {"hits": [], "answer": "x"}) == (1, 1)
    for reply in ("{not json}", "no json at all", '{"correct": 1}'):
        monkeypatch.setattr(run_eval.ans, "complete", lambda p, m, reply=reply: reply)
        with pytest.raises(ValueError, match="unreadable judge reply"):
            run_eval.judge(item, {"hits": [], "answer": "x"})
