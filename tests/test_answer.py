import pytest

from sheba_rag import answer as ans

HITS = [
    {"text": "মৌজা ম্যাপ = ৬৩০ টাকা", "score": 0.8, "title": "ভূমিসেবার ফি", "url": "u1", "doc_id": "land_fees"},
    {"text": "৭। কর ৩০ জুনের মধ্যে", "score": 0.6, "title": "ভূমি উন্নয়ন কর আইন", "section": "৭",
     "url": "u2", "doc_id": "land_tax_act_2023"},
]


@pytest.fixture
def fake(monkeypatch):
    calls = []

    def setup(reply, hits=HITS):
        monkeypatch.setattr(ans.store, "search", lambda *a, **k: hits)
        monkeypatch.setattr(ans, "complete", lambda prompt, model: calls.append(prompt) or reply)
        return calls
    return setup


def test_detect_lang():
    assert ans.detect_lang("মৌজা ম্যাপ কত?") == "bn"
    assert ans.detect_lang("How much is a mouza map?") == "en"


def test_answer_with_valid_citations(fake):
    fake("মৌজা ম্যাপের ফি ৬৩০ টাকা [1]। [9]")
    r = ans.answer("মৌজা ম্যাপ কত টাকা?", min_score=0.5)
    assert not r["abstained"]
    assert [c["n"] for c in r["citations"]] == [1]  # [9] matches no block and is dropped


def test_bangla_digit_citations(fake):
    fake("মৌজা ম্যাপের ফি ৬৩০ টাকা [১]।")
    r = ans.answer("মৌজা ম্যাপ কত টাকা?", min_score=0.5)
    assert not r["abstained"] and [c["n"] for c in r["citations"]] == [1]


def test_comma_and_range_citations(fake):
    fake("Tax is due by 30 June [1, 2]. See also [1-2].")
    r = ans.answer("When is the tax due?", min_score=0)
    assert not r["abstained"] and [c["n"] for c in r["citations"]] == [1, 2]


def test_low_score_abstains_without_llm_call(fake):
    calls = fake("should not be used", hits=[{**HITS[0], "score": 0.1}])
    r = ans.answer("Thai visa?", min_score=0.5)
    assert r["abstained"] and r["answer"] == ans.IDK["en"]
    assert calls == []


def test_abstain_token_and_uncited_answers_become_idk(fake):
    fake(ans.ABSTAIN_TOKEN)
    assert ans.answer("কিছু?", min_score=0)["abstained"]
    fake("An answer with no citation.")
    assert ans.answer("something?", min_score=0)["abstained"]


def test_prompt_asks_for_question_language_and_labels_sections(fake):
    calls = fake("x [2]")
    ans.answer("When is the tax due?", min_score=0)
    assert "Reply in English" in calls[0]
    assert "(ভূমি উন্নয়ন কর আইন, § ৭)" in calls[0]
