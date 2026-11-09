import pytest

from sheba_rag import store

CHUNKS = [
    {"text": "মৌজা ম্যাপ ৬৩০ টাকা", "title": "ভূমিসেবার ফি", "doc_id": "land_fees", "url": "u1",
     "section": None, "page": None},
    {"text": "৭। কর ৩০ জুনের মধ্যে", "title": "ভূমি উন্নয়ন কর আইন", "doc_id": "land_tax", "url": "u2",
     "section": "৭", "page": None},
    {"text": "হটলাইন ১৬১২২", "title": "সিটিজেন চার্টার", "doc_id": "land_charter", "url": "u3",
     "section": None, "page": 1},
]


@pytest.fixture
def temp_index(monkeypatch, tmp_path):
    monkeypatch.setattr(store.config, "CHROMA_DIR", tmp_path)
    store._client.cache_clear()

    def fake_embed(model_name, texts, kind):  # one axis per document, so the right chunk wins clearly
        axes = ["ম্যাপ", "জুন", "হটলাইন"]
        return [[1.0 if a in t else 0.01 for a in axes] for t in texts]
    monkeypatch.setattr(store, "embed", fake_embed)
    yield
    store._client.cache_clear()


def test_build_then_search_returns_metadata_and_exact_config(temp_index):
    assert store.index_size("sentence", "m") == 0  # not built yet
    assert store.build(CHUNKS, "sentence", "m") == 3
    hit = store.search("হটলাইন নম্বর", "sentence", "m", k=1)[0]
    assert (hit["doc_id"], hit["page"], hit["url"]) == ("land_charter", 1, "u3")
    assert "section" not in hit  # None values are dropped, Chroma rejects them
    hnsw = store._collection("sentence", "m").configuration_json["hnsw"]
    assert hnsw["ef_search"] == 200  # keeps top-k results repeatable between runs


def test_failed_embedding_keeps_the_old_index(temp_index, monkeypatch):
    store.build(CHUNKS, "sentence", "m")

    def out_of_memory(*a):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(store, "embed", out_of_memory)
    with pytest.raises(RuntimeError):
        store.build(CHUNKS[:1], "sentence", "m")
    assert store.index_size("sentence", "m") == 3


def test_index_fingerprint_changes_with_content(temp_index):
    store.build(CHUNKS, "sentence", "m")
    before = store.index_fingerprint("sentence", "m")
    store.build(CHUNKS, "sentence", "m")
    assert store.index_fingerprint("sentence", "m") == before  # same content, same fingerprint
    store.build([{**CHUNKS[0], "section": "২"}] + CHUNKS[1:], "sentence", "m")
    assert store.index_fingerprint("sentence", "m") != before  # a changed section label counts
