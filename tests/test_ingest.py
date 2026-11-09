import pytest
import requests

from sheba_rag import ingest

SRC = {"id": "nid_faq", "service": "nid", "type": "html", "title": "t", "url": "https://a.example/faq", "lang": "bn"}


def test_failed_download_stops_before_index_rebuild(monkeypatch):
    monkeypatch.setattr(ingest, "read_sources", lambda: [SRC])

    def offline(src):
        raise requests.ConnectionError("no network")
    monkeypatch.setattr(ingest, "download", offline)
    built = []
    monkeypatch.setattr(ingest.store, "build", lambda *a: built.append(a) or 0)
    with pytest.raises(SystemExit, match="nid_faq"):
        ingest.main(["--chunker", "sentence"])
    assert built == []  # the existing index was left alone


def test_changed_url_gets_a_new_cache_file():
    moved = {**SRC, "url": "https://a.example/faq-v2"}
    assert ingest.cache_name(SRC) != ingest.cache_name(moved)
    assert ingest.cache_name(SRC).startswith("nid_faq-")
