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


@pytest.mark.parametrize("html", [
    "<html><body><script>app()</script></body></html>",  # JavaScript-only page: no text at all
    "<html><head><title>Just a moment...</title></head><body><article><h1>Just a moment...</h1>"
    "<p>Checking if the site connection is secure. Enable JavaScript and cookies to continue.</p>"
    "</article></body></html>",  # bot-check page: a little text, but not a document
])
def test_empty_or_bot_check_source_fails_and_clears_its_cache(monkeypatch, tmp_path, html):
    monkeypatch.setattr(ingest, "read_sources", lambda: [SRC])
    monkeypatch.setattr(ingest.config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(ingest.config, "TEXT_DIR", tmp_path / "text")

    def fake_download(src):
        path = tmp_path / "raw" / f"{ingest.cache_name(src)}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return path
    monkeypatch.setattr(ingest, "download", fake_download)
    with pytest.raises(SystemExit, match="nid_faq"):
        ingest.load_documents()
    assert list(tmp_path.rglob("*.html")) == [] and list(tmp_path.rglob("*.json")) == []


def test_changed_url_gets_a_new_cache_file():
    moved = {**SRC, "url": "https://a.example/faq-v2"}
    assert ingest.cache_name(SRC) != ingest.cache_name(moved)
    assert ingest.cache_name(SRC).startswith("nid_faq-")
