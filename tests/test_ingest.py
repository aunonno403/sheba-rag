import pytest
import requests

from sheba_rag import ingest

SRC = {"id": "nid_faq", "service": "nid", "type": "html", "title": "t", "url": "https://a.example/faq", "lang": "bn"}
BOT_CHECK = "<html><body><article><p>Just a moment...</p></article></body></html>"


def fee_page(fee):
    rows = "".join(f"<p>Row {i}: the regular fee for a 48 page passport with {i} years validity is TK {fee}.</p>"
                   for i in range(1, 7))
    return f"<html><body><article><h1>e-Passport fees</h1>{rows}</article></body></html>"


@pytest.fixture
def dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest.config, "ROOT", tmp_path)
    monkeypatch.setattr(ingest.config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(ingest.config, "TEXT_DIR", tmp_path / "text")
    return tmp_path


def saved_source(tmp_path, html):
    """Write a hand-saved page and return its sources.csv row with the matching hash."""
    path = tmp_path / "manual" / "fees.html"
    path.parent.mkdir(exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return {**SRC, "file": "manual/fees.html", "saved": "2026-11-10", "sha1": ingest.content_hash(path)}


def test_failed_download_stops_before_index_rebuild(monkeypatch):
    monkeypatch.setattr(ingest, "read_sources", lambda: [SRC])

    def offline(src, refresh=False):
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
def test_empty_or_bot_check_download_fails_and_clears_its_cache(monkeypatch, dirs, html):
    monkeypatch.setattr(ingest, "read_sources", lambda: [SRC])

    def fake_download(src, refresh=False):
        path = dirs / "raw" / f"{ingest.cache_name(src)}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return path
    monkeypatch.setattr(ingest, "download", fake_download)
    with pytest.raises(SystemExit, match="nid_faq"):
        ingest.load_documents()
    assert list(dirs.rglob("*.html")) == [] and list(dirs.rglob("*.json")) == []


def test_saved_local_file_is_read_and_never_deleted(monkeypatch, dirs):
    src = saved_source(dirs, BOT_CHECK)
    saved = dirs / "manual" / "fees.html"
    assert ingest.download(src) == saved
    monkeypatch.setattr(ingest, "read_sources", lambda: [src])
    with pytest.raises(SystemExit, match="nid_faq"):  # too little text: fails...
        ingest.load_documents()
    assert saved.exists()  # ...but the hand-saved page is kept


def test_resaved_page_needs_a_new_date_and_then_gives_new_text(monkeypatch, dirs):
    src = saved_source(dirs, fee_page("4,025"))
    monkeypatch.setattr(ingest, "read_sources", lambda: [src])
    assert "4,025" in ingest.load_documents()[0][1][0]["text"]

    (dirs / "manual" / "fees.html").write_text(fee_page("5,000"), encoding="utf-8")  # fees changed, page re-saved
    with pytest.raises(requests.RequestException, match="set its `saved` date and sha1="):
        ingest.download(src)  # the old row no longer matches: the date cannot be forgotten

    src.update(saved="2027-01-05", sha1=ingest.content_hash(dirs / "manual" / "fees.html"))
    docs = ingest.load_documents()
    text = docs[0][1][0]["text"]
    assert "5,000" in text and "4,025" not in text  # not the cached old text
    assert ingest.chunk_documents(docs, "sentence")[0]["title"] == "t, saved 2027-01-05"


def test_missing_saved_file_is_a_clear_failure(dirs):
    with pytest.raises(requests.RequestException, match="saved file not found"):
        ingest.download({**SRC, "file": "manual/missing.html"})


def test_refresh_downloads_again_and_keeps_the_old_copy_on_failure(monkeypatch, dirs):
    class Reply:
        def __init__(self, body):
            self.content = body.encode()

        def raise_for_status(self):
            pass

    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: Reply(fee_page("20")))
    path = ingest.download(SRC)
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: Reply(fee_page("70")))
    assert "TK 20" in ingest.download(SRC).read_text(encoding="utf-8")  # cached copy reused
    ingest.download(SRC, refresh=True)
    assert "TK 70" in path.read_text(encoding="utf-8")
    assert "TK 70" in ingest.extract(SRC, path)[0]["text"]  # new content, so new text

    def offline(*a, **k):
        raise requests.ConnectionError("no network")
    monkeypatch.setattr(ingest.requests, "get", offline)
    with pytest.raises(requests.ConnectionError):
        ingest.download(SRC, refresh=True)
    assert "TK 70" in path.read_text(encoding="utf-8")  # the last good copy survives


def test_changed_url_gets_a_new_cache_file():
    moved = {**SRC, "url": "https://a.example/faq-v2"}
    assert ingest.cache_name(SRC) != ingest.cache_name(moved)
    assert ingest.cache_name(SRC).startswith("nid_faq-")
