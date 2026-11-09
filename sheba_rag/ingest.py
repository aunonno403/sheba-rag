"""Download sources, extract text (OCR when needed), chunk, embed, store.

    python -m sheba_rag.ingest                      # default chunker + embedding model from .env
    python -m sheba_rag.ingest --chunker all --embed BAAI/bge-m3 --embed intfloat/multilingual-e5-base
"""
import argparse
import csv
import hashlib
import json
import socket

import requests
import truststore
import urllib3.util.connection

from sheba_rag import config, store
from sheba_rag.chunking import CHUNKERS
from sheba_rag.loaders import html_pages, pdf_pages

# land.gov.bd publishes an IPv6 address that does not answer; force IPv4 for all downloads
urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET
# dip.gov.bd sends an incomplete certificate chain; the OS store (unlike certifi) can complete it
truststore.inject_into_ssl()


def read_sources():
    with open(config.SOURCES, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cache_name(src):
    # the URL hash makes a changed URL in sources.csv fetch fresh content instead of reusing the old file
    return f"{src['id']}-{hashlib.sha1(src['url'].encode()).hexdigest()[:8]}"


def download(src):
    path = config.RAW_DIR / src["service"] / f"{cache_name(src)}.{src['type']}"
    if not path.exists():
        r = requests.get(src["url"], timeout=60, headers={"User-Agent": "Mozilla/5.0 (sheba-rag)"})
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    return path


def extract(src, path):
    """Pages of text, cached in data/text/<id>.json so OCR runs once."""
    cache = config.TEXT_DIR / f"{cache_name(src)}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    pages = pdf_pages(path) if src["type"] == "pdf" else html_pages(path.read_bytes())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
    return pages


def load_documents():
    docs, failed = [], []
    for src in read_sources():
        try:
            path = download(src)
        except requests.RequestException as e:
            print(f"[fail] {src['id']}: download failed ({e})")
            failed.append(src["id"])
            continue
        print(f"[load] {src['id']}")
        pages = extract(src, path)
        if not any(p["text"].strip() for p in pages):
            print(f"[fail] {src['id']}: no usable text (cache cleared, the next run downloads it again)")
            path.unlink(missing_ok=True)  # otherwise every later run re-reads the same empty cache
            (config.TEXT_DIR / f"{cache_name(src)}.json").unlink(missing_ok=True)
            failed.append(src["id"])
            continue
        docs.append((src, pages))
    if failed:
        # building would replace the index without these documents; keep the old index instead
        raise SystemExit(f"Not rebuilding the index, these sources failed: {', '.join(failed)}")
    return docs


def chunk_documents(docs, chunker):
    chunks = []
    for src, pages in docs:
        for c in CHUNKERS[chunker](pages):
            chunks.append({**c, "doc_id": src["id"], "title": src["title"], "url": src["url"],
                           "service": src["service"], "lang": src["lang"]})
    return chunks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", default=config.CHUNKER, choices=[*CHUNKERS, "all"])
    ap.add_argument("--embed", action="append", help="embedding model (repeatable)")
    args = ap.parse_args(argv)
    chunkers = list(CHUNKERS) if args.chunker == "all" else [args.chunker]
    models = args.embed or [config.EMBED_MODEL]

    docs = load_documents()
    for chunker in chunkers:
        chunks = chunk_documents(docs, chunker)
        for model in models:
            n = store.build(chunks, chunker, model)
            print(f"[store] {store.collection_name(chunker, model)}: {n} chunks")


if __name__ == "__main__":
    main()
