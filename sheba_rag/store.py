"""Embeddings + Chroma. One collection per (chunker, embedding model) so configs can be compared."""
import hashlib
import json
import re
from functools import lru_cache

import chromadb
import torch
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

from sheba_rag import config


@lru_cache
def _model(name):
    cuda = torch.cuda.is_available()
    m = SentenceTransformer(
        name, device="cuda" if cuda else "cpu",
        model_kwargs={"torch_dtype": torch.float16} if cuda else {},
    )
    m.max_seq_length = 512  # chunks are <=1000 chars (~400 tokens); bge-m3 default 8192 wastes VRAM
    return m


def embed(model_name, texts, kind):
    """kind is 'query' or 'passage'; E5 models need that prefix, bge-m3 does not."""
    prefix = f"{kind}: " if "e5" in model_name.lower() else ""
    return _model(model_name).encode(
        [prefix + t for t in texts], normalize_embeddings=True, batch_size=16,
        show_progress_bar=len(texts) > 100,
    ).tolist()


def collection_name(chunker, model_name):
    return re.sub(r"[^A-Za-z0-9_-]", "-", f"{chunker}__{model_name.split('/')[-1]}")


@lru_cache
def _client():
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def build(chunks, chunker, model_name):
    """Replace the collection for this config with `chunks` (dicts with text + metadata)."""
    name = collection_name(chunker, model_name)
    # embed before deleting anything: if this crashes (e.g. GPU out of memory), the old index stays usable.
    # The doc title is embedded with each chunk so "NID law" style queries find the right document.
    vectors = embed(model_name, [f"{c['title']}\n{c['text']}" for c in chunks], "passage")
    if name in [c.name for c in _client().list_collections()]:
        _client().delete_collection(name)
    # default ef_search=10 made top-5 results vary between runs; at this corpus size 200 is effectively exact
    col = _client().create_collection(
        name, configuration={"hnsw": {"space": "cosine", "ef_search": 200, "ef_construction": 200}},
        embedding_function=None,
    )
    metas = [{k: v for k, v in c.items() if k != "text" and v is not None} for c in chunks]
    for i in range(0, len(chunks), 1000):  # Chroma caps batch size
        col.add(
            ids=[f"{c['doc_id']}-{j}" for j, c in enumerate(chunks[i:i + 1000], i)],
            documents=[c["text"] for c in chunks[i:i + 1000]],
            embeddings=vectors[i:i + 1000],
            metadatas=metas[i:i + 1000],
        )
    return col.count()


def _collection(chunker, model_name):
    return _client().get_collection(collection_name(chunker or config.CHUNKER, model_name or config.EMBED_MODEL))


def index_size(chunker=None, model_name=None):
    """Number of chunks in the index for this config, 0 if ingestion has not been run."""
    try:
        return _collection(chunker, model_name).count()
    except NotFoundError:
        return 0


def index_fingerprint(chunker, model_name):
    """Changes whenever the indexed chunks change (re-chunking, re-ingesting changed sources)."""
    data = _collection(chunker, model_name).get(include=["documents", "metadatas"])
    rows = sorted(zip(data["ids"], data["documents"], (json.dumps(m, sort_keys=True) for m in data["metadatas"])))
    return hashlib.sha1(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()[:8]


def search(query, chunker=None, model_name=None, k=None):
    model_name = model_name or config.EMBED_MODEL
    res = _collection(chunker, model_name).query(
        query_embeddings=embed(model_name, [query], "query"), n_results=k or config.TOP_K)
    return [
        {"text": doc, "score": 1 - dist, **meta}
        for doc, dist, meta in zip(res["documents"][0], res["distances"][0], res["metadatas"][0])
    ]
