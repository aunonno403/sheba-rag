"""Embeddings + Chroma. One collection per (chunker, embedding model) so configs can be compared."""
import re
from functools import lru_cache

import chromadb
import torch
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
    if name in [c.name for c in _client().list_collections()]:
        _client().delete_collection(name)
    # default ef_search=10 made top-5 results vary between runs; at this corpus size 200 is effectively exact
    col = _client().create_collection(
        name, configuration={"hnsw": {"space": "cosine", "ef_search": 200, "ef_construction": 200}},
        embedding_function=None,
    )
    # the doc title is embedded with each chunk so "NID law" style queries find the right document
    vectors = embed(model_name, [f"{c['title']}\n{c['text']}" for c in chunks], "passage")
    metas = [{k: v for k, v in c.items() if k != "text" and v is not None} for c in chunks]
    for i in range(0, len(chunks), 1000):  # Chroma caps batch size
        col.add(
            ids=[f"{c['doc_id']}-{j}" for j, c in enumerate(chunks[i:i + 1000], i)],
            documents=[c["text"] for c in chunks[i:i + 1000]],
            embeddings=vectors[i:i + 1000],
            metadatas=metas[i:i + 1000],
        )
    return col.count()


def search(query, chunker=None, model_name=None, k=None):
    chunker, model_name, k = chunker or config.CHUNKER, model_name or config.EMBED_MODEL, k or config.TOP_K
    col = _client().get_collection(collection_name(chunker, model_name))
    res = col.query(query_embeddings=embed(model_name, [query], "query"), n_results=k)
    return [
        {"text": doc, "score": 1 - dist, **meta}
        for doc, dist, meta in zip(res["documents"][0], res["distances"][0], res["metadatas"][0])
    ]
