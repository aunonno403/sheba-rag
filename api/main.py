"""FastAPI backend.  Run: uvicorn api.main:app --reload"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

import litellm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints

from sheba_rag import answer as ans
from sheba_rag import config, store

log = logging.getLogger("sheba_rag.api")


@asynccontextmanager
async def lifespan(app):
    store.embed(config.EMBED_MODEL, ["warmup"], "query")  # load the model now, not on the first question (~20 s)
    yield


app = FastAPI(title="Sheba RAG", description="Bangla Q&A over Bangladeshi public-service documents",
              lifespan=lifespan)


class AskRequest(BaseModel):
    # trimmed before the length check, so "   " is rejected instead of costing an LLM call
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=1000)]


@app.get("/health")
def health():
    chunks = store.index_size()
    return {"status": "ok" if chunks else "no index: run python -m sheba_rag.ingest", "chunks": chunks,
            "llm": config.LLM_MODEL, "embedding": config.EMBED_MODEL, "chunker": config.CHUNKER}


@app.post("/ask")
def ask(req: AskRequest):
    if config.LLM_MODEL.startswith("gemini/") and not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(503, "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    try:
        r = ans.answer(req.question)
    except litellm.RateLimitError as e:
        raise HTTPException(429, "The LLM rate limit was reached (free tier). Please try again in a minute.") from e
    except Exception as e:  # LLM/provider or vector-store failure: report it instead of a bare 500
        log.exception("answer failed")
        raise HTTPException(503, f"Could not answer right now: {type(e).__name__}: {e}") from e
    return {"answer": r["answer"], "citations": r["citations"], "abstained": r["abstained"]}
