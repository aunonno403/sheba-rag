"""Evaluate retrieval and answers on eval/testset.jsonl.

    python -m eval.run_eval retrieval                                   # every chunker x model, no LLM calls
    python -m eval.run_eval retrieval --models BAAI/bge-m3
    python -m eval.run_eval generate --config sentence:BAAI/bge-m3 --config fixed:BAAI/bge-m3

Collections must exist first: python -m sheba_rag.ingest --chunker all --embed <model> ...
"""
import argparse
import hashlib
import json
import re
import time
import unicodedata
from datetime import date
from pathlib import Path

import litellm

from sheba_rag import answer as ans
from sheba_rag import config, store
from sheba_rag.chunking import CHUNKERS

HERE = Path(__file__).parent
DEFAULT_MODELS = ["BAAI/bge-m3", "intfloat/multilingual-e5-base"]

JUDGE_PROMPT = """You grade a question-answering system about Bangladeshi public services.

Question: {question}
Reference answer: {expected}
Retrieved context:
{context}

System answer: {answer}

Return only JSON: {{"correct": 0 or 1, "grounded": 0 or 1}}
- correct = 1 if the system answer states the key facts of the reference answer without contradicting it (wording and language may differ; extra correct detail is fine).
- grounded = 1 if every factual claim in the system answer is supported by the retrieved context."""


def load_testset():
    return [json.loads(line) for line in (HERE / "testset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text))


def retrieval(models):
    items, rows = load_testset(), []
    for chunker in CHUNKERS:
        for model in models:
            rr, ev_hits = [], []
            for it in items:
                if not it["answerable"]:
                    continue
                hits = store.search(it["question"], chunker, model, k=5)
                rank = next((i for i, h in enumerate(hits, 1) if h["doc_id"] == it["expected_doc"]), None)
                rr.append(1 / rank if rank else 0.0)
                # stricter than doc match: the answering sentence itself must be inside a retrieved chunk
                ev_hits.append(any(norm(it["evidence"]) in norm(h["text"]) for h in hits))
            rows.append({
                "chunker": chunker, "embedding": model.split("/")[-1],
                "hit@1": f"{sum(r == 1 for r in rr) / len(rr):.2f}",
                "hit@5": f"{sum(r > 0 for r in rr) / len(rr):.2f}",
                "MRR": f"{sum(rr) / len(rr):.2f}",
                "evidence@5": f"{sum(ev_hits) / len(ev_hits):.2f}",
            })
    return table(rows, list(rows[0]))


def judge(it, result):
    context = ans.format_context(result["hits"])
    raw = ans.complete(JUDGE_PROMPT.format(question=it["question"], expected=it["expected_answer"],
                                           context=context, answer=result["answer"]), config.JUDGE_MODEL)
    try:
        # raw_decode reads one JSON object from the first "{" and ignores any text after it
        verdict, _ = json.JSONDecoder().raw_decode(raw[raw.index("{"):])
        return int(verdict["correct"]), int(verdict["grounded"])
    except (ValueError, KeyError, TypeError) as e:
        # never save a made-up 0: stop here, and a resumed run asks the judge again
        raise ValueError(f"unreadable judge reply for {it['id']}: {raw[:200]!r}") from e


TRANSIENT = (litellm.InternalServerError, litellm.ServiceUnavailableError, litellm.Timeout,
             litellm.APIConnectionError)


def paced(complete, rpm):
    """Wrap ans.complete: at most `rpm` calls a minute per model (Gemini free tier: 5), and wait out
    rate limits and transient server errors instead of crashing a long eval run."""
    last = {}

    def wrapper(prompt, model):
        for attempt in range(5):
            wait = last.get(model, float("-inf")) + 60 / rpm - time.time()
            if wait > 0:
                time.sleep(wait)
            last[model] = time.time()
            try:
                return complete(prompt, model)
            except (litellm.RateLimitError, *TRANSIENT) as e:
                pause = 60 if isinstance(e, litellm.RateLimitError) else 20
                print(f"  {type(e).__name__} on {model}, waiting {pause}s (attempt {attempt + 1}/5)")
                time.sleep(pause)
        raise RuntimeError(f"{model} still failing after 5 attempts")
    return wrapper


def setup_fingerprint(chunker, model):
    """Changes when a prompt, a retrieval setting or the indexed chunks change, so saved answers made
    under old settings or an old index are not reused."""
    text = (ans.PROMPT + JUDGE_PROMPT + f"top_k={config.TOP_K} min_score={config.MIN_SCORE}"
            + store.index_fingerprint(chunker, model))
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def generate(configs, rpm):
    ans.complete = paced(ans.complete, rpm)  # answer() and judge() both look this up at call time
    items, table_rows = load_testset(), []
    log_path = HERE / "results.jsonl"
    # resume: reuse finished items (free-tier daily quotas are tiny), but only if the question, the expected
    # answer, the prompts/settings and the index are unchanged, so any edit re-runs exactly what it affects
    key = lambda row: (row["config"], row["llm"], row["judge"], row["setup"], row["id"], row["question"],
                       row["expected"])
    done = {}
    if log_path.exists():
        done = {key(r): r for r in map(json.loads, log_path.read_text(encoding="utf-8").splitlines())
                if "setup" in r}
    with open(log_path, "a", encoding="utf-8") as log_file:
        for cfg in configs:
            chunker, model = cfg.split(":", 1)
            setup = setup_fingerprint(chunker, model)
            rows = []
            for it in items:
                row = done.get((cfg, config.LLM_MODEL, config.JUDGE_MODEL, setup, it["id"], it["question"],
                                it["expected_answer"]))
                if row is None:
                    r = ans.answer(it["question"], chunker, model)
                    c, g = judge(it, r) if not r["abstained"] else (None, None)
                    row = {"config": cfg, "llm": config.LLM_MODEL, "judge": config.JUDGE_MODEL, "setup": setup,
                           "id": it["id"],
                           "question": it["question"], "expected": it["expected_answer"],
                           "answer": r["answer"], "abstained": r["abstained"],
                           "citations": [x["doc_id"] for x in r["citations"]], "correct": c, "grounded": g}
                    log_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    log_file.flush()  # keep finished items if a later call fails
                print(f"{cfg} {it['id']}: abstained={row['abstained']} correct={row['correct']} grounded={row['grounded']}")
                rows.append({**row, "answerable": it["answerable"]})
            table_rows.append(summarize(rows, chunker, model))
    return table(table_rows, list(table_rows[0]))


def summarize(rows, chunker, model):
    ans_rows = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if not r["abstained"]]
    return {
        "chunker": chunker, "embedding": model.split("/")[-1], "LLM": config.LLM_MODEL.split("/")[-1],
        "correctness": f"{sum(r['correct'] or 0 for r in ans_rows) / len(ans_rows):.2f}",
        "groundedness": f"{sum(r['grounded'] for r in answered) / len(answered):.2f}" if answered else "-",
        "abstain on unanswerable": f"{sum(r['abstained'] for r in unans) / len(unans):.2f}",
        "false 'I don't know'": f"{sum(r['abstained'] for r in ans_rows) / len(ans_rows):.2f}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["retrieval", "generate"])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--config", action="append", help="chunker:embedding_model (generate mode)")
    ap.add_argument("--rpm", type=float, default=5, help="max LLM calls per minute per model (generate mode)")
    args = ap.parse_args()
    md = retrieval(args.models) if args.mode == "retrieval" else generate(args.config or [f"{config.CHUNKER}:{config.EMBED_MODEL}"], args.rpm)
    print("\n" + md)
    with open(HERE / "results.md", "a", encoding="utf-8") as f:
        f.write(f"\n## {args.mode}, {date.today()}, top-k {config.TOP_K}, MIN_SCORE {config.MIN_SCORE}\n\n{md}\n")


if __name__ == "__main__":
    main()
