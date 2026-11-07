"""Retrieve, prompt the LLM, return an answer with citations or an explicit "I don't know"."""
import re

import litellm

from sheba_rag import config, store

ABSTAIN_TOKEN = "NOT_IN_DOCS"
IDK = {
    "bn": "দুঃখিত, প্রদত্ত সরকারি নথিতে এই প্রশ্নের উত্তর পাওয়া যায়নি।",
    "en": "I don't know. The official documents I have do not cover this question.",
}
LANG_NAME = {"bn": "Bangla", "en": "English"}

PROMPT = """You answer questions about Bangladeshi public services using ONLY the numbered context blocks below.
Rules:
- Reply in {language}, even if the context is in another language.
- After every fact, cite its block number in square brackets, e.g. [1] or [2][3].
- If the context does not contain the answer, reply with exactly {abstain} and nothing else.
- Be concise: 1-4 sentences or a short list. Do not add facts that are not in the context.

Context:
{context}

Question: {question}"""


def detect_lang(text):
    return "bn" if re.search(r"[ঀ-৿]", text) else "en"


def source_label(hit):
    label = hit["title"]
    if hit.get("section"):
        label += f", § {hit['section']}"
    if hit.get("page"):
        label += f", p. {hit['page']}"
    return label


def format_context(hits):
    return "\n\n".join(f"[{i}] ({source_label(h)})\n{h['text']}" for i, h in enumerate(hits, 1))


def parse_citations(text, hits):
    """Citations the model used, ignoring numbers that match no context block."""
    nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    return [
        {"n": n, "title": hits[n - 1]["title"], "section": hits[n - 1].get("section"),
         "page": hits[n - 1].get("page"), "url": hits[n - 1]["url"], "doc_id": hits[n - 1]["doc_id"]}
        for n in nums if 1 <= n <= len(hits)
    ]


def complete(prompt, model):
    resp = litellm.completion(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0, num_retries=3, timeout=60,
    )
    return (resp.choices[0].message.content or "").strip()


def answer(question, chunker=None, embed_model=None, llm_model=None, min_score=None):
    lang = detect_lang(question)
    min_score = config.MIN_SCORE if min_score is None else min_score
    hits = store.search(question, chunker, embed_model)
    idk = {"answer": IDK[lang], "citations": [], "abstained": True, "hits": hits}
    if not hits or hits[0]["score"] < min_score:
        return idk  # nothing relevant enough: skip the LLM call entirely
    prompt = PROMPT.format(language=LANG_NAME[lang], abstain=ABSTAIN_TOKEN,
                           context=format_context(hits), question=question)
    text = complete(prompt, llm_model or config.LLM_MODEL)
    citations = parse_citations(text, hits)
    if ABSTAIN_TOKEN in text or not citations:
        return idk  # an uncited answer is treated as ungrounded
    return {"answer": text, "citations": citations, "abstained": False, "hits": hits}
