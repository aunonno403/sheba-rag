# Sheba RAG: Design Spec

Date: 2026-11-07
Status: implemented (see "Implementation notes" at the end for deviations)

## 1. Goal

A portfolio-grade question-answering assistant for Bangladeshi public services
(passport, NID, land). It answers in the user's language (Bangla or English),
cites the source document and section, and says "I don't know" when the
documents do not cover the question. An evaluation script makes chunking and
embedding choices comparable with numbers.

Success criteria:
- `python -m sheba_rag.ingest` builds the vector store from a clean clone.
- The Streamlit UI answers Bangla and English questions with citations.
- `python eval/run_eval.py` prints a results table comparing at least two
  chunkers and two embedding models.
- The README shows setup steps, an architecture diagram, and that table.

Constraints: Windows laptop, Python 3.13, RTX 3050 with 4 GB VRAM, Gemini free
tier only (low requests per minute), no paid services.

## 2. Corpus (findings from source research)

Researched on 2026-11-07. Key findings:
- **Bangla government PDFs mostly extract as broken text.** Word-exported PDFs
  in the Nikosh font produce wrong characters ("সেবা প্রদান" becomes
  "সসবা প্রোন"). Older SutonnyMJ PDFs give Latin garbage. Some are scans
  with no text layer. This affected all 8 Bangla PDFs tested.
- HTML pages on the national portal, the NID portal, land.gov.bd and
  bdlaws.minlaw.gov.bd give clean Unicode text.
- epassport.gov.bd is a JavaScript app that blocks scripted downloads, so its
  fee and instruction pages are not usable.
- land.gov.bd only answers over IPv4 from this machine; the downloader forces IPv4.

Initial sources (listed in `data/sources.csv`, fetched by a download script,
not committed):

| id | Service | Lang | Type | Source |
|---|---|---|---|---|
| passport_steps | Passport | bn | HTML | dip.gov.bd "পাসপোর্ট পেতে করণীয়" |
| passport_order_1973 | Passport | en | HTML | bdlaws act-print-445 (Bangladesh Passport Order 1973) |
| nid_faq | NID | bn | HTML | services.nidw.gov.bd/nid-pub/faq |
| nid_act_2023 | NID | bn | HTML | bdlaws act-print-1458 (জাতীয় পরিচয় নিবন্ধন আইন, ২০২৩) |
| land_fees | Land | bn | HTML | land.gov.bd/vumisheba-fee (mutation and record fees) |
| land_tax_act_2023 | Land | bn | HTML | bdlaws act-print-1449 (ভূমি উন্নয়ন কর আইন, ২০২৩) |
| land_charter | Land | bn | PDF | Ministry of Land citizen charter (Nikosh, broken text) |

`land_charter` exercises the PDF path: its text layer fails the quality gate,
so its pages are OCR'd with Tesseract (`ben+eng`, page segmentation mode 4,
300 dpi), which gives correct words with spacing. Table rows still interleave
columns; that is a known limit. Extracted text is cached in `data/text/` so
OCR runs once. Users can drop more PDFs or HTML files
into `data/raw/<service>/` and add a row to the CSV.

## 3. Architecture

```
sources.csv ──download──▶ data/raw/ ──ingest──▶ clean text ─▶ chunks ─▶ embeddings ─▶ Chroma
                                                                                       │
question ─▶ FastAPI /ask ─▶ embed query ─▶ top-k chunks ─▶ prompt + Gemini (LiteLLM) ─┘
                │                                              │
         Streamlit UI ◀──────── answer + citations ◀──────────┘
```

Units, each one file with one job:

| File | Job | Depends on |
|---|---|---|
| `sheba_rag/config.py` | Settings from `.env` (model names, chunk size, top-k, paths) | nothing |
| `sheba_rag/loaders.py` | HTML to text (trafilatura), PDF to text per page (PyMuPDF), Bangla quality gate | trafilatura, pymupdf |
| `sheba_rag/chunking.py` | Two chunkers plus section detection | nothing |
| `sheba_rag/store.py` | Embed and write/read Chroma; one collection per (chunker, embedder) | sentence-transformers, chromadb |
| `sheba_rag/ingest.py` | CLI: download if missing, load, gate, chunk, store | the above |
| `sheba_rag/answer.py` | Retrieve, build prompt, call LLM, parse citations, abstain | store, litellm |
| `api/main.py` | FastAPI `POST /ask`, `GET /health` | answer |
| `ui/app.py` | Streamlit chat calling the API | requests |
| `eval/run_eval.py` | Run test set over configs, write results table | answer, store |

## 4. Key design decisions

**Quality gate.** Correct Bangla words never start with a dependent vowel sign.
The gate computes the share of Bangla words that start with one or contain
doubled vowel signs. Clean sources scored 0.0 and broken PDFs scored 0.05 to
0.21, so the threshold is 0.02. Pages with no extractable text are also skipped.
Skipped files are logged, never silently ingested.

**Chunking.**
- `sentence` (default): split on Bangla and English sentence ends (`।`, `?`,
  `!`, `.` followed by space or line end) and on numbered law sections, then
  pack sentences up to about 400 tokens with one-sentence overlap. A chunk
  never crosses a section boundary.
- `fixed` (baseline): fixed 1,000-character windows with 150 overlap.
- Section detection: a line starting with a section number (`১।`, `১০।`,
  `1.`) or a chapter heading (`অধ্যায়`, `CHAPTER`) starts a new section.
  The section label is stored as metadata for citations, for example
  "ধারা ১৪" or "Article 7". HTML headings and PDF page numbers are used when
  no numbering exists.

**Embeddings.** `BAAI/bge-m3` (default) and `intfloat/multilingual-e5-base`
(comparison), run locally through sentence-transformers on the GPU in fp16,
falling back to CPU. E5 needs `query: ` and `passage: ` prefixes, handled in
`store.py`.

**LLM.** LiteLLM with the model name from `.env`, default
`gemini/gemini-2.5-flash`. The exact free-tier model is confirmed with a live
call during setup. Swapping provider is a one-line `.env` change. Calls use
LiteLLM retries with backoff for free-tier 429 errors.

**Answering and abstaining.**
- Retrieve top 5 chunks. If the best similarity is below a threshold tuned on
  the test set, return "I don't know" without calling the LLM.
- Otherwise the prompt gives numbered context blocks with source metadata and
  instructs: answer only from context, cite blocks as [1], [2], reply in the
  question's language, and reply with a fixed abstain token if unsupported.
- Language is detected by script: any Bangla characters means Bangla.
- The API returns `{answer, citations: [{doc_title, section, url, page}],
  abstained}`. Citation numbers that do not match a context block are dropped.

**Evaluation.**
- Test set: 30 items in `eval/testset.jsonl`, drafted from the corpus: 10
  passport, 10 NID, 10 land; about 6 in English; 5 unanswerable. Fields:
  `id, question, lang, expected_answer, expected_doc, answerable`.
  The user reviews and corrects expected answers before results are published.
- Retrieval metrics (no LLM, cheap): hit@5 and MRR on `expected_doc`. Run for
  every (chunker, embedder) pair.
- Generation metrics (LLM calls): correctness (judge compares answer to
  expected answer, 0/1), groundedness (judge checks every claim is supported
  by the retrieved context, 0/1), abstention accuracy on unanswerable items.
  Run for the best retrieval config and the baseline only, to stay within
  free-tier quota. The judge uses a cheaper Gemini model with a fixed prompt.
- Output: Markdown table to stdout and `eval/results.md`, raw rows to
  `eval/results.jsonl`.

## 5. Error handling

- Download failures: log and continue with files already on disk.
- Quality gate failures: log file and score, skip.
- LLM errors after retries: API returns 503 with a clear message; the UI shows it.
- Missing `GEMINI_API_KEY`: API starts, `/ask` returns a setup message.
- Windows console: scripts force UTF-8 output so Bangla prints correctly.

## 6. Testing

pytest, no network or API key needed:
- `chunking`: Bangla sentence split on `।`, section boundaries respected,
  overlap correct, fixed chunker sizes.
- `loaders`: quality gate scores clean vs broken sample strings.
- `answer`: abstain below threshold, citation parsing drops invalid numbers,
  language detection. The LLM call is replaced with a fake.
- An end-to-end smoke test ingests one small HTML fixture into a temp Chroma.

## 7. Repository layout

```
sheba-rag/
├── data/sources.csv
├── data/raw/            # downloaded, gitignored
├── data/chroma/         # gitignored
├── sheba_rag/           # config, loaders, chunking, store, ingest, answer
├── api/main.py
├── ui/app.py
├── eval/testset.jsonl, run_eval.py, results.md
├── tests/
├── docs/
├── requirements.txt, .env.example, .gitignore, README.md
```

## 8. Out of scope (v1)

Nikosh/SutonnyMJ character repair (OCR replaces it), table-aware OCR, hybrid
BM25 search, reranking, streaming, chat memory, auth, Docker, deployment.

## 9. Build order

1. Scaffold, config, download script.
2. Loaders and quality gate, with tests.
3. Chunkers, with tests.
4. Store and ingest; check retrieval by hand on a few questions.
5. Test set and retrieval-only eval across configs.
6. Answer module with Gemini, abstain, citations.
7. FastAPI and Streamlit.
8. Generation eval, README with diagram and results.

## 10. Implementation notes (2026-11-07)

- LLM default is `gemini/gemini-3.5-flash-lite`, judge `gemini/gemini-3.1-flash-lite`. `gemini-3.5-flash` allows only
  20 free requests a day (too few for one eval run), `gemini-2.5-flash` is closed
  to new keys and `gemini-3.8-flash` was overloaded (503s and timeouts) during development.
- The free tier allows 5 requests per minute per model, so `eval/run_eval.py` paces calls (`--rpm`) and waits
  60 s on a 429. The API fails fast with a 429 and a readable message instead.
- The similarity guard (`MIN_SCORE`) is 0.40, not tuned on the test set: answerable and unanswerable top-1
  scores overlap, so abstention relies on the prompt and on rejecting uncited answers.
- Retrieval eval gained `evidence@5` (exact answering phrase inside a retrieved chunk); document-level hit rates
  could not separate the chunkers on a 7-document corpus.
- Chroma `ef_search` raised to 200; the default of 10 made top-5 results vary between identical runs.
- Sentence chunks are capped at 1,000 characters (not ~400 tokens) so both chunkers compare at equal size.
- Downloads force IPv4 (land.gov.bd) and use the OS trust store via `truststore` (dip.gov.bd's incomplete chain).
