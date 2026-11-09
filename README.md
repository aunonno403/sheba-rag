# সেবা সহায়ক · Sheba RAG

A Bangla-first question-answering assistant for Bangladeshi public services (passport, NID, land),
built with retrieval-augmented generation. It answers in the question's language (Bangla or English),
cites the official document and section it used, and says "I don't know" when the documents do not cover
the question.

![stack](https://img.shields.io/badge/stack-FastAPI%20%7C%20Streamlit%20%7C%20Chroma%20%7C%20bge--m3%20%7C%20Gemini-informational)

## Architecture

```mermaid
flowchart LR
    subgraph Ingest["Ingestion (python -m sheba_rag.ingest)"]
        S[data/sources.csv] --> D[Download<br/>HTML / PDF]
        D --> X{Text layer<br/>readable?}
        X -- yes --> T[trafilatura / PyMuPDF]
        X -- "no (broken Bangla font, scan)" --> O[Tesseract OCR<br/>ben+eng]
        T --> C[Chunk<br/>sentence-aware or fixed]
        O --> C
        C --> E[Embed<br/>bge-m3, local GPU]
        E --> V[(Chroma<br/>one collection per config)]
    end
    subgraph Serve["Serving"]
        U[Streamlit UI] -->|POST /ask| A[FastAPI]
        A --> R[Retrieve top-5]
        V --> R
        R --> G{Score above<br/>guard?}
        G -- no --> IDK["I don't know"]
        G -- yes --> L[LLM via LiteLLM<br/>Gemini by default]
        L --> P[Parse citations<br/>drop invalid ones]
        P --> U
    end
```

| Part | Choice | Why |
|---|---|---|
| HTML extraction | trafilatura | Strips portal menus and footers |
| PDF extraction | PyMuPDF, with Tesseract OCR fallback | Most Bangla government PDFs have a broken text layer (see below) |
| Chunking | Own sentence-aware splitter (`।`, `?`, `!`, numbered law sections) vs fixed windows | Generic splitters ignore the danda and cut across law sections |
| Embeddings | `BAAI/bge-m3` (vs `intfloat/multilingual-e5-base`) | Best Bangla and cross-lingual retrieval in our eval |
| Vector DB | Chroma, local | No server; stores section and page metadata for citations |
| LLM | LiteLLM, default `gemini/gemini-3.5-flash-lite` | Any provider is a one-line `.env` change. Lite is the default because the free tier gives `gemini-3.5-flash` only 20 requests a day |
| API / UI | FastAPI, Streamlit | |

## What I learned about Bangladeshi government documents

- **Most Bangla PDFs extract as garbage.** Word-exported PDFs in the Nikosh font have a broken
  character map: PyMuPDF returns "সসবা প্রোন" for "সেবা প্রদান". Older SutonnyMJ PDFs give Latin
  gibberish, and some are scans with no text at all. All 8 Bangla PDFs I tested were affected.
- **A cheap detector catches it.** Correct Bangla never starts a word with a dependent vowel sign and never
  doubles one. The share of such words is 0.0 on clean text and 0.05-0.21 on broken PDFs, so pages above
  0.02 are sent to OCR (`sheba_rag/loaders.py`).
- **Tesseract page segmentation matters.** The default mode drops the spaces between Bangla words;
  `--psm 4` keeps them. OCR of tables still interleaves columns, which is a known limit.
- **Portal HTML hides content.** The NID FAQ puts every question in a bare `<strong>`, which trafilatura
  drops, leaving answers without questions. Converting those tags to paragraphs recovered all 44 questions.
- **Infrastructure quirks.** land.gov.bd advertises an IPv6 address that never answers, and dip.gov.bd (an
  earlier source) sent an incomplete TLS certificate chain. The downloader forces IPv4 and uses the OS trust
  store (`truststore`) instead of disabling verification.
- **The key passport pages block scripts.** epassport.gov.bd is a JavaScript app that refuses scripted requests,
  and fees and the latest rules live only there. Rather than work around its bot protection, the pages were
  saved by hand in a browser into `data/manual/`; `sources.csv` points at those files while citations keep the
  official URL and the date saved.
- **Sources disagree over time.** The older dip.gov.bd copy of the passport rules said applicants over 65 get a
  5-year passport and that super-urgent delivery needs a police clearance. The current epassport.gov.bd page
  cancels both. With both indexed the assistant could contradict itself, so the outdated page was removed.
- **Numbered lists are not law sections.** Treating every numbered line as a section turned the list of 70+
  passport offices into 140 chunks like "2. Jatrabari". Short sections are now packed together until a chunk
  reaches 300 characters, with range labels such as "§ 1–13".

## Corpus

| Service | Source | Lang | Type |
|---|---|---|---|
| Passport | [e-Passport fees](https://www.epassport.gov.bd/instructions/passport-fees), inside Bangladesh and at missions | en | HTML, saved by hand |
| Passport | [ই-পাসপোর্ট ফরম পূরণের নির্দেশাবলী](https://www.epassport.gov.bd/instructions/instructions) (application rules) | bn | HTML, saved by hand |
| Passport | [Documents checklist for e-Passport enrollment](https://www.epassport.gov.bd/landing/notices/160) | bn | HTML, saved by hand |
| Passport | [e-Passport required documents](https://www.epassport.gov.bd/landing/notices/34) | en, bn | HTML, saved by hand |
| Passport | [List of functional e-Passport offices](https://www.epassport.gov.bd/landing/notices/33) (last updated 2023) | en | HTML, saved by hand |
| Passport | [5 steps to your e-Passport](https://www.epassport.gov.bd/instructions/five-step-to-your-epassport) | en | HTML, saved by hand |
| Passport | [Bangladesh Passport Order 1973](http://bdlaws.minlaw.gov.bd/act-print-445.html) | en | HTML |
| NID | [NID FAQ](https://services.nidw.gov.bd/nid-pub/faq?tab=faq-general), Election Commission | bn | HTML |
| NID | [জাতীয় পরিচয় নিবন্ধন আইন, ২০২৩](http://bdlaws.minlaw.gov.bd/act-print-1458.html) | bn | HTML |
| Land | [ভূমিসেবার নির্ধারিত ফি](https://land.gov.bd/vumisheba-fee?lang=bn) | bn | HTML |
| Land | [ভূমি উন্নয়ন কর আইন, ২০২৩](http://bdlaws.minlaw.gov.bd/act-print-1449.html) | bn | HTML |
| Land | Ministry of Land citizen charter, 53 pages | bn | PDF, OCR |

The epassport.gov.bd pages were saved on 2026-11-10 and will go stale when the site changes; re-save them
and re-run ingestion.

Add a document by dropping a row into `data/sources.csv` and re-running ingestion. For a page that blocks
scripts, save it from your browser into `data/manual/` and put its path in the `file` column; the `url`
column stays the official address used in citations.

## Setup (Windows, tested; Linux/macOS similar)

Requirements: Python 3.11+, [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) with the Bengali
(`ben`) language pack, a free [Gemini API key](https://aistudio.google.com/apikey). A CUDA GPU is optional;
bge-m3 runs in fp16 on a 4 GB laptop GPU.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126   # skip on CPU-only machines
pip install -r requirements.txt
copy .env.example .env      # then put your GEMINI_API_KEY in .env

python -m sheba_rag.ingest  # downloads sources, OCRs PDFs (~2 min once), builds the index
```

Run the app in two terminals:

```powershell
uvicorn api.main:app --port 8000
streamlit run ui/app.py
```

API example:

```powershell
curl.exe -X POST localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"What is the NID helpline number?\"}"
```

Tests (no network or API key needed): `pytest`

## Evaluation

`eval/testset.jsonl` has 34 questions: 29 answerable (12 passport, 9 NID, 8 land; 8 in English, several
asking in English about Bangla-only sources) and 5 unanswerable ones, some deliberately close to the corpus
(for example the NID reissue fee, which no source states). Each answerable item has an expected answer, the
expected document, and an evidence phrase copied from the source.

```powershell
python -m sheba_rag.ingest --chunker all --embed BAAI/bge-m3 --embed intfloat/multilingual-e5-base
python -m eval.run_eval retrieval                     # all chunker x embedding pairs, no LLM calls
python -m eval.run_eval generate --config sentence:BAAI/bge-m3 --config fixed:BAAI/bge-m3
```

Metrics:
- **hit@k / MRR**: the expected document is among the top-k chunks.
- **evidence@5**: the exact answering sentence is inside one of the top-5 chunks. This is stricter, and
  it is the metric that separates chunkers.
- **correctness**: an LLM judge (`gemini-3.1-flash-lite`, a different model from the answerer) checks the
  answer against the expected answer.
- **groundedness**: the same judge checks every claim is supported by the retrieved context.
- **abstain on unanswerable / false "I don't know"**: how often the system declines when it should, and
  when it should not.

### Results

Retrieval, 29 answerable questions, top-5:

| chunker | embedding | hit@1 | hit@5 | MRR | evidence@5 |
|---|---|---|---|---|---|
| sentence | bge-m3 | 0.90 | 1.00 | 0.94 | **1.00** |
| fixed | bge-m3 | 0.93 | 1.00 | 0.95 | 0.97 |
| sentence | multilingual-e5-base | 0.83 | 0.90 | 0.84 | 0.86 |
| fixed | multilingual-e5-base | 0.72 | 0.90 | 0.80 | 0.83 |

Answers, all 34 questions, LLM `gemini/gemini-3.5-flash-lite`, judge `gemini/gemini-3.1-flash-lite`:

*Pending: being re-run after the corpus change of 2026-11-10 (new epassport.gov.bd pages, list packing).
The previous run on the 30-question set scored 0.96-1.00 on every answer metric; see `eval/results.md`.
The findings below still refer to that run until it is redone.*

Findings:
- **The embedding model matters most.** bge-m3 beats multilingual-e5-base by 8-12 points on every metric,
  and it handles English questions about Bangla documents well.
- **Sentence-aware chunking helps a little.** At document level the chunkers tie. With the stricter
  evidence check, fixed windows lose an answer that was cut across a window boundary.
- **The answer metrics saturate, and the judge is lenient.** Both setups score at or near 1.00, so this test
  set cannot separate them at the answer stage. Reading the answers shows the difference the judge missed:
  for "when is land development tax due without penalty?", the fixed-chunk answer starts mid-word
  ("ত বৎসরের ৩০ জুন...") and describes when tax becomes overdue instead of stating the 30 June deadline,
  yet it was graded correct. A harder test set and a stricter judge are needed to compare answer quality.
- **Abstention works, with one flaky refusal.** All 5 unanswerable questions got "I don't know", including two
  that sit close to the corpus (the e-passport fee and the NID reissue fee). One answerable question, the
  English one about a certified khatian copy, was refused in the latest sentence-chunk run although the same
  question was answered correctly in an earlier run and in two manual reruns. It has the weakest retrieval
  score in the set (0.43), so the model's answer flips between runs. This is the run-to-run noise the
  30-question set cannot average out; the recorded result is kept as measured.
- **A similarity cutoff cannot replace the LLM's judgement.** With bge-m3, the lowest-scoring answerable
  question (an English question about a Bangla fee table) scored 0.43, below three unanswerable questions
  (0.51-0.62). The cutoff is therefore only a guard at 0.40 against clearly off-topic questions; the prompt's
  abstain instruction and the "no citation means no answer" rule do the real work.
- **Run-to-run noise was a bug.** Chroma's default `ef_search=10` made top-5 results change between identical
  runs. Raising it to 200, effectively exact search at this size, made results reproducible.

Caveats: 30 questions is a small test set, so one question moves a score by 3-4 points. The answer run used
the free-tier `gemini-3.5-flash-lite`; `gemini-3.5-flash` was not evaluated because of its 20-requests-a-day cap. I drafted the
expected answers from the sources; they should be reviewed by someone who uses these services. An LLM judge
can be wrong; per-question verdicts are in `eval/results.jsonl` for inspection.

## Project layout

```
sheba_rag/   config.py  loaders.py  chunking.py  store.py  ingest.py  answer.py
api/main.py  FastAPI: POST /ask, GET /health
ui/app.py    Streamlit chat UI
eval/        testset.jsonl  run_eval.py  results.md
tests/       unit tests (chunking, loaders, answer logic, API)
data/        sources.csv, manual/ (hand-saved pages); downloads, extracted text and the index are generated
```

## Limitations and next steps

- The corpus is small, and the hand-saved epassport.gov.bd pages have to be re-saved by hand when the site
  changes. Step-by-step mutation instructions from upazila land offices are still missing.
- Table-aware OCR would make the citizen charter's fee and timeline tables much cleaner.
- Hybrid search (BM25 plus vectors) would likely help exact terms such as fee names and section numbers.
- Gemini's free tier is tight: 5 requests per minute per model, and a daily cap that was 20 requests for
  `gemini-3.5-flash`. The API returns a clear 429 message when a limit is hit. The eval script paces calls,
  retries transient errors, and resumes from `eval/results.jsonl`, so a run interrupted by a quota costs no
  repeated calls.

This is a portfolio project, not legal advice. Always confirm with the relevant office.
