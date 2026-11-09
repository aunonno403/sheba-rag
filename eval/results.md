# Evaluation results

Appended by `python -m eval.run_eval`. Test set: `eval/testset.jsonl` (30 questions).

## retrieval, 2026-11-07, top-k 5, MIN_SCORE 0.4

| chunker | embedding | hit@1 | hit@5 | MRR | evidence@5 |
|---|---|---|---|---|---|
| sentence | bge-m3 | 0.96 | 1.00 | 0.98 | 1.00 |
| sentence | multilingual-e5-base | 0.88 | 0.92 | 0.89 | 0.88 |
| fixed | bge-m3 | 0.96 | 1.00 | 0.97 | 0.96 |
| fixed | multilingual-e5-base | 0.84 | 0.92 | 0.88 | 0.84 |

## generate, 2026-11-09, top-k 5, MIN_SCORE 0.4

| chunker | embedding | LLM | correctness | groundedness | abstain on unanswerable | false 'I don't know' |
|---|---|---|---|---|---|---|
| sentence | bge-m3 | gemini-3.5-flash-lite | 1.00 | 1.00 | 1.00 | 0.00 |
| fixed | bge-m3 | gemini-3.5-flash-lite | 1.00 | 1.00 | 1.00 | 0.00 |

## retrieval, 2026-11-09, top-k 5, MIN_SCORE 0.4

| chunker | embedding | hit@1 | hit@5 | MRR | evidence@5 |
|---|---|---|---|---|---|
| sentence | bge-m3 | 0.96 | 1.00 | 0.98 | 1.00 |
| sentence | multilingual-e5-base | 0.88 | 0.92 | 0.89 | 0.88 |
| fixed | bge-m3 | 0.96 | 1.00 | 0.97 | 0.96 |
| fixed | multilingual-e5-base | 0.84 | 0.92 | 0.88 | 0.84 |

## generate, 2026-11-09, top-k 5, MIN_SCORE 0.4

| chunker | embedding | LLM | correctness | groundedness | abstain on unanswerable | false 'I don't know' |
|---|---|---|---|---|---|---|
| sentence | bge-m3 | gemini-3.5-flash-lite | 0.96 | 1.00 | 1.00 | 0.04 |
| fixed | bge-m3 | gemini-3.5-flash-lite | 1.00 | 1.00 | 1.00 | 0.00 |

## generate, 2026-11-09, top-k 5, MIN_SCORE 0.4

| chunker | embedding | LLM | correctness | groundedness | abstain on unanswerable | false 'I don't know' |
|---|---|---|---|---|---|---|
| sentence | bge-m3 | gemini-3.5-flash-lite | 0.96 | 1.00 | 1.00 | 0.04 |
| fixed | bge-m3 | gemini-3.5-flash-lite | 1.00 | 1.00 | 1.00 | 0.00 |

## retrieval, 2026-11-10, top-k 5, MIN_SCORE 0.4

| chunker | embedding | hit@1 | hit@5 | MRR | evidence@5 |
|---|---|---|---|---|---|
| sentence | bge-m3 | 0.90 | 1.00 | 0.94 | 1.00 |
| sentence | multilingual-e5-base | 0.83 | 0.90 | 0.84 | 0.86 |
| fixed | bge-m3 | 0.93 | 1.00 | 0.95 | 0.97 |
| fixed | multilingual-e5-base | 0.72 | 0.90 | 0.80 | 0.83 |
