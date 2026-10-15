# Evaluation results: naive RAG vs. agentic RAG

- Run: `20261015T204556Z` | dataset items: 43 (9 unanswerable/out-of-scope)
- LLM: `haiku` (claude_cli) | judge: `haiku` (temperature 0, prompt v1) | embeddings: `intfloat/multilingual-e5-small` | k=6
- **The dataset was drafted by an LLM-assisted process and needs human review before these numbers are trusted.** Judge and generator may share a model family.

| Metric | baseline | agentic [MIN_RELEVANT_DOCS=1] | agentic [MIN_RELEVANT_DOCS=2] |
|---|---|---|---|
| Retrieval hit rate@1 | 64.7% | 64.7% | 67.6% |
| Retrieval hit rate@3 | 97.1% | 97.1% | 100.0% |
| Retrieval hit rate@k | 100.0% | 100.0% | 100.0% |
| Answer correctness (LLM judge) | 97.1% | 94.1% | 55.9% |
| Groundedness (LLM judge) | 88.2% | 96.9% | 94.7% |
| Citation accuracy | 87.0% | 85.8% | 51.0% |
| Correct refusal rate (unanswerable) | 88.9% | 100.0% | 100.0% |
| False refusal rate (answerable) | 2.9% | 5.9% | 44.1% |
| Avg latency (s) | 5.76 | 35.93 | 65.45 |
| Avg LLM calls / question | 1.00 | 10.49 | 14.98 |
| Errors | 0 | 0 | 0 |

## Answer correctness by question category

| Category | baseline | agentic [MIN_RELEVANT_DOCS=1] | agentic [MIN_RELEVANT_DOCS=2] |
|---|---|---|---|
| cross_lingual | 100.0% | 50.0% | 0.0% |
| lookup | 100.0% | 100.0% | 56.2% |
| multi_chunk | 100.0% | 100.0% | 75.0% |
| vocab_mismatch | 87.5% | 87.5% | 50.0% |

Definitions: hit rate@k = an expected (file, page/section) is among the final retrieved chunks (hit@1 / hit@3: within the first 1 / 3 chunks in retrieval order); citation accuracy = mean share of returned citations that match an expected source (0 when the answer has none); groundedness = share of answered questions whose answer an independent judge finds fully supported by the chunks used; refusal = the system declined or routed out of scope.
