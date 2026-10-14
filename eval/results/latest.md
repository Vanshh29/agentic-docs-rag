# Evaluation results: naive RAG vs. agentic RAG

- Run: `20261010T132704Z` | dataset items: 43 (9 unanswerable/out-of-scope)
- LLM: `haiku` (claude_cli) | judge: `haiku` (temperature 0, prompt v1) | embeddings: `intfloat/multilingual-e5-small` | k=6
- **The dataset was drafted by an LLM-assisted process and needs human review before these numbers are trusted.** Judge and generator may share a model family.

| Metric | baseline | agentic [MIN_RELEVANT_DOCS=2] | agentic [MIN_RELEVANT_DOCS=1] |
|---|---|---|---|
| Retrieval hit rate@k | 100.0% | 100.0% | 100.0% |
| Answer correctness (LLM judge) | 97.1% | 38.2% | 97.1% |
| Groundedness (LLM judge) | 81.8% | 92.3% | 97.0% |
| Citation accuracy | 77.5% | 27.5% | 85.8% |
| Correct refusal rate (unanswerable) | 100.0% | 100.0% | 100.0% |
| False refusal rate (answerable) | 2.9% | 61.8% | 2.9% |
| Avg latency (s) | 5.23 | 67.08 | 36.65 |
| Avg LLM calls / question | 1.00 | 16.02 | 10.19 |
| Errors | 0 | 0 | 0 |

## Answer correctness by question category

| Category | baseline | agentic [MIN_RELEVANT_DOCS=2] | agentic [MIN_RELEVANT_DOCS=1] |
|---|---|---|---|
| cross_lingual | 100.0% | 0.0% | 100.0% |
| lookup | 100.0% | 37.5% | 100.0% |
| multi_chunk | 100.0% | 37.5% | 100.0% |
| vocab_mismatch | 87.5% | 50.0% | 87.5% |

Definitions: hit rate = an expected (file, page/section) is among the final retrieved chunks; citation accuracy = mean share of returned citations that match an expected source (0 when the answer has none); groundedness = share of answered questions whose answer an independent judge finds fully supported by the chunks used; refusal = the system declined or routed out of scope.
