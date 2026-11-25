# R3 — Small embeddings and rerankers for local memory retrieval

**Prepared:** 2026-10-19, using primary model cards, papers, and the project's Round 2 evidence. This is an orchestrator source synthesis; it is not a local model benchmark. No new model was loaded or installed.

## Decision summary

Keep `embeddinggemma` as the frozen comparison control because it won the project's preregistered Round 2 strict local ranking among the four measured models. It is not yet a release decision: live retrieval on the source-aggregated 165-query set remains below every quality floor.

Prioritize a reproducible E5/R3 comparison of (1) the current Gemma Ollama path, (2) the available Qwen3 and BGE-M3 Ollama variants, (3) a small multilingual ONNX candidate, and (4) an optional in-process runtime. Score the private exact-memory test set first; use MTEB only for candidate triage, never as a substitute.

The laptop's Round 2 results contradict a leaderboard-only choice: Gemma scored local Recall@5 .497 / MRR .2939 at p50 172.32 ms and 593.5 MB; Nomic scored .2182 / .2040 at 125.78 ms and 262.1 MB; BGE-M3 scored .2788 / .2206 at 222.68 ms and 1104.6 MB; Qwen3-0.6B scored .2667 / .2193 at 493.86 ms and 610.1 MB. These are the benchmark harness's recorded model comparisons, not Round 3 source-aggregated final retrieval scores, and their split/method must be preserved when quoting them.

## Candidate evidence

| Candidate | Size / shape | Published retrieval evidence | Fit and cautions |
|---|---:|---|---|
| Google EmbeddingGemma | 300M; 768 default, Matryoshka 512/256/128; 100+ languages | Multilingual MTEB v2 Mean(Task): 61.15/60.71/59.68/58.23 at those dimensions; English v2: 69.67/69.18/68.37/66.66 | Strong multilingual baseline; asymmetric search query/document templates are mandatory. Public score is not this project's local Recall@k. [Model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card) |
| Google EmbeddingGemma 2 | 740M total, optional modalities; 270M text encoder | Multilingual MTEB v2 Mean(Task) 61.36 at 768d; Code MTEB v1 78.68 | Do not count total model under a strict <=600M cap. Text-only loading is a separate branch and must be proven in the chosen runtime. [Model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2?hl=en) |
| Qwen3-Embedding-0.6B | 0.6B; 32k context; 32–1024 output dimensions; 100+ languages | MTEB multilingual v1 snapshot (2025-05-24): Mean(Task) 64.33; retrieval 64.64 | Instruction-aware asymmetric search; use query instruction and raw documents. Version/split differs from Gemma v2. [Model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| BAAI BGE-M3 | about 0.6B; 1024d; 8192 tokens; dense, sparse, and multi-vector modes | Official card exposes benchmark rows from different MTEB task/version snapshots, not one directly comparable aggregate | Multilingual and multi-functionality are attractive for mixed-language memory; sparse/multi-vector paths require extra code and memory beyond the current dense client. [Model card](https://huggingface.co/BAAI/bge-m3) |
| Nomic Embed Text v1.5 | 137M; Matryoshka-capable; 8192-token support | Published MTEB table commonly reports English retrieval; inspect the exact card revision/task before comparing | Small and fast candidate. Query/document prefixes differ; Round 2 local results were fastest/smallest but substantially worse on this corpus. [Model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) |
| BAAI BGE-small-en-v1.5 | Small English-only encoder; 384d, 512-token input | Official MTEB v1 table: retrieval 51.68, average 62.17 | Worth an English-only CPU/ONNX control if licensing and installed-runtime checks pass; prepend the model's retrieval query instruction. [Model card/table](https://huggingface.co/BAAI/bge-small-en) |
| Mixedbread mxbai-embed-large-v1 | 0.3B; 1024d, Matryoshka and binary quantization support | MTEB v1 table: retrieval 54.39, average 64.68 | English retrieval candidate with ONNX/OpenVINO/GGUF options; query prefix required, document prefix not required. Quantization must be rescored. [Model card](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) |
| Bekko Embedding v1-a8m (2026 preprint) | Claims <8M active parameters; 384d with 256/128/64 truncation | Preprint reports MMTEB Multilingual v2 Retrieval nDCG@10 56.2 | Newly published, not yet independently replicated here; compare only after checking artifact, license, model-card/runtime and the precise benchmark protocol. Treat author-reported figures as preliminary. [Paper](https://arxiv.org/abs/2607.25180) |
| mxbai-rerank-xsmall-v1 | 70.8M cross-encoder reranker | Model repository publishes task evaluation, but ranking-score metrics do not predict gain on this corpus | Candidate for a capped top-20/30 gated rerank only. Every added local model competes for RAM and heat budget. [Model card](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1) |
| mxbai-rerank-base-v1 | 0.2B cross-encoder reranker | Published reranker benchmark rows are candidate evidence, not local end-to-end latency | Test only if xsmall fails to improve MRR; do not run concurrently with embedder. [Model card](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1) |

## Interpretation

Public MTEB numbers are not a common ranking here unless model, MTEB version, language, task aggregation, dimension, prompt, quantization, and dataset split match. The Gemma card reports MTEB v2, while some BGE and Mixedbread tables are MTEB v1; Qwen's cited row is a dated leaderboard snapshot. Scores from multilingual and English-only tasks must not be merged.

Text-only parameter count, quantized file size, resident model memory, embedding vector dimension, and end-to-end service RAM are different quantities. Report all of them separately. Do not infer CPU latency from parameter count or vendor TOPS.

Asymmetric task prefixes are part of the model interface, not an optional prompt tweak: Gemma query `task: search result | query: ...` and document `title: ... | text: ...`; Qwen's retrieval instruction belongs on the query and not on documents; Nomic uses `search_query:` versus `search_document:`; Mixedbread asks for its retrieval prompt on queries. Persist the exact formatter in the model identity so old vectors are never silently mixed with new-format vectors.

Matryoshka dimension reduction saves vector/index storage and may improve throughput, but it can lower retrieval quality. Binary/int8 quantization changes similarity behavior and requires a paired local comparison. Quantization for the encoder weights and quantization of stored embeddings are separate ablations.

For Hindi/English or Hinglish, verify the query and source language labels and report per-language results. Multilingual support claims do not prove robust code-switching, exact paths, identifiers, or names.

The old service's exact-string queries need a lexical route even if dense semantic retrieval wins on natural-language questions. Use a hybrid lexical+dense result and test identifier accuracy, wrong-entity rate, and stale-state filtering independently.

## Ready experiment specification

1. **Question:** does an in-process small encoder improve end-to-end retrieval speed without reducing strict retrieval quality or breaching the 2 GB system RAM ceiling?
2. **Control:** same frozen `embeddinggemma` version and formatter, current chunking, current BM25+dense fusion, same 165 strict queries and relevance labels.
3. **Candidates:** installed Ollama variants first; then one <=600M ONNX/OpenVINO candidate with a verified local artifact. Never download a new model without an explicit runbook/user instruction and a quiet safety window.
4. **Prompts:** record model hash, tokenizer revision, query/document formatter, dimension, quantization, and chunk representation. Do not reuse vectors across candidate model/formatter changes.
5. **Warm-up and timing:** at least 20 warmups and 500 recorded queries in a quiet window; capture raw timings, p50/p95/p99, cold first-call time, RSS for MCP/embedder/harness separately, `ollama ps`, and watchdog samples.
6. **Quality:** Recall@1/5, MRR, exact-identifier Recall@1, paired bootstrap 10,000 resamples with 95% CIs; use McNemar for paired correctness outcomes if appropriate.
7. **Decision:** retain Gemma unless a candidate is non-inferior by the preregistered paired CI and materially improves latency or memory without crossing a safety or quality floor. The planned acceleration threshold for R10 is 2x.
8. **Repetition:** repeat the selected candidate from a clean model residency state; record the second run and any drift. One noisy run cannot establish a win.

## Adopt / test / reject

- **ADOPT:** Keep the current model and asymmetric formatting as the experimental control only. Keep lexical retrieval for exact paths, IDs, versions, and names.
- **TEST:** Re-run the private strict set with correct templates for the installed Qwen/BGE variants, a compact multilingual candidate, dimension truncation, int8/binary storage, and capped reranking. Test Hinglish only on independently verified language-labeled examples.
- **REJECT:** Selecting a model from a single MTEB leaderboard number; claiming EmbeddingGemma 2 is <=600M without its total-versus-text distinction; claiming a faster encoder from TOPS or parameter count; adopting any model from this report without local paired evidence.

## Sources

- [EmbeddingGemma v1 model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card)
- [EmbeddingGemma v2 model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2?hl=en)
- [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)
- [Nomic Embed Text v1.5 model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [BGE small English model card](https://huggingface.co/BAAI/bge-small-en)
- [mxbai-embed-large-v1 model card](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
- [mxbai-rerank-xsmall-v1 model card](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1)
- [mxbai-rerank-base-v1 model card](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1)
- [Bekko Embedding preprint](https://arxiv.org/abs/2607.25180)
- Project evidence: `evidence/r2-embedding-quality-selection.json`, `evidence/r2-embedding-all.txt`, `evidence/r3-live-retrieval-benchmark-pass2.json`.
