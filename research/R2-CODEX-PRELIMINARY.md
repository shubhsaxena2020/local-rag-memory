# R2 — Memory/RAG benchmarks and measurement protocol

**Prepared:** 2026-10-19 by the orchestrator from benchmark papers, official repos, and evaluation documentation. Independent Hermes research is pending. No external benchmark dataset was downloaded in this checkpoint.

## What to measure

Use separate layers and report them separately:

1. **Retrieval:** strict evidence Recall@1/5, MRR, exact-identifier Recall@1, stale-fact leakage, evidence coverage, and per-query latency.
2. **Answering:** answer correctness against reviewed gold, supported/unsupported/contradicted atomic claims, citation validity, abstain precision/recall, and answerable-question over-refusal.
3. **Operations:** cold start, p50/p95/p99, process and model RSS, RAM growth, time/token cost, failure rate, crashes, and recovery.
4. **Lifecycle:** update/contradiction accuracy, temporal “current vs historical” behavior, auditability, and whether old values leak as current.
5. **Benchmark protocol:** exact dataset revision/hash, split, model/version, reader and judge models, prompts, generation settings, retrieved-context/token budget, result denominator, exclusions, and scoring script commit.

Do not compare QA accuracy to retrieval recall. Do not report only an aggregate where a benchmark has abstention, temporal, or multi-hop slices.

## Candidate benchmarks

| Benchmark/tool | Scope and size | Practical local plan | Data/compute and caveats |
|---|---|---|---|
| LoCoMo (ACL 2024) | Public release: 10 long conversations with QA, event-summary, and multimodal annotations; commonly used 1,540-question non-adversarial subset | Smoke with one conversation; if the evaluator is stable, use all available QA rows in a fixed reader/judge configuration. It is feasible as an index/retrieval test; end-to-end QA may need hundreds to thousands of answer/judge calls. | Data license is CC BY-NC 4.0 in the release's license notice; review terms, attribute, and keep data local. The full paper describes an earlier 50-conversation version, so pin the exact file and hash. [Repo](https://github.com/snap-research/LoCoMo), [paper](https://aclanthology.org/2024.acl-long.747/), [license](https://raw.githubusercontent.com/snap-research/LoCoMo/main/LICENSE.txt) |
| LongMemEval-S (ICLR 2025) | 500 questions across six types; histories about 115k tokens per question; includes 30 abstention instances | Start with retrieval-only Recall@k on 50–100 questions and a fixed context budget. If local ingestion and scoring finish within a few hours, run all 500 retrieval questions. Then run end-to-end QA on a stratified subset with separately priced answerer and judge calls. | Official repo has a lightweight evaluator path, but end-to-end example uses GPT-4o and retrieval may require larger encoders. Dataset is published as MIT by the official cleaned HF release; confirm current upstream terms before redistribution. [Official repo](https://github.com/xiaowu0162/LongMemEval), [cleaned dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned), [paper](https://openreview.net/forum?id=pZiyCaVuti) |
| LongMemEval oracle | Same benchmark questions with only evidence sessions available | Use as a reader/answerer control, never as retrieval score; compare normal and oracle context under identical reader/judge. | Oracle removes retrieval difficulty and changes the task; do not place its answer accuracy beside normal full-history accuracy without clear labeling. Same upstream terms. [Official repo](https://github.com/xiaowu0162/LongMemEval) |
| LongMemEval-V2 (2026) | 451 manually curated web-agent questions; small/medium tiers; up to 115M tokens in the largest haystacks; static/dynamic state, workflow, gotchas, premise awareness | Too expensive for the current 24-hour/thermal window. Later, download the small tier only after license/resource review and test code-agent memory on trajectory recall. | Includes screenshots and substantial trajectory data; the README requires local preparation and separate reader/controller/embedding/judge endpoints. Verify dataset-specific license before use. [Official repo](https://github.com/xiaowu0162/LongMemEval-V2), [paper](https://arxiv.org/abs/2605.12493) |
| BEAM (2025) | 100 conversations, 2,000 validated questions, long contexts across 10 memory-ability categories; variants include 100K–10M-token scales | Start with its smallest available public tier only if installation and scoring are reproducible; not a one-hour fit for this laptop while local inference is occupied. | The 10M scale and multiple categories add substantial parsing/indexing/reader cost. Dataset notice reports CC BY-SA 4.0; inspect upstream notice before copying or publishing. [Paper](https://arxiv.org/abs/2510.27246), [dataset](https://huggingface.co/datasets/Mohammadta/BEAM) |
| MemBench (2025) | Factual and reflective memory, participation/observation tasks; effectiveness, efficiency, and capacity | Feasible as a later small API-only subset, but not a direct retrieval-only substitute for the current golden set. | Paper releases data and project; inspect repo license and prompt dependencies before use. [Paper](https://arxiv.org/abs/2506.21605), [repo](https://github.com/import-myself/Membench) |
| MTEB / BEIR subsets | General-purpose embedding/retrieval datasets and task scores, not personal-memory lifecycle tests | Use a small public retrieval subset only when selecting encoders, then validate on private memory questions. | Public scores vary by task, language, version, prompt, and aggregation; they are candidate-triage evidence only. [MTEB](https://github.com/embeddings-benchmark/mteb), [BEIR](https://github.com/beir-cellar/beir) |
| RAGTruth | Human span-level annotations for unsupported/contradicted claims in RAG responses; about 18k responses in published descriptions | Use a 100–300 response slice to exercise claim extraction/judge agreement; not a replacement for system outputs with matched retrieved facts. | Annotation task/domain differs from this app. Confirm data license and redistribution rules in the official repo before downloading. [Paper](https://arxiv.org/abs/2401.00396), [official repo](https://github.com/ParticleMedia/RAGTruth) |
| HaluEval | 35k examples: 5k general queries plus 30k task-specific QA/dialogue/summarization cases | Sample 100–300 examples to check hallucination detector/judge behavior; use the matching task subset only. | Not specifically a retrieval-grounded benchmark. Inspect repo `LICENSE` and per-source conditions before use. [Paper](https://arxiv.org/abs/2305.11747), [repo](https://github.com/RUCAIBox/HaluEval) |
| ARES | LLM-judge evaluation for context relevance, answer faithfulness, and answer relevance; prediction-powered inference | Not first choice for this 24-hour prototype: it expects at least 50 human-labeled triples, few-shot judging data, and a larger unlabeled result set. Reuse its distinction of dimensions if enough labels exist. | Some quick starts install API-key-dependent infrastructure; keep private data local and verify every evaluator call. ARES's classifier/PPI score is not the same as direct accuracy. [Paper](https://arxiv.org/abs/2311.09476), [repo](https://github.com/stanford-futuredata/ARES) |
| RAGAS | Open-source reference-free and reference-required RAG metrics | Use selected metrics as a secondary diagnostic on 50–100 rows, with a pinned library version and fixed judge. | Many metrics make judge calls and are nondeterministic; it is a framework, not a gold-label replacement. [Paper](https://aclanthology.org/2024.eacl-demo.16/), [metrics docs](https://docs.ragas.io/en/latest/concepts/metrics/) |

## License and privacy gate

- Read the upstream dataset license and any bundled-data terms before downloading or checking in data. A code repository license does not automatically license embedded third-party text.
- LoCoMo is marked CC BY-NC 4.0 in its data release license; treat it as noncommercial, attribute authors, and do not republish derived answer text without checking the terms.
- LongMemEval's cleaned dataset is published with MIT license metadata by its official release; pin the exact cleaned version/hash and preserve required notice.
- LongMemEval-V2 and BEAM have trajectories/screenshots and different terms; validate each dataset artifact's own license before use.
- RAGTruth, HaluEval, MemBench, and any model-generated subset need a file-level license/provenance check; this report does not grant permission to redistribute them.
- Do not write benchmark conversations or question/answer text into the user's personal vault or the production database.

## Feasible local plan

1. Keep the private gold set as the primary release gate; independently audit labels and freeze SEALED before tuning.
2. Use fixed, stratified 50-question pilots on one public benchmark to validate parser/scorer behavior before larger runs.
3. For retrieval-only comparisons, run 100–500 questions locally with a consistent index builder and report session-level and passage-level evidence metrics separately.
4. For end-to-end QA, separate retrieval output, answerer model, judge model, context token budget, prompt, and answer cost; use no-memory, grounded-memory, and oracle controls.
5. If judge labels are used, blind two model families on at least 25% overlap; report Cohen's kappa, discard a judge below .6, manually read 20 random verdicts, and detect templated answer/rationale content.
6. Bootstrap every headline proportion/rate with 10,000 resamples; paired bootstrap or McNemar for A/B outcomes. Keep unscorable or missing rows in the evidence log with a reason.
7. Version and hash raw data, model prompt, output, and scorer so later benchmark changes cannot silently rewrite the score.
8. Use retrieval recall and deterministic exact-match/citation rules as primary where possible; treat LLM judge as a second signal with uncertainty.

## Recommended experiment for this project

- **Primary:** private strict 165-query retrieval set plus the 102-item held-out abstention/grounding rows already available; report R@1/R@5/MRR, exact identifier R@1, grounded claim support, abstain precision/recall, and answerable over-refusal.
- **Secondary public replication:** LongMemEval-S retrieval-only pilot (100 questions) if the quiet window, data terms, and local storage budget allow; retain its source/session evidence definition and use no LLM answerer for retrieval recall.
- **Public QA pilot:** if tokens/quota permit later, 50 stratified LongMemEval questions with fixed Sonnet 5.5 answerer and an independent judge; include all six question types by stratified sampling and record abstention separately.
- **Do not attempt now:** full LongMemEval-M, LongMemEval-V2 largest tier, BEAM 10M, model training, full ARES PPI training, or a public leaderboard claim under current thermals.

## Adopt / test / reject

- **ADOPT:** strict local labels, separate retrieval and answer metrics, fixed answer/judge, source-level evidence, blind overlap adjudication, bootstrap CIs, data/model hashes, explicit licenses.
- **TEST:** small LongMemEval-S retrieval-only replication later, then a fixed 50-question end-to-end pilot; test RAGAS/ARES only as secondary diagnostics with local data controls.
- **REJECT:** ranking from vendor percentages without exact configs; treating LLM-as-judge faithfulness as ground truth; mixing data versions or tasks; publishing protected benchmark content; inferring production accuracy from a public benchmark.

## Sources

- [LoCoMo official repository](https://github.com/snap-research/LoCoMo), [ACL paper](https://aclanthology.org/2024.acl-long.747/), [license](https://raw.githubusercontent.com/snap-research/LoCoMo/main/LICENSE.txt)
- [LongMemEval official repository](https://github.com/xiaowu0162/LongMemEval), [cleaned dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned), [ICLR 2025 paper](https://openreview.net/forum?id=pZiyCaVuti)
- [LongMemEval-V2 repository](https://github.com/xiaowu0162/LongMemEval-V2), [paper](https://arxiv.org/abs/2605.12493)
- [BEAM paper](https://arxiv.org/abs/2510.27246), [data](https://huggingface.co/datasets/Mohammadta/BEAM)
- [MemBench paper](https://arxiv.org/abs/2506.21605), [repo](https://github.com/import-myself/Membench)
- [MTEB repo](https://github.com/embeddings-benchmark/mteb), [BEIR repo](https://github.com/beir-cellar/beir)
- [RAGTruth paper](https://arxiv.org/abs/2401.00396), [repo](https://github.com/ParticleMedia/RAGTruth)
- [HaluEval paper](https://arxiv.org/abs/2305.11747), [repo](https://github.com/RUCAIBox/HaluEval)
- [ARES paper](https://arxiv.org/abs/2311.09476), [repo](https://github.com/stanford-futuredata/ARES)
- [RAGAS paper](https://aclanthology.org/2024.eacl-demo.16/), [docs](https://docs.ragas.io/en/latest/concepts/metrics/)
