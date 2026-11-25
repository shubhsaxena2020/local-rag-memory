# R4 — Grounded answers: preliminary source synthesis

Status: orchestrator research synthesis written during watchdog pause. This is not the R4/R7 Hermes worker deliverable; compare and merge only after independently inspecting the worker report. No local experiment was run for this lane.

Research checked: 2026-10-19. Primary sources opened: Self-RAG (ICLR 2024 paper), ARES (NAACL 2024 paper), and RAGTruth (ACL 2024 paper).

## Decision summary

For this memory system, the first choices to TEST are (1) quote-first, claim-level support, (2) abstention from a separately calibrated evidence-support score, and (3) a low-cost lexical / exact-string route for identifiers. A small entailment verifier is a later TEST only if local quality and cost are measured. Reject raw dense similarity as calibrated answer confidence. Keep self-consistency and model self-critique out of the latency-sensitive default path unless paired tests show value.

The paper results below establish promising mechanisms, not guaranteed gains on our corpus, answerer, or local hardware. None can be transferred as a measured project result.

## Ranked techniques

### 1. Answer from retrieved quotes; attach support to each atomic claim

Require the answerer to emit concise claims with source IDs and exact supporting spans. A deterministic post-check should verify the cited source exists in the retrieved set and the span occurs in that source. Split compound sentences before judging support. Unsupported claims should be omitted or marked unverified; a citation match is necessary but is not sufficient evidence of entailment.

This is cheap to audit and makes failures inspectable. It may reduce completeness, so report answerable-question over-refusal and omission alongside unsupported-claim rate.

A short quote-first prompt is a low-cost TEST candidate. It must be compared against the current prompt using the same answerer model, isolated condition directories, held-out split, paired bootstrap or McNemar rule, and latency / token budget registered before execution.

### 2. Claim-level entailment verification

An entailment verifier can label each claim against its cited passage as supported, contradicted, or unknown. Only supported claims should be asserted as current facts. Contradiction and missing evidence should trigger an explicit caveat or abstention. Measure the verifier's precision and recall against a human-labeled subset before using it as a gate; an LLM judge alone is not a trusted gold label.

ARES evaluates context relevance, answer faithfulness, and answer relevance separately, using specialized judges and a human preference validation set. Its paper reports that fewer than roughly 100–150 human labels were not enough to reliably distinguish pseudo-RAG systems in relevant ranking settings. That result is a warning about small judge calibration sets, not proof that a verifier reaches a target error rate.

A small NLI cross-encoder is a possible local implementation, but no laptop latency, RAM, or project accuracy effect was verified in this lane. Measure model load, warm/cold latency, and end-to-end behavior before adoption.

### 3. Calibrate abstention using evidence features, not raw similarity as probability

Candidate features: top score, score margin, lexical/exact identifier match, number of independent supporting facts, evidence age, contradiction flag, and whether a direct answer string appears in a retrieved fact. Fit or choose a threshold on dev only; freeze it before held-out. Report precision, recall, and answerable over-refusal separately, plus 95% bootstrap confidence intervals.

Conformal risk control is a research candidate only if exchangeability and calibration assumptions can be defended for the probe population. A threshold selected after inspecting held-out or sealed data is invalid. No claim is made that conformal prediction automatically provides the desired risk bound under distribution shift.

### 4. Route by question type and retrieval need

Use a low-cost deterministic route for exact identifiers, paths, versions, dates, and names. Route factoid questions to hybrid retrieval with a small evidence budget. Treat subjective or open-ended questions differently from requests for stored personal facts. Route contradiction, stale-fact, or no-support cases to clarification / abstention.

Self-RAG shows a learned model can decide when to retrieve, critique retrieved relevance, and check whether a generated segment is supported. Its 7B / 13B models show improvements on several paper tasks, but reproducing its training and reflection-token inference is expensive for this app. The useful low-cost transfer is the control flow—retrieve only when needed, inspect relevance and support—not its reported scores.

### 5. Explicit support/refusal instructions

A prompt rule should say to answer only from supplied evidence, give source IDs, preserve uncertainty, and return a fixed abstention response when no passage supports the requested fact. The rule should not encourage the model to infer identity, dates, ownership, or causality from near-matches.

Prompt-only gains are model- and dataset-dependent. The Round 2 held-out result had abstention precision 0.2985 and answerable over-refusal 0.5732; a more restrictive prompt could worsen the already high over-refusal rate. Test paired on the same fixed answerer and report both error directions.

### 6. Self-consistency / multi-sample agreement

Multiple generations can expose instability, but agreement is not evidence: several samples from the same model may repeat the same unsupported claim. It also multiplies answerer cost and latency. REJECT as a default; TEST only on a small paired dev slice if quote / verifier methods fail, with no sealed access and a fixed generation count.

### 7. Post-hoc self-critique

A model can critique its response, but a model's own critique shares its blind spots. Self-RAG trains reflection behaviors and uses them during inference; that is distinct from asking an untrained model to rubber-stamp its own answer. REJECT self-critique as sole grounding control. A second independent reviewer can be an evaluation instrument, not a substitute for source-level checks.

## Reported effect evidence and limits

- Self-RAG paper reports 7B and 13B models outperforming compared baselines on several of six tasks and improving citation quality on ASQA. Its Table 2 includes dataset-specific scores; for example the table reports ASQA citation precision / recall of 66.9 / 67.8 for Self-RAG 7B and 70.3 / 71.3 for 13B. These are paper protocol values and are not comparable to our atomic-claim metric.
- Self-RAG's analysis reports that on the sampled correct predictions in TriviaQA and PopQA, 2% of Self-RAG answers had ground-truth strings absent from provided evidence, versus 15% for Alpaca 7B, 18% for Llama2-chat 13B, and 20% for Alpaca 30B. This is a paper-specific substring analysis, not the PLAN-24H unsupported-claim definition.
- ARES assesses three separate RAG properties and adds prediction-powered inference using human labels. Its reported ranking reliability degrades for small human validation sets; approximately 150 labels were needed in its tested setup. Judge agreement must still be independently measured for our panel.
- RAGTruth contains nearly 18,000 manually annotated RAG responses with case- and word-level labels and hallucination intensity. Its authors report that a relatively small finetuned model can be competitive with prompt-based detection by larger models. This supports using labeled samples to evaluate detectors; it does not establish that their detector meets our claim-level rubric.

## Ready experiment proposals (not run)

### R4-E1: Quote-first output format

Hypothesis: requiring one concise answer claim per line with an exact supporting source ID and span will reduce unsupported / contradicted claim rate versus the current grounded prompt without increasing answerable abstention by more than 5 percentage points.

Change: prompt/output schema only; same answerer model, retrieval results, and temperature in control and treatment. Conditions must be in separate directories and the no-memory condition must have no access to retrieved facts or gold answers.

Metrics: unsupported-claim rate, claim support precision, citation-span validity, abstain precision/recall, answerable over-refusal, answer completeness, p50/p95 end-to-end latency and tokens. Use PLAN-24H atomic claim rubric.

Split: tune on dev; run one registered paired held-out evaluation. Never inspect sealed examples or labels.

Cost budget: no extra model call; output <= 20% more tokens than control; same held-out sample size as the registered hallucination study.

Decision rule: adopt only if paired 95% CI for unsupported-rate change is below zero, absolute unsupported rate meets <=5%, abstain floors pass, over-refusal is not worse by >5pp, and latency remains within the registered budget. Report all CIs and error counts.

### R4-E2: Calibrated answerability gate

Hypothesis: a dev-calibrated multifeature evidence gate can raise abstain precision while maintaining recall >=0.85 and avoid the current high answerable-over-refusal rate.

Change: threshold features are frozen before held-out; baseline is current gating rule. Do not expose dense similarity as a probability.

Metrics: abstain precision/recall, answerable over-refusal, Brier score / calibration curve if the output is a probability, and unsupported answer rate. Report bootstrap 95% CIs.

Split: dev calibration, one held-out decision. The sealed set remains inaccessible until final scoring.

Cost budget: deterministic feature calculation; no new LLM call; p95 added overhead <=10 ms and RAM <=50 MB.

Decision rule: keep only if all abstention floors pass and paired held-out evidence supports the improvement. A small or underpowered held-out sample is inconclusive, not a pass.

### R4-E3: Small NLI verifier feasibility

Hypothesis: one compact verifier pass reduces unsupported claims enough to justify local inference cost.

Change: compare current pipeline to claim-split plus NLI pass, using identical answerer outputs first to isolate verifier behavior, then optionally paired end-to-end generation.

Metrics: verifier precision/recall/F1 on human-labeled claims, unsupported-claim rate, false rejection rate, latency, RAM, crash/timeouts. Include uncertainty intervals and confusion counts.

Split: labeled dev set to select thresholds; held-out once. Do not train/tune on held-out or sealed data.

Cost budget: max one loaded verifier; hard timeout <=10s/request; additional warm p95 <=100 ms and steady RSS <=250 MB or reject for this target device.

Decision rule: only adopt after human-labeled quality is adequate, the unsupported-claim reduction is significant, abstention/over-refusal floors pass, and total latency/RAM gates remain intact.

## Adoption decisions

- ADOPT: Exact source IDs and quote spans as a response-audit affordance, provided the source-span checker is implemented and tested. This is an output shape decision, not a measured accuracy claim.
- TEST: Quote-first generation; claim-level support checking; dev-calibrated abstention; deterministic question-type routes; small NLI verifier only after feasibility profiling.
- REJECT: Raw cosine score as calibrated confidence; self-critique as sole support verification; unregistered prompt changes; multi-sample consensus as default.

## Sources (opened primary sources)

- Asai et al., Self-RAG, ICLR 2024 / arXiv: https://arxiv.org/abs/2310.11511
- Self-RAG OpenReview proceedings: https://openreview.net/forum?id=hSyW5go0v8
- Saad-Falcon et al., ARES, NAACL 2024 / arXiv: https://arxiv.org/abs/2311.09476
- ARES source repository: https://github.com/stanford-futuredata/ARES
- Niu et al., RAGTruth, ACL 2024: https://aclanthology.org/2024.acl-long.585/
- RAGTruth source repository: https://github.com/ParticleMedia/RAGTruth

## Verification status

This file has more than 60 lines and contains primary URLs; it remains a preliminary orchestrator synthesis. The Hermes R4/R7 output is still missing and must be independently read, compared, and merged before G1. No effects or thresholds above are claimed as measured for this project. No experiment, test, local model call, or sealed access was performed while watchdog state is PAUSED.
