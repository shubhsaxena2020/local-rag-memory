# R6 — Hybrid retrieval and fusion: preliminary source synthesis

Status: Codex research notes for comparison with the assigned Hermes R5/R6 worker. This is not a measured result and does not replace research/R6.md.

## Retrieval problem and constraints

The current failure pattern includes weak exact identifier recall and weak overall strict recall, while raw dense cosine is being misused as answer confidence. Hybrid retrieval can improve recall across lexical and semantic evidence, but scores from different retrievers are not calibrated to a common scale.

Keep retrieval ranking, answerability, and claim support as separate decisions. A candidate's rank or similarity is not a probability that it answers the question.

Use stable fact IDs and deterministic final tie-breaking so benchmark output is reproducible. Scope/tenant and active-state filters must be applied before results reach the answering model.

## Fusion options

| Method | Data needed | Strength | Risk | Project status |
|---|---|---|---|---|
| Reciprocal Rank Fusion (RRF) | ranked lists only | Scale-insensitive; simple; robust starter | Rank constant and candidate depth still matter | Test as first baseline |
| Weighted normalized scores | representative dev labels / score calibration | Lets lexical, dense, exact signals be weighted | Score normalization and weights can be brittle / overfit | Test only after baseline |
| Learned ranking/fusion | adequate labeled query-fact judgments | Can exploit query features and interactions | Label cost, leakage, maintenance, calibration | Defer |
| Exact-match rules | identifier parser and equality index | Reliable for paths, versions, IDs, names | False positives from substring matching | Adopt carefully with normalized exact equality |

Cormack et al. (SIGIR 2009) report RRF outperformed individual component rankings and Condorcet fusion across their TREC experiments, and also outperformed the rank-learning approaches evaluated there. Those historical results motivate a low-cost baseline; they do not guarantee improvement on this personal-memory collection.

## Lexical and identifier handling

Use SQLite FTS5/BM25 for lexical ranking with quoted/escaped query construction. Verify tokenization on punctuation-heavy paths, package names, version strings, URLs, and mixed Hindi/English text.

Maintain a small exact-key index for normalized complete identifiers. Exact equality can be boosted deterministically; avoid broad substring boosts where `v1.2` could match `v1.20` or a short ID could match a longer token.

For filesystem paths and identifiers, test case sensitivity and Windows separator normalization explicitly. Keep the original source string to display and cite; normalization should create an additional key, not destroy evidence.

Trigrams can help with typos and partial paths but create collision risk and more index writes. Use them only as a fallback candidate channel with a low boost and a test for false neighbors.

Do not treat BM25 term score, cosine similarity, edit similarity, and answerability as one numeric quantity. RRF avoids raw-scale comparisons; if weights are later learned, the result is a ranker and still not a calibrated confidence.

## Query expansion and contextual methods

- Query rewriting can normalize abbreviations or split a compound query, but generated terms may alter intent. Test one rewrite and retain the original query as a candidate channel.
- Multi-query retrieval multiplies searches and increases latency. Reject as default until it provides significant paired retrieval improvement within the p95 budget.
- HyDE generates a hypothetical document and embeds it; it adds a generation call, potential unsupported content, and significant cold latency. Reject for a small memory system unless an API answerer already exists in the path and a measured benefit is compelling.
- Contextual retrieval prepends source/session context to chunks before embedding. This can disambiguate short facts but changes embedding inputs and index size. Test with paired chunk vectors; keep answerable evidence separable from synthetic context.
- Query-type routing can select exact identifier search for paths/versions/IDs and hybrid semantic search for paraphrases. Routing must have a safe fallback that searches both channels on uncertain classification.
- Graph expansion can retrieve connected facts for multi-hop questions but risks topical over-retrieval. No graph traversal result should be mistaken for direct evidence without a traceable source fact.

## Proposed baseline pipeline

1. Validate and normalize query while preserving the raw text.
2. Apply caller scope and active-fact filters.
3. Retrieve top lexical FTS results and top dense results independently.
4. Retrieve exact identifier matches if a full identifier is present.
5. Combine ranked candidates with RRF and a deterministic exact-match boost.
6. Collapse only proven duplicate records while retaining source IDs; do not merge contradictions.
7. Return structured evidence with fact ID, source, temporal status, snippet, and separate raw retrieval features.
8. Let a separately calibrated answerability gate and grounded answer step decide whether to answer.

For safety, retrieval fallback to FTS should be explicit in the output. It must not make a failed vector query appear equivalent to a successful hybrid query.

## Experiment specs (not run)

### R6-E1: lexical + dense RRF

Hypothesis: unioned lexical and dense candidate lists with RRF improve strict held-out R@1/R@5/MRR and identifier recall over dense-only while adding no more than 20ms warm p95.

Change: RRF only; same corpus, embedder, filters, top-k, and strict relevance checker. Query and result rows log answerer/model separately for any end-to-end phase.

Metrics: R@1/R@5/MRR, identifier R@1, candidate count, query latency, and storage/index overhead.

Decision rule: adopt only if paired bootstrap shows positive retrieval gain and all identifier and latency floors remain plausible; report CI and misses.

### R6-E2: exact identifier route

Hypothesis: full-key equality with a deterministic boost improves paths/version/name/ID R@1 without harming non-identifier queries.

Change: exact normalized key channel only; no trigrams at first.

Metrics: identifier R@1, non-ID retrieval metrics, false exact-match boosts, latency, Unicode/case/path normalization errors.

Decision rule: require >=.95 identifier R@1 and no statistically significant regression on non-ID classes beyond predeclared tolerance.

### R6-E3: rewrite or multi-query

Hypothesis: one rewrite channel improves multi-hop or paraphrase queries when baseline hybrid fails.

Change: add one fixed rewrite model/API call; keep original query channel and isolate conditions. Record answerer and rewriting model independently.

Metrics: paired recall/MRR, latency/cost, false retrieval and unsupported answer rates.

Decision rule: reject if the latency gate fails, if the gain CI includes zero, or if query drift increases false evidence.

## Adopt / test / reject

- ADOPT: exact full-string identifiers as a separate retrieval feature, with normalized and raw forms retained.
- TEST: RRF as the first rank fusion baseline; the project currently needs separate lexical/dense channels and RRF is independent of score calibration.
- TEST: weighted fusion only after score/feature calibration and on dev; freeze before held-out.
- DEFER: learned fusion until label volume and leakage-free splits are available.
- REJECT as default: HyDE, multi-query, graph expansion, and contextual retrieval until paired latency/quality ablations justify them.
- REJECT: score thresholds on raw cosine as probability of support.

## Sources

- Cormack, Clarke, Büttcher, Reciprocal Rank Fusion (SIGIR 2009): https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
- Google Research publication record for RRF: https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/
- Microsoft Research, Robertson & Jones on probabilistic relevance weighting: https://www.microsoft.com/en-us/research/publication/relevance-weighting-of-search-terms/
- SQLite FTS5 documentation: https://www.sqlite.org/fts5.html
- SQLite-vec official repository: https://github.com/asg017/sqlite-vec
- PLAID late interaction retrieval paper: https://arxiv.org/abs/2205.09707

## Review status

No experiment or test ran. No label data was inspected for tuning. This source synthesis is preliminary and waits for the Hermes R6 report and independent review. It contains no claimed project benchmark score.
