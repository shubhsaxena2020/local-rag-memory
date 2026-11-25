# R1 — State of agent memory systems (2026)

**Prepared:** 2026-10-19 by the orchestrator from papers, official repositories, and official product docs. This synthesis is provisional until compared with the assigned Hermes report. Scores below are attributed to their authors and are not a cross-system ranking.

## Executive synthesis

Current agent memory systems share four design choices: (1) separate compact durable memory from the active context; (2) make writes selective or asynchronous; (3) combine lexical and semantic lookup, often with structured relationships or time; (4) expose memory through tools, APIs, or MCP to multiple agent shells.

The best lesson for this repository is to borrow small, testable mechanisms: precise provenance and time validity, a hybrid index for exact names plus semantic paraphrase, an explicit memory budget, clear retrieval traces, reversible updates, and a separately evaluated write/consolidation path. Large LLM-on-every-write pipelines, hosted services, graph databases, or learned policies do not fit the current single-laptop/offline constraint without strong local evidence.

Published QA scores vary with dataset revision, context budget, ingestion choices, answer model, judge, prompt, and whether the reported score is end-to-end answer accuracy, retrieval recall, or an aggregate F1. They cannot be compared by putting percentages in one leaderboard column.

## Systems and evidence

| System | Design from primary source | Reported evidence / operating cost | Relevance here |
|---|---|---|---|
| Mem0 base | Extract, consolidate, and retrieve persistent user facts; API/SDK plus self-hosted stack | The 2025 paper reports gains over its chosen baselines on LoCoMo and lower token use/latency than its full-context comparison. The current official eval page reports LoCoMo 92.5 and LongMemEval 94.4 with mean context of 6,956 and 6,787 tokens respectively. Paper and current dashboard are different snapshots/protocols. | Copy selective fact extraction and explicit per-user scoping; do not copy hosted dependencies into a local-only design. [Paper](https://arxiv.org/abs/2504.19413), [official evaluation page](https://github.com/mem0ai/mem0/blob/main/docs/core-concepts/memory-evaluation.mdx) |
| Mem0 graph | Adds entity/relationship memory on top of Mem0 | Its paper reports about two percentage points above base on the evaluated LoCoMo setting; this is a within-paper comparison. | Graph only when a question needs explicit relationships or multi-hop traversal; graph cost is not justified for every fact. [Paper](https://arxiv.org/abs/2504.19413) |
| Zep / Graphiti | Temporal knowledge graph from unstructured conversation plus structured sources; temporal edges, invalidation, and hybrid graph/full-text/vector retrieval | Zep paper reports 94.8% vs 93.4% on DMR and up to 18.5% accuracy improvement plus 90% response latency reduction on the paper's LongMemEval comparison. Zep's current official page reports LongMemEval 90.2%, p50/p95 retrieval 104/162 ms, 4,408 median context tokens; LoCoMo 94.7%, 155ms. The current page is vendor-published and not the same protocol as the paper. | Borrow valid-time and learned-time separately plus current-vs-history filtering; likely too much graph infrastructure for the present corpus. [Paper](https://arxiv.org/abs/2501.13956), [official results](https://www.getzep.com/research/), [Graphiti](https://github.com/getzep/graphiti) |
| Letta / MemGPT | Stateful agent with explicit core/recall/archival memory and tool-controlled context movement (“virtual context management”) | The original MemGPT paper evaluates agent memory tasks but does not provide a directly comparable current LoCoMo/LongMemEval score for today's Letta product. No same-protocol price or laptop RAM comparison was found in the inspected primary sources. | Borrow explicit finite context management and inspectable memory tools; not a passive shared store for unrelated CLIs. [Paper](https://arxiv.org/abs/2310.08560), [Letta](https://github.com/letta-ai/letta) |
| Cognee | Open-source knowledge-engineering pipeline that turns data into a graph and vector/relational indexes; graph and RAG retrieval | Cognee reports LoCoMo/HotPotQA comparisons on its official evaluation pages and a BEAM paper/evaluation; numbers need source-run settings and judge review before comparison. No stable per-request local cost was established from those pages. | Borrow ingestion graph for domains where entities/relations matter; do not infer benefit for exact fact lookup without ablation. [Official evaluation results](https://www.cognee.ai/research-and-evaluation-results), [BEAM report](https://www.cognee.ai/benchmarking-cognee-on-beam) |
| A-MEM | LLM-authored structured notes, tags/keywords/context, dynamic links, and memory evolution in a Zettelkasten-like network | 2025 paper reports improvements against its chosen baselines across six backbones; the inspected abstract does not establish one common, reproducible operating cost. | Candidate patterns: relations and provenance around notes. Rewriting old memories on each write is risky for this project's append-only invariant. [Paper](https://arxiv.org/abs/2502.12110), [official implementation](https://github.com/agiresearch/a-mem) |
| LightMem | Separates short-, mid-, and long-term memory; small-language-model online retrieval/write modules and offline consolidation | ACL 2026 paper reports about +2.5 average F1 over A-MEM on LoCoMo, 83ms median retrieval and 581ms end-to-end in its evaluated configuration. These are paper-specific measurements and are not this product's latency targets. | Strong candidate for low-cost staged extraction/consolidation research; first test whether deterministic local extraction suffices. [ACL paper](https://aclanthology.org/2026.acl-long.588/) |
| Memori | Compact semantic triples and summaries, LLM-agnostic persistence | 2026 preprint reports LoCoMo 81.95% and about 1,294 tokens per query. The result is author-reported and still requires a matching harness before comparison. | A representation-compression idea, but triple extraction can lose exact source wording unless raw provenance is retained. [Paper](https://arxiv.org/abs/2603.19935) |
| AgeMem | Trains the agent's policy to choose store/retrieve/update/summarize/discard actions | 2026 preprint reports gains across five long-horizon benchmarks, but requires RL training and a substantial training/evaluation stack. | Research direction only; far beyond current local prototype scope. [Paper](https://arxiv.org/abs/2601.01885) |
| Supermemory | Hosted memory API plus MCP and a local open-source binary/engine option; document ingestion, namespace, search, extraction, reranking and query rewrite | API v5 was announced Oct 7 2026. Product docs describe endpoints and features, but no independently comparable score or one-laptop RAM profile was found. | Borrow namespaces, MCP packaging, and source/document lifecycle ideas; reject hosted service for private local-only constraint. [API v5 announcement](https://supermemory.ai/changelog/supermemory-api-v5/), [official docs](https://docs.supermemory.ai/memory-api/introduction) |
| MemClaw | Name currently refers to multiple distinct repositories/products, including personal local file-backed assistants and multi-agent fleet memory | No common paper, benchmark harness, or traceable score for one canonical project was located. Treat results as unknown until repository identity is pinned. | Avoid generic claims; inspect the exact codebase if a particular MemClaw project is proposed. [Example local repo](https://github.com/emanuilo/memclaw), [separate fleet repo](https://github.com/caura-ai/memclaw-long-run-fleet) |
| Claude Code memory | `CLAUDE.md` instruction/context files at user/project/local scopes; automatic loading and imports | Official Anthropic docs describe files and loading behavior, not a memory-retrieval benchmark or cost. | Useful for stable preferences/instructions; static files can be a fallback/export, not the authoritative fact lifecycle. [Memory docs](https://docs.anthropic.com/en/docs/claude-code/memory) |
| Codex persistent project guidance | `AGENTS.md` is discovered from global and repository paths and injected as instructions; workflow/context files are conventions | OpenAI docs describe instruction discovery/injection, not learned episodic recall or a RAG accuracy score. | Keep durable project rules there; use MCP for searchable changing facts. [Codex instruction guide](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.3-codex) |
| OpenAI Codex memory feature | The inspected official sources document persistent repo instructions and optional workspace-agent memory settings, but not an always-on Codex CLI episodic-memory database. | No comparable standalone score/cost was found. | Do not claim the native file instruction mechanism is a semantic memory store. [Codex usage/help](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan), [workspace agents](https://help.openai.com/en/articles/20001143/) |

## Evaluation caveats

- The official Zep paper score, the Zep current research page, the Mem0 paper, and the Mem0 current evaluation page are distinct experiment snapshots. Preserve source, date, dataset version, answerer, grader, memory budget, token count, and exact script for each number.
- LoCoMo and LongMemEval are end-to-end QA benchmarks; their accuracy does not equal retrieval Recall@k. Keep retrieval metrics and answer correctness separate.
- A memory system can score well by spending more context tokens or by using a stronger answerer/judge. Record context budget, latency, and cost alongside accuracy.
- Product claims are evidence of what the vendor publishes, not independent replication. Label them vendor-reported.
- Benchmark-specific optimization can inflate results. Keep the private 80-question SEALED set blind and report its score once after freezing.
- Exact paths, versions, dates, contradictions, deleted facts, and source attribution are underrepresented by broad conversation QA benchmarks; maintain product-specific tests.

## Low-cost mechanisms to borrow

1. Keep append-only raw evidence and separate derived/normalized entities so extraction never destroys source text.
2. Store provenance, observed time, valid time, and status on each fact; make supersession a new event and preserve the old row.
3. Route exact identifiers through FTS/trigram or a dedicated exact matcher and combine with dense search for paraphrases.
4. Return a small, capped, source-rich context with fact IDs, status, dates, and an explicit weak-evidence signal.
5. Separate fast online recall from slower offline consolidation; run consolidation only after an auditable policy exists.
6. Treat entities/graphs as an optional typed layer that augments, rather than replaces, source passages.
7. Give every CLI one stable MCP tool contract and test actual startup and cross-client recall before rollout.
8. Track memory token budget, p50/p95 latency, process RSS, stale leaks, exact identifier accuracy, and abstention precision.
9. Keep private memory local, bounded, inspectable, and exportable; never use a benchmark score as a privacy guarantee.

## Adopt / test / reject

- **ADOPT:** explicit lifecycle/provenance, hybrid search, bounded context, tool-based access, and frozen per-fact history.
- **TEST:** deterministic consolidation, query-type routing, compact structured attributes beside original text, small-model rerank/consistency checks, and optional temporal graph expansion on the existing dev set.
- **REJECT:** deploying vendor-reported leaderboards as a selection rule; replacing raw facts with LLM summaries; automatic deletion without audit; hosted memory in the local-only design; training an RL memory controller during this 24-hour build.

## Sources

- [Mem0 paper](https://arxiv.org/abs/2504.19413) and [official evaluation page](https://github.com/mem0ai/mem0/blob/main/docs/core-concepts/memory-evaluation.mdx)
- [Zep paper](https://arxiv.org/abs/2501.13956), [current research page](https://www.getzep.com/research/), and [Graphiti source](https://github.com/getzep/graphiti)
- [MemGPT paper](https://arxiv.org/abs/2310.08560) and [Letta repository](https://github.com/letta-ai/letta)
- [Cognee evaluation results](https://www.cognee.ai/research-and-evaluation-results)
- [A-MEM paper](https://arxiv.org/abs/2502.12110) and [repository](https://github.com/agiresearch/a-mem)
- [LightMem ACL 2026 paper](https://aclanthology.org/2026.acl-long.588/)
- [Memori paper](https://arxiv.org/abs/2603.19935)
- [AgeMem paper](https://arxiv.org/abs/2601.01885)
- [Supermemory API v5](https://supermemory.ai/changelog/supermemory-api-v5/) and [docs](https://docs.supermemory.ai/memory-api/introduction)
- [Anthropic Claude Code memory docs](https://docs.anthropic.com/en/docs/claude-code/memory)
- [OpenAI Codex AGENTS.md behavior](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.3-codex)
