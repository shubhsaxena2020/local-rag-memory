# R10 — Acceleration on this Core Ultra 5 225H laptop

**Prepared:** 2026-10-19 from vendor specifications, runtime documentation, and Round 2/3 local evidence. This is a research and test plan, not proof that an accelerated path is faster.

## Decision summary

No acceleration backend is adopted. Round 2 measured Ollama `embeddinggemma` on this machine, but there is no controlled CPU/GPU/NPU comparison, no per-provider execution trace, and no isolated server-plus-embedder RAM sample. The current result cannot support a claim that the NPU or iGPU is faster.

The most direct experiment is a pinned ONNX model through ONNX Runtime's OpenVINO execution provider on CPU, GPU, and NPU. Test DirectML as a separate DirectX 12 GPU path. Keep the current Ollama path as control and verify which device actually executed each graph. A provider name or package installation is not device-placement evidence.

The plan's adoption threshold is at least 2x measured latency improvement with non-inferior strict retrieval quality, stable repeat runs, and RAM/temperature within limits. It must also preserve embedding format and migration correctness.

## Hardware and software facts

- Intel lists the Core Ultra 5 225H at 14 CPU cores and 83 peak aggregate INT8 TOPS; the aggregate includes distinct compute engines.
- Intel's spec page lists Arc 130T graphics at 63 TOPS and Intel AI Boost NPU at 13 TOPS. These figures describe peak vendor throughput, not sentence-embedding latency or sustained thermally constrained performance.
- The CPU, iGPU, and NPU share a mobile power/thermal envelope. A sustained local benchmark can change clocks and invalidate a short comparison; record temperature and power context where available.
- The OpenVINO Execution Provider documentation describes CPU, GPU, and NPU device options and Windows installation via `onnxruntime-openvino`.
- OpenVINO can partition a model across supported/fallback devices. A run labeled `NPU` may still execute unsupported nodes elsewhere; collect provider placement and fallback diagnostics.
- ONNX Runtime DirectML is a DirectX 12 device provider. An Intel iGPU may be available through DirectML, but DirectML use alone does not establish NPU execution.
- ONNX Runtime's generic provider list includes CPU, DirectML, OpenVINO, and other providers; runtime availability is distinct from operator coverage for a specific exported model.
- The Ollama embedding API exposes `keep_alive`; the Round 3 MCP request uses `30m`. The API contract supports the field, but the running server's observed model residency and unload behavior still require local verification.
- The current source implementation uses loopback-only Ollama `/api/embed`, a 10 second MCP client timeout, EmbeddingGemma's asymmetric query/document formatting, and a 30-minute keep-alive. This source edit is unverified pending the watchdog and tests.

## Candidate paths

| Path | Why test it | Main risks / evidence required |
|---|---|---|
| Ollama EmbeddingGemma control | Existing measured baseline; simple lifecycle; currently wired in MCP code | Existing latency p95 was contaminated by a concurrent Qwen service and did not meet 20 warmup / 500 query protocol; separate model/server RSS is missing |
| ONNX Runtime + OpenVINO CPU | In-process call can avoid local HTTP and model-server overhead; Windows wheel documented | Conversion/export fidelity, thread tuning, model/tokenizer version pinning, startup cost, matched output normalization |
| ONNX Runtime + OpenVINO GPU | May use Arc iGPU; OpenVINO provider explicitly exposes GPU target | GPU placement must be shown; shared memory and CPU fallback; graphics activity may be invisible in total CPU numbers |
| ONNX Runtime + OpenVINO NPU | Direct Intel NPU path is documented | Operator support/fallback, graph partitioning, NPU compilation/cache startup, sustained rather than peak performance |
| ONNX Runtime + DirectML | Windows DirectX 12 route and separate comparison point | GPU provider only; not an NPU test; support and operator fallback need model-specific validation |
| FastEmbed/ONNX in-process | Simple CPU-oriented packaging for compact text encoders | Check model support, license, prompt behavior, dependency size, and actual Windows build; do not infer provider from package name |

## Available model candidates

- EmbeddingGemma 300M is the exact control already used by the app; its card provides correct retrieval prefixes and Matryoshka output sizes.
- Nomic Embed Text v1.5 (137M) and Qwen3-Embedding-0.6B are candidates present in prior local comparisons; both require their official retrieval formatting.
- BGE-M3 is around 0.6B and multilingual, but its dense/sparse/multi-vector options have materially different runtime and memory costs. Compare only the specific dense or hybrid path actually implemented.
- mxbai-embed-large-v1 supports ONNX and OpenVINO artifacts and Matryoshka/binary options, but its published aggregate English MTEB results do not establish a win on this corpus.
- Very compact or newly published encoders may be added after license, artifact, Windows operator, and local compatibility checks. Do not download them while the existing user workload is saturating Ollama or while the watchdog is paused.

## Local evidence and limitations

- The Round 2 selection artifact ranks EmbeddingGemma first by strict Recall@5 then MRR: `.497` / `.2939`, p50 `172.32 ms`, total recorded RAM `593.52 MB` in that benchmark. Nomic was faster and smaller but lower quality; BGE-M3 used more than 1.1 GB; Qwen3 had the slowest p50. See `evidence/r2-embedding-quality-selection.json` and its source text.
- The separate Round 3 live retrieval run reports source-context quality Recall@1 `.2545`, Recall@5 `.7091`, MRR `.4147`; its prior p95 retrieval measurement was `3194.57 ms`, almost all of it embedding inference. It used one warmup and 165 queries, so it does not meet the preregistered latency protocol.
- The old benchmark's 10,232.61 MB peak combined Python, Ollama, and multiple resident model runners; it did not isolate MCP, embedder, answerer, and harness. Do not compare it directly to the `<2 GB` whole memory subsystem gate.
- At this checkpoint, the local safety watchdog is paused and the user's existing VaultRAGAgent repeatedly invokes Qwen through Ollama. That workload is outside this project's authority and remains untouched. No local benchmarks or Ollama calls are appropriate during this contention.
- Current local acceleration status is **unmeasured**. No OpenVINO, DirectML, NPU, or in-process latency result is claimed.

## Ready-to-run experiment

1. **Freeze inputs:** use the same strict 165-query set, same facts and relevance labels, the same current production chunking, same BM25 fusion, and the same answerer-independent retrieval code for every candidate.
2. **Register first:** write the variant, device, model hash, export/conversion versions, dimension, quantization, formatter, batch size, threads, warm-ups, sample size, and pass rule in `EXPERIMENTS.md` before execution.
3. **Quiet gate:** start only when watchdog state is RUN, free RAM exceeds 12 GB, the temperature history is stable, and concurrent local inference is absent. If another user service keeps Ollama busy, do not stop it; defer the local experiment.
4. **Control sequence:** record `ollama ps`, process list, free RAM, CPU, and thermal samples; make at least 20 warm-up calls; collect at least 500 per-query timed requests per path. Cold start is a separate measurement.
5. **Single path:** run one Ollama or ONNX backend at a time; avoid parallel models. Cap client timeouts at 60 seconds, unload only a model loaded for this experiment, and check `ollama ps`/runner CPU immediately after.
6. **Device proof:** collect OpenVINO execution-provider diagnostics for every graph/node and explicit device selection; record fallback counts and whether all operators ran on the requested device. Do the same for DirectML GPU execution.
7. **Measurements:** report raw timings; embedding-only and end-to-end p50/p95/p99; cold-start; requests/sec; errors; paired Recall@1/5 and MRR; process RSS/working set for MCP, Python/runtime, Ollama server/model separately; model artifact size; thread/handle counts; thermal samples.
8. **Statistics:** compute bootstrap 95% CIs using 10,000 resamples. Compare candidate/control with paired query bootstrap. Treat no overlapping confidence interval or statistical significance alone as insufficient unless the predeclared non-inferiority margin is met.
9. **Decision:** adopt only if measured speed improves by >=2x, quality is non-inferior under the predeclared paired rule, system RAM stays below 2 GB, there are zero wrong-dimension/errors, and two independent quiet runs are stable.
10. **Cooling and recovery:** stop after 40 minutes or if the watchdog reaches its pause/emergency boundary; cool at least five API-only minutes before another heavy local run. Preserve all evidence and make no device/config change if any gate fails.

## Adopt / test / reject

- **ADOPT:** use the existing Ollama/Gemma route as the control only; preserve model+formatter identity in any vector schema and keep lexical matching for identifiers.
- **TEST:** OpenVINO CPU/GPU/NPU, DirectML GPU, and a supported in-process model under the exact matched protocol. Prioritize only after watchdog RUN and a genuinely quiet local window.
- **REJECT:** inferring model performance from Intel TOPS; calling DirectML an NPU result; accepting partial NPU placement without disclosure; using the old unisolated RAM or one-warmup latency numbers as a gate pass; installing or loading candidates while the machine is hot.

## Sources

- [Intel Core Ultra 5 225H specifications](https://www.intel.com/content/www/us/en/products/sku/241749/intel-core-ultra-5-processor-225h-18m-cache-up-to-4-90-ghz/specifications.html)
- [ONNX Runtime OpenVINO Execution Provider](https://onnxruntime.ai/docs/reference/execution-providers/OpenVINO-ExecutionProvider.html)
- [ONNX Runtime DirectML Execution Provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/)
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [EmbeddingGemma model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card)
- [mxbai-embed-large-v1 model card](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
- Project evidence: `evidence/r2-embedding-quality-selection.json`, `evidence/r3-live-retrieval-benchmark-pass2.json`, `EXPERIMENTS.md`.
