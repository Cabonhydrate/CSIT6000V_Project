# Evaluation: Retrieval Quality, Answer Quality, and Runtime Cost

## 1. Research questions

- **RQ1:** How do Basic, Hybrid and Adaptive strategies compare on source-grounded academic question answering?
- **RQ2:** What additional cost is incurred by multi-round Agent orchestration?
- **RQ3:** Which failure modes remain after evidence validation and controlled repair?
- **RQ4:** Can a locally hosted model support usable paper reading and external background learning without relying on commercial LLM APIs?

The experiments below answer these questions only within their limited sample sizes and configurations. They are **not** a claim of statistical significance or state-of-the-art QASPER performance.

## 2. Method and validity conditions

**Model and setting.** Local Ollama `qwen3:4b`; paper retrieval with lexical and/or dense retrieval. The evaluation material uses a fixed subset of the **QASPER v0.3 development split**. QASPER documents were imported as text/paragraphs, not artificially converted into PDFs.

**Compared configurations.** Basic uses one dense retrieval round; Hybrid uses one fused retrieval round; Adaptive can use multiple rounds and evidence-assessment calls. A previous Holdout audit found that Adaptive applied profile context even when the benchmarking flags disabled teaching and memory. A later code fix changed this behavior, but **the results reported below were not rerun under the corrected configuration**. Therefore, the early three-mode results are not a fully controlled causal comparison of retrieval strategies.

**Reported metrics.** Answer Token F1 is this project's simplified token-overlap proxy and is sensitive to Chinese/English mismatch; it is **not** the official QASPER scorer. Evidence F1 is evaluated only where gold paragraph mappings exist and does not measure citation entailment. Runtime status (`succeeded`, `partial`, `failed`) is not the same as factual answer correctness. Mean latency includes local service and retrieval overhead and can reflect cold starts.

## 3. QASPER development subset: first two trials

Three papers with five questions per paper were evaluated in three modes: **15 questions × 3 modes = 45 runs per trial**. The fixed selection and dataset-source verification are recorded in the independent benchmark workspace. The reported figures are taken from the saved metrics summary files.

### Trial 1

| Metric | Basic | Hybrid | Adaptive |
| --- | ---: | ---: | ---: |
| Successful runs / 15 | 14 | 13 | 9 |
| Answer Token F1 (including failures) | 0.127 | 0.139 | 0.059 |
| Evidence F1 (mapped cases) | 0.199 | 0.185 | 0.181 |
| Mean latency, successful runs (s) | 14.59 | 12.07 | 11.99 |
| Mean LLM calls per question | 2.00 | 2.00 | 3.67 |
| Mean retrieval rounds per question | 1.00 | 1.00 | 2.80 |

### Trial 2

| Metric | Basic | Hybrid | Adaptive |
| --- | ---: | ---: | ---: |
| Successful runs / 15 | 13 | 14 | 11 |
| Answer Token F1 (including failures) | 0.131 | 0.172 | 0.082 |
| Evidence F1 (mapped cases) | 0.199 | 0.185 | 0.180 |
| Mean latency, successful runs (s) | 13.77 | 12.18 | 12.50 |
| Mean LLM calls per question | 2.00 | 2.00 | 3.67 |
| Mean retrieval rounds per question | 1.00 | 1.00 | 2.80 |

**Interpretation.** The local model and all three orchestration modes ran on real QASPER content, but Adaptive incurred substantially more calls and did not lead the simplified Answer F1. Operational stability alone was not sufficient for good answers. The trials include runtime and response-quality problems, and should not be presented as final-post-fix system performance.

## 4. Cross-paper holdout: two unseen development papers

Two additional papers not used in the earlier three-paper selection were tested with three questions each: **6 questions × 3 modes = 18 runs**. The recorded preparation seed is `6001`.

| Metric | Basic | Hybrid | Adaptive |
| --- | ---: | ---: | ---: |
| Completed runs / 6 | 6 | 6 | 6 |
| Answer Token F1 | 0.026 | 0.104 | 0.102 |
| Evidence F1 | 0.151 | 0.167 | 0.106 |
| Mean latency (s) | 18.43 | 13.51 | 15.23 |
| Mean LLM calls | 1.83 | 2.00 | 3.50 |
| Mean retrieval rounds | 1.00 | 1.00 | 2.67 |

Representative error cases from the holdout review:

- The decoder-architecture question had a QASPER gold answer of **LSTM**, while one Adaptive answer claimed **Transformer**.
- Several answers discussed a paper generally rather than answering the requested architecture, translation tool, or precise score.
- The answer about the translation tool could correctly name **Apertium** while the evidence metric remained poor. Conversely, retrieved evidence could be present without the final answer being correct.
- Some responses contained internal analysis or unresolved reference placeholders.

**Conclusion:** The six-question holdout cannot establish a mode ranking, and no claim is made that Adaptive is superior to Hybrid. The result supports feasibility of the runtime workflow while exposing generalization and citation-quality limitations. The holdout's answer-quality review relied on the stored QASPER gold items and outputs; the original papers were not independently checked as full PDFs for every answer.

## 5. CC8: paper-specific development regression

An eight-question set covering formula definitions, algorithm explanation, tabular values, claims, supplemental experiments and unanswerable premises was repeatedly used for **development regression** on one research paper. Independent human review used five two-point dimensions, totaling 10 per question.

| Regression round | Mean human score / 10 | Successful runs | Partial runs | Failed runs |
| --- | ---: | ---: | ---: | ---: |
| Trial 1 | 6.38 | 7 | 1 | 0 |
| Trial 2 | 8.13 | 6 | 2 | 0 |
| Trial 3 | 7.50 | 6 | 2 | 0 |
| Trial 4 | 8.38 | 7 | 1 | 0 |

The improvement from 6.38 to 8.38 demonstrates progress on a **known regression set**, including better claim rebuttal and cross-passage citations. Trial 4's remaining `partial` was a cautious response to an out-of-scope question. Because the same eight questions guided multiple code changes, **the 8.38 score is not a generalization estimate** and should not be compared directly with QASPER metrics.

## 6. Live browser verification after targeted changes

Local Windows screenshots and backend logs demonstrated:

- Basic, Hybrid and Adaptive answering the CC definition with linked PDF evidence.
- Adaptive paper-to-external comparison of Chunk Coverage and code coverage with **separate PDF and external URL citations**; a source-derived recovery step was logged after two failed generated similarities.
- General Learning questions about machine learning and deep learning linking to the correct concept pages, without requiring an uploaded paper.
- Confirmed learning memory read/disable behavior, citation navigation, cancellation and user-facing error handling.
- Offline external search returning a conservative `partial` rather than inventing a web source.

These are **selected functional acceptance cases**, not a blind benchmark. The ability to recover a correct source-based response does not imply that the model generated the right response on its first attempt.

## 7. Efficiency and system trade-offs

Adaptive calls the model more frequently and conducts additional retrieval rounds, creating a quality-versus-cost trade-off. A local mixed-comparison run took approximately 43.57 seconds, including about 32.25 seconds attributed to a cold retrieval/embedding initialization, 7.18 seconds of model time, and remaining orchestration overhead. After warm-up, retrieval can be much faster. Single-run latency is not a stable benchmark of throughput or GPU efficiency.

## 8. Threats to validity

1. **Small, non-random samples:** six holdout questions and eight development questions are insufficient for broad inference.
2. **Repeated development-set exposure:** CC8 was used for error diagnosis and repair.
3. **Evaluation proxy limitations:** lexical F1 undervalues semantically correct cross-language answers and can overvalue keyword overlap.
4. **Control-variable differences:** the older holdout's Adaptive profile-context behavior differed from Basic/Hybrid. Post-fix mode parity was not re-measured on an untouched sample.
5. **Source evidence limitations:** a retrieved or linked paragraph is not a proof that each generated claim follows from it; PDF/OCR and QASPER table parsing also have gaps.
6. **Runtime environment:** model version, local cache state and warm-up can dominate single-run timings.

## 9. Reproducibility package guidance

Keep the original benchmark datasets, gold labels, run-level outputs and selection manifests **outside the public/official source ZIP unless the instructor explicitly requires them**. Retain hashes and summarized metrics in the report. Do not publish personal files, API credentials, model weights, SQLite databases or confidential uploaded PDFs. If permitted, provide selected, sanitized run traces and a small reproducible example separately.

## 10. Main finding

A local, multi-strategy paper-learning assistant is technically feasible and supports verifiable interface-level citations. **The principal remaining research challenge is grounded answer quality**, especially when a small local model must integrate retrieved evidence, respond to the correct question and reconcile multiple sources. More retrieval and more Agent steps are not automatically better.
