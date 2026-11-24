# Phase 2 — Formal QASPER v0.3 Evaluation Runbook

## Purpose and frozen design

Evaluate the unchanged local **v0.3.18-rc1** Paper Tutor implementation on unseen QASPER v0.3 development questions. Compare **Basic RAG**, **Hybrid RAG**, and **Adaptive Agentic RAG** under the same model and paper corpus. No TACC work, external web, beginner-skill enhancement, or learning memory is included in this experiment.

The standalone experiment driver **does not modify runtime application modules**. Place it in `artifact_evaluation/` of the source repository and commit the experiment harness before generating any official results. Keep downloaded datasets and run output outside the Git repository when possible.

## Key methodological controls

1. **Source**: official `qasper-train-dev-v0.3.tgz`, checked against the original `day2_qasper.py`'s pinned SHA-256. If using a locally extracted dev JSON, explicitly disclose that the archive itself was not reverified by the harness.
2. **Sampling**: fixed seed **62026**, **4 papers × 5 questions = 20 questions**. Exclude every paper containing any of the **21 previously evaluated question IDs** collected from historical Trial 1, Trial 2, and Holdout 6001 logs. This prevents reusing those papers even when different questions were selected. Both papers and questions are chosen with stable hashes. Each paper must yield at least two mappable gold evidence questions among the five selected cases; this conditional filter affects representativeness.
3. **Mode pairing**: every question is asked in all three modes; 60 observations. Mode order rotates by question to reduce systematic cache-order bias; all runs execute sequentially.
4. **LLM**: `qwen3:4b` on local Ollama, temperature 0 by default. Existing v0.3.18-rc1 repeat-error retry behavior remains unchanged. Log the Ollama version and GPU state separately.
5. **Retrieval**: Reranking OFF. Basic and Hybrid use their existing one-round policies; Adaptive may use extra rounds and extra calls. These are distinct *full pipeline configurations*, not equal-budget retrieval algorithms. No equal-retrieval-budget claim should be made.
6. **Learning controls**: `teaching=False`, `use_memory=False`, `allow_web=False`, `ENABLE_RERANK=false`; isolated `APP_DATA_DIR`, empty memories, English profile. If the logs reveal unintended memory/web use, do not publish the run as a valid controlled comparison.
7. **No test-driven tuning**: do not inspect selected gold questions to alter prompts/retrieval until after the official experiment and human review are complete. Any change in application source starts a **new version and new experiment**; do not blend incompatible runs.

## Environment and source preflight

From the **repository root** in the existing Python `Project` environment:

```powershell
python .\scripts\verify_release.py
cd backend
python -m compileall -q app
python -m pytest -q
python -m app.diagnose
cd ..
git diff --check
git status --short
```

The original RC offline regression target was **379 passed**. New evaluation-only files do not affect the runtime. Resolve tracked changes and **commit** before formal inference. If `git status` shows local runtime outputs, keep those excluded/untracked; never commit model artifacts or a local SQLite database.

Record the local GPU and server configuration in a separate plain-text terminal transcript:

```powershell
ollama --version
ollama ps
nvidia-smi
python --version
```

## Prepare (run once, never change the seed after seeing output)

Recommended: keep datasets/runs in the sibling `Day2` directory, not in the repository.

```powershell
$eval = ".\artifact_evaluation\formal_qasper.py"
$dataset = "..\Day2\formal_qasper_62026\dataset"
$runs = "..\Day2\formal_qasper_62026\runs"

# If an official archive has already been downloaded:
python $eval prepare --archive "..\Day2\dataset\qasper-train-dev-v0.3.tgz" --out $dataset

# OR omit --archive to download and SHA-verify the official QASPER archive:
# python $eval prepare --out $dataset

python $eval preflight --dataset $dataset --live --model qwen3:4b
```

Verify `selection_manifest.json` reports **4 / 20**, lists historically excluded papers, includes a pinned hash for `qasper_selected_dev.json`, and cites `verified_official_archive` where available. Selection is a fixed, development-set **conditional sample**. Do not describe it as unbiased random sampling of the full QASPER corpus.

## Smoke test and the actual official trial

```powershell
# Smoke tests are a separate trial, not included in the official report.
python $eval run --dataset $dataset --out $runs --trial smoke --max-runs 3
python $eval report --dataset $dataset --run-dir "$runs\smoke"

# The official trial must run all 60 observations.
python $eval run --dataset $dataset --out $runs --trial official
python $eval report --dataset $dataset --run-dir "$runs\official"
```

The benchmark creates an **isolated SQLite DB** in each trial's `runtime/` directory and appends **one JSON line per completed observation** to `runs.jsonl`. A restart with **the identical `run` command** resumes missing question/mode pairs while protecting the dataset, model name, code fingerprint, Git commit and evaluation driver hash. Do not delete or manually edit the ledger. `run` can produce failed or partial results: retain them as outcomes rather than silently rerunning only failures.

A complete official trial must show **60/60** unique question/mode observations. If an interrupted trial has fewer, run the same command again; the report marks incomplete output as ineligible for final publication.

## Automatically generated output (English)

| File | Purpose |
|---|---|
| `dataset/selection_manifest.json` | Sampling and provenance, excluded historical papers, SHA-256 |
| `runs/official/experiment_manifest.json` | Git, model, settings, application source hash |
| `runs/official/runs.jsonl` | Checkpointed answers, gold data, retrieval traces, citations, timing |
| `runs/official/metrics_summary.json` | Status, F1 proxies, Evidence F1, citation counts, latency P50/P95 |
| `runs/official/per_question_results.csv` | Paired question-level differences and failure inspection |
| `runs/official/formal_report.md` | Initial English experimental results and limitations |
| `runs/official/blind_review.csv` | Human review without system-mode labels |
| `runs/official/blind_review_mapping.json` | Private key to unblind the review after scoring |

### Manual review and scoring

Complete every row of `blind_review.csv` without opening the mapping file. Use integer scores **0, 1, 2** for each of three dimensions:

- **Correctness**: 0 = incorrect/unjustified; 1 = partially correct; 2 = supported and factually correct relative to the source/gold.
- **Completeness**: 0 = misses the question; 1 = covers some requirements; 2 = fully answers the requested facets.
- **Citation support**: 0 = no usable support or fabricated reference; 1 = support only for part of the answer; 2 = citations support all material factual claims. When necessary, open the full original paper rather than trusting the snippet.

For unanswerable questions, a correctly justified abstention can earn a high correctness score; an unsupported confident answer should be penalized. `succeeded` is a system status, not proof of correctness. A citation exists ≠ citation is relevant. For scores, a `partial` status is not automatically zero; review the content.

Once complete:

```powershell
python $eval finalize --run-dir "$runs\official" --review-csv "$runs\official\blind_review.csv"
```

`human_review_summary.json` is then created, and the English report receives a review section. For stronger reliability, an independent second evaluator can score the same blinded answers; report inter-rater disagreement rather than hiding it.

## Interpreting results honestly

- **Answer F1 proxy** uses the project's lowercase/ASCII token overlap and is **not the official QASPER F1**. English paraphrases and yes/no/long-form answers may be scored imperfectly.
- **Evidence F1** matches original QASPER paragraph IDs against the union of all retrieved paragraph IDs; multi-round Adaptive receives a potentially larger retrieval budget. Report retrieval-round and model-call cost alongside it.
- **Citation rate** means a citation metadata item exists, not that an answer is supported. Rely on blinded human citation checks for groundedness.
- **Latency** includes the locally hosted model, retrieval initialization, retry delays and local resource effects. Report mean, median, P95 and mean call counts; note first-run embedding initialization when interpreting.
- **Small sample**: 20 questions cannot establish reliable statistical significance or superiority. Be prepared to conclude Adaptive is more expensive and not consistently more correct if the measurements show it.
- **No PDFs are generated from QASPER JSON**: original QASPER structured papers are imported as text so that the official paragraph evidence IDs remain available.

## Completion gate for Phase 2

All 60 unique observations; complete `blind_review.csv` for all outputs; no source/dataset hash changes; no leaked memory/web calls; no missing evidence-mapping denominator; describe every failure and anomalous citation. After this gate, freeze the results and proceed to the separate Memory/Skills ablation (Phase 3).
