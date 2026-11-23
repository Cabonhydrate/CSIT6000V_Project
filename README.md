# Adaptive Agentic RAG: A Local Academic Paper Research and Learning Assistant

**CSIT6000V - Advanced AI Computing and Systems | Course Project**  
**Artifact version:** v0.3.11 local release candidate  
**Repository:** [CSIT6000V_Project](https://github.com/Cabonhydrate/CSIT6000V_Project.git)

This README is organized for artifact evaluation. **Getting Started Instructions** provide a minimal end-to-end check; **Detailed Instructions** explain the implementation, benchmark configuration, reproducibility, and limitations. The artifact is a single-user local research prototype; TACC and vLLM deployment are future work, not implemented results.

## Getting Started Instructions

### 1. What the artifact does

The assistant helps university students and beginning researchers read academic papers and learn technical concepts. It supports two knowledge scopes:

- **Paper QA:** Select an indexed PDF or imported QASPER paper, ask questions using **Basic RAG** (one dense retrieval), **Hybrid RAG** (one lexical/dense retrieval), or **Adaptive Agentic RAG** (bounded iterative retrieval and evidence assessment). Paper-grounded answers link to document passages or PDF pages where available.
- **General Learning:** Ask a basic concept question without selecting a paper. The system searches external knowledge sources and provides external URL citations, not fabricated PDF references.

Additional capabilities include an opt-in beginner-explanation skill, learner preferences and user-confirmed memory, reference navigation, run events, cancellation, and diagnostic metrics. Skills are file-backed instructions routed through the Agent; they are not independent models. A successful run status does not guarantee factual correctness.

### 2. Prerequisites and environment

The development environment is **Windows 11 with PowerShell**, Python **3.11 or 3.12**, Node.js **20 or 22**, npm, and locally installed **Ollama** with `qwen3:4b`. CPU, RAM, disk storage, and a local NVIDIA GPU are used for local inference and retrieval; GPU performance depends on the workstation. Linux/macOS instructions may require minor changes. A first-time setup requires internet access to fetch dependencies and model weights; external knowledge search also requires internet access. The application is **not suitable for direct exposure to the public internet** and has no production multi-user authentication.

The reviewer should have access to the GitHub repository and permission to download any required third-party model and dataset. Dependencies, QASPER data, and model weights are not embedded in the repository.

### 3. Install and start services (PowerShell)

Clone the repository, install the backend environment, and configure the local model:

```powershell
# In an empty work directory:
git clone https://github.com/Cabonhydrate/CSIT6000V_Project.git
cd CSIT6000V_Project

# Backend:
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-ai.txt
python -m pip install -r requirements-dev.txt
# Optional larger download for structured PDF parsing:
# python -m pip install -r requirements-pdf.txt
Copy-Item .env.example .env
$env:MODEL_PROVIDER = "ollama"
$env:MODEL_BASE_URL = "http://127.0.0.1:11434/v1"
$env:MODEL_NAME = "qwen3:4b"
$env:ENABLE_RERANK = "false"
```

Start or confirm the Ollama service in a **separate terminal**:

```powershell
ollama pull qwen3:4b
ollama list
# If Ollama is not already running as a desktop/background service:
# ollama serve
```

Start FastAPI from the `backend` directory:

```powershell
uvicorn app.main:app --reload --port 8000 --no-access-log
```

Start the frontend in **another terminal**, from the repository root:

```powershell
cd frontend
npm ci
npm run dev
```

The backend is available at <http://127.0.0.1:8000/api/health>, API docs at <http://127.0.0.1:8000/docs>, and the frontend at <http://localhost:3000>. **Expected health response:** JSON with `"status": "ok"`; model availability is reported separately. If model availability is false, start Ollama and verify `ollama list`. The `.env` file is not loaded automatically in every launch configuration; setting shell variables as above avoids ambiguity.

### 4. Minimal end-to-end example and expected behavior

From the `backend` directory, import the bundled **synthetic** QASPER-style fixture:

```powershell
python -m app.cli ..\examples\qasper-mini.json --split dev
```

The fixture describes *Synthetic Study of Adaptive Evidence Retrieval (Demonstration Only)*. It is fictional test material, **not** part of the official QASPER dataset and **not** evidence of research performance. In the web UI, select this imported paper and ask:

> What retrieval methods are combined?

**Expected behavior:** the system refers to the source passage saying that the synthetic method combines **dense similarity and BM25 using reciprocal rank fusion (RRF)**. The exact wording, retrieval scores, and answer latency may differ across model runs. Imported QASPER text uses paragraph references; it does not have the native page locations of an uploaded PDF.

For a PDF-specific check, upload a real research PDF, wait for `ready`, select **Paper QA**, ask an on-topic question, and click a PDF citation to open the cited passage. For an independent external-knowledge check, choose **General Learning** without any selected paper, ask `What is machine learning?`, and verify that its external citation opens a matching concept page rather than a PDF.

### 5. Quick automated checks

From `backend` with the virtual environment active:

```powershell
python -m app.diagnose
python -m compileall -q app
python -m pytest -q -k "not test_persistent_http_500_is_not_hidden and not test_http_400_not_retried"
```

From `frontend`:

```powershell
npx tsc --noEmit
npm run build
```

**Recorded development baseline:** 315 backend tests passed in the v0.3.11 workspace after excluding two previously known HTTP-error-message assertion tests. This is historical development evidence, not a guarantee that every clean checkout will produce the same count. The final Windows production build and clean-install test must be verified by the group before submission.

## Detailed Instructions

### 1. Architecture and implemented components

| Layer | Main components | Purpose |
| --- | --- | --- |
| L1 - Frontend | Next.js, React, React-PDF / PDF.js | PDF reader, chat, knowledge-scope and mode selection, citation navigation, learning settings |
| L2 - Application and Agent | FastAPI, LangGraph, SQLite, BM25, Sentence Transformers, FAISS | Document ingestion, chunking, retrieval, bounded agent loop, skills, memory, checks, run logging |
| L3 - AI service | Self-hosted Ollama, `qwen3:4b`, model-client abstraction | Local text generation and evidence evaluation over a model-service endpoint |
| L4 - Hardware | Local workstation CPU/RAM/storage and NVIDIA GPU | Model serving and retrieval execution |

The Paper QA pipeline ingests a PDF or original QASPER JSON, stores chunk text and source references, retrieves evidence, generates a response, and formats citations. Basic uses a single dense-retrieval round, Hybrid combines lexical and dense results, and Adaptive can plan additional retrieval rounds based on evidence sufficiency. Bounded recovery attempts seek to reject unsupported definitions, incorrect numeric claims, invalid citations, and certain malformed outputs; these checks do not eliminate hallucinations.

The **General Learning** path searches external knowledge independently from any PDF selection. For certain mixed paper/background comparison questions, Adaptive can use both paper and external sources with separate citation types. When a required external source is unavailable, a conservative `partial` result is preferable to fabricating evidence.

The memory feature stores notes explicitly confirmed by the user; it can be enabled or disabled in Adaptive mode. The system records run IDs, retrieval rounds, source identities, model calls, validation attempts, and timing. Sensitive input and private user data should not be committed to the public repository.

### 2. Test cases, reference data, and evaluation conditions

The principal quantitative evaluation uses the **QASPER v0.3 development split** (official dataset: <https://huggingface.co/datasets/allenai/qasper>). The supplied `artifact_evaluation/day2_qasper.py` script can download the official train/dev archive, **check its SHA-256**, deterministically select eligible papers/questions, import them into an isolated runtime database, run all three RAG modes against the same selected questions, and write comparison outputs. Original gold answers and gold evidence are used **only for evaluation**, not as context for the Agent.

The reported **holdout configuration** is:

| Setting | Value |
| --- | --- |
| Dataset | Official QASPER v0.3, development split |
| Selection | 2 papers x 3 questions = 6 questions; seed `6001` |
| Modes | `basic`, `hybrid`, `adaptive` |
| Total attempts | 18 (six per mode) |
| Model | Local Ollama `qwen3:4b` |
| Benchmark controls | Teaching disabled; learning memory disabled; academic web disabled; reranking disabled |
| Execution | Sequential per question/mode in a separate local data directory |
| Metrics | Simplified Answer F1, mapped Evidence F1, status, latency, LLM calls, retrieval rounds |

The selection policy requires eligible papers and favors questions with mappable gold evidence; **the six questions do not represent a statistically representative random sample of QASPER**. The project Answer F1 is a simplified token-overlap proxy, not the official QASPER leaderboard metric. Evidence F1 applies only to mappable references; Adaptive's cumulative retrieval can affect evidence metrics. Chinese/English output differences can bias token-overlap scores.

### 3. Reproduce the holdout experiment

Run these commands from the **repository root** in PowerShell. The `artifact_evaluation/day2_qasper.py` file must exist in the repository (it is provided together with this README for inclusion). On the first run, `prepare` downloads and verifies the official archive; you can also pass `--archive` with an already verified local archive.

```powershell
# Select the same six development-split questions:
python .\artifact_evaluation\day2_qasper.py prepare `
  --papers 2 --questions-per-paper 3 --seed 6001 `
  --output-dir .\artifact_evaluation\dataset\holdout_6001

# Use the same Python environment that has backend dependencies installed:
python .\artifact_evaluation\day2_qasper.py run `
  --backend .\backend `
  --dataset-dir .\artifact_evaluation\dataset\holdout_6001 `
  --output-dir .\artifact_evaluation\results\holdout_6001 `
  --trial holdout_reproduction --model qwen3:4b

# Summarize the new output (not the archived reference file):
python .\artifact_evaluation\day2_qasper.py report `
  --results .\artifact_evaluation\results\holdout_6001\qasper_dev_holdout_reproduction.json `
  --dataset-dir .\artifact_evaluation\dataset\holdout_6001 `
  --output-dir .\artifact_evaluation\results\holdout_6001
```

**Expected outputs:** `qasper_selected_dev.json` and `selection_manifest.json` after preparation; `qasper_dev_holdout_reproduction.json` after the model run; and `metrics_summary.json`, `per_question_results.csv`, and `comparison_report.md` after reporting. `run` imports into an isolated `APP_DATA_DIR`, rather than modifying the interactive demo database. The model and embeddings must be available locally. Depending on networking and cold starts, a complete run can take noticeably longer than a unit test.

For comparison against previously recorded outputs, see `artifact_evaluation/reference_results/holdout1_metrics_summary.json` and `holdout1_runs.json`. The reproduction is expected to show **directionally similar** findings, not identical model text or exact latency: local LLM sampling, environment, retrieved ordering, and software revisions can affect outcomes. The stored reference is historical and predates later answer-validation adjustments.

### 4. Main recorded results and how to interpret them

The six-question holdout completed 18/18 workflow attempts, but completion is **not** equivalent to factual accuracy:

| Metric | Basic | Hybrid | Adaptive |
| --- | ---: | ---: | ---: |
| Completed runs | 6/6 | 6/6 | 6/6 |
| Mean Answer F1 (proxy) | 0.026 | 0.104 | 0.102 |
| Mean Evidence F1 (mapped) | 0.151 | 0.167 | 0.106 |
| Mean latency | 18.43 s | 13.51 s | 15.23 s |
| Mean LLM calls | 1.83 | 2.00 | 3.50 |
| Mean retrieval rounds | 1.00 | 1.00 | 2.67 |

**Interpretation:** this small holdout does not demonstrate that Adaptive outperforms Hybrid; Adaptive incurs additional model calls. Manual review also found errors such as a wrong decoder architecture and answers that omitted requested facts despite successful execution. The simplified F1 results should not be reported as calibrated accuracy. Cold embedding initialization may dominate first-request latency; do not interpret the latency figures as an isolated model-inference benchmark.

Separately, an eight-question regression set about a Chunk Coverage paper was used repeatedly during development. The manually reviewed Trial 4 average was **8.38/10**, improved from **6.38/10** at Trial 1. These are **development-set regression scores**, not independent generalization results. Do not pool them with the QASPER holdout or infer statistically significant improvements from the eight reused questions.

### 5. Functional and failure-path evaluation

A practical evaluation can exercise these scenarios, using new conversations to avoid history contamination:

1. **Paper QA:** Import or upload a paper; compare Basic, Hybrid, and Adaptive on the same supported question and inspect the final citation. The modes differ in retrieval policy; their answers are not guaranteed to differ.
2. **Citation navigation:** Click each PDF reference and inspect the actual passage, not only its page number. A real citation can still be insufficient to support a proposition.
3. **Beginner skill and numerical examples:** Request an explanation with a clearly marked hypothetical numerical example; ensure hypothetical values are not misrepresented as paper experimental results.
4. **Memory:** Create a confirmed learning note, verify that opt-in retrieval can access it, turn memory off and verify it is excluded, then edit and delete it.
5. **General Learning:** Ask `What is machine learning?` with no selected paper. The source should be a relevant concept page, not a journal title or neighboring topic.
6. **Mixed sources:** With Adaptive and Academic Web enabled, compare the paper's Chunk Coverage with code coverage. Verify both paper PDF and external web citations, and distinguish common measurement structure from different coverage targets.
7. **Safety and error handling:** Attempt an invalid PDF upload, cancel a run, ask an unsupported paper-specific question, and temporarily disconnect an external search. Expected behavior includes a clear error, cancelled/partial status, or conservative abstention, not an invented source.

### 6. Configuration, expected variation, and limitations

- **Model:** Qwen3:4b served locally by Ollama; the backend model-client boundary could be adapted to a compatible self-hosted server, but a TACC/vLLM deployment has **not** been demonstrated.
- **Embedding and retrieval:** Sentence Transformers and a local FAISS/BM25 index; first-use model download or cache checks can add substantial cold-start time. The optional reranker is disabled in the reported benchmark.
- **Data:** QASPER's original full-text JSON is imported without inventing PDF page numbers. Uploaded PDFs may have real page references; complex tables, figures, and OCR grounding remain limited.
- **Correctness:** The local small model can hallucinate, omit a required fact, produce malformed output, or select an unrelated source. Some validation paths trigger a repair or return `partial`; neither `succeeded` nor a clickable link is a proof that every answer claim is correct.
- **Fairness:** Historical test runs may differ from current v0.3.11 behavior because validation logic was improved after initial benchmark collection. Rerun all modes against an unchanged version for a controlled new comparison.
- **Scope:** Single-user local demonstrator only; no production-grade authentication, multi-tenancy, durable distributed execution, or cloud GPU deployment.
- **Safety:** Do not expose this development server to the public internet or commit `.env`, logs, private documents, model caches, or local databases.

### 7. External resources and artifact integrity

- Official QASPER: <https://huggingface.co/datasets/allenai/qasper>
- LangGraph: <https://langchain-ai.github.io/langgraph/>
- Ollama: <https://ollama.com/>
- Docling (optional PDF parser): <https://docling-project.github.io/docling/>
- React-PDF: <https://github.com/wojtekmaj/react-pdf>
- NSDI '27 artifact evaluation guidance (used for README structure): <https://www.usenix.org/conference/nsdi27/call-for-artifacts>

The experimental script checks the official QASPER archive hash before selecting papers. Results are stored separately from retrieval context; the Agent does not receive gold annotations in its prompts. The repository's README and the README reproduced in the course artifact PDF must be **identical in content** at submission time. The evaluator should be able to clone the specific submitted Git revision, perform the minimal check, and rerun the documented metrics.
