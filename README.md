# Adaptive Agentic RAG: A Local Academic Paper Research and Learning Assistant

**CSIT6000V — Advanced AI Computing and Systems | Course Project**

**Release candidate:** v0.3.11 (local development)  
**Project scope:** A locally hosted academic paper assistant for university students and early-stage researchers. The current implementation uses a local Ollama model; TACC/vLLM deployment is proposed future work, **not** part of the implemented system.

## Overview

The system helps learners read academic papers, ask source-grounded questions, understand unfamiliar technical concepts, and retain explicitly confirmed study notes. It separates two knowledge scopes:

- **Paper QA:** The user selects one or more indexed papers. Basic RAG, Hybrid RAG, and Adaptive Agentic RAG provide different retrieval and orchestration strategies. Answers cite paper passages with page/paragraph references when available.
- **General Learning:** The user can ask a background question without selecting a paper. The assistant searches external knowledge sources and links to their actual web pages. External sources are not misrepresented as PDF citations.

The interface also supports an opt-in Beginner teaching skill, learner preferences, confirmed learning memory, run events, cancellation, and diagnostic information. **A successful run is not necessarily a factually correct answer.** Source attribution and manual verification remain important.

## Architecture

| Layer | Current components | Role |
| --- | --- | --- |
| L1 — Frontend | Next.js, React, React-PDF/PDF.js | Document reading, question entry, result rendering, reference navigation, learning settings |
| L2 — Application and Agent | FastAPI, LangGraph, SQLite, BM25, Sentence Transformers, FAISS | Paper ingestion, retrieval, agent orchestration, skills, memory, output checks, run tracking |
| L3 — AI Service | Locally hosted Ollama with `qwen3:4b`; configurable model-client boundary | Model inference through a self-hosted endpoint |
| L4 — Hardware | Local workstation, CPU/RAM, local NVIDIA GPU | Compute and storage for model serving and retrieval |

Observability and evaluation span these layers. See [Architecture](docs/ARCHITECTURE.md) for the detailed flow and trust boundaries.

## Implemented capabilities

| Capability | Current behavior |
| --- | --- |
| PDF ingestion | Upload, asynchronous parsing/indexing, paper status, and reader |
| QASPER ingestion | Import original QASPER JSON; retain paragraph references independently from evaluation gold |
| Basic RAG | Single-round dense retrieval |
| Hybrid RAG | Single-round lexical/dense fusion |
| Adaptive Agentic RAG | Bounded retrieval planning, evidence assessment, answer construction, controlled recovery |
| Beginner learning | Skill-based explanations and illustrative examples; examples must not be passed off as reported study data |
| Learning memory | User-confirmed notes, edit/delete, opt-in retrieval, learning preferences |
| Academic and general web | Supplemental academic sources for relevant Paper QA tasks; external background search for General Learning |
| Citations | PDF excerpt navigation for paper evidence; separate external URL references |
| Diagnostics | Run IDs, event history, response status, model/retrieval timing and validation metadata |

The Skills are file-backed task instructions and answer contracts, **not** independent agents or separate LLMs.

## Local requirements

- Windows 11 + PowerShell (tested by the developer); equivalent Linux/macOS setups may require environment adjustments.
- Python 3.11/3.12, Node.js 20/22, and npm.
- [Ollama](https://ollama.com/) serving `qwen3:4b` locally.
- An internet connection **for first-time dependency/model downloads and external knowledge search**. Paper-only QA can work with local indexed content and cached dependencies.
- Sufficient CPU/RAM, disk space, and local model-serving resources. The system is a **single-user local demo**, not a public production service.

### 1. Set up the backend

From the repository root in PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-ai.txt
python -m pip install -r requirements-dev.txt
# Optional: higher-fidelity PDF parsing with Docling
python -m pip install -r requirements-pdf.txt
Copy-Item .env.example .env
```

The application does not automatically read `.env` in every launch configuration. Set variables in the current shell or run Uvicorn with `--env-file` after installing the necessary dotenv support:

```powershell
$env:MODEL_PROVIDER = "ollama"
$env:MODEL_BASE_URL = "http://127.0.0.1:11434/v1"
$env:MODEL_NAME = "qwen3:4b"
$env:ENABLE_RERANK = "false"
uvicorn app.main:app --reload --port 8000 --no-access-log
```

Backend health: `http://127.0.0.1:8000/api/health`  
Interactive API documentation: `http://127.0.0.1:8000/docs`

### 2. Start the local model

Install Ollama, then in another terminal:

```powershell
ollama pull qwen3:4b
ollama list
```

Ensure the Ollama service is running. On systems where Ollama has not been started as a service, run `ollama serve`. The project invokes a **local endpoint** and does not require paid LLM API access. From `backend/`, run `python -m app.diagnose` for a model-level smoke test.

### 3. Set up the frontend

In another terminal, from the repository root:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3000`. The default backend URL is `http://127.0.0.1:8000`; see `frontend/.env.example` for configuration.

### 4. Run tests and production-build checks

Backend:

```powershell
cd backend
python -m pytest -q -k "not test_persistent_http_500_is_not_hidden and not test_http_400_not_retried"
python -m compileall -q app
```

Frontend:

```powershell
cd frontend
npx tsc --noEmit
npm run build
```

**Verification status:** The v0.3.11 development workspace previously passed 315 backend tests with two older model-error-message tests excluded. Real local-browser runs confirmed multiple Paper QA and General Learning scenarios. **The latest complete Windows production build and a clean-checkout reinstallation still need to be recorded for the final submission.** See [Release Checklist](docs/SUBMISSION_CHECKLIST.md).

## Quick demonstration

1. Start Ollama, FastAPI, and Next.js.
2. Upload a research PDF, wait until indexing reports ready, and select it under **Paper QA**.
3. Ask a paper-specific question in **Basic**, **Hybrid**, and **Adaptive** modes, then open a PDF citation.
4. Enable **Beginner skill** and request an explanation with an explicitly hypothetical example.
5. Switch to **General Learning**, deselect all papers, and ask `What is machine learning?` Open the external source link.
6. Demonstrate confirmed learning memory and inspect a run's event history.

The scripted demo, acceptance evidence, and fallback steps are in [Demo Guide](docs/DEMO_GUIDE.md).

## Project documentation

- [Architecture and implementation](docs/ARCHITECTURE.md)
- [Evaluation and evidence](docs/EVALUATION.md)
- [Final report draft](docs/FINAL_REPORT_DRAFT.md)
- [Demo guide](docs/DEMO_GUIDE.md)
- [Submission and release checklist](docs/SUBMISSION_CHECKLIST.md)

## Evaluation summary

We evaluated the three retrieval modes on a fixed QASPER v0.3 development subset and a separate six-question, two-paper holdout. We also used an eight-question paper-specific regression set during development. The **holdout does not establish that Adaptive outperforms Hybrid**; the higher agent-call cost and observed answer errors are reported explicitly. Token-overlap F1 is a project-specific proxy, not the official QASPER leaderboard score. See [Evaluation](docs/EVALUATION.md).

## Known limitations and safety boundaries

- The small local model can give an incorrect or incomplete answer even if source retrieval succeeds; validation and conservative fallback reduce but do not eliminate this risk.
- The system is intended for a single-user local course demonstration. It has no production authentication, persistent distributed queue, or multi-tenant security controls. **Do not expose the API directly to the public Internet.**
- PDF figure/OCR grounding and complex table extraction are limited; QASPER text imports do not have native PDF page locations.
- Embedding initialization may be slow on a cold start, particularly when checking or downloading cached weights.
- Citation links identify sources but are not a guarantee that every generated proposition is supported by those sources.
- Actual TACC deployment, GPU cluster orchestration, vLLM migration, and large-scale production serving are **future work**.

## External resources

- [QASPER dataset](https://huggingface.co/datasets/allenai/qasper)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Ollama](https://ollama.com/)
- [Docling](https://docling-project.github.io/docling/)
- [React-PDF](https://github.com/wojtekmaj/react-pdf)

**Academic integrity:** Only documented measurements and verifiable local runs are presented as results. Gold annotations are used for evaluation and are not injected into the Agent's answer prompts.
