# Adaptive Agentic RAG — Day 1 Integration Baseline (v0.1.27 source)

> **Day 1 verified on 2026-11-22:** 156 backend tests passed, a genuine 20-page PDF ingestion/search/citation flow passed, three HTTP RAG modes passed with a **mock LLM**, and `tsc --noEmit` passed. **Real Ollama/GPU inference and Windows Next production build still require host-side verification.** See [DAY1_ACCEPTANCE.md](DAY1_ACCEPTANCE.md).

**CSIT6000V course project · research-paper learning assistant for beginning university/graduate students**

> **Scope of this release:** Local application implementation only. No TACC configuration, cluster scheduler, TACC SSH tunnel, or remote GPU deployment. LLM inference is through a **self-hosted model endpoint** (Ollama native `/api/chat` or OpenAI-compatible `/v1/chat/completions` for a future vLLM server), not a paid third-party API. The demo dataset is synthetic; it is **not** QASPER.

## 1. Architecture

```text
Next.js + React-PDF frontend (port 3000)
    │ REST / SSE
    ▼
FastAPI application backend (port 8000)
    ├── Papers / Conversations / Run management
    ├── Knowledge service
    │   ├── QASPER JSON ───────────────────┐
    │   ├── PDF ── Docling (PyMuPDF backup)├─► structure-aware chunks
    │   ├── BM25 / Dense (SBERT + FAISS)   │   + page/paragraph references
    │   └── Hybrid RRF + optional reranker ◄┘
    ├── Agent service (LangGraph)
    │   ├── Basic / Hybrid / Adaptive modes
    │   ├── Evidence checks / controlled retries
    │   ├── Beginner / Comparison / Reading Skills
    │   ├── Optional OpenAlex literature metadata
    │   └── ModelClient ─► Ollama native /api/chat or OpenAI-compatible model service
    ├── SQLite: papers, chunks, conversations, messages, runs, events
    ├── Local rotating log: data/logs/app.log (run_id / paper_id)
    ├── User-confirmed learning memory
    └── QASPER benchmark (only this module loads gold annotations)

AI service boundary (self-hosted Ollama/vLLM) ─► local GPU or CPU
```

The L2 application code does **not** import CUDA or know the server GPU allocation. The `ModelClient` is the swap point for later GPU serving. External academic search is opt-in for interactive Adaptive RAG and forcibly off in benchmarking.

## 2. Development environment

Recommended: Python 3.11 or 3.12, Node.js 20/22, npm; Ollama for local inference. First-time dependency/model downloads require Internet access. Use one backend worker; this release's background tasks are **in-process**, not a distributed job queue.

### Backend (PowerShell / WSL / macOS / Linux)

```bash
cd backend
python -m venv .venv
# Linux/WSL/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-ai.txt
# Optional, larger download: structured PDF parsing via Docling
pip install -r requirements-pdf.txt
cp .env.example .env     # Windows PowerShell: Copy-Item .env.example .env
uvicorn app.main:app --reload --port 8000 --no-access-log
```

`.env` is an example for your shell/environment; `python-dotenv` isn't loaded automatically in this release. **Export any edited variables** in the shell, or start uvicorn with `--env-file .env` (requires `python-dotenv` / `uvicorn[standard]`). For a quick test, the built-in defaults already work with Ollama's usual port. If you prefer custom values:

```powershell
$env:MODEL_BASE_URL="http://127.0.0.1:11434/v1"
$env:MODEL_NAME="qwen3:4b"
$env:APP_DATA_DIR="./data"
uvicorn app.main:app --reload --port 8000 --no-access-log
```

Swagger API docs: <http://127.0.0.1:8000/docs>. Backend health: <http://127.0.0.1:8000/api/health>.

The **minimal** `pip install -r requirements.txt` variant supports PDF/QASPER import, BM25, and API testing; it does **not** support Dense/Hybrid/Adaptive RAG until `requirements-ai.txt` is installed and the local embedding weights are available. There is deliberately no fake dense/LLM fallback for production requests.

### Local model (no TACC)

Install Ollama on your local workstation and run:

```bash
ollama pull qwen3:4b
ollama serve
```

If Ollama is already running as a background service, `ollama serve` may report that the port is already in use — leave the existing service running. `MODEL_BASE_URL` defaults to `http://127.0.0.1:11434/v1`, and `MODEL_NAME` defaults to `qwen3:4b`.

**v0.1.4 model fix:** The local Ollama client automatically uses Ollama's *native* `POST /api/chat` with `think:false`, instead of OpenAI-compatible streaming. This avoids Qwen3 filling the output budget with reasoning and leaving `content` empty. You can force it with `MODEL_PROVIDER=ollama` (PowerShell: `$env:MODEL_PROVIDER="ollama"`) when Ollama is on a non-standard port. Native mode also checks that `MODEL_NAME` appears in `/api/tags`; the green status dot does not prove a successful generation. Run the diagnostic below to test that.

For a self-hosted vLLM server instead, set `MODEL_PROVIDER=openai`, `MODEL_BASE_URL=http://127.0.0.1:8001/v1` and `MODEL_NAME` to the **exact served model ID**; start the vLLM process yourself. For models requiring disabling reasoning via the compatible endpoint, `MODEL_REASONING_EFFORT=none` is available, provided your serving engine supports it. The backend does not start or download LLMs.

For Dense retrieval, `sentence-transformers` downloads `all-MiniLM-L6-v2` on first use (or reads a cached model). The reranker is optional (`ENABLE_RERANK=true`) and needs `cross-encoder/ms-marco-MiniLM-L-6-v2`. All document/query embedding computations currently run on CPU for predictability; this can be changed in the independent encoder adapter.

### Frontend

In a separate terminal:

```bash
cd frontend
npm install
# Optional: copy .env.example to .env.local
npm run dev
```

Open <http://localhost:3000>. Frontend accesses <http://127.0.0.1:8000> by default; customize `NEXT_PUBLIC_API_BASE_URL` if needed.

## 3. Test without downloading real QASPER

From the **backend** folder:

```bash
python -m app.cli ../examples/qasper-mini.json --split dev
```

This imports one **fictional** paper and a fictional QA label only to check ingestion and citation mapping. The paper appears in the UI as QASPER text (not a PDF). To test retrieval without an LLM:

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"BM25 reciprocal rank fusion","paper_ids":["qasper-demo-paper-001"],"strategy":"keyword"}'
```

For Windows PowerShell, use `Invoke-RestMethod` instead of the `curl` syntax above.

## 4. Import the official QASPER dataset

Official QASPER v0.3 release (not bundled):

- Dataset: <https://huggingface.co/datasets/allenai/qasper>
- Training/development archive: <https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz>
- Original JSON filenames include `qasper-train-v0.3.json` and `qasper-dev-v0.3.json`.

Download and extract the archive **outside** of Git, then run from `backend/`:

```bash
python -m app.cli path/to/qasper-dev-v0.3.json --split dev --limit 10
```

Remove `--limit 10` after basic testing. Data import is idempotent for previously imported papers. The QASPER JSON is parsed directly; no artificial PDF conversion is performed. If you later import a test split, use `--split test` and keep it separate.

**Leakage prevention:** corpus stored in `chunks` contains only abstract/full text; `question`, `answers`, and gold `evidence` are stored separately in `eval_cases`, which is never used by the runtime retrieval and Agent modules. Original QASPER paragraph refs such as `s4-p2` are kept for evidence checking; PDF page numbers come only from parsers. Some annotation strings cannot map exactly to paragraphs; such cases keep missing gold refs, and the evaluator records missing mapping rather than inventing a negative label.

## 5. API highlights

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/papers` | Upload PDF, asynchronously parse/index |
| GET | `/api/papers` | List papers and processing status |
| GET | `/api/papers/{id}/document` | Display uploaded PDF |
| GET | `/api/papers/{id}/chunks` | Inspect structured paper text |
| DELETE | `/api/papers/{id}` | Delete paper, chunks and cached indexes |
| POST | `/api/search` | Keyword / Dense / Hybrid retrieval |
| POST | `/api/conversations` | Create conversation |
| GET | `/api/conversations/{id}` | Conversation history |
| POST | `/api/agent/runs` | Start an AgentRun (returns run ID) |
| GET | `/api/agent/runs/{id}` | Poll final state, answer and metrics |
| GET | `/api/agent/runs/{id}/events` | SSE event replay and live stream |
| POST | `/api/agent/runs/{id}/cancel` | Request cancellation |
| GET/PUT | `/api/learning/profile` | Beginner level, language, goals |
| GET/POST/PATCH/DELETE | `/api/learning/memories/...` | Explicitly confirmed learning memory |
| GET | `/api/health` | API/model availability and provider |
| GET | `/api/debug/runs` | Latest 15 run statuses and errors (local demo only) |
| GET | `/api/agent/runs/{id}/debug` | Run metadata + persisted events + log path |

Example Agent run JSON:

```json
{
  "conversation_id":"<create-conversation-first>",
  "question":"Explain the main method in simple language",
  "paper_ids":["qasper-demo-paper-001"],
  "mode":"adaptive",
  "teaching":true,
  "use_memory":true,
  "allow_web":false
}
```

Events: `run_started`, `step_started`, `tool_completed`, `answer_delta`, `run_completed`, `run_failed`, and `stream_end`. Every SSE event has a persisted numeric ID; reconnect using `Last-Event-ID` or `?after=<id>`. A disconnected browser does not automatically cancel the AgentRun.

## 6. Evaluation

Import a **real** QASPER development split first; then:

```bash
python -m app.benchmark --split dev --limit 10 \
  --modes basic hybrid adaptive --output results/qasper-dev.json
```

The harness uses the same paper corpus, question set, model endpoint, and response generator for all three modes. Teaching skills, long-term memory and academic Web search are disabled **by construction** in evaluation mode. Output is per-question JSON with answer token-F1 proxy, retrieved paragraph evidence precision/recall/F1 where mapping is available, latency and model/tool/retrieval call counts. These scores are **not claimed to exactly reproduce official QASPER evaluator results**; an official evaluator comparison remains to be implemented. Adaptive mode can retrieve more chunks over multiple rounds, so compare both quality **and** retrieval cost; do not call its cumulative evidence recall a fixed-budget Recall@K.

## 7. Local tests

```bash
cd backend
pip install -r requirements-dev.txt
PYTHONPATH=. pytest -q      # Linux / WSL / macOS
# Windows PowerShell: python -m pytest -q
python -m compileall -q app
```

Tests use a temporary SQLite database and a fictional QASPER record. A fake local model double is used only **inside tests**, never for real user answers.

## 8. Known limitations / next milestones

1. **Not yet end-to-end verified with a real GPU Qwen endpoint:** code supports the interface, but `vLLM`, Ollama and model weights were not started during package creation. Check local model health and model streaming before relying on answers.
2. **Not verified as a built Next.js bundle in the offline authoring environment:** npm packages were unavailable; run `npm install && npm run build` on a networked machine and fix integration issues as they arise.
3. **LangGraph short-term checkpoints are in-memory** in this release. SQLite persists business records, run events, answers and explicit long-term memories, but not resumable LangGraph node state. Restart fails interrupted runs rather than silently restoring model execution.
4. **Background jobs are in-process:** a single uvicorn worker is required. For deployment, migrate to Redis/Celery/Arq (or another persistent queue) and add database-level dispatch/leases.
5. **Security:** this is explicitly **single-user local development**, with no authentication, rate limiting, request quotas, multi-tenant isolation or web deployment hardening. Do not expose the API to the Internet.
6. **Retrieval:** no table-image OCR or figure-grounded answers; PDF fallback can lose structural detail; evaluation gold evidence mapping is approximate when evidence is not an exact paragraph. FAISS is built from cached vectors per query, not a large-scale persistent global ANN index. Model downloads require connectivity.
7. **Skills / academic web:** current skills are prompt templates; OpenAlex supplementary metadata is not verified against papers and is disabled in evaluation; advanced source relevance grading and citation verification remain future work.
8. **Observability:** SQLite events, a local rotating console/file logger, and run-level debug endpoints are implemented. Langfuse and distributed tracing are deliberately out of scope for this demo.
9. **TACC (intentionally omitted):** future stage adds vLLM GPU deployment/configuration and verified model endpoint routing, without modifying the Agent workflow.

## 9. Key code locations

```text
backend/app/main.py                  HTTP API, runs, SSE, profiles
backend/app/db.py                    SQLite DDL and persisted events
backend/app/knowledge/ingest.py      QASPER, Docling/PyMuPDF, chunking
backend/app/knowledge/retrieval.py   BM25, dense, RRF and rerank
backend/app/agent/workflow.py        LangGraph adaptive state machine
backend/app/agent/model.py           local Ollama native + OpenAI-compatible client
backend/app/logging_setup.py          console + rotating log file
backend/app/diagnose.py               live model smoke test
backend/app/agent/skills.py          skill prompts
backend/app/evaluation.py            evaluation metric functions
backend/app/benchmark.py             QASPER batch comparison
frontend/app/page.tsx                interactive paper+chat UI
frontend/components/PdfReader.tsx   PDF.js and QASPER reader
```

## References

- [QASPER dataset](https://huggingface.co/datasets/allenai/qasper)
- [Docling document converter](https://docling-project.github.io/docling/reference/document_converter/)
- [LangGraph StateGraph documentation](https://reference.langchain.com/python/langgraph/graph/state/StateGraph)
- [React-PDF library](https://www.npmjs.com/package/react-pdf)

## Next.js 15.5 development warnings (frontend hotfix v0.1.1)

If your repository root and `frontend/` both contain a `package-lock.json`,
Next.js may choose the parent as its workspace root. `frontend/next.config.mjs`
now pins `outputFileTracingRoot` to the frontend directory. Keep the root
lockfile if it belongs to another package; don't delete files blindly.

`frontend/app/globals.css` also replaces `align-items: end` with
`align-items: flex-end` in the flex composer row, to avoid an Autoprefixer
compatibility warning. These warnings were not fatal startup errors.

After applying the patch, stop `npm run dev`, delete `frontend/.next` if a
cache warning persists, then start `npm run dev` again. A successful `Ready`
message indicates that the dev server is listening; visit the browser to
check compilation and runtime behavior.

## Browser compatibility fix (v0.1.2)

On Chromium 150+ and Edge 150+, DOM scrolling APIs (including `scrollIntoView`) can return a Promise. React `useEffect` cannot return a Promise as its cleanup value. The chat auto-scroll effect in `frontend/app/page.tsx` now uses a block body and returns `undefined`. No behavior change is intended. The PDF viewer's page-sync effect was also made explicitly return-free.

If you previously ran the frontend, replace the two changed TSX files or use this ZIP, stop the dev server, and run `npm run dev` again from `frontend/`. If Next.js retains stale compilation output, delete only `frontend/.next/` and restart. You do not need to delete `node_modules` or reinstall packages for this fix.

This patch was syntax/static inspected; full browser/Next.js build tests depend on locally installed frontend packages.

### Next.js PDF viewer compatibility

If the browser shows `Object.defineProperty called on non-object` in `pdfjs-dist/build/pdf.mjs`, see `frontend/TROUBLESHOOTING.md`. The frontend pins `react-pdf` to `10.1.0`; update an existing lockfile using `npm install react-pdf@10.1.0 --save-exact` in the `frontend` directory.


## v0.1.4 — Model response debugging and lightweight logs

The original `Model produced an empty response` generally means retrieval has already run but the model did not produce an **answer** in its streamed `content` field. Qwen3's reasoning can consume the generation budget even though the server returns HTTP 200. The local Ollama path now uses native `/api/chat` with `think:false`; other serving engines keep the compatible interface. No thinking text is exposed as the final answer.

**Quick diagnostic from `backend/`:**

```powershell
ollama list
python -m app.diagnose
# When the diagnostic succeeds, run FastAPI without noisy GET access logs:
uvicorn app.main:app --reload --port 8000 --no-access-log
```

The diagnostic sends both a non-streaming and a streaming question to the real model. A healthy `/api/health` alone is *not* sufficient. If streaming remains empty, check the model/version and `backend/data/logs/app.log` (or `$env:APP_DATA_DIR/logs/app.log` if configured).

To inspect the last run, open <http://127.0.0.1:8000/api/debug/runs>, copy its `id`, and visit `http://127.0.0.1:8000/api/agent/runs/<id>/debug`. On an Agent failure, the frontend also shows a **Run details** link. For live log monitoring in PowerShell:

```powershell
Get-Content .\data\logs\app.log -Tail 80 -Wait
```

The single `app.log` file records upload/ingestion, selected retrieval strategy, hit count, top chunk IDs, model provider/model, model response duration, output/reasoning character counts, run result and exception stack trace. Every Agent-related line includes the same `run_id` (and each paper ingestion line uses `paper_id`). Rotating logs use 5 MB per file and retain 2 older files; no ELK, queue, or additional logging service is needed. Logs do **not** contain entire user questions, prompts or model answers, only diagnostic lengths/IDs. SQLite still retains messages and SSE events for this local demo.

The UI polls `/api/papers` every 3 seconds **only while a paper is still processing**, instead of polling continuously after all papers are ready. Basic/Hybrid/Adaptive keep their same retrieval architecture. In Adaptive mode, questions containing an uppercase acronym (such as Chinese questions about `CC`) initially use hybrid retrieval, since pure English sentence embeddings may struggle with Chinese queries. This is a retrieval heuristic, not a guarantee of answer quality.

**Verification performed for this package:** 14 backend tests passed with model HTTP responses mocked (including `think:false` and failure diagnostics). Real Ollama/Qwen3 inference and a complete Next.js browser build were not available in the packaging runtime; validate with the commands above on your Windows machine. The existing `backend/data/` database, imported papers, `.env.local`, `node_modules` and virtual environment should not be overwritten when upgrading an existing installation.


## v0.1.5 fixes (local demo)

- PDF `document_path` is stored as a relative path (`pdfs/<paper_id>.pdf`) for new uploads. Existing absolute paths are resolved as before; if an old absolute path no longer exists and the PDF is present in the current `data/pdfs/` location, the backend re-links the record automatically. **If the PDF bytes were not copied, a `404` is expected: re-upload the original PDF**. SQLite `ready` means the extracted chunks are available, not that the source file is physically present.
- React-PDF now displays the backend load error in the reader instead of emitting an unhandled console error.
- Adaptive RAG evidence checks request short JSON output (`format=json` in Ollama, `response_format=json_object` in the compatible client). Invalid JSON no longer automatically forces three unnecessary retrieval rounds; a short definition question allows at most two.
- Short factual answers have a 400-token output budget; longer questions use 800 tokens. The answer prompt says not to echo instructions, source lists, or repeat the question. This reduces prompt-copying but cannot guarantee a small model always answers correctly.
- The first Dense embedding call may be much slower due to initial SentenceTransformer model initialization. Subsequent retrievals should be faster.

### Checking a source-PDF 404 in PowerShell

From the backend directory, with the backend running:

```powershell
# Substitute the paper ID shown in the browser error.
$paperId = "06c5aba283264aed82967cb66847e129"
Invoke-RestMethod "http://127.0.0.1:8000/api/papers/$paperId"
Test-Path "./data/pdfs/$paperId.pdf"
```

If the file is missing, retrieve it from the original project's `backend/data/pdfs/` directory or re-upload the PDF from the UI. **Do not delete `backend/data/app.sqlite3` while upgrading**. Do not nest the new project inside the old one when applying update-only files: copy the `adaptive-agentic-rag/` subtree into your existing project root.

Run local tests with `python -m pytest -q`. Full real-model UI testing must be done in a working local Ollama environment; these unit/API tests use a simulated LLM.


## v0.1.6 answer quality update

See [CHANGES_v0.1.6.md](CHANGES_v0.1.6.md) for the concise prompt, grounded acronym lookup, bounded retry, validated SSE output and regression test instructions.

## v0.1.6 — Answer reliability and conversation management

This local update introduces an **All Conversations** page (left navigation) for browsing all previous discussions, searching titles, reopening a thread, renaming it, and deleting it. The sidebar still shows the six most recently active conversations. The conversation API includes `PATCH /api/conversations/{id}` and `DELETE /api/conversations/{id}`. Deleting a conversation also deletes its messages, runs and run events; an in-progress run must finish or be cancelled first (HTTP 409). No database migration or external service is needed.

Adaptive RAG also handles an explicitly defined acronym directly from paper text (e.g. `Chunk Coverage (CC)` in an actual selected chunk), rather than asking a small LLM to repeatedly judge a simple full-name question. Other answers use shorter prompts with a basic quality gate and **one** bounded repair attempt if the model echoes instructions or truncates a short answer. A failure after repair is reported as an error, not stored as a successful answer. This is intentionally lightweight for the local demo. **The generated answer is checked before it is emitted, so answer SSE events are chunked output rather than live token streaming in v0.1.6.**

To upgrade, unpack the *update-only* ZIP into the running project's **parent** directory so that `adaptive-agentic-rag/` aligns with the existing code. Do not copy a whole new project directory into the existing one. Keep `backend/data`, your PDFs, SQLite database, `.env` files and `frontend/node_modules`. Restart FastAPI and Next.js. The existing conversations remain stored in SQLite and become visible on the Conversations page.

Backend checks: `cd backend; python -m pytest -q` (26 tests in the package, including conversation delete cascades and answer-quality regressions). TypeScript/TSX files have been parsed with TypeScript's compiler API, but no complete Next.js production build or real Ollama inference has been run in the packaging environment. See [CHANGES_v0.1.6.md](CHANGES_v0.1.6.md) for details.

### v0.1.7 — Beginner-friendly acronym definitions

In Adaptive Agentic RAG, a verified acronym expansion no longer forces an overly terse reply when **Beginner skill** is on. The model contributes short JSON-formatted educational explanation/example/role fields, while the original document supplies and anchors the literal expansion and clickable citation. If the model cannot provide usable educational text, the answer safely falls back to the verified expansion. Toggle **Beginner skill** off for the faster one-sentence answer. See [CHANGES_v0.1.7.md](CHANGES_v0.1.7.md).


### Source-grounded table QA (v0.1.15)

For numbered PDF tables, the Adaptive mode can use the preserved row/column structure to answer numeric questions and compute percentage changes or speedups without asking the LLM to transcribe numbers. These values are extracted from the PDF; ambiguous cells fall back to the normal workflow. To inspect an uploaded paper's table index, use `GET /api/papers/{paper_id}/tables?number=5` (omit `number` to see all tables). Older uploads are indexed on first inspection; the database is extended automatically at startup. This does not change Basic/Hybrid retrieval baselines or enable TACC. See `CHANGES_v0.1.15.md` for limitations and regression tests.

## Answer skills (v0.1.19)

Adaptive mode selects optional local Markdown-backed skills in `backend/app/skills/`:
`verifyClaims`, `explainFormula`, and the composable `explainForBeginner` style.
The Agent owns retrieval, evidence checks, model calls, and citation mapping.
The skills shape structured JSON answers; existing `answer` / `citations` JSON
remains a fallback for locally hosted small LLMs. See `CHANGES_v0.1.19.md`.
