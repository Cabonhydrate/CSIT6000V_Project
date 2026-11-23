# Adaptive Agentic RAG — 本地论文助手与 TACC 模型部署指南

> **项目现状（2026-11-23）**：本地 Ollama Qwen3:4b、Basic/Hybrid/Adaptive、PDF 上传与引用、General Learning（无论文问答）、外部 Wikipedia/OpenAlex 链接、Skills 和 Memory 已有实际浏览器运行记录。后续 P0-A v0.3.4 的离线回归为 242 通过、2 项既有错误文案测试排除。**TACC GPU Job、TACC 与本地 FastAPI 的网络连通尚未实测；本文提供的是基于课程原始包核对后的可执行操作指南。** 早期的 Day1 实现与测试历史保留在下文。

**CSIT6000V course project · research-paper learning assistant for beginning university/graduate students**

> **当前运行范围**：本地 Paper Tutor 和 General Learning 已实现；TACC 采用课程提供的 `tcloud` + vLLM 公共环境启动方案，文档已集成，**不能将尚未提交的 GPU 作业写成已验证结果**。本项目调用自部署 Ollama / vLLM，不依赖付费模型 API。下文某些历史 Day1 示例数据为 synthetic，不代表正式 QASPER。

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

**增量架构（v0.3.2–v0.3.4）**：前端可选 `Paper QA`（需 PDF）和 `General Learning`（无 PDF）；后者有外部知识搜索及外部 URL 证据、独立 `explainBackgroundConcept` Skill。本地模型服务可以通过 `ModelClient` 切换 Ollama / OpenAI-compatible vLLM。课程 TACC 运行时独立 GPU Serving + CPU Client，联网路径取决于平台许可。

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

### Local model (Ollama; TACC deployment is documented below)

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

<!-- TACC-COURSE-SDK-START -->
## TACC GPU / vLLM 部署（CSIT6000V 课程环境，逐步操作）

> **推荐路线：公共环境 `project-release-1(1).zip`**。课程正式提供 `tcloud` SDK + `tuxiv.conf` + `qwen-public-serve`/`qwen-public-client`，无需自行安装 GPU 驱动或编译 vLLM。  
> **详细版（包括私有 vLLM 构建、报错、回滚）**：[`tacc/README_TACC.md`](tacc/README_TACC.md)。  
> **当前真实状态**：提供了可执行脚本及文档，**尚未在你的 TACC 账号上验证 GPU Job/真实网络连通性**。

### TACC 实际配置与路线选择

| 路线 | 使用场景 | 具体工具 | 资源与约束 |
| --- | --- | --- | --- |
| **公共版（先做）** | 使用已有 Qwen3.8-27B 推理服务并验证模型调用 | `tcloud submit`、`qwen-public-serve`、`qwen-public-client` | GPU 服务：1 节点、2 GPU、8 CPU、1 小时；CPU Client：1 CPU、10 分钟 |
| 私有 vLLM 版 | 自己修改并构建 vLLM 的 Python 源码 | `prepare-source.sh`、`submit-build.sh`、`qwen-private-serve` | 先构建（8 CPU、45 分钟），再启动与公共版相近的 GPU 服务 |
| Windows 本地 Paper Tutor → TACC | 用本地 FastAPI Agent 调用远程 GPU 模型 | OpenAI-compatible ModelClient | **只有网络访问/转发被课程允许且实测可达时才做** |

**必须区分**：课程包通过 `tcloud` 来提交作业，其 `tuxiv.conf` 已指定 Slurm 调度资源。**不要直接用**先前通用的 `tacc/jobctl.py`、`tacc/job_template.sbatch` 取代课程脚本。公共版使用 SDK 环境 `proxy`，私有版使用 `mpi-gemm-v1`，不可交叉复制 `job.sh` / `tuxiv.conf`。

### 1. WSL2 / Linux 准备

用 Windows PowerShell 打开 WSL2 Ubuntu：

```powershell
wsl -d Ubuntu-22.04
```

以下均在 **WSL2 Bash** 执行：

```bash
mkdir -p ~/csit6000v-tacc
cd ~/csit6000v-tacc

# 改为你实际的课程包 ZIP 路径（示例 E:\Downloads）
cp "$(wslpath -u 'E:\Downloads\project-release-1(1).zip')" .
unzip 'project-release-1(1).zip'
cd qwen27b-student-release

# 课程 tcloud SDK v0.4.1 需先安装并加入 PATH
tcloud --help
```

若 `tcloud: command not found`，从课程所附说明的 [Linux x86_64 SDK v0.4.1](https://github.com/turingaicloud/quickstart/releases/download/v0.4.1/linux-amd64-v0.4.1.zip) 安装。在 WSL2 中使用 Linux SDK 是可尝试的路径，但是否受到课程平台正式支持需以 TA 说明为准。切勿把私钥放在项目仓库。

### 2. 首次配置账号与公共环境

```bash
# 用你注册的 TACC 用户名和对应私钥路径替换下面占位符
tcloud config -u YOUR_TACC_USERNAME -f "$HOME/.ssh/YOUR_PRIVATE_KEY"
tcloud init
tcloud env ls

PROJECT_DIR=$(pwd -P)  # 必须在 qwen27b-student-release 根目录
bash setup-sdk-account.sh
```

检查是否成功：`tcloud env ls` 能返回平台环境，脚本提示注册 `proxy`；不要为课程公共运行时自行安装私有 vLLM。若遇到现存 `.condarc` 冲突，按课程 README 联系 TA。

### 3. 提交 2 GPU Serving Job

```bash
cd "$PROJECT_DIR/qwen-public-serve"
tcloud submit

# 从上面命令输出中复制你的实际数字作业 ID
SERVICE_JOB_ID=填写服务作业ID

tcloud ps -j "$SERVICE_JOB_ID"
tcloud cat "slurm_log/qwen-public-serve-${SERVICE_JOB_ID}.log"
```

**必须等到日志出现 `Application startup complete`**。同时找到：

```text
SERVICE_BASE_URL=http://实际计算节点名称:8000/v1
```

这不是固定地址，**每次重新提交服务都要重新复制**。模型名固定为课程文档中的 `qwen3.8-27b`。注意 `Resources` 状态表示等待调度，可能不是报错。

### 4. 复制服务地址并提交 CPU Client Job

```bash
cd "$PROJECT_DIR"
# 把下一行替换成最新服务日志里 SERVICE_BASE_URL= 右侧的完整地址
SERVICE_BASE_URL='http://实际计算节点名称:8000/v1'
printf '%s\n' "$SERVICE_BASE_URL" > qwen-public-client/server-url.txt
cat qwen-public-client/server-url.txt

cd "$PROJECT_DIR/qwen-public-client"
tcloud submit
CLIENT_JOB_ID=填写客户端作业ID

tcloud ps -j "$CLIENT_JOB_ID"
tcloud cat "slurm_log/qwen-public-client-${CLIENT_JOB_ID}.log"
```

成功的 Client 日志通常会回答 `37+58=95` 和“中国的首都=北京”，并显示 Token Usage、每次请求耗时以及结果文件位置：

```text
RESULT_FILE=CSIT6000V-project-results/job-<CLIENT_JOB_ID>.json
```

Client 是**独立 CPU Job**，调用课程 GPU Service，确保 GPU Serving Job 仍在运行。

### 5. 下载运行结果，结束时释放 GPU

```bash
cd "$PROJECT_DIR"
tcloud download "slurm_log/qwen-public-client-${CLIENT_JOB_ID}.log" "./client-${CLIENT_JOB_ID}.log"
tcloud download "CSIT6000V-project-results/job-${CLIENT_JOB_ID}.json" "./result-${CLIENT_JOB_ID}.json"
cat "result-${CLIENT_JOB_ID}.json"

# 测试结束主动释放 GPU
tcloud cancel -j "$SERVICE_JOB_ID"
```

这些命令成功后，才可以在项目报告里写“**TACC 模型服务通过课程内部 CPU Client Job 实测**”。下载结果里的 JSON 包含模型回答、Token usage 和请求耗时。

### 6. 如何接到现有 FastAPI？

Paper Tutor 的 `backend/app/agent/model.py` 已支持 OpenAI-compatible API。**但课程只明确保证平台内部 CPU Client → GPU Service**；Windows 直接访问计算节点是否被允许，需要提前询问 TA，不能把 `SERVICE_BASE_URL` 原样复制到本地后端就当作能连。

若**学校批准并已建立受控隧道**，且模型服务能被该隧道访问，在启动后端的同一个 Windows PowerShell 中设置：

```powershell
$env:MODEL_PROVIDER = 'openai'
$env:MODEL_BASE_URL = 'http://127.0.0.1:8001/v1'  # 经批准的隧道本机示例端口
$env:MODEL_NAME = 'qwen3.8-27b'
$env:MODEL_API_KEY = 'local'  # 如服务要求鉴权，使用实际密钥（不要写进 Git）
$env:MODEL_TIMEOUT = '240'
$env:MODEL_REASONING_EFFORT = ''
cd backend
uvicorn app.main:app --reload --port 8000 --no-access-log
```

**还有一项重要兼容性**：课程 `qwen_client.py` 给 Qwen 的请求带 `chat_template_kwargs={"enable_thinking": false}`；当前 Paper Tutor 的 OpenAI-compatible `ModelClient` 并未默认添加这个字段。必须进一步核对 Chat 非空、JSON mode、Streaming，才能称 Agent 真正接入 TACC；不能只看到 `/v1/models` 返回模型名就算完成。具体检查、私有源码构建、错误处理见 [`tacc/README_TACC.md`](tacc/README_TACC.md)。

**回退本地 Ollama**：

```powershell
$env:MODEL_PROVIDER = 'ollama'
$env:MODEL_BASE_URL = 'http://127.0.0.1:11434/v1'
$env:MODEL_NAME = 'qwen3:4b'
$env:MODEL_API_KEY = 'local'
```

从同一终端重启 FastAPI。TACC 27B 与本地 4B 的结果应分别统计，不能把模型规模差异归因于 RAG 策略本身。

### 7. 关键验收节点

- [ ] `tcloud --help` 成功，`tcloud env ls` 成功。
- [ ] `setup-sdk-account.sh` 注册公共 `proxy` 环境。
- [ ] GPU 服务日志出现 `Application startup complete` 和本次 `SERVICE_BASE_URL`。
- [ ] CPU Client Job 通过，下载 `result-<CLIENT_JOB_ID>.json`。
- [ ] 用 `tcloud cancel -j "$SERVICE_JOB_ID"` 释放 GPU 资源。
- [ ] **可选**：网络获准后由 FastAPI 完成真实模型请求、JSON/Streaming 和一个论文 AgentRun。

<!-- TACC-COURSE-SDK-END -->

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

1. **Historical Day1 status (superseded by later local testing):** initial package was not tested with a real Qwen endpoint; local Ollama Qwen3:4b has since been exercised. Check local model health and model streaming before relying on answers.
2. **Historical Day1 build status:** subsequent Windows `tsc`/`next build` passed before the later General Learning updates. Re-run `npm run build` for the currently installed source version.
3. **LangGraph short-term checkpoints are in-memory** in this release. SQLite persists business records, run events, answers and explicit long-term memories, but not resumable LangGraph node state. Restart fails interrupted runs rather than silently restoring model execution.
4. **Background jobs are in-process:** a single uvicorn worker is required. For deployment, migrate to Redis/Celery/Arq (or another persistent queue) and add database-level dispatch/leases.
5. **Security:** this is explicitly **single-user local development**, with no authentication, rate limiting, request quotas, multi-tenant isolation or web deployment hardening. Do not expose the API to the Internet.
6. **Retrieval:** no table-image OCR or figure-grounded answers; PDF fallback can lose structural detail; evaluation gold evidence mapping is approximate when evidence is not an exact paragraph. FAISS is built from cached vectors per query, not a large-scale persistent global ANN index. Model downloads require connectivity.
7. **Skills / academic web:** current skills are prompt templates; OpenAlex supplementary metadata is not verified against papers and is disabled in evaluation; advanced source relevance grading and citation verification remain future work.
8. **Observability:** SQLite events, a local rotating console/file logger, and run-level debug endpoints are implemented. Langfuse and distributed tracing are deliberately out of scope for this demo.
9. **TACC course SDK guide added:** See [TACC GPU/vLLM deployment](#tacc-gpu--vllm-部署csit6000v-课程环境逐步操作) and [`tacc/README_TACC.md`](tacc/README_TACC.md). Service and client scripts are supplied by the course; actual TACC job/remote backend verification is pending.

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

## 历史版本与修复记录（保留供回溯）

以下为早期版本记录；当前部署与 TACC 以本 README 顶部和 TACC 章节为准。

### Next.js 15.5 development warnings (frontend hotfix v0.1.1)

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
