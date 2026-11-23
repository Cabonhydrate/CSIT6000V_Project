# Adaptive Paper Tutor — v0.3.3 Quality Consolidation

**Scope:** single small quality consolidation on top of installed **General Learning v0.3.2**. No database migrations, model-server changes, retrieval changes, new services, or frontend source changes.

## 1. What changed

| General issue | Change | Why |
|---|---|---|
| False `partial` when comparing Chunk Coverage with *code coverage* | Replace the blanket ban on any mention of “代码覆盖率/代码执行” with a narrower check for a definition that **identifies CC/document chunks with source-code chunks** | Legitimate cross-domain comparisons must not be rejected merely for mentioning code coverage. Invalid definitions are still rejected. |
| Qwen analysis/planning displayed as a final answer | Expand the generic answer-quality gate to detect “首先，问题是 …”, “让我…分析…” and similar internal narration; always repair responses terminated by `finish=length` | A successful HTTP/model call is not an acceptable final answer if the content is truncated or internal deliberation. |
| Adaptive benchmark receives user profile even with Teaching=false, Memory=false | Suppress personalized learning context **only when both request options are off** | The three-mode retrieval comparison should not silently add profile content to Adaptive's answer-generation prompt. |
| General Learning checkbox off still got a beginner-specific prompt | Make the General Learning answer prompt honor `request.teaching` | Style preference and factual source selection should be independent. |

**Still not solved:** semantic hallucinations (e.g. LSTM vs Transformer), per-claim entailment, narrow CC8 overfitting, live network errors, HF cold start, and detailed frontend visual acceptance. Do not claim those were repaired by this patch.

## 2. Local verification completed before packaging

- Python syntax check on changed modules: PASS.
- Targeted tests including newly authored cases: **23 passed**.
- Backend regression: **217 passed, 2 deselected**, run with:

```powershell
python -m pytest -q -k "not test_persistent_http_500_is_not_hidden and not test_http_400_not_retried"
```

- The two *unchanged* tests require exact HTTP status descriptions `500 Internal Server Error` and `400 Bad Request`, while the previously installed `model.py` now reports `Model server HTTP 500: ...` / `Model server HTTP 400: ...`. Running all tests without deselection produces 217 passes and those 2 failures. Fixing these test assertions is deferred because it does not affect the user-facing quality changes.
- All tests above are **offline/mocked**; they do not prove Qwen, Windows, network or browser behavior.

## 3. Install on Windows

Stop the FastAPI process first. Use the inner repo root containing both `backend` and `frontend`.

```powershell
cd "E:\School\PTG\Adv computing and system\Project\adaptive-agentic-rag\adaptive-agentic-rag"
$zip = "E:\Downloads\release_v033_quality_patch.zip"   # change location
Expand-Archive -Path $zip -DestinationPath ".\release_v033_quality_patch" -Force
powershell -ExecutionPolicy Bypass -File ".\release_v033_quality_patch\install_patch.ps1" -RepoRoot "."
```

The installer first checks full SHA256 for **both existing source files**, before making any changes. If a hash differs, stop and inspect your local modifications; do **not** use an unconditional overwrite. Backups are placed under `patch_backups/release_v033_<timestamp>/`.

## 4. Verify in one batch

In `backend`:

```powershell
python -m py_compile app\agent\workflow.py app\skills\teaching.py
python -m pytest -q tests\test_release_quality_v033.py
python -m pytest -q -k "not test_persistent_http_500_is_not_hidden and not test_http_400_not_retried"
```

In `frontend` (the original v0.3.2 update still needs a Windows production-build record):

```powershell
npx tsc --noEmit
npm run build
```

Restart FastAPI and Next.js and run the following **once each**, using separate conversations:

1. **General / Beginner ON** (no paper): ask `请给初学者解释 Q-Learning，并举一个假设例子`; verify an external URL (not a PDF page).
2. **General / Beginner OFF** (no paper): ask `什么是时间差分学习？`; verify answer remains grounded and `answer_skills` excludes `explainForBeginner`.
3. **Paper QA / Adaptive / Beginner ON**: ask `文中的 Chunk Coverage 和传统软件测试中的代码覆盖率有什么区别？`; verify it does **not** auto-fail merely because code coverage is discussed, and inspect PDF page citation(s). An answer that actually equates chunks with code should still fail validation.
4. **Paper QA / Basic and Hybrid**: recheck a stable CC-definition question with PDF citation. One query per mode is enough.
5. **General / network unavailable**: temporarily disable external access; verify `partial` with **no invented sources** (G04). Restore connection afterward.

For failed scenarios, save the Run ID and `/api/agent/runs/{id}/debug` JSON. **Do not retry just to obtain a lucky success.**

## 5. Finish criteria and next steps

- If tests pass and demos are usable: **freeze v0.3.3**. Do not continue per-question patching.
- Report observed factual limitations faithfully using pre-existing QASPER Holdout and CC8 results, including the finding that Adaptive was not better than Hybrid on the six new QASPER questions.
- Prepare reproducible README, architecture diagram, Demo screenshots, model/embedding environment, evaluation methodology and sanitized source archive. Defer Mixed Scope, additional search providers, TACC and model migration.
