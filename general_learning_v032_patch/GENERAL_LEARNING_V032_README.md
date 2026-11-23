# General Learning v0.3.2 — installation and acceptance

This is an incremental **patch**, not a complete project snapshot. It was generated from the uploaded `adaptive-agentic-rag(1).zip` plus P1/P2/P2.1 source patches. Existing data, ModelClient, PDF parsing, QASPER data and the prior CC8 fixes are not replaced.

## 1. Scope and behavior

- **Paper QA (default):** still requires 1–8 ready paper IDs; Basic/Hybrid/Adaptive keep their existing behavior. No previous QASPER evaluation input contract changes.
- **General Learning:** requires no paper ID, uses a separate external search and local `qwen3:4b` for an explanation. In this release General uses the Adaptive mode only (the mode selector is disabled while General is selected).
- **Search:** English or Chinese Wikipedia introductory excerpts. When appropriate or if no encyclopedia result is available, falls back to OpenAlex abstract metadata. Search is bounded, requires internet, and accepts only canonical Wikipedia or DOI source URLs. These are public references, not a general search-engine service.
- **Citations:** General response citations are marked `source_type=external`, with actual `source_url` and `paper_id=null`. They open a URL, never a PDF. If no external source, or the model fails the JSON/source-index contract, the run ends `partial` with no fabricated citations.
- **Skill:** `explainBackgroundConcept` is a new file-backed Skill, separate from the existing `explainForBeginner` style.
- **Memory:** existing confirmed memory semantics remain in force; language and study level may personalize the explanation, but are never evidence.

Source provenance and semantic entailment are different: the implementation validates the URL/source index, **not** that every generated claim is mathematically entailed by the referenced text. Verify factual claims in a browser. Existing Paper QA Academic Web is unchanged in this patch.

## 2. Installation (Windows PowerShell)

**Stop Uvicorn and Next.js first.** Do not install while background runs are active. Your repo root is the *inner* `adaptive-agentic-rag` directory containing both `backend` and `frontend`.

```powershell
cd "E:\School\PTG\Adv computing and system\Project\adaptive-agentic-rag\adaptive-agentic-rag"
$patch = "E:\Downloads\general_learning_v032_patch.zip"  # change this
Expand-Archive -Path $patch -DestinationPath ".\general_learning_v032_patch" -Force
powershell -ExecutionPolicy Bypass -File ".\general_learning_v032_patch\install_patch.ps1" -RepoRoot "."
```

`install_patch.ps1` checks the SHA-256 of all existing files **before** copying anything, and makes timestamped backups in `patch_backups`. It refuses to overwrite unexpected local modifications. In particular it requires the current P2.1 `workflow.py` baseline. Check `patch_manifest.json` if a hash does not match.

You can also inspect `general_learning_v032.diff` before installing.

## 3. Smoke tests

In the inner `backend` folder:

```powershell
python -m py_compile app\schemas.py app\agent\workflow.py app\agent\background_learning.py
python -m pytest -q tests\test_general_learning_scope_v032.py tests\test_general_learning_search_mock_v032.py
uvicorn app.main:app --reload --port 8000 --no-access-log
```

In another terminal at the inner `frontend` folder:

```powershell
npx tsc --noEmit
npm run build
npm run dev
```

The patch author verified 203 backend tests (excluding the two pre-existing HTTP error-text tests) and TypeScript compilation; **the Linux test environment could not run `next build`** because it only had Windows-packaged node modules and could not download Linux SWC. The build must be verified on your Windows system.

## 4. Browser acceptance (once each)

| ID | Setup | Expected |
| --- | --- | --- |
| G01 | Knowledge scope = `General Learning`; zero papers selected; ask `请向初学者解释 Q-Learning，并举一个假设例子` | No PDF retrieval, external URL button(s), concise source-based answer |
| G02 | Click an External citation | Opens actual Wikipedia or DOI HTTPS URL; never highlights unrelated PDF |
| G03 | Disable Beginner skill; ask a new general topic | `answer_skills` contains `explainBackgroundConcept` but not `explainForBeginner` |
| G04 | Turn off network/proxy while asking a General question | `partial` and **zero citations**, not fabricated PDF/URL |
| P01 | Switch to `Paper QA`; select the CC PDF; ask CC definition | Works as before, PDF page citations; Basic/Hybrid/Adaptive selection available |
| M01 | With General Memory on/off in fresh sessions | `learning_memory_candidates` changes only when a matching confirmed memory is opted in |

Check individual run metrics through `/api/agent/runs/{id}/debug`. In General scope `knowledge_scope=general`, `retrieval_rounds=0` and `external_sources_found` reflects only external tool output. When a network search produces no sources, the system correctly abstains.

## 5. Known limitations / non-goals

1. Public Wikipedia/OpenAlex APIs can be rate-limited or blocked by local network conditions. This is a bounded **reference lookup**, not a full web crawler, no page scraping/search engine integration.
2. General Learning currently supports **Adaptive mode only**, deliberately preserving the existing Paper QA three-mode evaluation baselines.
3. Mixed Paper + Web knowledge synthesis is **not** implemented; users explicitly choose Paper QA vs General Learning.
4. The original Paper QA `Academic web` setting retains its prior conditional OpenAlex logic; this patch does not claim to have solved every mixed-source semantic attribution issue there.
5. Do not use a high completion rate as proof of answer correctness. Human evaluation on unseen questions is still needed.

If a test fails, preserve the `run_id`, `debug` JSON and screenshot; batch fixes only after checking several representative results.
