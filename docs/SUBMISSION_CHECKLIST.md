# Release and Submission Checklist

**Release candidate:** v0.3.11 local prototype  
**Documentation language:** English  
**Assumption:** The official instructor handout or course submission rubric has **not** been supplied in this packaging pass. Confirm the required file formats, upload platform, naming convention, team information, page limit, demo format, and deadline before final submission.

## A. Version and working tree

- [ ] Confirm that the repository contains the latest source files from the v0.3.11 tested local version.
- [ ] Record the Git commit hash: `git rev-parse HEAD`.
- [ ] Inspect `git status --short`, `git diff --check`, and `git diff --cached --stat`.
- [ ] Commit only intentional changes. Do not include runtime databases, downloaded models, cache directories or private annotations.
- [ ] Keep all **formal submitted documentation, report text, screenshot labels, and slides in English**.
- [ ] Keep runtime answers multilingual if needed; multilingual user support is a product feature, not a documentation-language violation.
- [ ] Remove obsolete temporary files and unrelated old drafts from the final artifact. The repository may retain development history in Git; historical Chinese-language working notes need not be bundled as current official deliverables.

## B. Final Windows build evidence (must be executed on the submitted checkout)

Backend PowerShell:

```powershell
cd backend
python -m pytest -q -k "not test_persistent_http_500_is_not_hidden and not test_http_400_not_retried"
python -m compileall -q app
python -m app.diagnose
```

Frontend PowerShell:

```powershell
cd frontend
npm ci
npx tsc --noEmit
npm run build
```

- [ ] Save a screenshot or terminal transcript of the **final** frontend build.
- [ ] Save the final backend test summary and explicitly report the two pre-existing model HTTP error-text assertion tests excluded in the documented development test command.
- [ ] Verify `/api/health` and at least one real local Ollama generation.
- [ ] Record model identifier and Python/Node versions; record local GPU type where relevant.
- [ ] Confirm the application can start from documented commands without relying on untracked source files.

## C. Minimal live acceptance (one run per scenario)

- [ ] PDF upload / parsing / `ready` state works.
- [ ] A paper question can be answered in Basic, Hybrid and Adaptive modes.
- [ ] At least one PDF citation opens the relevant passage and location.
- [ ] An explicitly hypothetical arithmetic example is not falsely attributed to the paper.
- [ ] An external concept can be explained with no selected paper and a working **relevant** URL.
- [ ] A paper-plus-background comparison keeps PDF and external citations separate.
- [ ] Turning confirmed learning memory off prevents retrieval of that saved note.
- [ ] A missing external source or disconnected service yields a safe, readable error/partial result.
- [ ] Capture run IDs for any failed or partial case; never hide an error behind a success-only screenshot.

## D. Academic reporting integrity

- [ ] The final report does not claim TACC, vLLM, distributed serving or GPU performance experiments were completed.
- [ ] The QASPER tables report the actual selected development subsets and experimental settings.
- [ ] Answer Token F1 is called a simplified **proxy**, not the official QASPER leaderboard score.
- [ ] CC8 scores are described as **development-set regression**, not unseen generalization performance.
- [ ] The 6-question cross-paper holdout and observed answer errors are reported, not omitted.
- [ ] Adaptive's extra retrieval/model cost is disclosed; no unsupported superiority claim is made.
- [ ] Previously noted benchmarking profile-context differences are acknowledged when interpreting old results.
- [ ] All example screenshots, quotations and statistics refer to real captured runs; any synthetic illustration is labeled as hypothetical.
- [ ] Citations in the report correspond to actual referenced papers or documentation.

## E. Source and privacy hygiene

Avoid adding the following to the submission:

- `.env`, `.env.local`, credentials, tokens, API keys or personal authentication material.
- `node_modules/`, `.next/`, `__pycache__/`, `.pytest_cache/`, `.venv/`, build caches and `.tsbuildinfo`.
- `backend/data/`, SQLite databases, uploaded private PDFs, local logs or personal learning memory.
- Model weight files, downloaded dataset archives, Hugging Face caches and large binary indexes.
- Private evaluation gold, reviewer annotations or test-question answer keys not authorized for submission.
- Personal identifiers in screen captures, run logs or file paths.
- Temporary patch ZIPs, duplicate `model` copies and outdated internal progress notes unrelated to reproducing the submitted release.

Useful Git inspection commands from the repository root:

```powershell
git status --short
git diff --check
git ls-files | Select-String -Pattern '(^|/)(data|node_modules|\.next|__pycache__|\.venv|\.pytest_cache)/|\.env$|\.env\.local$|\.sqlite|\.db$|\.log$|\.zip$'
```

**Important:** `.gitignore` does not retroactively remove files already tracked by Git. If a sensitive or unnecessary file is tracked, stop and remove it from the submission index intentionally; inspect the impact before using `git rm --cached`.

## F. Submission structure

A suggested **English-only formal document set** (adjust to instructor requirements):

```text
README.md
backend/
frontend/
examples/
scripts/
docs/
  ARCHITECTURE.md
  EVALUATION.md
  FINAL_REPORT_DRAFT.md   # Rename to FINAL_REPORT.md after approval
  DEMO_GUIDE.md
  SUBMISSION_CHECKLIST.md
```

The report may need to be exported to PDF or DOCX if required by the instructor. Do **not** assume Markdown alone is an accepted academic report format. Keep screenshots in a separate `docs/figures/` folder if they are required and have been cleared for submission.

## G. Release gate

| Gate | Acceptance evidence | Status |
| --- | --- | --- |
| Source commit | SHA and clean/intentional tracked files | Pending final local confirmation |
| Backend tests | Latest test command and result | Previously 315 passed in development; rerun on final commit |
| Frontend type check | `npx tsc --noEmit` | Previously passed; rerun on final commit |
| Frontend production build | `npm run build` | **Pending final-version confirmation** |
| Model diagnostic | Real local Qwen generation | Previously exercised; rerun on final commit |
| Paper QA | Correct cited answer, 3 modes | Previously observed |
| General Learning | Relevant external concept URL | Previously observed |
| Privacy scan | No sensitive or runtime files | Pending clean package audit |
| Instructor requirements | Actual rubric and deliverable formats | **Not yet verified** |

**Freeze rule:** Once all mandatory gates are satisfied, make no further functional changes unless a genuine blocking issue is found. Any post-freeze change requires a new test/build evidence capture.
