# v0.3.18-rc1 — Final Targeted Hardening Candidate

## Scope

This is a **local release candidate**, prepared from the uploaded v0.3.17 source on November 24, 2026. It is not a claim of final Windows, live-Ollama, or live-web acceptance. The three RAG mode algorithms, PDF ingestion/indexed chunks, learning-memory selection, and TACC directories are unchanged. TACC/vLLM work is explicitly out of scope.

### Changed runtime files

- `backend/app/agent/background_learning.py` — reject Wikipedia disambiguation pages even if their title exactly matches the query; use MediaWiki page metadata where available; verify explicitly named components in technical source text; on demand, fetch relevant paragraphs from the **same** canonical Wikipedia article if its lead paragraph is too shallow. Never invent a source or treat a title alone as evidence.
- `backend/app/agent/workflow.py` — put actual English instructions first for general-learning generation; require English-visible output and explicitly named answer components for multi-component questions; bound a failed attempt to one source-preserving retry. English-language failure/partial messages are now localized in several previously hardcoded Chinese paths.

### Regression tests

- Existing `backend/tests/test_external_memory_routing_v0317.py` has one artificial encyclopedia fixture upgraded to contain actual technical evidence, because the release gate correctly refuses title-only content.
- Added `backend/tests/test_release_candidate_source_quality.py`: disambiguation, deeper-source retrieval, insufficient technical detail, multi-component question parsing, language validation, and bounded retries.
- Baseline before changes: **369 passed**, `python -m pytest -q` in the provided local container.
- RC checks: **379 passed**, `python -m pytest -q` in the same environment; `python -m compileall -q app` completed.
- No tests were excluded. These are mocked/offline tests, not a live model or live Wikipedia factual-quality score.

### Artifact packaging

- Added the previously missing `artifact_evaluation/day2_qasper.py` standard-library benchmark driver referenced by `README.md`.
- Archived previously supplied **historical** QASPER holdout results under `artifact_evaluation/reference_results/`. They are diagnostic baselines, not measurements of v0.3.18-rc1 and not valid model prompts.
- Updated this version's source checksums and release checklist. The source package excludes `.git`, model weights, runtime databases, downloaded PDFs, and node/build caches.

## Important remaining limitations

1. A topical title plus explicit component mentions is **not** proof that every generated proposition is entailed by the source. Human spot-checks remain necessary.
2. The source coverage gate is intentionally conservative and currently primarily recognizes clearly enumerated English technical components. Complex questions may return a safe partial answer rather than a confident unsupported explanation.
3. Source requests require working internet access. Offline tests use deterministic network mocks.
4. A source excerpt may omit figure/equation context; users must inspect the full PDF for important claims.
5. No live Ollama or final Next.js build was verified in this packaging environment. The final release gate requires local Windows confirmation before tagging a final version.
6. The original QASPER trial scores must not be presented as the v0.3.18-rc1 results; the new release requires a fresh, fixed-config benchmark.

## Windows acceptance and freeze commands

Verify file integrity first using `python scripts/verify_release.py` (from the repository root).

Run from the actual local Git repository **after** replacing files with this complete package, not from the download directory:

```powershell
# Repository root
 git status --short
 git diff --check
 git rev-parse HEAD

# Backend (existing Project environment, with dependencies installed)
 cd backend
 python -m compileall -q app
 python -m pytest -q
 python -m app.diagnose
 cd ..

# Frontend
 cd frontend
 npm ci
 npx tsc --noEmit
 npm run build
 cd ..
```

Manual live regression (record a run ID and screenshot for each failure):

1. Paper QA: `What is Chunk Coverage?` in Basic, Hybrid, Adaptive; verify definition and PDF citation page.
2. Paper QA: `Compare Chunk Coverage with code coverage in software testing.` with Academic Web on; verify **separate** source identities and no false shared execution claims.
3. General Learning: `Explain the query, key, and value vectors in self-attention.`; reject a Wikipedia disambiguation citation; verify each explanation against a real source passage.
4. General Learning: `What is a B-tree index in SQL?` and `Explain the difference between set union and set intersection.`; verify external links.
5. With English preference, induce a source/model failure and confirm English error text.
6. With confirmed memories enabled and disabled, compare a matching topical question and an unrelated question. Do not infer better learning from a mere `learning_memory_applied` flag.
7. Exercise PDF invalid/cancel/backend-unavailable cases and confirm earlier regressions remain fixed.

**Freeze only after** backend unit tests, Windows frontend build, model health, at least one live paper citation, at least one live General Learning citation, and relevant Memory ON/OFF tests pass. Record a Git commit and use it for all formal QASPER experiments. Any subsequent code change invalidates the previous release gate and must be re-tested.
