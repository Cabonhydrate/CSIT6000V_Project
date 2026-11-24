# v0.3.17 — General Learning Search and Read-Only Memory Commands

## Scope

This change addresses the failed T03, T04, T06, and T07 manual acceptance cases on November 24, 2026. It does not change PDF ingestion, chunk IDs, vector indexes, QASPER data, Basic/Hybrid retrieval, or GPU/Ollama deployment.

## Changes

1. **Search topic extraction:** English instructional prose is removed before title matching. For example, a question about query/key/value vectors in self-attention resolves to `self-attention`, rather than `and value vectors in self-attention`.
2. **Bounded multi-topic retrieval:** Comparison questions yield distinct candidate topics. A small explicit list of encyclopedia title variants supports `self-attention` / `Attention (machine learning)`, `B-tree index` / `B-tree`, and `set union` / `Union (set theory)` (with the corresponding intersection variant). These are *search aliases*, not facts or answer templates.
3. **Verified source gate:** Results are checked against the extracted topic candidates, never accepted solely because a requested word appears in a paper abstract or page body. The existing rejection rules for journals and unrelated subfields remain in place.
4. **Diagnostics:** General Learning run metrics now include `external_search_topics`, `external_queries_attempted`, `external_source_filter_reasons`, `external_network_errors`, and `external_failure_type`. Search failure is classified as `network_error`, `filtered_all`, or `no_results` when appropriate.
5. **Read-only Memory Management routing:** A direct request such as `List my saved learning notes.` is handled before the Paper RAG or General Learning workflow. It reads the local user-owned SQLite memory records, returns up to 100 recent notes and the total count, creates **no external/PDF citation**, and invokes **no LLM or external search**. The existing `/api/learning/memories` endpoint remains available for full list management.
6. **Frontend:** A recognized memory-list command can be sent even when no PDF is selected. Saved notes are viewable regardless of the optional personalization toggle.

## Safety and limitations

- Source lookup is still restricted to the existing configured encyclopedic and scholarly providers. Search aliases are validated against the *returned* article title. No fabricated citations or unsourced model-only fallback are introduced.
- Wikipedia query results and article excerpts may be unavailable because of network conditions or insufficient indexed text. A successful offline regression cannot guarantee live access.
- Topic extraction remains a bounded syntactic heuristic; semantically equivalent but unfamiliar names can still be missed.
- Memory listing is a **read-only** intent. Notes are data, never high-priority model instructions or verified facts about a paper.
- Memory lists show a maximum of 100 recent entries through the chat UI; the Learning & Memory page exposes the full collection.
- The existing definition-citation precision and unrelated Paper QA improvements have not been modified in this release.

## Automated validation

- Backend: `python -m pytest -q` → **369 passed** (358 inherited + 11 new regression cases).
- Python compilation: `python -m compileall -q app` → passed.
- Modified frontend TypeScript/TSX: syntax and transpilation passed with TypeScript.
- Full Next.js build and live Ollama/network behavior **must be verified on the user's Windows machine**; this environment did not install the frontend's node_modules.

## Apply to the local Git checkout (PowerShell)

Make a Git commit or stash any changes you need to preserve before overwriting files.

```powershell
$outer = "E:\School\PTG\Adv computing and system\Project\adaptive-agentic-rag"
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\adaptive-agentic-rag-v0317-source.zip" `
  -DestinationPath $outer -Force
cd "$outer\adaptive-agentic-rag"
git diff --check
cd backend
python -m pytest -q
cd ..\frontend
npx tsc --noEmit
npm run build
```

No PDF re-upload or memory deletion is necessary.

## Suggested minimal live acceptance checks

1. **General Learning:** `Explain the query, key, and value vectors in self-attention.` → expect relevant external source(s), `external_sources_found > 0`, answer in English, and no PDF citations. The output may still be partial if no verified source is found.
2. **General Learning:** `What is a B-tree index in SQL?` → expect a relevant B-tree source, and `learning_memory_candidates == 0` when there are no SQL memories.
3. **General Learning:** `Explain the difference between set union and set intersection.` → expect concept-relevant results for both (when reachable); verify memory selection does not confuse user notes with external facts.
4. **Either scope:** `List my saved learning notes.` → expect the actual saved notes, `memory_management_action == "list"`, `llm_calls == 0`, `retrieval_rounds == 0`, and `citations == []`. A PDF must not be required.
5. **Fail-closed regression:** `What is machine learning?` must never cite a page about the *Journal of Machine Learning Research* or a *quantum machine learning* subfield as the general definition without direct topic-matching evidence.
