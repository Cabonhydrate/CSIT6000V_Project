# v0.3.15 — Strict Learner-Focus Verification

**Scope:** A targeted, local fix for an observed Memory A/B failure. The version does not change PDF ingestion, chunk IDs, citation records, retrieval indexing, web-search behavior, frontend code, or TACC deployment.

## Observed failure

With a saved note indicating the learner understands the RAG pipeline but has not learned **set union**, an Adaptive + Beginner answer gave a general explanation of Chunk Coverage and an illustrative `90%` coverage example but did not explain set union. The Memory ON response looked similar to Memory OFF, despite selecting a saved note.

The implementation used `learning_focus_visible()`, which considered **any single word** of a learner's longer difficulty statement sufficient. A note such as `not familiar with set union and how Chunk Coverage is calculated` could thus be considered addressed because the answer mentioned `Chunk Coverage` and contained a generic example.

## Change

- Extract a short, first named difficulty from a compound note, instead of treating any token in the note as proof of personalization.
- Require the answer to name the distinctive concept, including tolerant handling of ordinary hyphenated English terms and reversed short phrases.
- Require a nearby, recognizable illustration or hypothetical worked example; an example somewhere else in the answer is not sufficient.
- When an otherwise valid answer fails this **teaching-style** check, reuse the existing one-time focused repair. A user-authored note remains untrusted data in the USER portion of the prompt; the SYSTEM prompt contains only generic policy.
- If the repair is unsuccessful, preserve the already fact-checked answer but record `learning_focus_rewrite_failed=true` and `learning_focus_covered=false`. These metrics distinguish **workflow success** from **successful personalization**.

The check is deliberately lexical and bounded: it does not evaluate semantic correctness or claim that the learner has mastered the concept. It may still produce false positives or negatives on paraphrases. It is never used to justify a paper claim or create a source citation.

## Validation

In the local offline test environment:

- **348 backend tests passed**, including five new tests for this release.
- A previous false positive (`Chunk Coverage ... For example, a test suite might cover 90% ...`) now fails the learning-focus check for a saved `set union` gap.
- An explanation with a named set-union operation and a two-query worked set example passes.
- End-to-end mocked Agent runs confirm targeted rewrite, observable failure when the rewrite is empty, and no memory injection when it is disabled.
- Python source compilation passed.

This does **not** certify actual Ollama behavior or a full frontend production build. The frontend files were unchanged in this release.

## Apply to the Windows Git project

The ZIP contains complete source files under the `adaptive-agentic-rag/` folder; extract it to the existing outer project folder after committing or stashing local edits.

```powershell
cd "E:\School\PTG\Adv computing and system\Project\adaptive-agentic-rag\adaptive-agentic-rag\backend"
python -m pytest -q
python -m compileall -q app
cd ..\frontend
npx tsc --noEmit
npm run build
```

There is no need to re-upload indexed PDFs or delete the existing SQLite database for this update.

## Real Memory A/B acceptance

Use two **new conversations**, the same indexed paper and identical question, Adaptive Agentic RAG + Beginner enabled, and English profile. Toggle only `Use confirmed learning memory`.

**Saved note:**

> I already understand the RAG retrieval pipeline, but I am not familiar with set union and how Chunk Coverage is calculated.

**Question:**

> Explain what Chunk Coverage measures, why repeatedly retrieving the same chunk does not increase coverage, and why high coverage does not guarantee correct answers.

**With Memory ON:** The response should explain the saved gap, ideally with two hypothetical query result sets such as `{A, B}` and `{B, C}`, their union, and why `B` is counted once. In Run details check:

- `learning_memory_applied=true`
- `learning_focus_requested=true`
- `learning_focus_covered=true` (if the answer actually addresses the gap)
- `learning_focus_rewrite_attempted=true` only if the initial draft omitted the focus
- `learning_focus_rewrite_failed=true` means the model did **not** complete the adaptation successfully, even if the paper-grounded answer was retained

**With Memory OFF:** `learning_memory_applied=false`, `learning_focus_requested=false`, and no memory-based rewrite. A small-model response may still independently mention set union, so text difference alone is not proof of causal Memory use.

Keep source grounding separate from learning adaptation: an illustrative example must be labeled hypothetical and should not be cited as an empirical result. Verify that the PDF citation supports the paper facts in the answer, particularly the coverage definition and its distinction from answer correctness.
