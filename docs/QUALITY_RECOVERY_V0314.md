# v0.3.14 — Numeric Safety and Memory-Aware Answer Recovery

**Scope:** Local Day 2 quality improvements. This change does not update PDF chunk
IDs, the QASPER importer, the retrieval indexes, or the TACC integration.

## What changed

1. **English illustrative percentages.** Explicit, modal hypothetical examples,
   such as “For example, a test suite might cover 90% of chunks”, are no longer
   mistaken for measured values from the paper. The marker must occur in the
   same English sentence, and it does not authorize ungrounded experimental
   numbers in the next sentence. Existing checks for source-reported values and
   inconsistent arithmetic are retained.
2. **Safe partial answers.** A candidate rejected by numeric-source checks can
   no longer reappear as `best_partial` after a failed repair call. If no
   grounded alternative survives, the system returns a safe explanation of
   its inability to verify the answer, without attaching a misleading paper
   citation. Earlier valid, cited but incomplete material may still be kept.
3. **Memory-aware teaching.** If an opted-in, confirmed learning note states a
   particular difficulty, the system records a learning focus and asks for a
   short, clearly hypothetical worked example explaining it. It can make one
   extra focused rewrite **within the existing two-generation limit**, while
   recording `learning_focus_covered`, `learning_focus_rewrite_attempted`,
   and `learning_focus_rewrite_failed` where applicable.
4. **No factual regression on failed personalization.** If a first answer
   passes all content and citation checks but omits the requested teaching
   focus, and the optional rewrite fails, that initial answer is retained as
   an operationally successful **fact-checked but not fully personalized**
   response. The failure to personalize is visible in run metrics; it is not
   silently counted as a personalized success.
5. **Language handling.** The answer-partial and no-verified-answer notices in
   the analytical flow now follow the saved language preference or explicit
   question-language override. English-default users no longer receive a
   Chinese-only disclaimer on this path.
6. **Untrusted memory boundaries.** User-provided learning gaps remain in the
   user data portion of the prompt. The trusted system prompt contains only
   the generic adaptation policy, not verbatim note content.

## Test evidence

- **343 backend tests passed** (offline mocks; no real Ollama or Internet).
- Eight new tests cover the precise 90% example, isolation from a subsequent
  unsourced 97.3% claim, safe partial answer fallback, language notices,
  learning-focus detection, successful personalized rewriting, and failed
  rewriting that preserves an earlier valid answer.
- Existing multi-question tests now explicitly set their target language,
  removing dependence on previous test execution order.
- Python source compilation passed.

## Verification required on the developer workstation

From the repo root in PowerShell:

```powershell
cd backend
python -m pytest -q
python -m compileall -q app
cd ..\frontend
npx tsc --noEmit
npm run build
```

**The full frontend TypeScript/Next.js build was not run in this environment**
because the dependency tree is not available. No frontend source files changed
in v0.3.14.

### Memory A/B smoke test

Use the same newly indexed paper and exactly the same English question in two
fresh conversations:

> Explain what Chunk Coverage measures, why repeatedly retrieving the same
> chunk does not increase coverage, and why high coverage does not guarantee
> correct answers.

Keep **Adaptive + Beginner** enabled. Change only **Use confirmed learning
memory**. When enabled, a saved note about understanding RAG but not set union
should trigger a visible explanation of set union and a short illustration.
When disabled, saved notes must not enter prompts. For a controlled comparison,
use the same paper, model, retrieval settings and question; Qwen sampling may
still make individual answers differ.

In **Run details**, inspect `learning_memory_candidates`,
`learning_memory_applied`, `learning_focus_requested`,
`learning_focus_covered`, `learning_focus_rewrite_attempted`,
`learning_focus_rewrite_failed`, `numeric_grounding_issue`, and
`validation_attempts`. A selected note alone is **not** proof that the answer
was pedagogically improved. The focus check is a lightweight textual proxy,
not a semantic entailment test.

### Numeric and fallback smoke test

Ask for a hypothetical illustration using an arbitrary coverage percentage,
e.g. “A test suite **might** cover 90% of chunks”. This should be labeled as a
hypothetical scenario, never a measured result from the paper. Ask for an
unavailable precise experimental score and verify that `partial` does not
repeat that unsupported number with a paper citation.

## Deployment notes

- This release contains **complete source files**, not patch fragments.
- Existing `backend/data/` is intentionally not packaged or modified.
- Existing PDF indexing is unchanged. **Do not re-import papers** solely to
  apply this fix.
- Review `git status --short` and `git diff --check` before committing.
- TACC/vLLM remains outside this local implementation phase.
