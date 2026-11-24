# English Defaults and Learning Memory — v0.3.12

## Changes

This update uses the source snapshot supplied on November 24, 2026. It modifies the current project files **in place**; it is not a series of installers or source-code patch files.

- **English is the default response language** for newly created learning profiles and for the frontend's initial state. A previously saved Chinese preference is respected; change it to English and press **Save preferences** to switch. A request explicitly asking for an answer in Chinese or English overrides the saved preference for that question only.
- **A single language policy** is appended to model-generation prompts in Basic, Hybrid, Adaptive, general knowledge, verification/recovery, and the Basic/Hybrid streaming path. Existing citation, numeric, definition, and source-validation rules have not been removed. Note that language steering is a prompt-level constraint, so a small local model is still capable of ignoring it; check a real Ollama run before final acceptance.
- **English starter prompts** replace Chinese suggestions in both Paper QA and General Learning. The Learning UI now shows an unsaved/saved preferences notice.
- **Explicit saved-learning-note queries** such as "Based on my saved learning note" are recognized when selecting memories, even when the query lacks a literal topic match. Ordinary topic questions prefer relevant notes. This is bounded to at most three user-confirmed records and never creates an inferred note or assumes mastery.
- **Actionable learner context** distinguishes `mentioned`, `studied`, and `confirmed`; asks the explanation to bridge from known concepts rather than repeat prerequisites. Memory remains user-provided background and must not be treated as paper evidence.
- **Observable memory usage:** the Agent records `learning_memory_candidates`, `learning_memory_topics`, `learning_memory_ids`, `learning_memory_applied`, `learning_context_applied`, and `response_language`; the frontend displays the count of relevant notes selected. The displayed count is not a guarantee that the LLM used the memory accurately.
- **No new Skill is required.** Memory is an input to existing explanatory and learning-path Skills. Adding another Skill before fixing recall/injection would add overhead without solving the underlying issue.
- **Fair benchmark isolation is preserved:** Basic and Hybrid do not receive confirmed learning-memory context; Adaptive receives it only when the request permits it and the saved profile enables it. All modes honor the language preference. Compare personalization on/off within Adaptive; compare retrieval strategies with memory and teaching disabled.

## Reproduction checks

1. In **Learning & Memory**, set `Language = English`, set a domain and goal, then click **Save preferences**. Confirm the saved message is shown.
2. Save a confirmed note: Domain `RAG`, Topic `Chunk Coverage`, Note `I already know the RAG retrieval pipeline but have not mastered set union.`, Mastery `Studied`.
3. In **Paper QA / Adaptive Agentic RAG**, turn on Beginner skill and choose the paper. Ask: `Based on my saved learning note, explain what Chunk Coverage measures and why it does not guarantee correct answers.`
4. Inspect Agent activity and run details: one or more memory candidates should be selected, `learning_memory_applied=true`, and `response_language=en`. The explanation should build on prior RAG knowledge without treating the note as a paper claim.
5. Disable `Use confirmed learning memory in Adaptive RAG`, save the preference, start a fresh conversation, and repeat the **same** question. Expect `learning_memory_candidates=0` and `learning_memory_applied=false`. The response should no longer refer to the saved learner context. The factual paper-grounded claims should remain consistent.
6. Repeat in **Basic** and **Hybrid** with English profile: answers should follow the English preference, but saved memory should not be injected. Explicitly request Chinese in one question to test the override.
7. Start a **General Learning** conversation and ask `What is machine learning?`. The starter suggestions, question/answer language and source labels should be English. Verify the external URL points to a relevant page.

## Validation and remaining limits

- The Python backend test suite passed **321 tests** in the package preparation environment; compilation succeeded.
- The frontend TSX parser found **no syntax errors**. A complete `tsc --noEmit` and Next.js production build could **not** be verified here: the npm registry was unavailable and the dependency installation was incomplete. Run these on the final Windows checkout.
- The previous tests used a mocked LLM to validate memory-flow wiring and observe response changes. They do not prove that the live local Qwen3:4b model will always personalize correctly. A real paired-memory test is a required manual acceptance gate.
- Existing records and learning data are not included in the source archive and should remain in the local `backend/data/` directory. Do not overwrite them as part of source-only deployment.

## Local commands

```powershell
cd backend
python -m pytest -q
python -m compileall -q app

cd ..\frontend
npm ci
npx tsc --noEmit
npm run build
```

The formal submission report should describe memory as **explicit, consent-controlled, relevance-filtered learner personalization**, not automatic mastery tracking or a second academic evidence source.
