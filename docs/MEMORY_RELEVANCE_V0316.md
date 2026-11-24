# v0.3.16: Relevant and Bounded Learning Memory

## Observed issue

In a live Memory ON/OFF comparison, a saved Chunk Coverage note was selected, and the final response was marked `learning_focus_covered=true`, although the actual answer did **not** teach the explicitly named difficulty (*set union*). The previous lexical check could be satisfied by mentioning a loose token such as "coverage" plus an unrelated example. Separately, memory recall was too permissive: a note with a broad domain such as RAG or generic content words like "retrieval" could be injected into questions unrelated to that note. With more memories, this behavior becomes increasingly misleading.

## Implementation

### Source selection

The memory gate uses deterministic topic/content matching, bounded by `limit=3`:

1. **Named topic:** Strongly prefer the full topic title or meaningful matched topic words. `Q-Learning` and `Q Learning` are treated consistently.
2. **Explicit learning gap:** A distinctive phrase in a saved note can match a question even if the note's topic label is broad. Generic words alone do not qualify.
3. **Domain:** Only a tie-breaker; a note about Chunk Coverage is not selected just because an unrelated question also concerns RAG.
4. **Explicit memory recap:** `List my saved learning notes` is allowed to return a small recent overview. `Based on my saved note, explain X` **never** falls back to unrelated recent notes when X cannot be matched.
5. **Selection bounds:** Up to three notes total, no more than two with the same topic. A single-user SQLite lookup inspects up to 500 newest records, rather than only the previous 30. Newer records take priority within equal-relevance scores.

Memory remains user-controlled and **is never treated as authoritative paper evidence**. It is inserted as untrusted learner data into answer prompts, not as a new System instruction. If memory is disabled or no relevant note exists, no saved note is passed to the model. Basic and Hybrid remain memory-free for benchmark isolation.

### Teaching focus

The preceding v0.3.15 change is included in this complete source package: a named learning gap must actually be addressed, with a nearby relevant worked illustration. A generic `90% coverage` example cannot satisfy a `set union` learning gap. This is a best-effort lexical teaching signal, **not** a proof of semantic understanding. When more than one note is selected, only the primary relevant note may require a teaching-focused rewrite; the others are context, so the Agent does not attempt to satisfy three unrelated teaching gaps at once.

The answer remains subject to paper/source validation. An unsuccessful optional teaching rewrite must not invalidate a previously fact-checked answer or invent citations.

### Diagnostics

Inspect the Run details (available in English):

- `learning_memory_available`: the number of records examined for this run (0 when memory is disabled)
- `learning_memory_candidates`: number of records selected for use
- `learning_memory_filtered`: examined records not selected
- `learning_memory_topics`: selected note topics
- `learning_memory_applied`: selected notes were actually placed into a generation prompt
- `learning_focus_requested`: at least one learner-stated gap was selected for explanation
- `learning_focus_covered`: the final answer met the **lexical** teaching-focus check
- `learning_focus_rewrite_attempted` / `learning_focus_rewrite_failed`: whether a bounded optional teaching correction was attempted and whether it succeeded

Do not equate `learning_memory_applied` or `learning_focus_covered` with objectively improved answer correctness or user mastery.

## Verification

The source package passed **358 / 358 backend tests**, including ten new tests for memory selection, multi-topic questions, explicit recaps, old notes beyond the former 30-record window, opt-in context and a false-positive teaching example. Python compilation also succeeded. These tests use mocks and deterministic fixtures; **a new real Ollama A/B run is still required** to confirm actual teaching behavior. Frontend files were not changed in this release; the full Windows production build was not rerun here.

## Recommended live test (English)

Create these three saved memories and enable confirmed memory:

| Topic | Note |
| --- | --- |
| Chunk Coverage | I know RAG retrieval basics, but I am not familiar with set union. |
| Self-Attention | I understand Transformer layers, but I struggle with the query/key/value mechanism. |
| Clustering | I have studied K-means, but I am not familiar with silhouette scores. |

Run each prompt in a **fresh conversation**:

1. `Explain Chunk Coverage and why retrieving the same chunk twice does not increase coverage.` Expected: only the Chunk Coverage note is selected; a worked set-union illustration is preferred when Beginner skill is enabled.
2. `Explain the query, key, and value vectors in self-attention.` Expected: only the Self-Attention note is selected, not Chunk Coverage or Clustering.
3. `Explain the differences between Chunk Coverage and code coverage.` Expected: relevant Chunk Coverage information, but no unrelated Clustering note; paper and web citations remain distinct when Academic Web is enabled.
4. `What is a B-tree index in SQL?` Expected: **zero** learning memories selected, unless an explicitly relevant note exists.
5. `List my saved learning notes.` Expected: a bounded, recent summary of available notes, not fabricated learning progress.

For each run, verify the number of selected notes and the answer's actual content. Relevance rules are intentionally conservative, so semantic paraphrases not sharing technical terms may fail to recall a relevant note; a future embedding reranker could address this if the user library becomes large.

## Apply safely

This is a **complete source overwrite**, not a binary patch. From the Git repository's parent folder, extract the ZIP and then run the backend and frontend checks. Preserve the local SQLite database; **no PDF reindex and no memory migration are required**. Commit or stash intentional Git changes before extraction.

```powershell
cd 'E:\School\PTG\Adv computing and system\Project\adaptive-agentic-rag\adaptive-agentic-rag\backend'
python -m pytest -q
python -m compileall -q app
cd ..\frontend
npx tsc --noEmit
npm run build
```

This version does not modify PDFs, citation IDs, retrieval indexes, the frontend, or TACC deployment.
