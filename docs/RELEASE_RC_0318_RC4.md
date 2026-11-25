# v0.3.18-rc4 — Student-Provided Set Arithmetic and Task Routing

## Confirmed regression after RC3

A learner asked (Paper QA, Adaptive + Beginner):

> Suppose a corpus has 6 chunks in total. Query 1 retrieves {c1, c2},
> Query 2 retrieves {c2, c3}, and Query 3 retrieves {c1, c3}.
> What is the Chunk Coverage? Please explain it step by step for a beginner.

Memory OFF returned a definition with no calculation. Memory ON returned a
placeholder instead of a calculation. Both outputs missed the correct
set union `{c1,c2,c3}` and result **3/6 = 50%**. The question splitter
incorrectly treated "Please explain it step by step..." as an independent
subquestion; the multi-question validator did not verify the requested math.
Previously the deterministic arithmetic helper accepted only `C={...}` and
`q1={...}` and could not parse a natural-language corpus size and query sets.

## Focused fixes

- `backend/app/agent/workflow.py`: Keep dependent English pedagogical instructions
  attached to the preceding question. Do NOT merge two independent technical
  questions. Route a fully specified set-union coverage example as a calculation
  even when phrased as "What is the coverage?".
- `backend/app/skills/task_outputs.py`: Extend existing set calculation to
  accept bounded natural-language inputs such as "corpus has 6 chunks" and
  "Query 1 retrieves {c1,c2}". Require at least two nonempty query sets,
  reject duplicate query IDs, reject a union larger than the corpus, and
  preserve explicit-corpus-set support. The user-supplied numeric example is
  never represented as a paper result.
- `backend/app/agent/workflow.py`: For an explicitly provided, complete
  set-calculation problem, calculate the union deterministically **only when
  retrieved paper evidence independently supports the CC definition**. Cite
  the paper definition, not the student example. Do not rely on the 4B LLM to
  calculate a simple set union. When a relevant confirmed learning note
  explicitly mentions difficulty with set union or duplicate counting,
  include a targeted non-duplicate-counting explanation; without memory,
  preserve a short step-by-step answer. Track actual memory use accurately.
- `backend/app/agent/learning_context.py`: Recognize explicit personal
  learning gaps phrased "I often confuse X with Y". This affects learning
  context only, not retrieval evidence or model system instructions.
- `backend/tests/test_explicit_coverage_arithmetic_rc4.py`: Seven new
  isolated regression tests for arithmetic parsing, rejection conditions,
  routing, source-backed citation and Memory ON/OFF behavior.

This is a general mechanism for explicitly supplied corpus/set inputs, not a
hardcoded answer for a paper, model, prompt ID, or benchmark row. Paper
facts are still taken only from retrieved PDF evidence.

## Verification

- Backend: **406 passed** (`cd backend && python -m pytest -q`).
- Formal QASPER harness: **9 passed** (`python -m unittest discover -s artifact_evaluation -p test_formal_qasper_offline.py`).
- Python compilation, integrity manifest and ZIP structure: checked.
- Actual Windows + Ollama + indexed PDF: still requires local acceptance.

## Windows installation and minimal acceptance

Back up local uncommitted edits, overlay the whole source package onto your
existing checkout. Preserve `.git`, `.env`, `backend/data` and the external
`Day2` experiment directory; restart FastAPI after copying.

```powershell
cd backend
python -m pytest -q
cd ..
python scripts/verify_release.py
```

Test the exact student question above in Paper QA / Adaptive Agentic RAG /
Beginner ON / Academic Web OFF. With Memory OFF, expect `3/6 = 50%`, a
step-by-step union, `status=succeeded`, and a paper definition citation. With
Memory ON and a relevant user-confirmed "often confuse set union" note,
expect the same correct result with a clearer explanation of why
`2 + 2 + 2` is not the number of distinct chunks. Check `Run details`:
`question_intent=calculation`, `deterministic_calculation=true`, no
`subquestions_total`, and the appropriate `learning_memory_applied` flag.

Do not delete or rerun the frozen **official-rc2** QASPER experiment. RC4 is a
post-benchmark maintenance version; any RC4 QASPER measurements must use a
new trial ID and be reported separately.

## Known limitations

This deterministic path only handles explicit, small user-provided set
examples with verifiable denominator and literal query sets; arbitrary
mathematical reasoning still uses the LLM and validators. A model response
can still be factually wrong on other question types. The heuristic source
check requires an actual PDF definition passage and may decline to compute
when retrieval did not select that passage. Strict source SHA verification
remains opt-in; default experiment preflight records actual source SHA and
permits new trials, but does not permit source changes within one trial.
