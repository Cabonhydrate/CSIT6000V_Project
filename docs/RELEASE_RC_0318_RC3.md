# v0.3.18-rc3 — Conservative Numeric-Example Recovery (After Phase 2)

## Context

The completed 60-run QASPER benchmark was executed on **v0.3.18-rc2**.
**RC3 is a new, post-benchmark maintenance revision.** Do not overwrite the original
`official-rc2` run journal or claim that RC3 produced the existing formal results.

A live Paper QA query with confirmed learning Memory **disabled** failed:

> Can you explain how Chunk Coverage is calculated in this paper?

The first model answer gave a correct, explicitly hypothetical worked example:

> For example, if a test suite ... 5 unique chunks ... 20 chunks,
> CC = 5/20 = 0.25.

However, the conservative numeric provenance check rejected `0.25` as an
unverified measured result, because its English hypothetical-marker expression
recognized `for example if` but not `For example, if`. The second (bounded)
repair attempt yielded empty JSON and the Agent returned `partial`. The run
metrics proved that Memory was disabled (`learning_memory_applied=false`); the
issue was **not caused by Memory**.

## Changes

- `backend/app/skills/numeric_provenance.py`: recognize common explicitly
  hypothetical English forms including `For example, if`, `For instance, if`
  and `e.g., if`, with optional colon or comma. *Reported paper results* do not
  acquire an exemption, and incorrect arithmetic still fails validation.
- `backend/tests/test_numeric_provenance_p0a.py`: five new regressions for the
  exact failure and neighboring safety boundaries, including an end-to-end
  mocked AgentRunner test with Memory OFF and a single successful answer attempt.
- `artifact_evaluation/formal_qasper.py`: update **optional strict** packaged
  SHA-256 reference for RC3. This does not change sampling, scoring or the
  selected QASPER cohort. Default preflight remains informational on source hash;
  the journal-resume check still prevents mixing multiple source versions.

No retrieval, Chunking, Memory selection, Skills routing, model prompting,
QASPER data, or TACC deployment was modified.

## Offline checks

- Backend: `399 passed` (`python -m pytest -q`).
- Formal harness: `9 passed` (`python -m unittest discover -s artifact_evaluation -p test_formal_qasper_offline.py`).
- Python compile and packaged-source integrity check passed.
- Actual Windows/Ollama end-to-end behavior requires local acceptance.

## Installing into an existing local repo

Back up or commit uncommitted edits; extract and overlay the **whole** RC3
source directory. Do not erase `.git`, `backend/data`, `.env`, or the external
QASPER data. Restart the backend service to load the new validation code.

```powershell
cd backend
python -m pytest -q
cd ..
python scripts/verify_release.py
```

Rerun the original Paper QA + Beginner Skill query with Memory OFF and inspect
`numeric_provenance` in Run details. A synthetic, correctly calculated example
should report `kind=hypothetical`, `verified=true`, with no numeric grounding
issue. Test an unsupported reported experiment score separately; it must still
be rejected. No PDF reupload or index migration is required.

### Protect the completed formal experiment

The existing `official-rc2` report is valid only for the RC2 source that
produced it. Do not `run --trial official-rc2` or try to resume its journal
using RC3 code. To evaluate RC3, always choose a **new** trial name.
The model-independent human review of the RC2 results is not changed by this
small maintenance fix.
