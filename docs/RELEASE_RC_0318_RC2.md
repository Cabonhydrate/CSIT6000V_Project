# v0.3.18-rc2 — Final-Answer Output Integrity (Phase-2 Candidate)

## Why a second release candidate was necessary

The first QASPER Smoke Test exposed a transport-layer output defect: Ollama `qwen3:4b` sometimes put internal planning text in `message.content` even though the request contained `think: false`. In the observed stream the text ended with an orphan `</think>` tag and was followed by the final answer. RC1 preserved all of it, causing severe Answer F1 proxy distortion. RC1 Smoke outputs are diagnostic only.

## Scope (strictly limited)

- `backend/app/agent/model.py`: `final_answer_only` filters model `content` on both chat and stream paths. It accepts only text after the last explicit closing reasoning tag, strips complete `<think>` / `<analysis>` blocks, and **fails closed** on unclosed blocks, reasoning-only outputs, or recognizable planning preambles without tags. The model's `think:false` setting is unchanged.
- The streaming client accumulates model chunks before producing a sanitized final segment. The current Agent already buffers stream output until its own checks complete, so this does not introduce another user-visible streaming delay. Failed/incomplete transport streams now emit **no partial text**. OpenAI-compatible endpoints use the same boundary.
- `artifact_evaluation/formal_qasper.py`: records the **actual backend/app SHA-256** for each experiment. A differing local tree prints a warning rather than blocking a new trial; opt in to strict packaged-source matching with `--strict-source`. Resuming an existing trial still rejects changes in source/data/model/driver, preventing mixed-version results. It also stops the experiment if a reasoning tag somehow reaches persisted final text. Other sampling, mode-ordering, F1 scoring and data selection algorithms are unchanged.
- `backend/tests/test_reasoning_content_boundary_rc2.py`: synthetic transport and real AgentRunner/SQLite integration tests for Basic/Hybrid, JSON responses, split closing tags, OpenAI-compatible responses, incomplete tags, and failed streams.

We did **not** tune the model, prompts, retrieval, Adaptive routing, question sampling, data, or gold answers. The output was corrected at the model-client boundary rather than by manipulating evaluation answers.

## Offline validation results

- Backend: **394/394 passed** (`python -m pytest -q`) plus Python compile checks.
- Phase-2 harness: **9/9 passed** (`python -m unittest discover -s artifact_evaluation -p test_formal_qasper_offline.py`).
- Packaged application SHA-256 reference: `d2269c3e43628ba4627e88ea6f84c21ad3b65295f14c882eb97f01172d4bd9d6` (informational by default; `--strict-source` enables enforcement).

## Validation boundary

The offline tests exercise synthetic HTTP responses. They do not prove that a real local Qwen3:4b run will be correct. Verify with live Ollama before starting the 60-run official comparison. Re-run the whole backend test suite and `python scripts/verify_release.py` after overlay. The frontend is unchanged, so its Windows production build remains a separate acceptance check.

## Preserve experiment integrity

1. Back up or commit local edits. Overlay the **entire** RC2 source package into your repository. Remove any redundant `.py` copies inside `backend/app` (e.g. `model - 副本.py`), otherwise the tool prints a local-source warning. For a reproducible official trial, avoid keeping source copies inside `backend/app`. Do not erase `.git`, `backend/data`, `.env`, or external QASPER data.
2. Run `python scripts/verify_release.py`, `cd backend; python -m pytest -q; cd ..`, and `python -m unittest discover -s artifact_evaluation -p test_formal_qasper_offline.py`.
3. Prefer committing the updated code **and experiment harness** as a new Git commit. A dirty/no-Git tree produces a provenance warning, not a hard failure. Run `python artifact_evaluation/formal_qasper.py preflight --dataset <existing dataset path> --live --model qwen3:4b`.
4. Keep the existing deterministic `qasper_selected_dev.json` and `selection_manifest.json` — **do not prepare a new dataset or select new questions**.
5. Run a fresh `--trial smoke-rc2 --max-runs 3`; inspect final answers (no reasoning trace) and scores. Then run the full `--trial official-rc2` (60 calls). RC1's `smoke` output must **not** be reused as a checkpoint.
6. During the official trial, do not edit application code or evaluation driver. The program independently records source hash, Git commit, dataset hash and experiment driver hash. A new trial does not require matching the package SHA; an existing trial still cannot be resumed using a different source hash.

## Known limitations

- The sanitizer addresses **explicit reasoning delimiters** and common untagged planning openings. It cannot reliably classify every natural-language internal thought without delimiters. It intentionally refuses ambiguous outputs rather than inventing a final answer.
- Basic/Hybrid answers may still be factually incorrect or overly verbose. Sanitizing output does not improve their retrieval or the underlying model's factual reasoning.
- Adaptive abstention and validation failures remain as observed; they are **not fixed or tuned using the Smoke question**. They must be measured in the formal benchmark.
- Answer F1 remains a **token-overlap proxy**, not the official QASPER metric. Human factual and citation assessment remains essential.
- The RC1 Smoke Test's original low F1 values must never be reported as RC2 formal measurements.
- This package does not establish that the Windows frontend or live local Ollama passed acceptance.
