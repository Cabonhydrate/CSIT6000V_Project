PHASE 2 FORMAL QASPER — EVALUATION-ONLY ADD-ON

INSTALL: Copy the artifact_evaluation directory contents into the existing v0.3.18-rc2 repository.
The evaluation harness itself does not modify backend/app or frontend. RC2 source bundle separately includes the model-client output fix. The original day2_qasper.py remains unchanged.

READ FIRST: artifact_evaluation/FORMAL_QASPER_RUNBOOK.md

USAGE: python artifact_evaluation/formal_qasper.py --help

An unchanged copy of day2_qasper.py may be included for standalone testing. In an
existing RC repository the original can be kept unchanged.

Includes prior_exposed_qasper_ids.json: 21 old QASPER dev question IDs used
in Trial1, Trial2 or Holdout6001. Excludes their entire source papers.

This package contains NO live formal model results or real new QASPER gold labels.

RC2: Reuse the previously selected dataset, but run fresh `smoke-rc2` and
`official-rc2` trial directories; do not merge with RC1 `smoke`.
