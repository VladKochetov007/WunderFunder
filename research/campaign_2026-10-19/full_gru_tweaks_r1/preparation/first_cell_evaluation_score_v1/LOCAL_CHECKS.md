# Local checks

The final local preparation was checked with the repository's literal `.venv/bin/ruff` and `.venv/bin/python` executables.

- `.venv/bin/ruff format research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v1` — passed; 7 files unchanged.
- `.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v1` — passed.
- `WNN_SOURCE_ROOT=/Users/nb/Documents/wunder/WunderFunder-strong-gru-tweaks-20261019 .venv/bin/python -B -m pytest research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v1/tests/test_score_first_cell.py -q` — 6 passed. The fixtures use tiny synthetic arrays and the pinned source68 metric helpers; they cover canonical batch parity, source order, nonfinite rejection, zero-weight and zero-scored rows, repeated component membership, and the registered-source metadata contract.

The focused tests do not invoke the scorer CLI, read an export, map predictions, open model/data/cache payloads, run inference, or perform scoring on B. No unchanged suite was run. The local scorer remains source preparation only.
