# Local checks — score result capture V1

Scope: one new synthetic metadata/boundary fixture and Ruff on this folder's reader and fixture only. No B/SSH access, remote queries, actual receipt binding, scorer execution, old reader test replay, model/data/cache/prediction payload access, or quality evaluation.

Commands run from the maintenance repository root:

```text
.venv/bin/ruff format research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_capture_v1/SCORE_RESULT_CAPTURE_TEMPLATE_V1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_capture_v1/test_score_result_capture_v1.py
.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_capture_v1/SCORE_RESULT_CAPTURE_TEMPLATE_V1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_capture_v1/test_score_result_capture_v1.py
.venv/bin/python -B -m unittest discover -s research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_capture_v1 -p 'test_score_result_capture_v1.py' -v
```

Results: Ruff formatting reported both files unchanged; Ruff check passed. All six new synthetic fixture cases passed: unbound gate ordering, reused-helper operational-AST correspondence, synthetic receipt joins and invocation rejection, strict terminal-success/result-read gating, bounded log tail/final-symlink rejection, and no-stdin/no-submission/no-payload-import boundary. The old result-reader fixtures and all unchanged suites were not run.
