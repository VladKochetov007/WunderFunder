# Local checks

The V2 metadata-only fixture passed:

```text
.venv/bin/python -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v2/test_score_v2_binding_metadata.py
Ran 1 test
OK
```

It read only the two small local binding JSON files and scorer source text. It verified the new binding bytes/SHA, the two flat support paths, equality of all other binding fields, the V2 constants, and AST equality for the scorer core after normalizing only the two binding constants. It checked the V2 dependency manifest pins for the scorer core and fixture. The scorer was not imported or executed; no source tree, WNN, model, prediction, target, dataset, cache, or B payload was accessed. No V1 tests were rerun.

Ruff passed for the five new V2 Python files:

```text
.venv/bin/ruff format --check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v2/first_cell_score_contract.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v2/first_cell_score_core.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v2/first_cell_score_predictions.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v2/score_first_cell.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v2/test_score_v2_binding_metadata.py
5 files already formatted
.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v2
All checks passed!
```

The independent producer/consumer review is [EXPORT_SOURCE_REVIEW_V2.json](../first_cell_evaluation_export_v2/EXPORT_SOURCE_REVIEW_V2.json), 10,250 bytes / SHA-256 `f736f60a6b5481e94be380a183ba840aad850958c9b863b439bc708e65e997af`, status `PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR`. It is source-only and does not authorize export or score execution.
