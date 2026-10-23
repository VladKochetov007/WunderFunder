# Local checks

The only executed checks were the focused fixtures for this new stop template
and Ruff on these two Python files. No prior packet tests or broad suites ran.

- `.venv/bin/python -m pytest research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_stop_v1/test_stop_score_template_v1.py -q`: **11 passed**.
- `.venv/bin/ruff format research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_stop_v1/STOP_SCORE_TEMPLATE_V1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_stop_v1/test_stop_score_template_v1.py`:
  both files already formatted on the final pass.
- `.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_stop_v1/STOP_SCORE_TEMPLATE_V1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_stop_v1/test_stop_score_template_v1.py`:
  **All checks passed**.

The source remains conditional and unbound. No B/systemd query, stop, score,
model, prediction, dataset, or cache access occurred. Independent source review
is pending and required before any future ROOT binding or use.

Final source pin: `STOP_SCORE_TEMPLATE_V1.py`, 11,984 bytes,
SHA-256 `c9647858157b4c1d33e104a89321556a5ce67dfa19e26cae53ad7972faad03ef`.
Final fixture pin: `test_stop_score_template_v1.py`, 4,752 bytes,
SHA-256 `ac798b12cf027791de504ff1f880e0109604a3e9a37625710fc88af1824ae5bd`.
