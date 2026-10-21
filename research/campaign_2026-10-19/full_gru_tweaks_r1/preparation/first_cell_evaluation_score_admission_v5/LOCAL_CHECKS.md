# Local checks

The focused V5 receipt-preflight fixture passed using only the exact local AllThree fit admission/launch receipts plus synthetic stat rows:

    .venv/bin/python -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v5/test_score_admission_v5_fit_receipt_preflight.py
    Ran 4 tests
    OK

The inverse-byte proof passed:

    .venv/bin/python -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v5/prove_score_admission_v5_inverse.py
    PASS_EXACT_NARROW_V4_RECONSTRUCTION

Ruff formatting and lint passed for only the new V5 Python files:

    .venv/bin/ruff format --check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v5
    4 files already formatted
    .venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v5
    All checks passed!

No V4 or other historical fixtures/suites were rerun. The V5 renderer was not invoked. There was no B query, staging, admission, launch, score execution, or quality result. Exporter V3, scorer V3, and the next export's root acceptance/exact-stop are still required before rendering.
