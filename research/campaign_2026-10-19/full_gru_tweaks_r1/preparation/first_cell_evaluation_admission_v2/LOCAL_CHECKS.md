# Local checks

Only the new ownership boundary fixture was run; the unchanged V1 fixture and broader suites were not replayed.

```sh
./.venv/bin/python -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v2/test_gpu_owner_boundary_v2.py
./.venv/bin/python -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v2/prove_v2_inverse.py
./.venv/bin/ruff format --check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v2/REMOTE_SUBMISSION_V2.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v2/ADMISSION_BODY_TEMPLATE_V2.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v2/render_admission_remote_v2.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v2/test_gpu_owner_boundary_v2.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v2/prove_v2_inverse.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/V5_STDLIB_HELPERS_EXTRACTED.py
./.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v2/REMOTE_SUBMISSION_V2.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v2/render_admission_remote_v2.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v2/test_gpu_owner_boundary_v2.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v2/prove_v2_inverse.py
```

The new fixture passed 3/3 tests. The byte proof returned `PASS_EXACT_SINGLE_FUNCTION_INVERSE`, with identical bytes before and after `gpu_owners` and an exact reconstruction of V1. Ruff format and check passed. The V1 7-test fixture was not rerun. No production-body import/execution, model or data payload read, B access, staging, admission, or remote action occurred.
