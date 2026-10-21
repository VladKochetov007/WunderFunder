# Local checks

This V2 packet remains conditional and unbound. The focused fixture imports only the new stdlib-only body template; it does not execute `main`, inspect `/proc`, query a GPU, import WNN/scorer code, access model/data/cache/prediction payloads, or contact B. The five V1 fixtures were not replayed.

The new V2 policy fixture passed:

```text
.venv/bin/python -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v2/test_score_admission_v2_owner_scope.py
Ran 1 test
OK
```

It verifies no reader calls for a foreign UID, exactly bounded cgroup/cmdline reads for a same-UID owner, the `check_selected_gpu` UID arguments, and that other top-level body AST nodes match the preserved V1 source.

Ruff was run only on new V2 Python files:

```text
.venv/bin/ruff format --check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v2/SCORE_ADMISSION_BODY_TEMPLATE_V2.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v2/render_score_admission_v2.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v2/test_score_admission_v2_owner_scope.py
.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v2
```

No renderer invocation, remote staging/admission, score execution, or quality result is claimed.
