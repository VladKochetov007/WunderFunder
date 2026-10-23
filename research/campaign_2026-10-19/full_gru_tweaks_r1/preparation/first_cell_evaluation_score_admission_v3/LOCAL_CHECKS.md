# Local checks

This V3 packet remains conditional and unbound. The new focused fixture imports only this stdlib body module and uses injected readers for synthetic paths. It does not execute `main`, inspect `/proc`, query a GPU, import WNN/scorer code, access model/data/cache/prediction payloads, or contact B. V1 and V2 fixtures were not replayed.

The V3 policy fixture passed:

```text
.venv/bin/python -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v3/test_score_admission_v3_owner_scope.py
Ran 1 test
OK
```

It verified that a foreign UID causes one bounded cgroup read and zero cmdline reads, a same-UID owner causes bounded reads of both files, the worker call-site passes the observed UID and current UID, the campaign cgroup predicate rejects any UID, and other top-level body AST nodes match the preserved V2 source.

Ruff passed only on the new V3 Python files:

```text
.venv/bin/ruff format --check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v3/SCORE_ADMISSION_BODY_TEMPLATE_V3.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v3/render_score_admission_v3.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v3/test_score_admission_v3_owner_scope.py
3 files already formatted
.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v3
All checks passed!
```

No renderer invocation, remote staging/admission, score execution, or quality result is claimed.
