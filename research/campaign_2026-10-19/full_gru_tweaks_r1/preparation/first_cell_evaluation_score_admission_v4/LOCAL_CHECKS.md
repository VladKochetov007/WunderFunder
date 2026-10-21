# Local checks

The new V4 binding-join fixture passed:

```text
.venv/bin/python -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v4/test_score_admission_v4_binding.py
Ran 1 test
OK
```

It read only the small binding/review/dependency metadata and source text. It checked exact V2 binding bytes/SHA and flat support paths, the independent producer/scorer review pin, V4's binding/review/scorer constants, matching renderer source pins, the dependency manifest/module pins, and V3-to-V4 body AST equality after normalizing only those source pin assignments. It did not run the renderer, import the scorer or source tree, inspect `/proc`, query a GPU, or access model/data/cache/prediction payloads. No previous fixtures were rerun.

Ruff passed for the three new V4 Python files:

```text
.venv/bin/ruff format --check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v4/SCORE_ADMISSION_BODY_TEMPLATE_V4.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v4/render_score_admission_v4.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v4/test_score_admission_v4_binding.py
3 files already formatted
.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v4
All checks passed!
```

V4 remains a local conditional template with root-owned export acceptance and exact-stop pins unbound. No renderer invocation, B staging/admission, score execution, or quality result is claimed.
