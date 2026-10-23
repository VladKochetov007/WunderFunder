# Local checks

This packet is still conditional: the expected ROOT export acceptance and exact-stop files have not been created, and their byte/SHA-256 pins remain unset. The renderer correctly refuses to produce a submission body while those receipts are unbound. No remote host was queried, no staging or launch occurred, and no export, prediction, target, model, cache, or score payload was opened.

Focused local metadata tests passed:

```text
.venv/bin/python -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v1/test_score_admission_v1.py
.....
Ran 5 tests
OK
```

The fixtures exercise synthetic export-acceptance and exact-stop metadata only. They cover strict native exit integers, matching acceptance/stop identity, zero stop result with empty streams, failed/incorrect stop rejection, the no-premature-score condition, and the renderer's fail-closed unbound state. They do not import the scorer, load source/model/data modules, inspect arrays, contact B, or execute an admission body.

Ruff passed for the new packet:

```text
.venv/bin/ruff format --check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v1/SCORE_ADMISSION_BODY_TEMPLATE_V1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v1/render_score_admission_v1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v1/test_score_admission_v1.py
3 files already formatted
.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v1
All checks passed!
```

No unchanged repository suite was run. These checks validate only the local conditional receipt gate and do not authorize rendering, staging, admission, scoring, or a quality claim.
