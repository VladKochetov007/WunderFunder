# V5 local checks

Run the focused commands with the literal maintenance virtual environment:

```text
/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/ruff format render_admission_remote_v5.py test_admission_v5_actual_receipts.py test_admission_v5_export_review_contract.py prove_admission_v5_inverse.py
/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/ruff check render_admission_remote_v5.py test_admission_v5_actual_receipts.py test_admission_v5_export_review_contract.py prove_admission_v5_inverse.py
/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/python -B test_admission_v5_actual_receipts.py
/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/python -B test_admission_v5_export_review_contract.py
/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/python -B prove_admission_v5_inverse.py
```

The unrendered body template intentionally relies on helper names prepended by the pinned V1 stdlib extraction. Ruff checks apply to the standalone local scripts and the complete rendered body, not to the partial template. The focused tests exercise the actual small receipt metadata and the actual V3 review `reviewed_files` shape. They do not import the model/project, read source/model/checkpoint/array/data payloads, replay unchanged suites, or contact B.

The rendered `REMOTE_SUBMISSION_V5.py` is 248,948 bytes / `e24a6fc87e72a43056d30e1b4edf2cb12e410a37f549a858e89420e54cdb38c0`; its binding is 17,800 bytes / `76359e9a889ed9f0647d4aeb901c01e1b7cfa9b44bf885481bf1cd7c87b7fd1c`. The inverse proof is 4,645 bytes / `030c9d58e9ca89e3775ed93e790ec664e1cce2883c01234fd4cf4146b40f5d05`, status `PASS_EXACT_V4_BODY_INVERSE_OUTSIDE_FIXED_PAYLOAD_AND_IDENTITY_SLOTS`. Both focused fixtures passed (3/3 actual-receipt checks and 4/4 review/history checks); Ruff format and check passed for the standalone scripts and rendered body.

Preserved history in the V5 binding separates the earlier V2/V3 admission preflights (zero worker units) from the first actual export unit failure. The latter is root-only historical metadata; the worker's old unit is not queried, stopped, retried, or restaged. A separate independent exact-body source review and root-owned fresh admission are required before any remote action.
