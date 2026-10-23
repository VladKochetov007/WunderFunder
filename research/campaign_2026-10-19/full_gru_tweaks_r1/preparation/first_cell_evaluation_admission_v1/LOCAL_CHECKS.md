# Local checks

The exact admission stdin body was rendered from the fixed five payloads and the frozen V5 stdlib helper spans. Its local source binding also pins the passing exporter/scorer review `EXPORT_SOURCE_REVIEW_V1.json` at 9,633 bytes, SHA-256 `cf44af7624e81ee99e9705bf08e69b359f7048354df069cad35b26c7ec927c15`.

Final focused validation used the literal maintenance venv paths:

```sh
./.venv/bin/python -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/test_admission_packet_v1.py
./.venv/bin/ruff format --check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/REMOTE_SUBMISSION_TEMPLATE.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/render_admission_remote_v1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/extract_v5_stdlib_helpers.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/test_admission_packet_v1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/ADMISSION_BODY_TEMPLATE_V1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/V5_STDLIB_HELPERS_EXTRACTED.py
./.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/REMOTE_SUBMISSION_TEMPLATE.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/render_admission_remote_v1.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/extract_v5_stdlib_helpers.py research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/test_admission_packet_v1.py
```

The focused fixture passed all 7 tests. Ruff format and check passed on the exact rendered body and the local renderer/extractor/fixture. The body template and extracted helpers were format-checked as compositional source fragments; Ruff lint ran on their complete rendered body because the separate fragments intentionally rely on each other's imports and helper definitions.

Earlier exploratory commands were not treated as validation: a generic `python -m unittest research.campaign_2026-10-19` invocation discovered 0 tests, and bare `ruff` / `python -m ruff` were unavailable. No installation was attempted. Those observations do not describe the final literal maintenance-venv checks above.

No WNN/Torch/NumPy/exporter import, model or data payload read, unchanged suite, B access, staging, admission, or remote action was performed by this packet author.
