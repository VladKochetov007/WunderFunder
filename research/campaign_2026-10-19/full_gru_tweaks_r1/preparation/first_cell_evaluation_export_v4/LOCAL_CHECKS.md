# V4 local source checks

The focused V4 input-lead metadata fixture passed five tests with the maintenance virtual-environment interpreter:

```text
/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/python -B -m unittest discover -s research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v4 -p 'test_export_v4_input_lead_metadata.py' -v
Ran 5 tests — OK
```

Ruff formatting and checking passed on `export_first_cell_v4.py` and `test_export_v4_input_lead_metadata.py`, using `/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/ruff`.

The fixture reads the exact local CONTROL `ADMISSION.json` and `LAUNCH.json` metadata, verifies their fixed byte/SHA pins and trainer argv digest, and reads source text from commit `68ff682977b822f38351aac0f621146f5f13b440` only after matching each file to the bundled source-manifest row. It AST-compiles only the pure V4 argument-check helper. It does not import the exporter, canonical training code, WNN, PyTorch, or model code, and does not read checkpoint, cache, parquet, target, prediction, or array payloads. No B action occurred.

An initial unittest invocation by dotted module path failed before discovery because the campaign directory name contains a hyphen. The corrected `unittest discover` command above ran the five focused fixtures successfully; no production or legacy test suite ran.
