# E488 result renderer source note

This is a result-only renderer for the eventual frozen `eda/duration_v3`
artifacts. It reads the score JSON, fit-only proof JSON, fit and grade attempt
receipts, and root arithmetic reproduction. It hashes the saved selected
prediction NPZ for provenance but never deserializes it. It does not read the
parquet data, checkpoints, or test rows, invoke the scorer, or recompute score,
contrast, or bootstrap arithmetic.

Before rendering, it requires the exact 24 seed/arm/epoch score cells, 27
global contrasts, 27 contrast records with two targets each (54 target
values), 2,000 shared component-bootstrap draws, successful fit and grade
attempt receipts, artifact path/hash/size matches, and a root reproduction
bound to the score/export hashes. Outputs are the full arm/contrast CSVs,
three figures, and a report that states the exposed-panel and conditional
interval limits. The script refuses to overwrite existing outputs.

No E488 result JSON or raw grade proof was consumed during renderer
development. Three synthetic tests passed under the hard-limited 1 CPU / 2
GiB `scripts/limited` runner, covering complete-matrix validation, a missing
arm rejection, and output row counts. Ruff, `py_compile`, and `git diff
--check` also pass.

- `render_report_v3.py`: SHA-256
  `b5c5235324dcaffbde622f76ba281c974af29b1b73311f06bed2b7179720a81b`
- `test_render_report_v3.py`: SHA-256
  `cbcbb304223155f754279213491841e67ff16a08499f2fbfe3d091d57821ed50`

Render only after the parent controller confirms the official grade and root
arithmetic reproduction are complete. Supply all saved receipt paths explicitly
to the CLI; no default path is assumed.

## Final v5 rendering addendum

After the official grade and independent raw-export reproduction completed, the
final result-only renderer validated both the raw arithmetic receipt and the
root provenance wrapper. It checked all 24 snapshots, 27 global contrasts, 54
target-specific contrasts, all 2,000-draw summary arrays, and the score/export
hash bindings. The reproduced maximum absolute error was `4.771194290348335e-15`.
The renderer never opened the NPZ contents or loaded training/evaluation data.

The v5 plots use horizontal interval segments and point markers, so they remain
valid if a percentile interval does not contain its point estimate. Duration
interaction rows are labeled `E3−E1`, not as an epoch-4 snapshot. The v4
display artifact with its `nan` field and incorrect tick is preserved in
`REPORT_v4_DISPLAY_INCIDENT.md` and `eda/duration_v3/rendered_v4_display_incident/`;
the canonical report and plots are the v5 files.

- `render_report_v5.py`: SHA-256 `fa00ae17e4b533387b1812989466cbeb9b744c325e8c6f35fb0aea21eb2e1ba7`
- `test_render_report_v5.py`: SHA-256 `f6150865e399cd7d0f950901ba3d651767c0585303f1d1958f7a15ffb5664ae3`
- `REPORT.md`: SHA-256 `e5457e66b2f6e88332a26258a640b8099cda02f25e348de30b0f29bd9543eb78`
- Root raw arithmetic receipt: `checks/context_duration_export_reproduction.json`, SHA-256 `d94a44ed9eca5435db07c7b0a8e4b046bb6b5700412550edc23513b9df6281f6`
- Root provenance wrapper: `checks/context_duration_export_root_reproduction.json`, SHA-256 `366bc223c2759c7aa600f328ba20e4f83fa9a3d98634a6a7bfcd383d8ad5421a`

Six renderer tests passed under `scripts/limited` at one CPU and 2 GiB RAM;
Ruff passed. The full temporary-root pipeline fixture validates receipts,
renders figures, and checks the formatted maximum error without reading raw
prediction arrays.
