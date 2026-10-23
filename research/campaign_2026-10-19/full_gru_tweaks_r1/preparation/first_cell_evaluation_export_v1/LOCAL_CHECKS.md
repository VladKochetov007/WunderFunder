# Local checks

The focused fixtures use tiny synthetic CPU metadata, IDs, and NPY values. They do not import the WNN source tree, load any checkpoint/model/cache/data payload, run an unchanged suite, or contact B.

Validated locally:

- `ruff format` on the exporter and its focused fixture.
- `ruff check` on the exporter and its focused fixture.
- Fifteen focused unit fixtures passed on the immediately preceding source revision, including exact local AllThree admission joins for all six source identities, the approved portable-registry byte/SHA and source-entry join, the fixed registry environment path, legacy CONTROL representation metadata, strict reader sequence-ID typing, the registered 1,873-row-group component inventory, the exact validation-data parent alias, and the pre-mutation disk-space reserve gate. After adding the six-path stat-only observer, its new focused synthetic fixture passed separately with `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest test_export_first_cell_v1.FirstCellExportContractTests.test_six_source_stat_observer_checks_registered_aliases_without_reading` from this directory. The unchanged fixtures were not replayed after that narrow addition.

No prediction export, scorer, metric, or quality evaluation was run. No B staging, admission, SSH, or remote execution was performed by this local implementation step.
