# Exact failed R1 source snapshot

This immutable four-file snapshot preserves the exact recipe, runner, manifest and admission used by the failed first execution. The recipe-key mismatch stopped before data values; see the [failure receipt](../EXECUTION_R1.json) and [traceback](../EXECUTION_R1.log). R2 fixes the integration contract without rewriting these bytes.

- [DIRECTION_PROBE_ADMISSION_R1.json](DIRECTION_PROBE_ADMISSION_R1.json) — exact R1 input; SHA-256 `b7a2fd33e735fd143db940aefe839fa44e9fabaf563e466b57674916c540432f`.
- [DIRECTION_PROBE_RECIPE_R1.json](DIRECTION_PROBE_RECIPE_R1.json) — exact R1 input; SHA-256 `1db1dfd44835956f4d297c469d2e1d9e6cbd6de9d9f63cd28c74510b9c314c65`.
- [DIRECTION_PROBE_SOURCE_MANIFEST_R1.json](DIRECTION_PROBE_SOURCE_MANIFEST_R1.json) — exact R1 input; SHA-256 `ba7bafea473ae81a64921fbc145b7360577a2a6a27d5b8d6d9af297df02d7951`.
- [run_direction_probe.py](run_direction_probe.py) — exact R1 input; SHA-256 `9e3174a3bd2cd1f5c243e4437087559af13230cda8b5ee360ba3898e237afcd0`.
