# Exact original fbc1k native replay

Generated receiver output for the [original-member diagnostic](../README.md), using [the frozen ZIP receipt](../receipt.json) and [one-CPU offline command](../run_cpu.sh). `predictions.npy` is written incrementally and must not be consumed before `report.json` exists with `validation_completed: true` and a matching prediction SHA256. The report will bind all 1,873 source row groups, official scored rows, state resets, native library and measured timing. This is exposed train-only validation evidence, not a new holdout or portal result.
