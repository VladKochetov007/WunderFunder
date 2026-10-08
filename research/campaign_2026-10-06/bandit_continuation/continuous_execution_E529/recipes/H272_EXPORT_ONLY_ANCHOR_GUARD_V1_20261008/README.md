# H272 export and frozen-comparison successor

H272 inherits a parent-anchor guard defect from H233. The actual C1 producer
sidecar binds its valid prediction to `model_ep1.pt` SHA
`7d34ee5649e124948f0b651aa786083fa9ee66a01c439b034345a1d9df9ec3f0`, while
the distinct `model.pt` alias has SHA
`5f21361d47cc1f010abb363e6aca6e3cbf50a3d5d40d88ad81851aed7519fc4d`. The
frozen H272 exporter compares the sidecar field to the alias hash, so it
rejects the real producer artifact.

`h272_export_only_anchor_guard.py` verifies the unchanged 55-member H272
runtime lock and recipe, then permits only valid-split `export`. It replaces only
`_validate_anchor_reference`'s checkpoint-hash comparison with the pinned
producer checkpoint hash. The original exporter still verifies the alias
file hash and tensor equality, parent prediction bytes, row order, fixed
epoch, and destination. The actual H233 producer sidecar is copied byte for
byte into `tests/fixtures` and pinned by SHA for a data-free regression.

`H272_DIRECT_LABEL_COMPARISON_PREREG_V1_20261008.json` and
`grade_h272_direct_label_readouts.py` freeze the H233 comparison design for
H272. They reuse the exact reviewed H233 mask/alignment helpers, fixed E540
paired scorer, H218 row/metric helpers, and official scorer, each hash-pinned.
The primary comparisons remain each trained readout against H272's hidden-only
head. The secondary comparisons remain each head against the same frozen C1
parent at raw blend weights 0, 0.1, 0.25, 0.5, and 1, with official clipping
inside the unchanged metric after blending and the same 2,000 paired-component
bootstrap. The grader additionally validates the actual parent sidecar's
producer, valid split, epoch, FP16 precision, and complete row order.

The adapter and grader are source preparation only. Root review, fit/export
terminal artifact admission, and a separate root grade authorization are
required; neither export nor grading has been run here.

After fit and root export review, the existing contract's export argv should
use this adapter path instead of the frozen exporter path:

```sh
PYTHONPATH=research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/recipes/H272_DIRECT_LABEL_READOUT_V1_RUNTIME/src \
.venv/bin/python research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/recipes/H272_EXPORT_ONLY_ANCHOR_GUARD_V1_20261008/h272_export_only_anchor_guard.py export \
  --checkpoint artifacts/runs/bd26_h272_direct_label_c1s3_rank32_f0_v1/model_ep1.pt \
  --split valid \
  --output-dir research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/exports/H272_C1S3_DIRECT_LABEL_E1_VALID_V1
```

That command remains gated by the existing H272 export resource contract and
fresh root authorization. The grade command is intentionally omitted until
root has reviewed the frozen preregistration and grader source and admitted the
actual fit/export artifacts.
