# Causal-stem fixed-endpoint readout R3

`readout_stem_r3.py` re-exports and independently scores the six already-completed
E6 checkpoints. It does not train, retry, or modify any fit output. R1 and R2
attempts remain preserved under their original paths.

## Why R3 exists

The native R2 attempt stopped in `validate_fit` before any prediction export.
The plain cells were rejected because source-normalized configuration omits the
default `model.input_stem_kernel: 0`; R2 compared the missing key as `None` to
zero. R3 applies the same effective default already used by its admission check.
The K1/K16 cells were rejected because R2 reconstructed train membership as
`train[0:10325]`. The official train roster has 10,325 sparse sequence IDs; the
verified split manifest, not a contiguous range, defines the exact roster.

R3 compares each run's full `fit_members` and `evaluation_members` lists against
the exact lists in the SHA-pinned chronological manifest via the frozen,
SHA-pinned R2 common validator. It retains source-normalized config identity,
raw config and admission receipts, model/checkpoint hash, exact split/data
fingerprints, no-teacher and training-only ancestry, all-fit normalization,
assigned L4 identity, E1–E6 event coverage, deadline, prediction manifest and
independent-score checks. No membership or ancestry check is relaxed.

## ROOT launch

ROOT stages this new receiver separately as
`/data/lc-epanov_s/tabdata-stem-20261110-r1/readout-receiver-r3/readout_stem_r3.py`.
It consumes the same six on-time fit results and the original first-profile
clock/admission. The helper remains the exact frozen R2 common file at
`/data/lc-epanov_s/tabdata-next70-20261110-r1/readout-receiver-r2/readout_r2.py`.

```bash
/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python \
  /data/lc-epanov_s/tabdata-stem-20261110-r1/readout-receiver-r3/readout_stem_r3.py \
  --root /data/lc-epanov_s/tabdata-stem-20261110-r1 \
  --source /data/lc-epanov_s/tabdata-stem-20261110-r1/source-r1 \
  --python /tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python \
  --old-src /data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1/source-gru-monitor-resume-r1/src \
  --r2-helper /data/lc-epanov_s/tabdata-next70-20261110-r1/readout-receiver-r2/readout_r2.py \
  --admission-sha256 <ROOT-provided-64-hex-SHA256>
```

R3 uses only new paths: `control/readout-r3/` and
`runs/{cell}/final-readout-r3/`. It preserves the original 3-hour fit and
4-hour collection cutoffs, per-cell export/scoring limits, 2.5 GB aggregate
readout output cap, and 4 GB free-disk reserve. No new fit is started.

## Interpretation

Each complete cell must export all 1,873 official-valid sequences with shape
`[20000, 2]` and float32 predictions, then score all 3,528,273 selected rows.
The independent scorer must agree with that run's fixed-E6 full-valid monitor
within `1e-5`. Paired contrasts remain withheld unless all six cells pass.
This re-evaluates repeatedly exposed validation for prediction/scorer agreement;
it is not fresh confirmation or a public-score result.

The local R3 regression covers normalized K0 config identity, exact sparse
membership acceptance, rejection of a contiguous-range substitute, receipt
time cutoffs, paired contrasts, and pinned-helper namespace rebinding.
