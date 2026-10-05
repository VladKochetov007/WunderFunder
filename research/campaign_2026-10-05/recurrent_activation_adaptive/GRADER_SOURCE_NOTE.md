# E479 independent grader source receipt

`score_recurrent_adaptive.py` is ready as a source-only independent grader for
the sealed E479 v2 fit. It reconstructs the stacked reset-after GRU equations
directly, applies the two learned bounded alpha values as Tensor inputs to its
own scripted recurrence, checks the strict checkpoint state layout, and
requires the model's normalization buffers to remain identity values. Feature
normalization is recomputed from the frozen 24 fit components before test rows
are read. The checkpoint's fit-order record must equal the seed-derived
permutation of those exact fit groups.

The grader fails closed unless the runner attempt is finished successfully,
unsignaled, torn down, bound to the exact sealed spec digest, and attests the
single sealed `fit_manifest.json` artifact by path, size, and SHA256. It also
checks sealed code/input hashes, the runner runtime source map, fixed E479
source and preflight-v4 identities, full 16-arm matrix, all 474 updates,
telemetry cadence, cohort roles, and the expected data/cohort identities. Raw
selected targets and predictions are checked for finiteness before metric
clipping. The exported NPZ contains predictions for every arm, numeric-sorted
component and row-group IDs, selected raw targets, selected row indices, and
the full need/scored/combined score masks for all 16 components. Both the NPZ
and JSON receipt use exclusive atomic creation and refuse overwrite.

All 28 registered contrasts are kept: 16 within-seed/optimizer shape contrasts,
8 MuonAdamW-minus-AdamW contrasts, and 4 optimizer-by-learned-shape
differences-in-differences. The 2,000 bootstrap samples resample the same
numeric-sorted component multiplicities jointly for every contrast in a seed.
The intervals condition on the fitted weights and exposed components; they do
not quantify seed or search uncertainty and do not establish generalization.

At source validation time the sealed attempt
`e479-learned-candidate-sixteen-fit-v2` was still `running`, with no exit,
teardown, or artifact receipt yet. Therefore no real run, checkpoint, or test
label was opened by this grader work, and no score claim is made. After the
controller records successful completion, the intended command is:

```bash
.venv/bin/python research/campaign_2026-10-05/recurrent_activation_adaptive/score_recurrent_adaptive.py \
  --run-dir research/campaign_2026-10-05/recurrent_activation_adaptive/runs/sixteen_v1 \
  --seal research/campaign_2026-10-05/recurrent_activation_adaptive/e479_sixteen_fit_v2.sealed.json \
  --attempt research/campaign_2026-10-04/runner/attempts/e479-learned-candidate-sixteen-fit-v2.json \
  --output research/campaign_2026-10-05/recurrent_activation_adaptive/eda/sixteen_v2/RESULT.json \
  --device cpu
```

The grader also validates the complete campaign-spec schema and successful
systemd unit result/status. It compares the independently recomputed fit-only
normalizer exactly against the normalization embedded in the fit-manifest
cohort receipt. Both output paths are resolved and checked before any sealed
input, checkpoint, or dataset reads.

Source hashes:

- grader: `83ec9a2a64b34b1d6532e1785ff2cb3d95d5bfec686f20e13b84558a45018346`
- synthetic tests: `26e5abf173e7332ba3b7de66fda4fc73ab240fd16eeb9e3f7e2e4462e11607eb`
- bound actor: `6a11025735bb8df7d88b08bdcf7dfd9d8879992a82c1c64d40b43ada7456c7bf`
- bound CPU preflight v4 receipt: `92fcb33ea416527dc742911605da9059eb1d7c728a4a594ed9e5a65dae5bbb66`

Validation was synthetic-only: Ruff format/check passed, and the four grader
tests passed in 10.67 seconds after adding fixed01/learned01 output and state
parity from shared initialized weights. Tests exercise model parameter counts, alpha
gradient connectivity and initialization, identity buffers, fixed/learned parity
at both alpha initializations, carried chunk state, future causality, sorted-component
bootstrap registration, no-overwrite behavior, invalid unit/spec receipts,
normalizer mismatches, and occupied output paths. No GPU was used and no
validation target rows were opened.
