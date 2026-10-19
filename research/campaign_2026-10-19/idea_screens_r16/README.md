# R16 conditional-innovation graph screen

**Current status: LOCAL PREPARATION / REMOTE HOLD.** See the [allocation hold](ALLOCATION_HOLD_20261019.md). No R16 staging or job occurred; the user-driven joint candidate is a separate [fixed CPU comparison](../idea_combo_r1/DESIGN.md).

This folder contains source-prepared code for the fixed H-NF19-10 input-only screen in [DESIGN.md](DESIGN.md). No sample arrays, models, or checkpoints have been opened locally, and no B-side job has been run by this preparation task. Root owns staging, resource admission, and any actual execution.

## Fixed question and scope

The screen asks whether any of six prespecified directed input-block pairs has one-step innovation information beyond its destination block's own past and a fitted three-component common linear control. It uses `x[t-1]` to predict `x[t] - x[t-1]`, with fit-only scaling, PCA control, source residualization, and ridge-100 baseline/correction fits. It is an input-predictive proxy, not a target-label score, causal graph, fresh-fold result, or full-panel claim. The exact edge gate is frozen in the design: at least 1% relative SSE reduction in both component-half directions and a positive median reduction over all 32 held components in both directions; any undefined component denominator makes that median/gate undefined.

## Pinned panel and counts

The bound R3 sample is `b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64`, membership is `e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217`, and the exposed split is `c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7`. The metadata describes 96 complete 20,000-row sequences: 64 fit components and 32 evaluation components. The estimator reads only the 64 fit `x` and scored-mask arrays (128 NPY files hashed/mapped for integrity), selecting 497 original-mask stride-40 rows per fit component. Each fixed half has 15,904 rows; each direction trains on one half and evaluates on the other. No target-value or evaluation-array payload is used.

There are six ordered edges, evaluated in two directions (12 directional evaluations): six baseline ridge fits, 12 source projections, and 12 correction ridge fits. The PCA covariance uses `n_fit_rows - 1`; the source residualizer applies the frozen relative eigenvalue cutoff described in the design. This is not a Torch model or a training run.

## Source and local checks

[PREPARATION_MANIFEST.json](PREPARATION_MANIFEST.json) pins 18 files, including the nine-module runtime closure. Its SHA-256 is `88a584baceeac9efba05c1697cece83baf5fa55cfcaa990523ae16f9af97074a`. The independent design review is [DESIGN_REVIEW.json](DESIGN_REVIEW.json); the earlier proposal and review are preserved as `DESIGN_V1.md` and `DESIGN_REVIEW_V1.json`. [TEST_RECEIPT.json](TEST_RECEIPT.json) records 11 passing synthetic tests, Ruff checks, the no-payload cold-import smoke, and raw stdout hashes. The smoke matched all nine expected module paths; `numpy.load` and `numpy.memmap` sentinels were untouched, and Torch/PyArrow were absent. Six NumPy matrix-multiplication RuntimeWarnings appeared during a synthetic transform test; outputs remained finite and the cause was not established. No actual sample/model execution or remote action occurred.

## B-side invocation contract

Root's planned source location is `B/support-source/eda19-r16-v1`; the accepted sample is `B/research_tmp/nf19-sample-r3-v2`, with output at `B/research_tmp/nf19-r16-v1`. The required CLI is:

```sh
PYTHONPATH="<B>/support-source/eda19-r16-v1" \
  python -B -m idea_screens_r16.run_probe \
  --sample "<B>/research_tmp/nf19-sample-r3-v2" \
  --output "<B>/research_tmp/nf19-r16-v1" \
  --expected-sample-sha256 b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64 \
  --expected-preparation-manifest-sha256 88a584baceeac9efba05c1697cece83baf5fa55cfcaa990523ae16f9af97074a
```

The declared unit is `computer-vision-homework-eda19-r16-v1.service`, capped at 0.5 CPU, 1.4 GB memory-high, 1.5 GB memory-max, no swap, and 600 seconds, with output required under physical `research_tmp`. This README describes the preparation contract only; admission and execution remain root-owned.

The [implementation review](IMPLEMENTATION_REVIEW.json) covers source-only preparation. The [prepared scalar-audit implementation](audit_scalar.py) has not been run; there is no R16 result to audit. These files do not supersede the [current remote hold](ALLOCATION_HOLD_20261019.md).
