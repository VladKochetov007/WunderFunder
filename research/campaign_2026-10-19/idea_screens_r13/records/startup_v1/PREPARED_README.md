# R13 prepared source

R13 tests one fixed input-only context recipe: append the mean of each sequence's first 99 raw feature rows to the existing current-raw plus signed-EMA features. Three same-capacity arms use zero, duplicated current-raw, and warmup-mean side blocks. The shared R7 fit-row normalizer is applied to every raw block; each side is exact zero before row 99. All arms use the original R3 masks and fixed 12-epoch training schedule.

The expanded 336-to-64 mean path computes separate base-224 and side-112 affine projections, adds them, then applies GELU. That preserves exact initial parity when side weights and inputs are zero. It is the same operation across these three arms; this report does not claim bitwise equivalence to a single 336-wide matrix multiply or to an old R7 model. R7/R3 endpoint scores are not matched controls for this comparison.

The accepted panel and its split are pinned in [DESIGN.md](DESIGN.md), [SELECTION_HISTORY.json](SELECTION_HISTORY.json), and [PREPARATION_MANIFEST.json](PREPARATION_MANIFEST.json). The manifest covers the complete 30-module local/canonical import closure, tests, frozen design/review, and the R3 producer's metadata-only sample/membership fixtures. [TEST_RECEIPT.json](TEST_RECEIPT.json) records source-only checks. No sample array or model payload has been read by this local implementation task; the R3 fixture test reads metadata JSON only.

The runtime will hash and map all 288 NPY files, including evaluation targets, for source integrity. Evaluation target values are not used for fit, normalization, arm construction, or selection; they are used once for the final canonical score after all three arms finish. Model weights and normalizer arrays are written only to the private runtime archive on the execution host.

Root-owned execution command shape:

```sh
python -B -m idea_screens_r13.run_probe \
  --sample /B/research_tmp/r3-sample \
  --output /B/research_tmp/nf19-r13-v1 \
  --expected-sample-sha256 b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64 \
  --expected-preparation-manifest-sha256 <sealed-manifest-sha256>
```

This folder is prepared for independent source review and root-owned admission; no R13 fit or evaluation has run locally.
