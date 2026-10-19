# R15 sparse dictionary screen — source prepared (V2)

This folder contains the fixed implementation for H-NF19-09. V2 corrects the downstream freeze guard: the dictionary fingerprint captured **after AE training and freezing** is the reference checked after both predictor fits. The pre-training fingerprints remain initialization provenance. The fix has a synthetic regression showing a trained dictionary differs from initialization, passes against its frozen fingerprint, and fails after a later mutation.

This is source preparation for root review and B-side admission; **no R15 data arrays, weights, or model fits have been run here**. The design is one sparse top-four dictionary pipeline against a same-width dense dictionary pipeline. Both retain the reconstruction residual, use the same causal signed EMA and predictor recipe, and are scored once on the accepted exposed R3 panel. It tests this exact pipeline only; it does not identify stable latent factors or establish fresh-fold performance.

## Frozen inputs and counts

The accepted R3 sample has 96 complete 20,000-row sequences: 64 fit components and 32 component-disjoint evaluation sequences. R15 uses 31,808 original-mask stride-40 fit rows and 59,436 original scored evaluation rows. Histories are formed over each complete sequence before masks or stride are applied. The sample and membership metadata are pinned in the preparation manifest; all 288 NPY files (including target files) are mapped/hashed by the R1 loader for integrity. Fit labels are used only by the two supervised predictor fits; evaluation labels only for the final endpoint. Dictionary fitting is label-free, and input/feature normalizers use fit rows only.

Each of the four fixed fits (two dictionaries and two predictors) uses 12 epochs and 756 AdamW updates. The total budget is 3,024 updates. There is no cross-validation, hyperparameter search, outer retuning, or second sample. The prospective gate is pooled sparse-minus-dense WP at least 0.005 and both target deltas positive; this is an allocation screen, not a significance rule.

## Source and checks

[DESIGN.md](DESIGN.md), [DESIGN_REVIEW.json](DESIGN_REVIEW.json), and [SELECTION_HISTORY.json](SELECTION_HISTORY.json) freeze the question and scope. [PREPARATION_MANIFEST.json](PREPARATION_MANIFEST.json) pins 31 implementation, test, panel-metadata, canonical, and transitive runtime files; its SHA-256 is `582171973f77d88348409218d974c41481a509af70e6b5a91b1dcd08cf8897dd`. The cold import proof is in [cold_import_smoke.json](records/preparation_v2/cold_import_smoke.json): all 25 expected runtime modules resolved to pinned source paths, and NumPy array, Torch checkpoint, and PyArrow parquet/dataset read sentinels were not called. CUDA stayed uninitialized. No sample payload was loaded during that proof.

[TEST_RECEIPT.json](TEST_RECEIPT.json) records eight focused synthetic checks and final Ruff results. Their final raw stdout is preserved under [records/preparation_v2](records/preparation_v2/). The earlier candidate, receipt, smoke proof, test logs, and changed source/test copies are preserved under [records/source_v1](records/source_v1/). README, receipt, and execution logs are operational evidence outside the scientific source manifest.

## B-side invocation

Root owns source staging, source rehash, resource admission, and actual execution. The fixed source location is `B/support-source/eda19-r15-v1`; canonical code is `B/source-a266b036/src`; accepted sample is `B/research_tmp/nf19-sample-r3-v2`; output is the new directory `B/research_tmp/nf19-r15-v1`. The CLI is `python -B -m idea_screens_r15.run_probe` with exactly these required arguments:

```sh
PYTHONPATH="<B>/source-a266b036/src:<B>/support-source/eda19-r15-v1" \
  python -B -m idea_screens_r15.run_probe \
  --sample "<B>/research_tmp/nf19-sample-r3-v2" \
  --output "<B>/research_tmp/nf19-r15-v1" \
  --expected-sample-sha256 b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64 \
  --expected-preparation-manifest-sha256 582171973f77d88348409218d974c41481a509af70e6b5a91b1dcd08cf8897dd
```

The reviewed unit is `computer-vision-homework-eda19-r15-v1.service`, capped at 0.5 CPU, 1.4 GB memory high, 1.5 GB memory max, zero swap, and 600 seconds on physical `research_tmp`. Root performs all B-side gates and operations. Expected create-only output consists of `START.json`, `SOURCE.json`, `PROGRESS.jsonl`, `RESULT.json`, `DONE.json`, and the private `model_parameters.npz`; a bounded `FAILURE.json` records a pre-completion error. Model arrays and predictions are never emitted to the local machine.
