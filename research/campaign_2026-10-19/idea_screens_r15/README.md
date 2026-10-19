# R15 sparse dictionary screen — bounded run complete

[Current allocation update](ALLOCATION_UPDATE_20261019.md): the user retains the small positive R15 endpoint as a transfer/interaction candidate. Original numeric results and failed +.005 gate remain unchanged; EMA overlap prevents adding screen deltas.


This folder contains the fixed implementation for H-NF19-09. V3 fixes fresh-process CPU initialization: after verifying source pins and the runtime import closure, the CLI sets Torch intra-op and inter-op threads to one and disables matmul/CuDNN TF32 before creating the executor or mapping sample arrays. Autocast and CUDA-initialization checks remain fail-closed. The source-only subprocess test checks the resulting flags with data-read sentinels installed.

V3 retains the V2 dictionary freeze-guard correction: `dict_initial` is pre-AE initialization provenance, a separate fingerprint is captured after both AEs train and freeze, and the post-predictor state must match that frozen fingerprint. A synthetic regression verifies changed initial-to-trained hashes, passes the frozen comparison, and detects a later mutation.

The V3 source bundle was admitted and run once on B. The compact endpoint and limits are in [RESULTS.md](RESULTS.md). The comparison is one sparse top-four dictionary pipeline against a same-width dense dictionary pipeline. Both retain the reconstruction residual, use the same causal signed EMA and predictor recipe, and are scored once on the accepted exposed R3 panel. The result tests this exact pipeline only; it does not identify stable latent factors or establish fresh-fold performance. No model arrays or predictions were copied to this machine.

## Observed result

Pooled sparse-minus-dense WP was +0.0025681, below the frozen +0.005 gate; both target deltas were positive. The exact variant is closed without retuning. [RESULTS.md](RESULTS.md) gives the endpoint, component sensitivities, fit counts, and capture provenance.

## Frozen inputs and counts

The accepted R3 sample has 96 complete 20,000-row sequences: 64 fit components and 32 component-disjoint evaluation sequences. R15 uses 31,808 original-mask stride-40 fit rows and 59,436 original scored evaluation rows. Histories are formed over each complete sequence before masks or stride are applied. The sample and membership metadata are pinned in the preparation manifest; all 288 NPY files (including target files) are mapped/hashed by the R1 loader for integrity. Fit labels are used only by the two supervised predictor fits; evaluation labels only for the final endpoint. Dictionary fitting is label-free, and input/feature normalizers use fit rows only.

Each of the four fixed fits (two dictionaries and two predictors) uses 12 epochs and 756 AdamW updates. The total budget is 3,024 updates. There is no cross-validation, hyperparameter search, outer retuning, or second sample. The prospective gate is pooled sparse-minus-dense WP at least 0.005 and both target deltas positive; this is an allocation screen, not a significance rule.

## Source and checks

[DESIGN.md](DESIGN.md), [DESIGN_REVIEW.json](DESIGN_REVIEW.json), and [SELECTION_HISTORY.json](SELECTION_HISTORY.json) freeze the question and scope. [PREPARATION_MANIFEST.json](PREPARATION_MANIFEST.json) pins 31 implementation, test, panel-metadata, canonical, and transitive runtime files; its SHA-256 is `ad26d2589ed34101b4eed1cd4655f7924c11349f5cb5ea3fd4a345189cd8f798`. The cold import proof is in [cold_import_smoke.json](records/preparation_v3/cold_import_smoke.json): all 25 expected runtime modules resolved to their pinned source paths, and NumPy array, Torch checkpoint, and PyArrow parquet/dataset read sentinels were not called. CUDA stayed uninitialized; both Torch thread counts were one and both TF32 flags were disabled. No sample payload was loaded during that proof.

[TEST_RECEIPT.json](TEST_RECEIPT.json) records nine focused synthetic checks and final Ruff results. Raw final stdout is preserved under [records/preparation_v3](records/preparation_v3/). Earlier source candidates and receipts remain under [records/source_v1](records/source_v1/) and [records/source_v2](records/source_v2/); [records/SOURCE_HISTORY.md](records/SOURCE_HISTORY.md) indexes the preserved versions. The preparation log sets are indexed in [records/preparation/README.md](records/preparation/README.md), [records/preparation_v2/README.md](records/preparation_v2/README.md), and [records/preparation_v3/README.md](records/preparation_v3/README.md). README, receipt, and execution logs are operational evidence outside the scientific source manifest.

## B-side invocation

Root owned source staging, source rehash, resource admission, and execution. The source location was `B/support-source/eda19-r15-v1`; canonical code was `B/source-a266b036/src`; accepted sample was `B/research_tmp/nf19-sample-r3-v2`; output was `B/research_tmp/nf19-r15-v1`. The CLI was `python -B -m idea_screens_r15.run_probe` with these required arguments:

```sh
PYTHONPATH="<B>/source-a266b036/src:<B>/support-source/eda19-r15-v1" \
  python -B -m idea_screens_r15.run_probe \
  --sample "<B>/research_tmp/nf19-sample-r3-v2" \
  --output "<B>/research_tmp/nf19-r15-v1" \
  --expected-sample-sha256 b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64 \
  --expected-preparation-manifest-sha256 ad26d2589ed34101b4eed1cd4655f7924c11349f5cb5ea3fd4a345189cd8f798
```

The reviewed unit was `computer-vision-homework-eda19-r15-v1.service`, capped at 0.5 CPU, 1.4 GB memory high, 1.5 GB memory max, zero swap, and 600 seconds on physical `research_tmp`. Root performed the B-side gates and operations. Captured create-only output consisted of `START.json`, `SOURCE.json`, `PROGRESS.jsonl`, `RESULT.json`, `DONE.json`, and the private `model_parameters.npz`. The JSON captures and root validation/stop receipts are linked from [RESULTS.md](RESULTS.md) and indexed in [records/terminal_capture](records/terminal_capture/). The independent [scalar audit](records/RESULT_AUDIT.json) passed; [CONTINUATION.md](CONTINUATION.md) records execution boundaries and next-work limits. Model arrays and predictions were not emitted to the local machine.
