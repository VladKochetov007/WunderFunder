# Combo R1: NLL improves the stored sparse+EMA endpoint on this panel

The one fixed NLL fit scored **0.5044141 pooled WP**, improving by **+0.0219729** over the stored R15 sparse-WMSE endpoint and **+0.0245410** over the stored dense-WMSE endpoint. Higher WP is better. Both targets improve against both stored endpoints. This is a positive result for this exact sparse+EMA/NLL recipe on the accepted exposed panel, not an interaction or generalization estimate.

| Fixed endpoint | Target 0 WP | Target 1 WP | Pooled WP |
|---|---:|---:|---:|
| New sparse+EMA NLL | 0.5169595 | 0.4918687 | **0.5044141** |
| Stored R15 sparse WMSE | 0.4993565 | 0.4655259 | 0.4824412 |
| Stored R15 dense WMSE | 0.4983145 | 0.4614317 | 0.4798731 |
| New minus sparse WMSE | +0.0176030 | +0.0263427 | **+0.0219729** |
| New minus dense WMSE | +0.0186450 | +0.0304370 | **+0.0245410** |

The comparison used the accepted R3 panel: 64 fit components and 31,808 original-mask stride-40 fit rows, plus 32 disjoint evaluation components and 59,436 scored rows. The model ran 12 epochs, 756 successful updates, and 381,696 row exposures, with zero nonfinite events. The initial predictor fingerprint matched R15; its variance head began at zero and changed under NLL. Recomputed fit-only R15 normalizers matched exactly, and the restored sparse dictionary fingerprint remained unchanged. The official metric was `global_wp_clip2_v1`.

Per-component contrasts show a consistent but not universal direction. Candidate-minus-sparse-WMSE WP was positive on 22/32 components (10 negative), ranging from −0.04262 to +0.08474; the target-0 contrast was positive on 21/32 and target-1 on 19/32. Candidate-minus-dense-WMSE WP was positive on 20/32 components (12 negative), ranging from −0.06044 to +0.15770; target-0 was positive on 22/32 and target-1 on 16/32. These are component descriptions of the same fixed endpoints, not independent fits or an estimated win probability.

The fixed-prediction leave-one-component comparisons stayed positive for every omission. Relative to the sparse-WMSE endpoint, all 32 omissions improved pooled WP, with deltas from +0.01990 to +0.02394; target-0 ranged from +0.01385 to +0.02012 and target-1 from +0.02186 to +0.02900. Relative to dense WMSE, all 32 were positive as well: pooled +0.01893 to +0.02853, target-0 +0.01398 to +0.02470, and target-1 +0.01892 to +0.03522. This only tests sensitivity of the same predictions to one omitted component; it is not replication or a confidence interval.

The run restored and independently hashed R15's B-side archive (`9f3404f9…d4269f`, 178,584 bytes), checked its full key manifest, and loaded the six fixed raw-input/normalizer and sparse-dictionary arrays. It did not load old predictor arrays, or reinfer or regrade R15 predictions. The new model archive remains on B (`79a49200…bea4b01`, 90,386 bytes) and contains parameters/normalizers, not predictions.

This remains one adaptively selected recipe, seed, and exposed evaluation panel. The missing dense+NLL arm means the result cannot estimate feature/objective synergy. It does not establish a GRU, public, full-fold, or CP30 gain. Previously recorded R7 NLL (0.5142969) and R3 signed-EMA ridge (0.5360037) have higher absolute WP, but use different recipes and are not matched controls; this is not a new best. Retain this as a positive combination candidate; no additional fit or outer-score retuning is currently queued.

## Run records

The sole CPU unit was `computer-vision-homework-idea-combo-r1-v1.service`, invocation `66d47211356441deba6a6ef9b712e844`. It started at 12:12:49 UTC, exited with status 0 at 12:13:43, was captured and root-validated at 12:14:34.283865, then exactly stopped at 12:14:47.521280 UTC.

- [Compact result](records/terminal_capture/REMOTE_RESULT.json), SHA256 `9d6109b4181e9666195011478b83f77c9319549b162651ff5df8af6c89c7de05`.
- [Source provenance](records/terminal_capture/REMOTE_SOURCE.json), [completion receipt](records/terminal_capture/REMOTE_DONE.json), and [root capture validation](records/ROOT_CAPTURE_VALIDATION.json).
- [Exact stop receipt](records/STOP_RECEIPT.json). `REMOTE_RESULT.json` reports 45.12639868189581 seconds elapsed wall time and 22.554326948 seconds CPU time for the probe process; these differ from the service and capture timestamps above.
- [Frozen design](DESIGN.md), [source manifest](PREPARATION_MANIFEST.json), and [source-only test receipt](TEST_RECEIPT.json).

The [independent compact scalar audit](RESULT_AUDIT.json) reconstructs the pooled, component and fixed-prediction drop-one results within1e-15, without opening any model/array or rerunning scoring. [Resource reconciliation](RESOURCE_RECONCILIATION.json) and [metadata review](RESOURCE_METADATA_REVIEW.json) retain separate systemd/helper/guard measurements and zero new parent/local memory events. The captured systemd MemoryPeak is not substituted for the larger producer RSS/current samples; their difference remains unexplained.
