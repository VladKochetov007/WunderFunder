# R7: a learned variance objective helps one small MLP pair

The fixed weighted Gaussian-NLL arm improves final mean-prediction WP by **+0.0123755330** over its matched WMSE arm. Both targets improve, with most of the contrast on target1. The independent [compact audit](RESULT_AUDIT.json) reconstructs the two endpoints and component sensitivities from captured sufficient statistics; it does not repeat fitting or read model/array payloads.

| Final fixed endpoint | Target0 WP | Target1 WP | Official pooled WP |
|---|---:|---:|---:|
| WMSE control | 0.5196829254702040 | 0.4841598129320762 | 0.5019213692011400 |
| Mean + variance Gaussian NLL | 0.5242803195323170 | 0.5043134849177928 | 0.5142969022250550 |
| Candidate minus control | +0.0045973940621130 | +0.0201536719857167 | **+0.0123755330239149** |

All32 leave-one-component pooled-WP contrasts remain positive, from+0.00864432145 to+0.01378090380. Individual component WP contrasts are19positive/13negative. These are sensitivity summaries of the same fixed predictions, not32 training replications, an estimated population win rate or a confidence interval.

## What was compared

[Pre-code design](DESIGN.md), [independent design review](DESIGN_REVIEW.json) and [selection history](SELECTION_HISTORY.json) froze one paired seed19031. The existing R3 exposed-F0 panel supplies64 fitting components/31,808 stride40 fitting rows and32 evaluation components/59,436 scored rows. No sample replacement, hyperparameter grid, CV selection, early stopping or evaluation-driven endpoint choice occurred. Signed EMA representation was chosen adaptively after R3; this is not a fresh holdout or an unbiased full-fold score.

Both arms use the same224inputs (raw112 + causal signed-delta EMA with half-life32), fit-only normalization,224→64GELU trunk, separate mean/logvariance heads, full initialization, epoch batches and optimizer. Each executes12epochs ×63batches =756successful AdamW updates, with zero nonfinite events. Same examples/updates do not mean matched FLOPs or wall time. The control logvariance head is present but skipped and unchanged; the candidate changes it.

The control uses canonical clipped-label magnitude-weighted MSE. With `w=abs(clip(y,-2,2))`, the candidate minimizes `sum(w*(exp(-s)*(mean-clip(y,-2,2))**2+s))/sum(w)` over rows and both targets, where `s=clamp(logvariance,-4,4)`. Variance is not passed into the official metric and does not remove evaluation rows. At zero variance logits the mean loss-gradient scale matches WMSE; variance gradients and subsequent global clipping can immediately change the effective updates. Thus this is a finite-recipe objective/optimization comparison, **not an isolated proof that recovering input-dependent noise caused the gain**.

## What the result changes

Idea22 now has a positive matched nonlinear pilot and merits one discriminating follow-up. A useful next control is learned per-target constant logvariance under otherwise fixed conditions, separating global target/loss scaling from feature-dependent variance. That control has not run or been admitted here; floor, seed, width and epoch retuning on these outer scores would not answer it.

This is not the strongest absolute endpoint on this panel: the stored [R3 selected signed-EMA ridge](../idea_screens_r3/RESULTS.md) scores0.5360037389, above both small MLP endpoints. That is a descriptive comparison across different model/optimization recipes, not a matched rejection of the new objective. There is no GRU, public-score, full-recipe or CP30 improvement claim.

Fit diagnostics also differ from evaluation: final fit WMSE is0.990083 for control versus1.080348 for candidate, and fit WP is0.600739 versus0.548720. The candidate's better evaluation score does not establish better fit or a unique regularization/noise mechanism. Both full12epoch curves are retained in the result. Candidate logvariance reaches the lower clamp on19/20evaluation rows for the two targets (0.03197%/0.03365%); these descriptive counts do not identify a causal explanation or justify tuning the clamp.

## Evidence and execution

- [Exact result](records/probe_v1/terminal_capture/RESULT.json), SHA`dd36e82a51672a9b9e2ab694dd92fcd827a2445d607927548200d333d1f4b9e0`; [source/provenance](records/probe_v1/terminal_capture/SOURCE.json), SHA`9f396b1e0c9a3bbb696b5d5fb32d9a4908a6bf8e14f551f6a8dc899f06a907a0`.
- [Independent audit](RESULT_AUDIT.json), SHA`99f96f71c965d94a7383c4cbb130d10c72ab2e422030a908d8e89e134a524ea8`: both global endpoints,64 component and64 drop-one score records, provenance/counters/caps/stop; numerical reconstruction error below8e-16. Thirty preparation pins match; no array/model replay.
- [Preparation manifest](PREPARATION_MANIFEST.json), [implementation review](IMPLEMENTATION_REVIEW.json), [11tiny checks](TEST_RECEIPT.json) and [startup review](ROOT_STARTUP_REVIEW.json) passed before the sole actual invocation. These checks did not fit actual data locally.
- [Execution records](records/README.md) and [resource reconciliation](RESOURCE_RECONCILIATION.json): exact CPU unit exited0 at01:51:55UTC and was captured before exact-unit stop at01:54:16UTC. No new parent/F3 memory events; existing F3 deep continuation and controller remain populated. This is not a new optimizer-batch or terminal-quality check of F3.

Arrays and final models stay on Beleriand. `B/research_tmp/nf19-r7-v1/FINAL_MODELS_AND_NORMALIZER.npz` is123,084bytes, producer SHA`63141a6a14a0b840d3c15c3b15b0865301c3b98f3d7ad3d61470f18c8d5fc750`. Capture checked stat size and producer receipts only; the archive was not opened, independently rehashed or copied to the Mac. It contains both model states and shared normalization, no prediction payload. Integrity mapping before fitting can read evaluation files for source verification; no evaluation labels or scores enter normalization, fitting or selection.
