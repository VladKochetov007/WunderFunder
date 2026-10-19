# R15 results: sparse dictionary with retained residual

The fixed sparse pipeline produced a small positive pooled endpoint difference, but it missed its predeclared spending gate. Sparse-minus-dense pooled WP was **+0.0025681**, below the required **+0.005**; both target differences were positive. This closes the exact R15 recipe without promoting it or retuning its seed, sparsity, or training schedule.

## Endpoint

The one run used the accepted exposed R3 panel: 64 fit components and 31,808 original-mask stride-40 fit rows; then 32 disjoint evaluation components and 59,436 scored rows. The official metric was `global_wp_clip2_v1`.

| Endpoint | Dense pipeline | Sparse top-four pipeline | Sparse minus dense |
| --- | ---: | ---: | ---: |
| Target 0 | 0.4983145 | 0.4993565 | +0.0010420 |
| Target 1 | 0.4614317 | 0.4655259 | +0.0040943 |
| Pooled WP | 0.4798731 | 0.4824412 | +0.0025681 |

The frozen gate required pooled WP improvement of at least 0.005 and positive differences for both targets. The target sign condition passed; the pooled threshold did not.

Across the 32 individual evaluation components, sparse-minus-dense WP was positive on 13 and negative on 19. Across 32 leave-one-component-out pooled comparisons, 30 deltas were positive and 2 negative, ranging from −0.0019702 to +0.0075391. These are component and sensitivity descriptions, not independent replications or uncertainty intervals.

## What was compared

Both pipelines learned a 16-code dictionary from the same normalized 112-channel fit inputs. The sparse arm retained the four largest-magnitude signed code values per row; the dense arm used all 16. Both appended the exact reconstruction residual, formed the same causal signed EMA with half-life 32, and trained the same downstream mean predictor recipe. Each arm had its own fit-row downstream normalizer. Thus this endpoint compares the complete sparse-versus-dense retained-residual pipelines; it does not isolate a sparse-code effect from the nonlinear representation and its downstream normalization.

The fit-stride dictionary reconstruction MSE was 0.4901009 for sparse and 0.4241688 for dense. All 16 sparse factors were used on fit rows, with uneven activation counts. This is a reconstruction summary, not evidence of factor stability or identifiable market sources. Because the residual is retained, the sparse transformation does not discard the original normalized input.

Each of the four fits (two dictionaries, two downstream predictors) completed 12 epochs, 756 updates, and 381,696 row exposures. Total updates were 3,024; nonfinite count was zero. The helper's `RESULT.json` reports 95.725 seconds wall and 47.692 CPU seconds; native systemd accounting separately reports 105.0575 seconds wall and 52.17586 CPU seconds. These are distinct timing scopes; no cause is inferred for the difference. This fixed small-panel comparison does not establish fresh-fold performance, equal compute cost, or factor stability across refits. The R3 fit/evaluation components were already exposed in the preceding campaign work.

## Provenance and capture

The captured result JSON SHA-256 is `ab149a3fcf9d58c1221300ddca3eba356a16181fe5239ea1b3d7646bfe9099c3`; the captured source digest is `f8abfe9102941f3471f924cfdab9066823c64b079286177473cc155b3729dd01`. The run used the V3 preparation manifest SHA-256 `ad26d2589ed34101b4eed1cd4655f7924c11349f5cb5ea3fd4a345189cd8f798` and sample SHA-256 `b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64`.

The root capture validator accepted all nine payloads at 08:54:58.288932 UTC. The exact stop check at 08:55:07.840018 UTC found the unit absent and inactive. The model archive metadata records a 178,584-byte NPZ on the B-side. Its payload was not read or copied to this machine; it contains model parameters only, not predictions. The [independent scalar audit](records/RESULT_AUDIT.json) passed score and drop-one reconstruction from recorded sufficient statistics without opening arrays or replaying a fit. See [the captured result](records/terminal_capture/REMOTE_RESULT.json), [source record](records/terminal_capture/REMOTE_SOURCE.json), [completion record](records/terminal_capture/REMOTE_DONE.json), [root validation](records/ROOT_CAPTURE_VALIDATION.json), [stop receipt](records/STOP_RECEIPT.json), and [continuation record](CONTINUATION.md).
