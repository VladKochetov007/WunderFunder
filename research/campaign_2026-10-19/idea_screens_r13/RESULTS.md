# R13 results: fixed first-99-row mean context

The warmup-mean arm did not improve the final pooled score over either matched control. Its pooled `global_wp_clip2_v1` score was **0.462646**, versus **0.499683** for the duplicated-current-input control and **0.507387** for the zero-side-block control. The prespecified spending gate required the warmup arm to exceed both controls by more than `+0.005`; it failed both comparisons.

| Arm | Target 0 | Target 1 | Pooled WP |
|---|---:|---:|---:|
| Zero side block | 0.519757 | 0.495018 | 0.507387 |
| Current-input bypass | 0.510076 | 0.489290 | 0.499683 |
| First-99-row mean | 0.489864 | 0.435427 | 0.462646 |

| Pooled contrast | Delta WP |
|---|---:|
| Warmup mean − current bypass (primary) | −0.037037 |
| Warmup mean − zero side block | −0.044742 |
| Current bypass − zero side block | −0.007704 |

All three models used the same 336-to-64-GELU architecture, shared fit-row normalizer, fixed epoch order, and 756 successful updates. Their initial states were identical and initial prediction difference was exactly zero. The models completed 12 epochs each with no nonfinite updates. The first-99-row context was an immutable input-only mean, broadcast only from row 99 onward; the original causal raw-plus-signed-EMA history and scored masks were unchanged.

Individual component-level contrasts were mixed: warmup-minus-current was positive for 17 of 32 evaluation components, warmup-minus-zero for 13, and current-minus-zero for 10. The all-32 drop-one pooled contrasts were negative for each comparison. For the primary warmup-minus-current contrast, drop-one values ranged from −0.043024 to −0.023279; warmup-minus-zero ranged from −0.052887 to −0.031779; current-minus-zero ranged from −0.010684 to −0.005679. These are sensitivity checks on the same evaluation panel, not 32 independent replications or a significance test.

This closes the exact fixed first-99-row mean recipe under this one exposed panel. It does not reject all context summaries, establish a general warmup or recurrent-memory result, or estimate full-fold/public performance. The R7 and earlier R3 scores are not matched controls for this comparison. No retuning or follow-up model was used to revise this result.

## Execution and provenance

The accepted R3 panel had 64 fit components and 32 evaluation components. Training used 31,808 stride-40 fit rows per arm; final evaluation used the original 59,436 scored rows. All 288 sample NPY files, including evaluation-target files, were hashed and mapped for integrity. Evaluation target values were not used for fit, normalization, arm construction, or selection; they were used for the final scoring pass.

The bounded CPU run completed 2,268 successful updates in 65.76 seconds of wall time. It used Torch 2.14.0+cu130 on CPU with CUDA uninitialized, autocast and TF32 disabled, and one Torch thread. Root validated the captured bytes before stopping the exact unit. The model archive remained on the execution host; its array payload was not opened or copied to this workspace.

- Frozen protocol: [DESIGN.md](DESIGN.md), [DESIGN_REVIEW.json](DESIGN_REVIEW.json), [SELECTION_HISTORY.json](SELECTION_HISTORY.json)
- Source and test receipts: [PREPARATION_MANIFEST.json](PREPARATION_MANIFEST.json), [IMPLEMENTATION_REVIEW.json](IMPLEMENTATION_REVIEW.json), [TEST_RECEIPT.json](TEST_RECEIPT.json)
- Captured scalars and provenance: [REMOTE_RESULT.json](records/terminal_capture/REMOTE_RESULT.json), [REMOTE_SOURCE.json](records/terminal_capture/REMOTE_SOURCE.json), [REMOTE_DONE.json](records/terminal_capture/REMOTE_DONE.json), [ROOT_CAPTURE_VALIDATION.json](records/ROOT_CAPTURE_VALIDATION.json), [STOP_RECEIPT.json](records/STOP_RECEIPT.json)
- Independent scalar audit: [RESULT_AUDIT.json](RESULT_AUDIT.json) (PASS; compact scalar reconstruction only)

Result SHA-256: `7219d98d0856e399dcde390f08b19cbc1cc89e675e75eb6c4fee7c58cf3802dd`. The captured result was produced by invocation `cc91e56effe941faada48ac9f83e3e5e`; root capture validation passed before the exact service stop, and the unit was subsequently absent (`LoadState=not-found`).
