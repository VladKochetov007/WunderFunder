# B300 native execution and receiver repairs, 2026-11-12

Eight Transformer fits are actually running on all eight B300 cards. These are individual fits, including matched controls, rather than eight independent mechanism comparisons. The active scientific bundle is commit `ef5a89299bbe3e4f40bdb701eaca44aae91eea45` (packet R5; archive SHA `9ad6c222e783f69bf676d956443b98cf08f1172aff19eebeec61bc8cf51a9195`). Runtime source is pinned PR75 commit `8626000a4b965763a70011a0291a891aa6535d7e`.

The 40-cell inventory remains explicit: seven Transformer cards have 28 planned fits; three GRU cards have 12. The first Transformer panel uses seed 204101, both arms for numeric PLE, private target heads and intermediate supervision. The group-mixer treatment failed its real-shape profile; its control continues and the freed lane started future-target supervision. Those unmatched arms cannot support paired claims yet. Remaining Transformer cells are ordered in the bounded native queue. GRU import failures are execution failures, not negative quality evidence; the repaired GRU bundle is separately qualified and is not reported as trained.

## Resource and runtime amendment

Measured placement retains global microbatch 256 and four accumulated chunks, using one GPU with local B256 instead of two GPUs with B128 each. The engineering comparison gave about 12.5% more forwarded rows per GPU-second. Fresh per-cell 50-update profiles precede fits. Current full fits occupy approximately 230 GiB/card, with 95–100% GPU utilization in the captured native snapshot; exact throughput and resources remain in live journals.

The original five-hour wave end is unchanged: 08:20:19 UTC. New B300 fitting receives 4h35 of that window, until 07:55:19 UTC, leaving 25 minutes for canonical readouts. This prospective CLOCK_B300_R2 does not rewrite old CLOCK_R1, old receipts, or active L4 fitting deadlines. Cells that cannot finish retain partial checkpoints and monitoring with an explicit incomplete label. No public score or significance claim follows from a partial monitoring curve.

BF16/BF16x9 is the declared Transformer compute mode; parameters, gradients, losses and optimizer state remain FP32 tensors. Process-wide cuBLAS BF16x9 is not an IEEE-FP32 arithmetic claim. Native subprocesses are detached from SSH and use exact PID/birth ownership, absolute deadlines, RSS/output/free-space guards, retained epoch checkpoints, and independent full valid evaluation after each epoch.

## Preserved execution failures and repairs

| Surface | Evidence and repair |
|---|---|
| Serialized config identities | Initial Windows line endings changed raw config hashes. Later packets are archived with Git autocrlf disabled and all 40 descriptors are checked after extraction. |
| Receiver dependency | Real Git was absent on the B300 node. A user-local git-core package was extracted; no system package or credentials were installed. |
| Synthetic attention checks | BF16 efficient attention rejects head dimension 12. Engineering probes now use hidden64/heads4; production model sizes are unchanged. |
| Scheduler check | A one-update synthetic warmup-cosine fixture violated the two-update minimum. The fixture now uses two updates and zero warmup. |
| Future-label check | A nonfinite injected label correctly removes one eligible target. The fixture expected count was corrected from 18 to 17 for that target; objective code is unchanged. |
| GRU lifecycle import | PR79 exposes `load_resume`, while PR75 used `_load_resume`. The compatible import is source-qualified; both signatures were checked and a native import preflight was added. |
| Full-shape group mixing | Flattening B256 × T256 creates 65,536 independent attention batches and fails the selected CUDA kernel. Future packet R7 chunks independent rows; CPU output/gradient parity passed. Its CUDA component stress and a new scientific attempt remain separate admissions. Active R5 fits are unchanged. |

Source R6 (`cfd23f0788e28a11d7d492e39c70e96b2585f3f4`) and R7 (`0c1fe5b06cf81d8a1fd97e305be198b905090fa6`) are repairs/handoffs, not retroactive identities for active R5 fits. Each failed attempt is retained in the native run and controller receipts. There are no blind retries or resumed historical fits.

## Validation and interpretation

All fitting and fitting-only normalization use the official earlier train partition: 10,325 whole sequences, no valid ancestry. Evaluation uses all 1,873 independent official valid sequences and 3,528,273 scored rows, split SHA `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`. Shuffled sequence identifiers do not define additional chronological folds. Main evaluation is unchanged pooled `global_wp_clip2_v1`, with both targets reported.

This validation has been inspected repeatedly. Report each fixed endpoint and same-seed delta; do not treat seeds as independent time periods or infer public score from local deltas. Train-evaluation and valid use different row masks/populations and are labelled accordingly. Canonical prediction export/score is distinct from the monitor score. Readout/source failures cannot be reclassified as quality losses.
