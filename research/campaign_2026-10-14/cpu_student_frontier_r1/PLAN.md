---
kind: method
status: admitted
created_utc: 2026-10-13T23:43:28Z
updated_utc: 2026-10-13T23:43:28Z
hypothesis_ids: [H-CPU-POLICY-20261014, H-STUDENT-LADDER-20261014, H-STATE-INIT-20261014]
experiment_ids: []
evidence_paths: [research/campaign_2026-10-13/qat_renewable_r1/final_self_distill_r2/PUBLIC_TRANSFER_REVIEW_20261014_R1.md]
---

# Find the strongest student architecture for the competition CPU

This is the complete user-supplied implementation plan, recorded before new experiments. The October 14 follow-up adds: CPU uploads will not happen immediately; prepare checked probes, study theory while awaiting target-host evidence, and continue independent research rather than blocking on uploads. Root owns delivery. Prior healthy work and immutable receipts remain preserved.

## 1. Objective and evidence

Build a measured frontier of **prediction quality versus actual competition runtime**, then distill stronger teachers into the best architecture on that frontier. Target **50–55 minutes**, using all available competition submission slots for useful CPU-policy probes when the user can upload them.

The current official test contains **39.4 million rows**. That permits approximately **76–84 µs per row** at our target duration, including scorer overhead; the hard 60-minute ceiling is **91.4 µs per row**. The allocation is one vCPU, despite the host having 48 logical CPUs. Sources: [official FAQ](https://wundernn.io/connectome/docs/faq), [submission environment](https://wundernn.io/connectome/docs/submission_guide). These are plan premises; refreshed official-page evidence is recorded separately, and no unauthenticated portal state establishes remaining slots.

Starting evidence:

- The existing four-member Q8 endpoint completed in **50m16s**. It is a useful serving baseline.
- Current Q4 deployment expands weights into Q8 kernels at loading. Its smaller ZIP has **not demonstrated additional CPU capacity**. [Latest QAT result](../../campaign_2026-10-13/qat_renewable_r1/final_self_distill_r2/PUBLIC_TRANSFER_REVIEW_20261014_R1.md).
- Existing depth and activation experiments do not establish the best modern student architecture.
- Local model comparisons provide empirical robustness evidence; laptop-to-EPYC extrapolation remains provisional until checked through submissions.

The deliverable is a supported capacity range for each architecture and precision, followed by the strongest validated student or small ensemble. A maximum parameter count alone cannot determine the winner.

## 2. Measure CPU policies through matched submissions

**Freeze the original `xteam_h4` weights, coefficients, preprocessing, quantization and architecture for the first engine comparison.**

| Policy | Change being tested |
|---|---|
| A: control | Current shipped Q8 VNNI implementation |
| B: vector width | Genuine 256-bit EVEX VNNI versus current 512-bit VNNI |
| C: matrix scheduling | Register blocking, packed layout and accumulator scheduling |
| D: recurrent fusion | Fuse compatible cell operations while preserving separate scales and correct old-state use |

Zen 4's internal vector execution and cache hierarchy make both vector widths worth testing; instruction width alone does not establish throughput. [AMD architecture documentation](https://www.amd.com/content/dam/amd/en/documents/products/epyc/4th-gen-amd-epyc-processor-architecture-whitepaper.pdf).

For each policy:

- Inspect generated assembly, then run matrix, GRU-cell and complete-callback probes locally. Include activation quantization, nonlinearities, preprocessing, Python/native boundary and resets.
- Use intrinsics initially. Introduce handwritten assembly only for a demonstrated compiler or scheduling limitation; package the resulting compiled library with reproducible source/build provenance.
- Require identical complete-sequence predictions for claims of exact optimization. Treat changed arithmetic as a numerical variant requiring its own quality evaluation.
- Assert the selected implementation at startup so a successful probe cannot silently measure a fallback.
- Measure actual ZIP bytes and resident memory independently of matrix payload size.

Use all **remaining team slots**, currently subject to the official shared limit of five daily. Start with **A–B–C–D–A**, then reverse candidate order and repeat useful comparisons. Submit sequentially and record ZIP hash, policy, score, duration, memory and failure details. Control repetitions estimate host variation; one faster run is provisional. The user's follow-up places immediate uploading with the user; prepare the order and continue without assuming a submission date or quota.

Later probe families compare BF16 matrix paths and genuine packed-W4 execution. For W4, compare expanded-byte execution, unpacking inside the kernel, and a lookup-table implementation using **identical represented weights**. [T-MAC](https://arxiv.org/abs/2407.00088) supplies a concrete lookup method, but its LLM speedups are hypotheses for our smaller recurrent matrices.

Keep predictions independent of elapsed time. If the portal remains unauthenticated, deliver checked ZIPs in submission order and continue local training while the user uploads them.

## 3. Search architecture and precision together

Preserve the current **211 causal features**, normalization, projection/skip structure and teacher targets in the initial architecture comparison.

| Comparison | Student candidates | Approximate matrix parameters |
|---|---|---:|
| Compact depth/width | H256 × 1 versus H192 × 2 | 0.58M / 0.56M |
| Current control | H256 × 2 | 0.97M |
| Larger depth/width | H512 × 1 versus H384 × 2 | 2.21M / 2.15M |
| Near current ensemble compute scale | H640 × 1, H512 × 2, H448 × 3 | 3.41M / 3.78M / 4.11M |

These are shape-derived matrix counts, excluding biases and packing overhead. **They are candidates to measure, not certified runtime limits.** A single H512×2 has roughly the dense arithmetic of four current H256×2 members, making it a particularly useful comparison.

Profile all shapes before fitting. Compare both approximately equal parameter counts and equal measured serving cost. Universal approximation establishes representability; it does not decide finite-size accuracy, trainability or latency. [Depth/width literature](https://arxiv.org/abs/1709.02540).

After the initial ladder, run controlled branches:

- **GRU versus MLP allocation:** expose independent encoder, recurrent and head widths. Hold the other dimensions fixed when changing one, then compare reallocations at matched serving cost. Test one versus two hidden head layers.
- **One model versus several:** compare a large student, two smaller independently fitted students, and a jointly trained two-branch recurrent model with a shared encoder and concatenated readout. Include the existing shared-backbone/multiple-head option where tensors genuinely match.
- **Activations:** first compare GELU and ReLU in encoder/readout. Separately test a ReLU recurrent candidate while retaining sigmoid gates. Measure state growth and long-sequence stability; changing the recurrent activation requires training and native support.
- **Cheaper recurrence:** reserve bounded exploration for minGRU and structured low-rank recurrence. Their training or compression advantages must survive the single-row CPU callback. [minGRU](https://arxiv.org/abs/2410.01201), [compact recurrent networks](https://arxiv.org/abs/1604.02594).

For quantization, compare **a stronger larger W4 student against a smaller W8 student at matched runtime**, alongside same-model FP32/Q8/W4 controls. Use fitting-only calibration and exact native export.

Start with FP32 persistent state, gate calculations, biases and output accumulation. Compare W8 throughout against W4 input matrices/W8 recurrent matrices and broader W4 QAT. Measure layer/gate sensitivity before allocating higher precision. Activation-aware training must reproduce the deployed rounding boundaries.

Teacher improvement runs alongside this search: preserve the current clean teacher cohort, add larger recurrent teachers and heterogeneous family targets, and measure both teacher quality and student benefit. Freeze each teacher version during an architecture comparison; redistill the leading two students when a better teacher cohort arrives.

## 4. Test predicted GRU state as a separate research arm

The current baseline starts from **zero state**. Compare:

1. Zero state.
2. A learned constant state for each GRU layer.
3. A small model predicting the initial state from the first available row.
4. A prefix model adjusting the state **after rows 0–98, before processing row 99**.

For the prefix arm, use normalized prefix mean, standard deviation and last observation, passed through a 64-unit MLP with a bounded residual output for each recurrent layer. Continue processing every warmup row normally; apply the adjustment once. Reset the prefix statistics and recurrent state together.

Compare joint training with training only the initializer against a frozen backbone. Optimize end-to-end prediction/distillation loss; hidden-state reconstruction alone does not establish usefulness.

This has direct precedent in [Contextual Recurrent Neural Networks](https://arxiv.org/html/1902.03455v1). The repository also contains an uncertain positive prefix-adapter experiment, which becomes a related control rather than evidence that initialization works.

Measure:

- Full official score.
- Effects over positions 99–199, 200–999, 1,000–4,999, 5,000–9,999 and 10,000–19,999.
- Decay of hidden-state and prediction perturbations.
- Once-per-sequence cost and package overhead.

Use only available same-sequence inputs. Future rows, sequence IDs, targets and scoring masks cannot enter the initializer. If initialization effects disappear quickly, test persistent prefix conditioning as a distinct successor.

## 5. Execution, validation and research records

Run continuous local GPU work over the next few days, with Kaggle handling larger teachers and parallel repetitions. The prior read-only check found approximately **9.25 GPU hours and 20 TPU hours remaining**, renewing October 17. These are snapshots; refresh before admission.

- **Local GPU:** architecture profiles, compact matched fits, state-initializer experiments and exports.
- **Kaggle GPUs:** wider teachers/students and fold/seed repetitions. Repair the diagnosed cache-path failure before resubmitting its successor.
- **Kaggle TPU:** collect the queued bridge outcome, then require actual-data update, checkpoint reload, causal-state and export checks before assigning substantial training.
- **CPU:** inference development, local probes, evaluation and packaging within the existing five-core aggregate limit.

Retain one local GPU worker, the 5 GB observed VRAM cap, 17 GB committed-memory target, 18/20 GB memory limits and no swap. No paid capacity is needed.

Start the eight-architecture screen on F0 with seed 3 and identical teacher targets, training exposure, update budget and checkpoint rule. Advance four useful candidates to seed 4, then two leading alternatives to F3 with both seeds. Complete finalists across all four source-qualified folds, rebuilding missing clean teacher ancestry where necessary. Preserve an exploration candidate when an inconclusive screen warrants follow-up.

Acceptance checks cover:

- Causal chunk/full-sequence parity, warmup behavior, finite outputs and complete state reset.
- Quantizer boundaries, exact code/scale transport, SIMD padding and native/reference differences.
- Actual Python 3.11 ZIP import, ABI, determinism, size and server completion.
- Official `global_wp_clip2_v1`, paired component uncertainty and distinct training-seed variation.

All existing folds remain exposed. Cross-validation measures robustness; public scores confirm transfer. CPU probe scores do not select blend weights or architectures, and confidence intervals crossing zero do not automatically terminate a direction.

Root owns delivery through fitted checkpoints, predictions, grading and packages. Use at most three Luna/max subordinate roles for inference, training delivery and independent review. Reallocate after each completed wave.

During implementation, record the contract, literature findings, hypotheses, complete attempted set, negative results, capacity curves and submission receipts in the existing ledgers and unified research navigation. The final handoff includes the strongest supported package, its fallback, and the measured reasons for choosing its depth, width, precision and state initialization.

## Evidence and interpretation

This note records authorized work, not completed experiments or a new best. A/B/C/D submission results, architecture quality/capacity curves and initializer benefits remain to be measured. The original Q8 endpoint remains the serving reference. Existing folds are exposed, and public feedback cannot choose blend coefficients or erase fitting ancestry.

## Decision and next step

Prepare the independent CPU policy implementations and the F0 seed3 matched architecture ladder concurrently, with an independent reviewer. Root refreshes quota, recovers cloud outcomes and admits profiles against actual resource headroom. CPU uploads are an asynchronous evidence dependency, not a gate on local work. Maintain theory and nonblocking alternatives while waiting.

## Source records

See [campaign README](README.md), [contract](CONTRACT.json), the prior [QAT frontier](../../campaign_2026-10-13/qat_renewable_r1/FRONTIER.md), [official team ancestry decision](../../../specs/adr/0035-official-team-code-and-weight-sharing.md), and [cross-fold validation decision](../../../specs/adr/0031-cross-fold-top-tier-submission-release.md).
