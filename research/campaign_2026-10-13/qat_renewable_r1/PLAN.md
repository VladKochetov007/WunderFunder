---
kind: experiment
status: admitted
created_utc: 2026-10-13T15:44:27Z
updated_utc: 2026-10-13T15:44:27Z
hypothesis_ids: [H-QAT-W4-20261013, H-QAT-A8-20261013, H-QAT-LSQ-20261013]
experiment_ids: [E-QAT-DEPLOYMENT-20261013-R1]
evidence_paths:
  - research/campaign_2026-10-04/E419_RECOVERED_QAT_REPORT.md
  - specs/adr/0036-qat-renewable-capacity.md
---

# QAT research using local and renewable Kaggle capacity

## Objective and authorization

Find the strongest deterministic served predictor through quantization-aware training, compression and validated ensemble improvements. The user explicitly requests implementation of this plan and its preservation as an indexed Markdown file.

There is **no campaign-wide GPU-hour cap**. Use the local GPU as needed and all available Kaggle GPU/TPU allowance, continuing across weekly resets while research remains authorized. The proposed 4/2/2-hour allocations are removed. Per-job resource estimates, recovery windows and collection allowances are operational safeguards, not campaign spending ceilings. No paid capacity, rental, top-up or billing change is authorized. Preserve healthy admitted jobs and the single local GPU-worker authority.

Keep 5 aggregate CPU cores, a 17 decimal GB committed-memory target, 18 GB memory-high, 20 GB hard RAM and no swap. Count parent cgroup memory and unused admitted reservations. Profile actual GPU shapes and memory with operating headroom. CPU is for analysis and execution support, not large model fits. Stage large immutable inputs on physical disk, using hardlinks/memmaps/streaming; `/tmp` is RAM-backed on the current host.

## Research basis and hypotheses

[E419](../../campaign_2026-10-04/E419_RECOVERED_QAT_REPORT.md) established no improvement for weight-only Q8 QAT over matched floating continuation followed by PTQ. Native serving also rounds recurrent activations, which the current trainer does not simulate. The earlier Q4 degradation was a small diagnostic, not a QAT result.

Start with two policies: **uniform recurrent W4**, quantizing every GRU input/recurrent matrix to four bits, and **mixed W4/W8**, using W4 input-to-hidden and W8 hidden-to-hidden. Both retain a Q8 first head layer, existing projection/biases/final head, and FP32 recurrent state.

Version the quantization specification: selected matrices, bit widths, exact signed codes/scales, rounding, zero rows, packed representation and activation semantics. Preserve old checkpoint and CLI behavior. Connect exact code/scale import to native `DynQ8::from_rows`, avoiding constructor requantization. Support packed Q8 and Q4 archives; unpack Q4 once into the existing byte kernels. Report archive savings separately from resident memory and runtime: this is not INT4 execution.

Literature components are hypotheses rather than evidence of gains here:

- [Quantized RNNs, arXiv:1611.10176](https://arxiv.org/abs/1611.10176): placement of quantization within gates and recurrence; motivates the separately controlled activation-aware arm.
- [Four-bit RNN-T, arXiv:2206.07882](https://arxiv.org/abs/2206.07882): mixed precision and recurrently aware training; simulated hardware gains do not establish CPU gains for this runtime.
- [LSQ, arXiv:1902.08153](https://arxiv.org/abs/1902.08153): learned weight scales, conditional on exact scale-preserving export.

## 1. Establish deployment baselines

Use the available clean-T3 H256 CP5 F0 seed-3 student. Freeze checkpoint identity, all teacher/initializer ancestry and source-qualified fitting/evaluation membership before scoring.

Compare unchanged-parent FP32, native Q8, exact packed-Q8 transport, uniform W4 and mixed W4/W8. Calibration and sensitivity analysis use fitting components. Evaluate complete 20,000-row trajectories and the fixed full fold, recording official overall/target scores, recurrent-state error accumulation, actual archive bytes, resident weights and callback runtime. Separate deterministic representation/parity tests from model quality and host-specific performance.

## 2. Matched weight-only QAT

From the same parent, train floating continuation, uniform-W4 QAT and mixed-W4/W8 QAT. Quantize the floating continuation into each candidate format for the primary paired comparisons.

- Identical teacher targets, loss weights, auxiliary terms and sampling.
- FP32, AdamW, maximum learning rate `1e-4`, existing one-cycle schedule.
- Batch 16, chunk 1,000, after actual-shape profiling.
- One epoch, fixed final checkpoint, matched optimizer and averaging initialization.
- Hashed nested 10%, 30% and full fitting-component subsets, retaining complete sequences and causal histories.
- Fixed full-fold evaluation across subset sizes; separate subset fidelity from optimization/update budgets.

Small subsets diagnose implementation and estimate cost. Advance useful mechanisms to full data; retain a scale-dependent follow-up when the screen is inconclusive. Initial fits are explicitly **weight-only QAT**: native activation A8 rounding remains outside training. Do not use an interval crossing zero as an automatic rejection rule.

## 3. Investigate remaining compression error

Run separately controlled extensions:

1. **Activation-aware QAT:** reproduce native activation quantization at recurrent matvec boundaries while retaining FP32 state. First verify unquantized recurrence against fused PyTorch; then compare quantized behavior to native inference.
2. **Learned-scale QAT:** compare learned row scales against fixed row scales at matched budget with exact export.

Profile compiled recurrent forward/backward before scaling. Do not expand a slow Python timestep loop into long fits. Preserve backend/throughput failures separately from model-quality failures. After initial comparisons, test longer schedules and quantization-aware distillation for promising recipes. Duration, teacher targets and quantization curricula are explicit arms with matched controls.

## 4. Spend demonstrated savings on complete endpoints

Test compressed replacements inside existing fixed blends. Then measure whether real headroom admits an additional complementary member or stronger compressed student. A packed INT4 kernel is conditional on profiling a useful CPU opportunity; archive compression alone cannot support a speed claim.

## Compute orchestration and execution ownership

| Resource | Continuing role |
|---|---|
| Local RTX 3060 | Actual-data profiles, QAT fits, exports and reproductions through the existing single-worker queue |
| Kaggle GPUs | Parallel matched candidate/control fits, full-data continuations and independent fold/seed repetitions |
| Kaggle TPUs | Actual-data JAX bridge, then matched same-backend QAT/control repetitions and larger batches |

Recheck provider quota before admission and reconcile settled usage. At exhaustion, collect useful artifacts, continue locally and resume Kaggle after renewal. Available free capacity does not justify repetitive, uninformative fits.

The TPU bridge must repair its recorded provenance mismatch and pass actual-data forward/backward, optimizer update, checkpoint reload and prediction export. Synthetic parity is insufficient. Root owns admission, integration, actual run delivery, artifact collection and allocation. Scoped actors implement quantization/native changes; a separate critic/grader checks them before result promotion. Provider access failures are execution evidence, not scientific negatives.

Every runnable job receives a frozen recipe/source manifest, measured resource estimate, recovery checkpoint policy and collection allowance. Preserve original failures and correct them with new attempt identities. Do not change sources under a running job.

## Validation, allocation and delivery

Test boundaries, zero rows, invalid representations, exact import, cache invalidation and checkpoint continuation. Verify sequence reset/replay, causal state and native instruction-path consistency. Keep `global_wp_clip2_v1` unchanged.

Shortlisted recipes extend across F0/F3 and fitting seeds 3/4 with clean teacher/initializer ancestry; complete finalists extend to all four folds. Report paired native-endpoint deltas, component-bootstrap uncertainty and fitting-seed variation separately. All existing folds are exposed: cross-fold robustness is not pristine confirmation or a numerical public-score forecast.

Rank complete deterministic native endpoints jointly by quality, archive size and measured callback cost. Reallocate after each completed wave, deepening informative mechanisms, retaining exploration and closing repetitive low-information recipes rather than whole families. Preserve immutable recipes, failures, weights and predictions. Index every useful source/result in the unified library and append hypothesis/experiment/allocation history.

Deliver reproducible comparisons and the strongest supported package after broader validation. **Public upload remains a separate action.** Until those measurements exist, code/test/profile completion is not a QAT gain or a new best submission.

## Source records and live progress

[Campaign entry point](README.md), [research contract](CONTRACT.json), [frontier](FRONTIER.md), [authorization ADR](../../../specs/adr/0036-qat-renewable-capacity.md), [earlier strategy campaign](../strategy_execution_r1/README.md). The earlier campaign and its allocations remain historical evidence; this QAT plan controls new admissions for its scope.
