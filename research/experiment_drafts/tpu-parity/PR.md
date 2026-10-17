# [Experiment] tpu-parity: Resolve native versus TPU full-FP32 parity

1. **Status, lineage and ownership**

   Implementation handoff / proposed comparison, not a fitted result. Lineage: **engineering**. IDs: `H-DRAFT-TPU-PARITY-20261017`, `E-DRAFT-TPU-PARITY-20261017-R1`. Author/source: team repository plus this October17 draft, based on main/research-team `18407ecfa3b02d4db70cfb3c34abbacd6653e79d`. Receiving researcher owns device adaptation, source-clean bindings, profiling and complete run/evaluation return. New code in this PR: none; reuse linked existing implementation, preserving original source.

2. **Why it should work**

   Operand/accumulation precision may explain native-versus-TPU output drift. First isolate identical-input matmul and recurrence behavior before backend quality experiments. The strongest general alternative is extra capacity, changed optimization or target calibration rather than the proposed mechanism; freeze the indicated matched controls. Current source evidence: OneTPU update/export ran; reset passed, native parity failed max0.00777 versus1e-4. Cause unresolved.

3. **Where more compute buys quality**

   More compute buys complete fitting horizons, the30 paired cells, clean teacher cohorts and explicit capacity/context ladders; there is no measured scaling law. For student mechanisms use a single card per fit when it fits, and compare downstream official student/blend quality rather than teacher score alone. Large Transformer teachers can use multiple cards and207M/682M starting branches, with1.65B/long-context extensions conditional on evidence and memory. Do not assume all8 cards accelerate a small GRU. No leaderboard impact is measured or numerically predicted.

4. **Frozen comparison and training proposal**

   Arms: `native-fp32`, `tpu-explicit-fp32`. All use FP32 parameters, activations, states, gradients and optimizer state; autocast and TF32 off, no quantization. The adjacent recipe records **starting proposals**, not a final frozen training command: AdamW, preclip norm1,12-epoch WSD with2% warmup and20% cooldown to0.1×LR. Proposed peak LR is 0.001, with parameter-group decay exclusions for biases/norms/gates to be recorded. Exact batch/accumulation/context, source-matched targets, inner checkpoint policy and successful-update count must be bound before fits. Changing the historical FP16/two-epoch recipe to this route is an explicit successor, not a faithful rerun. Scale/context/horizon sweeps are separate frozen arms, not post-result tuning hidden within these30 cells.

5. **Thirty paired runs**

   Not applicable: this is an engineering parity repair; no30-run quality claim. No runs in this matrix have started. `0` fits are required for all named arms; this is a plan, not a compute allocation.

6. **Validation and statistical significance**

   Official `global_wp_clip2_v1`, unchanged ±2 evaluation clipping and original clipped-label magnitude weights. Primary statistic: one quarter times each fold's mean paired endpoint WP delta, then summed; compute scores per fit, not from seed-averaged predictions. Use the [workflow's seed/component procedures](https://github.com/VladKochetov007/WunderFunder/blob/experiment/tpu-parity/research/unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md),10000 resamples, RNG17017. Separate conditional component uncertainty, fitting-seed variability, observed-fold sensitivity and unknown public transfer. Specify primary contrasts/multiplicity (Holm within frozen primary contrast family) before results. Thirty cells do not ensure power; do not treat rows, overlapping fits or four folds as many independent markets. All existing folds are exposed.

7. **Future leakage and representative validation**

   Inputs/feature transforms must be causal with sequence reset and fitting-only scaling. Audit source-qualified component overlap, aliases and teacher/initializer/preprocessing ancestry. True competition labels may reference future horizons; AR feedback accepts only own predictions. Check prefix invariance, reset/reload and intended chunk semantics before expensive runs. Report by source, fold, target, sequence position and activity/target-magnitude regimes; unknown anonymous clocks limit forward-market claims. No clean-validation claim is made for reused fitted groups.

8. **Decision-changing outcome**

   **Useful-effect threshold is deliberately pending**, to be justified from baseline variance, student transfer and GPU cost before fits. Success: a practically useful paired endpoint gain with acceptable fold/seed behavior and relevant downstream transfer. Uncertainty: an unresolved effect or heterogeneous regimes can justify a bounded targeted follow-up. Negative result: no useful gain or a concrete failure mode narrows this recipe; an interval crossing0 does not close the whole direction. Record whether the result changes teacher, student or target-link allocation. No retrospective threshold selection.

9. **How to run and remaining integration**

   **Engineering diagnostic, not a quality experiment.** Engineering diagnostic only: identical operands, explicit backend precision, bounded complete-sequence reset checks. No30-fold quality claim and noGPU/TPU launch in this delivery.

   From the repository root, install the repository's pinned Python/Torch environment on the receiving host and record the actual CUDA/driver/Torch/container identities. Mount only permitted supplied data and source-qualified split/teacher/cache assets on physical storage. Inspect actual B300 topology and FP32 kernel support; no claimed throughput or wall time. Run the implemented **plan command**:

   ```bash
   python research/experiment_drafts/tpu-parity/run_plan.py
   ```

   This prints the30-cell matrix and missing bindings; it does **not** train. Existing implementation entry points are linked below.  

   A receiving-host multi-GPU teacher launch will use `torchrun --standalone --nproc_per_node=<measured_count> <completed_teacher_trainer> ...`; this is a **shape of command, not an existing executable trainer**. Existing `scripts/train_rnn.py` handles GRU fitting and exposes `--precision fp32`, `--seed`, `--valid-fold`, `--split-manifest`, `--grad-stats`, `--save-training-state`; keep its cache/provenance validation. No automatic launch or queue is provided. For initializer work reuse the linked phase runner rather than pretending it accepts generic trainer flags. Resume only from matching source/config and complete optimizer/scheduler/RNG/sampler/state, otherwise restart a declared boundary.

10. **Evidence and archives to return**

   Explicitly hook `wnn.research_reporting.RunRecorder`; it does not instrument trainers automatically. Save source/config/ancestry/environment; attempted/completed steps, losses/components, param-group LR, preclip gradient norm, clipping/nonfinite counts, scores/masks/unit counts, timing/throughput and peak memory. Outer validation reports cannot choose checkpoints. Save all cells, failures, checkpoints/prediction manifests and an English terminal result note.

   ```bash
   python scripts/research_result_bundle.py --root <run_dir> --output-dir <archive_dir> \
     --evidence run_metadata.json --evidence metrics.jsonl \
     --evidence terminal_result.json --weight checkpoints/final.pt
   ```

   The explicit-file bundler writes separate evidence.tar and weights.tar plus SHA256 manifest; add prediction/grade/provenance files explicitly as more --evidence arguments. Keep datasets out. Return private archive locations/hashes separately from this PR, and keep the provenance manifests/narrative notes indexed in Git.

11. **Results, provenance and merge trace**

   No new fits or hardware checks; score/LB impact unknown. Source: [canonical prior record](https://github.com/VladKochetov007/WunderFunder/blob/experiment/tpu-parity/research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/README.md). [Inventory](https://github.com/VladKochetov007/WunderFunder/blob/experiment/tpu-parity/research/unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md), [workflow](https://github.com/VladKochetov007/WunderFunder/blob/experiment/tpu-parity/research/unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md). Add actual PR URL, run source SHA, run manifests, archives, result notes, critic findings, research-team merge SHA and later main promotion SHA. Merge useful negative/inconclusive evidence too; never overwrite frozen recipes or append-only history.

## Implementation sources

- [research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/check_feature_forward_r4.py](https://github.com/VladKochetov007/WunderFunder/blob/experiment/tpu-parity/research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/check_feature_forward_r4.py)
- [research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/check_native_compat_r2.py](https://github.com/VladKochetov007/WunderFunder/blob/experiment/tpu-parity/research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/check_native_compat_r2.py)
