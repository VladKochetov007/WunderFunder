# [Experiment] E-GALAM-QUALITY8-20261028: quality mechanisms on eight L4s

1. **Status, lineage and owner.** Root preparation, parent H-GALAM-L4-20261028;
   source base aac25c7c. Eight locally checked recipes, zero fits/profiles.
   User approved RSNA deletion after storage consultation; deletion completed.
   Full launch remains pending consultation. Original GALAM dimensions unknown.
2. **Why it should work.** Capacity, recurrent depth, recurrence type, attention
   history, gradient horizon, LR and duration are distinct possible constraints.
   Each candidate changes one named package relative to an engineering anchor;
   overfit, mask/schema defects and optimizer granularity are alternatives.
3. **Compute advantage.** Eight independent fits exploit eight L4s without DDP
   synchronization for tiny models. More compute buys candidate breadth and a
   longer fixed training horizon. No empirical gain, utilization or ETA exists.
4. **Frozen comparison.** README and eight JSONs specify the prospective panel:
   FP32/AdamW, warmup-cosine-v2, train-only normalization, common seed1001 and
   fixed final5/10 epochs. Batch16 awaits profiles; changes require an explicit
   pre-fit amendment and new hashes. No teacher, KD or valid-fitted initializer.
5. **Bounded fitting panel.** Exactly8new fits:1baseline+7candidates, each one
   seed and one GPU. Zero reused controls/teacher/student fits. Seven5-epoch and
   one10-epoch recipe =45fit-epochs. User explicitly supersedes the older6-fit
   plan for this wave. Engineering profiles accounted separately.
6. **Validation and uncertainty.** Official chronological train then full valid;
   pooled official WP plus both targets and comparable explicitly labelled
   train-eval diagnostics. Single-seed deltas are descriptive; validation is
   exposed, no independent confirmation or seed-variance estimate. No best-valid
   epoch selection. Report all outcomes and exclusions.
7. **Leakage and coverage.** Whole sequences/component exclusions, fitting-only
   preprocessing and teacher ancestry, past-only inputs and explicit state reset.
   seq_ix is shuffled. Complete source audit must reconcile historical corrupt
   group528 handling and target-mask semantics; neither can be assumed away.
8. **Decision-changing result.** A prospective +0.0005 full-valid WP delta is a
   spending trigger for targeted confirmation, not significance. Report target
   regressions, actual compute and every negative. NaN/OOM/data interruption is
   infrastructure-censored, not a zero scientific score. Mechanisms can remain
   unresolved at this one-seed budget. Combine only after separate evidence.
9. **How to run.** No dispatch script is claimed ready. After data/runtime/mask
   checks, profile each exact recipe with scripts/train_sequence.py and approved
   source-bound profile receipts; then one CUDA_VISIBLE_DEVICES slot per fit.
   Use --config, --data-root, --split-manifest, --output, --profile-receipt.
   Durable detached jobs, unique output paths, per-job exact ownership and
   checkpoint/resume receipts are required before launch. Inspect literal
   command lines and measured ETA with the user before admission.
10. **Evidence to return.** Source/config/env/data/manifest hashes, every fit's
    train and full-valid loss/WP/t0/t1, actual masks, LR, preclip norms, updates,
    epoch/elapsed curves, throughput/data wait, GPU/RAM/disk, checkpoints,
    predictions and scorer receipts. Preserve evidence and weights separately
    with hashes. Keep interrupted status distinct from completed endpoints.
11. **Results and trace.** Local schema/model construction checks passed for all
    eight recipes; no forward, training payload or scientific score. Server
    audit and authorized cleanup receipts are linked in README. Independent
    portfolio/data reviews are preserved. No public-score claim or GitHub URL.
