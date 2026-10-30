# [Experiment] Heterogeneous teacher transfer and fixed ensembles against fresh TL3

1. **Status, lineage and owner.** E-HETEROGENEOUS-TL3-20261030-R1,
   H-HETEROGENEOUS-TEACHERS-20261029. The user requests actual execution of the
   proposed six-fit screen and a TL3-baseline ensemble comparison. Root owns
   execution/collection; this packet prepares the exact recipes and fitting-only
   cache consumer. Parent source is PR49 plus the preserved checkpoint-publication
   and GitHub candidate review. Actual immutable source/profile/launch receipts
   are separate. Code completion and CPU checks are not fitted results.
2. **Mechanism.** A plain LSTM and causal dilated TCN may provide useful errors
   beyond the existing GRU/GALAM pool. Test whether a fixed heterogeneous target
   improves identical-capacity GRU students at paired seeds. Low prediction/error
   correlation alone is insufficient; actual fixed-pool/student WP is decisive.
   Additional teacher compute and architecture-specific optimization are alternative
   explanations; this first screen does not isolate diversity from extra compute.
3. **Compute advantage.** Reuse two existing train-only GRU root teachers and
   one immutable GALAM-LSTM128 snapshot. Add two teachers/four students, rather
   than automatically training28base+28fine-tuned models. Teacher prediction
   exports/cache construction/full-valid grading remain real support costs.
   Eight existing L4s permit independent lanes subject to actual memory/disk
   profiles; no unmeasured utilization, duration or leaderboard gain is claimed.
4. **Frozen comparison.** All new fits: event211,4epochs/fixed zero-based3,
   B32/C1000, AdamW.001/WD.01/beta2.999, OneCycle5%, clip1, FP32, EMA1/.999/eval,
   next/structural.3/.3, since32/floor.25, no initializer/propensity. LSTM256×2,
   seed1201; TCN128×12/mix0/RF4096, seed1202. Students GRU256×2, seeds1301/1302;
   same label/teacher target mixture alpha.5, bootstrap0/familyloss0. Control
   teacher is equal fin256_s3/s4 epoch3; candidate is35%each GRU +10%GALAM human
   epoch3/zero-based2 +10%new LSTM +10%new TCN. Pooled raw FP32 targets sum1 before
   the canonical training-time clip. No low-LR correlation stage is admitted here.
5. **Bounded fitting panel.** Six total new fits:2teacher+2control+2candidate,
   24configured epochs/154560expected optimizer updates. Three reused teacher
   checkpoints are not new replications. Canonical TL3 requires the separately
   recovered W18 node; that pending historical fit is not counted among these six.
   Profiles are disposable engineering, not scientific repetitions. No automatic
   extra seed, sweep or fit retry is admitted by this packet.
6. **Validation and uncertainty.** Frozen official train-only/full-valid split,
   10325/0fitting train/valid members, all1873official-valid sequences and3528273
   scored rows. Use pooled official `global_wp_clip2_v1` and report t0/t1, each
   student's paired delta and fixed ensemble delta versus fresh canonical TL3.
   Full valid is development-exposed. Two student seeds do not establish
   significance, independent temporal folds or teacher-training variance.
7. **Leakage and coverage.** All normalizers, EMA teachers, initializers and
   fitting targets exclude official valid. Whole sequences remain intact;
   shuffled seq_ix does not define chronology. New caches join source rows by
   actual group IDs, covering10607legacy or10325GALAM export rows without
   positional truncation. Both construction and consumption verify actual pinned
   checkpoints, raw producer provenance/epoch-index, qualified memberships,
   source inventories, FP32 prediction bytes and cache hashes. GALAM retains its
   native producer schema. Neither approximate overlap exclusions nor source
   separation independently proves hidden competition timestamps.
8. **Decision-changing result.** Working transfer threshold is+.0005WP. Compare
   both matched student seeds and complete declared blends. Mixed/negative
   results narrow this composition/weighting, not every heterogeneous teacher
   family. A stronger pool without stronger students motivates target-fusion or
   capacity work. Do not infer generalization from correlation or add a local
   delta to an old public score.
9. **How to run.** See README, RECIPE.json, recipe materializer and new cache
   builder. Root claims free GPUs through the existing persistent SSH session,
   pins an immutable source, runs representative finite-update profiles, then
   launches exact canonical trainer arguments in detached bounded workers.
   New caches use `wnn.heterogeneous-teacher-pool.v1`; the W18 .8125/family-head
   cache contract is unchanged. New-format-only consumer checks reject student
   bootstrap/scale/architecture or optimization drift. No speculative fallback.
10. **Evidence to return.** Source/config/data/split/producers and cache identities;
    profiles, starts/exits and actual attempted/successful updates; objective,
    train-eval/full-valid curves, LR/preclip gradients/nonfinite diagnostics;
    CPU/GPU/RAM/disk and timing; all fixed checkpoint and prediction hashes;
    complete fixed-ensemble table with exact coefficients/baseline/member counts.
    Preserve source-qualified failures and separate weights from small evidence.
11. **Results and trace.** Focused local recipe/cache checks and Ruff/structure
    validation are recorded in README. They are engineering evidence only.
    An expanded archival-cache suite finds missing sparse-checkout historical
    snapshots/receipts and a Windows-only `resource` import dependency; do not
    hide these or modify frozen legacy fixtures to manufacture a pass. Actual
    receiver/source/profile/launch/result evidence is appended by root. No exact
    historical GALAM or TL3 reproduction, served/public score or completed
    baseline is inferred from this preparation.
