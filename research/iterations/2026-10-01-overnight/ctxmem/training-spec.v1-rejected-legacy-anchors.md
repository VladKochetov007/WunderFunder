# CTXMEM01 fixed eight-fit execution

The native cost preflight passed its unchanged 3% limit (median paired ratio
1.023038596225507). It provides no forecasting evidence. This screen requires
independent source/plan review and an explicit root allocation before execution.

`build_trainer.py` derives `train_ctx.py` from the exact original
`scripts/train_rnn.py` SHA256
f97047e6ef51f32097f2cbe2eefc7c57766771ebaed12ae8798a3c1b7efcd52f.
The receipt records every substitution. Only the model factory, strict recipe/
identity guard, immutable output creation and validation/fixed-endpoint handling
change. The original batch construction, objective including all auxiliaries,
autocast, backward, GradScaler, trainable-parameter filtering, fused AdamW,
gradient clipping, OneCycleLR, state detach and checkpoint payload remain intact.
The original evaluation-loader CPU seed draw occurs without reading its rows.
No placeholder score is emitted; epoch0 and epoch1 are retained and model.pt is
exactly the fixed epoch1 state. There is no EMA/SWA/teacher in ordinary H2.

Four existing ref/refs4 epoch3 parents initialize matched current/memory heads.
The exact source checkpoint hashes and every argument appear in execution-plan.json.
Eight fresh runs cover seeds3/4 and held-out chain folds0/1, two epochs each.
Both arms of each seed/fold share the same parent and ordinary initialization/
shuffle RNG. Assignment of arms to the two GPU lanes alternates across pairs.
The context-model implementation, clock, half-lives, projection and zero-E policy
remain the already reviewed native-spec version; no feature fitting occurs.

The exact parsed namespace must equal the registered job, including defaults.
Guarded checkpoints record the plan hash, parent hash, arm, fold and complete
projection/recurrence metadata. The initial original-head gradients match ordinary
H2 at zero E. Extra gradients can increase the joint clipping norm, so this is
not a claim of an identical ordinary-H2 training trajectory. The paired control
controls the extra optimization path with equal parameter count.

Prediction uses only `SequenceFile.read(..., targets=False, mask=False)` on the
held-out fold. NPZ metadata access materializes only row_group, split and fold.
Before and after prediction, full file hashes bind valid.parquet, valid_targets.npz,
chain_folds.npz and the exact data reader. Hashing compressed file bytes establishes
identity; no target/scoring arrays are decoded or used. Each immutable output
retains the full original valid-cache order, fills only its own fold, and uses NaN
outside that fold. Receipts bind source, checkpoint, plan, exact groups/positions
and prediction bytes. Only the root's guarded grader may evaluate the caches.

After all sixteen train/predict stages succeed, `check_actual_fits.py` verifies
their exact ledger commands/plan hashes and logs before opening any model.
It checks all epoch snapshots and model.pt; exact parsed arguments; fixed parent
backbone/normalization/P; nonzero learned E and changed ordinary heads; finite
parameters; absence of validation scores; fixed-epoch prediction lineage and input
hashes. It does not open prediction contents or target arrays. A failed fit/audit
is preserved and cannot silently restart under the same plan.

Four fixed guarded readouts are listed explicitly in the plan. Each uses equal
thirds in [plain128 seed3, plain128 seed4, c500] order. Joint memory must exceed
joint current and the plain anchor by at least .0003 pooled, each fold nonnegative.
Each single-seed replacement must exceed its matched current replacement pooled
strictly above zero and on neither fold be negative. These diagnostics cannot
rescue a joint failure. No standalone readouts, coefficient tuning, epoch selection,
rank/clock/half-life searches or combinations with SAM/Q3 are allocated.

Root launches the two existing queue lanes on GPUs6/7 with --timeout1200. The runner
requires at least6200MiB free at each stage, .34 GPU memory fraction, four Torch
threads, two loader workers and4GiB loader budget,20GiB host memory scope, no swap.
The root GPU watcher enforces5.5GiB per job and at most two jobs. New screen jobs
stop at06:00UTC; any conditional final heads stop launching at07:00UTC. Expected
screen duration is about10–20 minutes including input identity checks, not a bound.

Only all gates passing can allocate the two final memory heads on existing full-data
seed3/4 parents, again fixed epoch1. Their concrete parent hashes/commands require
a separate finalist freeze. Export reuses the exact fixed3 wrapper/native source,
adds P/E and metadata only to those two members, retains unchanged c500, and applies
the reviewed ordinary FP16 head-weight policy. Trained-model native/AMP parity,
arbitrary prefix/reset behavior, all packaged ISA fallback correctness and complete
ZIP projected39.4M-row runtime at most50 minutes remain mandatory before delivery.
