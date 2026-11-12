# PR82 #06: per-update GRU weight EMA

This packet adds one isolated treatment to the PR80 K16 labels-only sequence
recipe: update a parameter EMA with decay `0.999` after every successful optimizer
step, then use the averaged parameters for full validation and inference
checkpoints. The EMA starts from the fresh normalized model weights at update
zero. It is parameter-only; buffers remain at their trained model values. The
AdamW model remains the training state and is saved together with the EMA shadow
at every resumable epoch boundary.

The comparison uses four fresh source-qualified configs: EMA and raw K16 for
seeds 203301 and 203302. This makes each same-seed comparison use the identical
new source, split and runtime contract; the completed PR80 K16 controls remain
historical reference values and are neither refit nor used as admission inputs.
Configs preserve the official split and six-epoch endpoint: fit and
normalize on 10,325 whole official train sequences; fit on zero validation
sequences; evaluate all 1,873 valid sequences / 3,528,273 scored rows after every
epoch. Validation is exposed development data. No teacher or initializer is
used, and `seq_ix` is never treated as chronology.

The per-update callback runs only after the optimizer step has passed the
existing synchronized finite-gradient and finite-parameter checks. EMA state is
included in the safe resume checkpoint and the actual-shaped profile reload
proof. Profiles must verify the exact source, config and split for 30 updates;
ROOT owns native admission and fit launch. The four configs each received
exact-source 30-update profile receipts, including the two raw K16 controls.
The cohort used the existing fixed CLOCK_R1 bytes and cutoff; no new clock was
created. The execution status below supersedes this draft's original prelaunch
wording.

This is a standalone EMA screen. It does not combine EMA with Muon, since-event
weighting, a teacher, or a different checkpoint rule. The full-valid monitor
evaluates EMA weights every epoch and records the weight source in event context.
The final model and all retained epoch snapshots contain EMA parameters; the
resumable training state retains raw parameters plus EMA state.


## Execution status (2026-11-12)

ROOT admitted four same-source cells under CLOCK_R1. Both raw K16 controls completed E6 and reproduced historical monitor scores to about 1e-9. Both EMA treatments reached 38,760 updates but timed out during final E6 evaluation; their E6 checkpoint weights were not committed, and only E1-E5 checkpoints are retained. Treat them as partial fits. Canonical R3 rejected the two completed raw controls because their fit roots lacked the project path, then ROOT launched the unchanged R3 receiver into new R4 output roots after creating verified symlinks. The R4 result is pending; see the [combined execution packet](../l4-followups-ema-aux-20261112/README.md).
