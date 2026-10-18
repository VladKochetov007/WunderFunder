# Next matched diagnostic for the tl3 incumbent

## Decision

The local evidence does not establish an exact clean F0 counterpart of
`tl3.zip`. The incumbent is now identified from the user transcript as
submission `1lmehfi3`, score `0.66832`. Its fixed members are 0.34
`w18final_distilled_s5@ep1`, 0.33 `fttall_tlead_s5@ep3`, and 0.33
`fttall_tlead_s6@ep3`. The local archived package is 12,538,394 bytes with
SHA-256
`d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569` as
recorded in the October 17 inventory (the user-supplied portal bytes have not
been hashed).

All four production-member rows in `CONTROL_ASSET_MAP.json` say
`train_source=train+valid`, `include_heldout_valid=true`, and `valid_fold=-1`.
Their valid scores are therefore full-fit diagnostics. Selecting F0 rows from
those checkpoints cannot turn them into clean controls. The map's nearest
tied-lead endpoints are explicitly marked “not fully certified”; their logs
show 467 evaluation groups, 10,606 train and 1,406 valid fit groups, different
teachers (`h256ft_f0,h256s4ft_f0`), and seeds 3/4. Those are not this tl3
recipe and do not match the current F0 membership of 468 evaluation, 10,535
train, and 1,405 fit-valid groups.

## Next evidence-producing comparison

With the F0 residual decomposition already run, and after the captured Main
T3 prediction has been aligned and graded on those same rows, make one
held-out-F0 reconstruction of the **three-member tl3 recipe**. The current
export receipt is an inference capture, not that grade. This is a single
diagnostic fold/seed cell, not a new public submission or a full-fit estimate.
Do not repeat the residual decomposition or tune blend weights against the
public score.

Use `F0_MEMBERSHIP_COMPARISON.json` as the membership contract: train on the
10,535 `fit.train` plus 1,405 `fit.valid` groups, and exclude all 468
`evaluation.valid` groups from every fitted student, parent, teacher, teacher
target cache, propensity/calibration object, and learned preprocessing step.
The old Main split (`09263ad4…79f9316a3d`) and current canonical split
(`c09f7240…2d2294b7`) have different manifest and `chain_folds.npz` identities,
but the receipt proves their ordered memberships and component mapping are
equal. Use the canonical split for newly trained artifacts; retain explicit
ordered membership digests in every new provenance record. Keep the old split
only when reading the already-restored Main T3 checkpoint, whose provenance
pins it.

Rebuild the tl3 members at their archived endpoints and then apply the
unchanged coefficients:

| Weight | Member to reproduce | Fixed recipe facts to preserve |
|---:|---|---|
| 0.34 | `w18final_distilled_s5@ep1` | 211 inputs (112 raw + 99 causal-event); GRU 256×2; warm start `fbc1k_s5@ep3`; `w18final_teacher` distillation, alpha 0.7; family-teacher list `w18final_fbc1k_s5,w18final_family_full,w18final_family_raw,orgbase`. |
| 0.33 | `fttall_tlead_s5@ep3` | 263 inputs (211 + 52 lead); tied-lead shared instrument encoder; GRU 256×2; scratch start; teachers `fin256ft_s3,fin256ft_s4`; alpha 0.5; bootstrap hidden width 8; Muon/AdamW and EMA family. |
| 0.33 | `fttall_tlead_s6@ep3` | Same recorded family/config as tied-lead s5 with seed 6; retain the seed-specific training state and archived endpoint. |

For the distilled member, reconstruct its warm initializer and every named
teacher with F0 excluded; do not reuse `fbc1k_s5@ep3` or teacher weights unless
their exact provenance proves that exclusion. For both tied-lead members,
reconstruct or certify the `fin256ft_s3/s4` teachers under the same exclusion.
Preserve the archived feature assembly, training schedule, optimizer, EMA,
teacher-target generation, and endpoint selection. The local map is not a
complete run card: recover the original config/source snapshots and resolve
defaults before launch. If the `orgbase` branch or any other upstream asset
cannot be shown to exclude F0, stop and label the three-member clean replica
blocked; silently dropping a teacher would make it a different recipe.

Export each reconstructed endpoint on the same ordered 468 evaluation groups,
then compute the fixed 0.34/0.33/0.33 mixture without fitting a new combiner.
Compare it row-for-row with the current Main T3 control and the existing F0
candidate; report each member and mixture on identical rows, plus the existing
component-paired uncertainty summary. This measures one exposed fold and one
seed realization only. It does not estimate the public score, establish a
winner, or meet the standing 30 paired fold/seed cells per arm requirement for
a general scientific comparison.

The cross-validation product and the public incumbent must remain distinct in
reporting: the public tl3 package is a final `train+valid` fit, while this new
F0 analogue has F0 held out and trains only on the prescribed fit groups. It
can test whether the *recipe* transfers under clean F0 exclusion; it is not the
production checkpoint nor an exact prediction replica of the submitted ZIP.

## Inputs not yet source-qualified locally

- The local ZIP inventory binds its eight members, fixed coefficients, size,
  and archive SHA. It contains `members.json`, `solution.py`, three runtime
  binaries, and the three NPZ weight files; it contains no training configs or
  provenance sidecars. The trace joins this local/server archive to the user
  supplied portal row by filename, not by portal-payload hash.
- `CONTROL_ASSET_MAP.json` records remote source paths and selected config
  fields for the production members, but its scope explicitly says the large
  checkpoint payloads were not rehashed and full ancestry was not certified.
  The exact creative source snapshot, complete resolved run configs,
  checkpoint bytes/hashes, training provenance, and teacher-target cache
  identities must be recovered and pinned before the clean reconstruction.
- The initializer and teacher closure is not locally bound to the tl3 recipe:
  `fbc1k_s5/s6`, `w18final_teacher` and its listed family teachers, and
  `fin256ft_s3/s4` need exact source/config/checkpoint identities and
  F0-exclusion proof. The asset map also flags historical teacher-cache epoch
  binding and propensity-model ancestry as not re-certified; resolve the
  production defaults and provenance for every learned preprocessing or
  weighting input rather than inferring them from a nearby run.
- The `dftt_tlead_s3/s4` historical logs and checkpoints are available in the
  metadata inventory, but their 467-group fold, different teachers and
  uncertified ancestry make them useful only as recipe-near context. They do
  not fill the missing exact tl3 F0 counterpart.

The immediate readiness gate is a compact, reviewable source-closure record:
one immutable code/config/data identity per member and ancestor, exact ordered
F0 fit/evaluation membership, and a graph showing that no evaluation group
contributed to any fitted ancestor or cached target. Once that record closes,
the one-cell reconstruction above is the next informative scientific step.

## Evidence read

- [`public_submission_trace_r1/README.md`](public_submission_trace_r1/README.md)
  and [`INCUMBENT_BINDING.json`](public_submission_trace_r1/INCUMBENT_BINDING.json):
  portal row, local/server ZIP identity, members and caveats.
- [`research/unified/reviews/audit-2026-10-17/submission_inventory.json`](../../unified/reviews/audit-2026-10-17/submission_inventory.json):
  ZIP entry inventory and member coefficients.
- [`research/unified/models/CURRENT_MODEL_CATALOG.md`](../../unified/models/CURRENT_MODEL_CATALOG.md):
  family feature dimensions, architectures and full-fit caveat.
- [`CONTROL_ASSET_MAP.json`](CONTROL_ASSET_MAP.json):
  `production_members`, `creative_nearest_siblings`, and `remaining_limits`.
- [`export_preparation/F0_MEMBERSHIP_COMPARISON.json`](export_preparation/F0_MEMBERSHIP_COMPARISON.json):
  exact ordered membership/component-map equality and differing source
  identities.
- [`diagnostics/results/README.md`](diagnostics/results/README.md): the already
  completed F0 residual decomposition. The separately captured Main T3 export
  receipt is in
  [`export_preparation/export_execution_r1/records/computer-vision-homework-quality-gap-control-export-v1/EXPORT_RECEIPT.json`](export_preparation/export_execution_r1/records/computer-vision-homework-quality-gap-control-export-v1/EXPORT_RECEIPT.json);
  it records a feature-only full-valid export, not a metric result.

No fitting, inference, package build, payload read, or remote action was
performed for this memo.
