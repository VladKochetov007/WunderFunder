---
kind: review
status: complete
created_utc: 2026-10-14T02:23:12Z
updated_utc: 2026-10-14T02:23:12Z
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: [E-STUDENT-F0S3-H256L2-20261014-R2]
evidence_paths:
  - research/campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h256_l2_r2_fit_validation.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_SEED3_B8_R2_CONSUMER_CONTRACT.json
  - worktrees/student-frontier-ops-r4-20261014/scripts/validate_student_frontier_fit.py
---

# F0 student-frontier R4 actual-trainer fit-gate review

## Question or claim

Does the R4 post-fit validator correctly check the frozen H256×2 fit against the actual trainer and optimizer semantics, without rejecting it for a `beta1` field that the trainer does not serialize?

## Findings

**Validator and receipt: GO.** The preserved R3 validator attempt failed after 5.03 seconds at one assertion: it expected `args.beta1 == 0.9`, while the frozen trainer has no `beta1` CLI argument and therefore records no such field. The failure happened during post-fit validation; it is not a training failure, OOM, or evidence against the recipe. The full H256×2 fit and checkpoint remain unchanged.

R4 removes that incorrect CLI-field comparison. It pins constructor beta1 `.9` in the consumer contract, rejects a serialized beta1 override, checks `cycle_momentum=true` and the 14,920-step OneCycle horizon, and reconstructs the beta1 cycle under the PyTorch version recorded by the actual run. The consumer contract pins `src/wnn/training_dynamics.py` at `041e3afdf11b064ba35b6ee021116789c6222a4c73a992ea1398ad69f65c0463`; that source constructs AdamW with `(0.9, args.beta2)` and enables OneCycle momentum cycling for AdamW. The checked runtime path is `.95 → .85 → .95`: the scheduler changes beta1 during initialization, before the first optimizer update, so `.9` is the constructor value rather than the first update's value. The validator also checks the recorded PyTorch version and the OneCycle defaults used by that source.

The repaired validator passed on `student_frontier_f0s3_h256_l2_r2` under the unchanged R2 contract SHA `9476373fc3c32aea2ae48e9a2c48d211f0f152d13d086fa68efd1e8b1a02e768`. Its canonical receipt records 14,920 attempts and completed updates, zero AMP skips, 14,920 gradient-stat steps with zero nonfinite steps, and 16 finite terminal model tensors. Cache/teacher, split and source-qualified sampler, propensity, normalization-source, source-hash and checkpoint bindings remain checked. It explicitly does not claim to check per-step loss finiteness.

## Evidence and interpretation

The frozen validator packet is clean at commit `352ee724c378de01fe8566dc9d901c230915b41c`. Validator SHA-256 is `902ba98bf37abc3713ef04c276496dc9834307777e175f90bd9c271d1252c75a`; the canonical fit-validation receipt SHA-256 is `1042e31a73456edd88f40a8bd03b11c91ff346907ace16c50abc10b58fecdf43`. The preserved R3 failure attempt and traceback are in the same commit.

The actor's R4 static-preflight receipt reports that all eight one-shape sequences bind the R4 validator path and matching fit receipt while keeping the previously frozen profile contract and consumer contract hashes. Its SHA-256 is `f6b26da736be22db368adf7d130c477e48ca44cb924f32aa482ddec3bdb647a0`; checker SHA-256 is `46eafa600544500271a9e94b8f114440ed3bd1e5408f57dae187c0304f73222b`. I inspected the R4 validator diff, the H256×2 plan delta, the static receipt, and the actual validation receipt read-only. No CUDA, fit, export, grade, or provider action was performed by this reviewer.

## Limits and validity

This review clears the fit assertion and its actual H256×2 receipt only. The fit is an exposed fold-0 development run and makes no quality or clean-holdout claim. The R4 receipt does not establish profile/export resource safety or quality; root retains the separate B16 profile terminal-counter and VRAM gate before the corresponding export. The eight-plan static-preflight receipt is recipe evidence, not execution evidence for the other seven shapes.

## Decision and next step

Use the R4 validator and canonical receipt as the corrected post-fit gate for the existing H256×2 control. Continue only through the separately reviewed per-shape B16 profile and root's terminal resource checks. This review adds no user-approval gate and does not authorize automatic progression to another shape.

## Source records

- [R4 validator source](../../../../worktrees/student-frontier-ops-r4-20261014/scripts/validate_student_frontier_fit.py), [preserved R3 failure log](../../../../worktrees/student-frontier-ops-r4-20261014/research/campaign_2026-10-14/cpu_student_frontier_r1/results/student-frontier-f0s3-verify-h256l2-r3.log), [failure attempt receipt](../../../../worktrees/student-frontier-ops-r4-20261014/research/campaign_2026-10-14/cpu_student_frontier_r1/results/student-frontier-f0s3-verify-h256l2-r3-attempt.json), and [R4 eight-plan preflight receipt](../../../../worktrees/student-frontier-ops-r4-20261014/research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/R4_SEQUENCE_STATIC_PREFLIGHT.json), from commit `352ee724c378de01fe8566dc9d901c230915b41c`.
- [Canonical H256×2 fit-validation receipt](../../campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h256_l2_r2_fit_validation.json) and [frozen R2 consumer contract](../../../../worktrees/student-frontier-sequence-20261014/research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_SEED3_B8_R2_CONSUMER_CONTRACT.json).
- [Prior R3 operations review](CPU_STUDENT_FRONTIER_OPS_R3_REVIEW_20261014_R1.md) covers the fit→assert→B16 profile→validated export→grade source sequence; this R4 review addresses only the corrected actual-trainer fit assertion.
