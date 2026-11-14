# PR54 final FBC teacher versus clean reproduction audit

**Date:** 2026-11-14
**Scope:** receipt and source audit only; no model loading, fit, profile, provider action, or remote write.

## Verdict

PR54 provides a valid, hash-closed full-data `tl3final_fbc1k_s5` artifact in its 13-role W18 ancestry. Its declared core recipe, fixed epoch-3 endpoint, and EMA selection align closely with the clean train-only `tt29rep_fbc1k_s5` run. This is a **hyperparameter-aligned final-data refit**, not an exact source, membership, or teacher-ancestry match. It has no independent score.

The final-data model therefore cannot inherit the clean run's `.6914204799` score. That score belongs to the clean run on the exposed 1,873-sequence official valid set. PR54 fitted those 1,873 valid sequences and explicitly forbids an independent-score claim.

## Archive and 13-role closure

The downloaded [13-role evidence archive](</home/vlad/development/wunder/research_tmp/q14_team_artifacts_r3/tabdata-w18-final-13-role-evidence-20261103-r1.zip>) has SHA-256 `2dc00d8a16f316775ddb979382312ee8a1d60b3c3555128726b8c4a44786c29e`, matching the adjacent [delivery manifest](</home/vlad/development/wunder/research_tmp/q14_team_artifacts_r3/tabdata-w18-final-delivery-manifest-20261103-r1.json>) (manifest SHA-256 `d24715f7c15de929b55c35b0de84275a9ebd953774099e5260327af738be2d2f`). All 43 files declared by the archive manifest match their recorded sizes and SHA-256 values. The 13 role identities agree across the delivery manifest, archive manifest, and final model receipt.

For `tl3final_fbc1k_s5`, the command hash is `7a87eab97778e7f9c784b87a6dc1a8f7247c2e4cecc484808e1ee35b2aebf4ea`, process receipt hash `05a5fb3f133af28ad1b90a0d3abce4ae94889b3c7816c5deae6a651cc227c7fc`, and fixed `model_ep3.pt` hash `d012e69e2617847cf89775161d7c2180d557cc1c7395f1432c8c19183013fdd9`. Each was checked against the archive and delivery bindings. The final-fit membership manifest closes at `3f766945e00a2fa5edc6230d9c0bed125da112584f11e3d82458167a011bf543`: 10,607 train plus 1,873 valid groups, 12,480 total, zero independent evaluation groups; score computation is false and an independent-score claim is forbidden.

## Comparison

| Dimension | Clean `tt29rep_fbc1k_s5` | PR54 `tl3final_fbc1k_s5` | Audit finding |
|---|---|---|---|
| Core recipe | GRU H256×2, seed 5; 4 epochs, fixed ep3; batch 32/chunk 1000; LR .001, OneCycle/warmup .05; WMSE/chunk; weight decay .01, clip 1, beta2 .999; TL3-Muon ×3; aux-next/struct .3/.3, aux weight .5; since weight 32/floor .25; two fin256 teachers, alpha .5, boot-h8 | Same declared values and optimizer/loader flags in the final role command | Strong core hyperparameter match. More fitted groups mean a different number/order of optimizer updates and a different realized OneCycle trajectory. The clean provenance records 25,760 completed updates; the final plan specifies 31,200 expected updates. |
| Endpoint / EMA | Fixed epoch 3; EMA teacher 1, decay .999, `eval_ema=true`; score/checkpoint selected from EMA evaluation | Fixed epoch 3; command includes `--ema-teacher 1 --ema-decay .999 --eval-ema` | EMA endpoint semantics match in the declared recipes. |
| Membership | 10,325 train-only fit groups; zero valid fit groups; 1,873 official-valid evaluation groups; split SHA `6d23a1b7…` | All 10,607 train and all 1,873 valid groups fit; no evaluation groups | Expected final-data expansion, but not a clean evaluation counterpart. |
| Teacher ancestry | `tt29rep_fin256_s3` / `_s4`, train-only parent checkpoints `9009230ecb036945ba14afc9675932ea235e7a2ee1316b2a9d07554044144452` / `fe496fb2dfdf2579bce808efd551ece67b2a0ea906c6859f513be6250b54ea90` | `tl3final_fin256_s3` / `_s4`, final-data parent checkpoints `b0e4bce354749f4ac0fc1362304fdef79f4c13692e463bee5c2a99079c1b2474` / `d81db16463453b99e6b54db4bd97c1b1fcae4bfb4a464a192ba110edefa05bc3` | Parent weights and their fitting membership differ. This can affect the teacher even with matching distillation settings. |
| Source identity | PR49 worktree commit `188c0788e571c6f981cdc6827fef0d50bd38cf14`; clean trainer implementation hash `d174a35541ebfe6e0efc66fd124d491daf5c5643c338f5d0702c33f404418889` | Source recovery capsule SHA `83e307643088844d50fd219d1fdb145fd2322e018b251b411f7b996b91e4407f`; final role receipt has no Git commit | Not an exact same-code run. The `train_rnn.py` content hashes differ (final `4b641867…`, clean `1f6d980a…`). The `training_dynamics.py` and `tl3_optimizer.py` texts match PR49 after newline normalization; the final source also adds the final-fit bridge. The source binding's one-file cache-validator amendment is relative to its own base capsule, not proof of PR49 source identity. |
| Precision | Clean launcher logged TF32 off and cuDNN RNN precision `ieee`. With `precision=fp32`, the TL3-Muon factory selects FP32 Newton–Schulz intermediates. | Command specifies `--precision fp32`; the retained TL3-Muon factory maps this to strict FP32 Newton–Schulz. The final source precision setup disables general TF32 flags, but does not explicitly set `torch.backends.cudnn.rnn.fp32_precision="ieee"`; no such runtime setting is recorded in the process receipt. The source lock pins Torch `2.14.0+cu130`. | Optimizer Newton–Schulz is source-path-supported as FP32; strict IEEE cuDNN GRU execution is **not certified** for the final run. If its locked Torch runtime was used, the missing RNN-specific override leaves the documented Torch 2.14 TF32 default as a material concern. This runtime claim remains unverified. |

The November 14 legacy-`Muon` BF16 Newton–Schulz correction does not directly describe this pair: both use the separate `tl3-muon` factory, whose strict precision argument is derived from `precision=fp32`. The remaining precision uncertainty is specifically the final run's cuDNN GRU mode and absent runtime precision telemetry.

## Research use and limits

The receipt supports identifying PR54's FBC checkpoint as a provenance-closed final-data teacher role. It does not show that those full-data weights outperform the clean teacher, and the clean score cannot be transferred to them. PR54's W18 receipt includes this FBC checkpoint as a 0.3-weight teacher role; per the latest root steering, this full-data teacher has **not** been used in the separate clean three-teacher target diagnostic now starting. This audit does not gate other quality hypotheses or change the original eight-local-GPU-hour cap; root retains ownership of any teacher experiment.

For a future direct quality comparison, hold the target pool and ancestry fixed where possible, use the corrected strict recurrent-kernel setup, and grade only a train-only candidate on the exposed development panel. Keep any final-data refit unscored on that same valid set.

Evidence read: [artifact scout](PR_SCOUT_R1.md), [clean teacher track and logs](../../../campaign_2026-10-29/teacher_track_r1/README.md), [precision audit/amendment](../PRECISION_AMENDMENT_R2.json), and the archive/manifest above. Metadata was inspected without importing PyTorch or unpickling weights.
