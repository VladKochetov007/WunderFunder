# fin256 ancestry findings

Both seed-specific parents are now exact checkpoint/config identities:

| Parent | Bytes | SHA-256 |
|---|---:|---|
| fin256_s3/model_ep3.pt |3933599|8590dae1c0fb9289c5b56bd4c4520383be089c010baa1a260919b75977f388fb|
| fin256_s4/model_ep3.pt |3933599|24d0f4b0edc2a800702b205e32cf61e9d36b213f2477cf852679155d0472abd7|

The actual [metadata receipt](TL3_FIN256_METADATA_R1.json) reports epoch3,211input channels and GRU256×2 settings. The two recorded args dictionaries differ only in name and seed(3/4). Both have empty `init` and `distill`, so no further weight-parent or teacher alias is recorded for these two roots. This config evidence is distinct from realized historical execution/producer proof.

Their stored recipe includes4epochs, chunk2000,batch32,lr.001,weight_decay.01,WMSE/chunk normalization,`aux_next=.3`,`aux_struct=.3`, `since_weight=32`, `since_floor=.25`, `valid_true_mask=.03`, propensity `prop_f0` with power2/epsilon.03/cache enabled. Precision is fp16, `fused_adam=true`, `ema_decay=.999`, `eval_ema=true`. The args do not include an explicit optimizer-class field; a current trainer default alone is not an authenticated historical optimizer binding.

The fitting options are `train_source=train+valid`, `include_heldout_valid=true`, `valid_train_fraction=1`, `valid_fold=-1`, `full_valid=true`, with evaluation source`holdout`. These are production configs, not evidence of F0 exclusion. The checkpoint stores no ordered training-membership provenance or source map. No strong clean F0 conclusion follows from the368historical holdout groups or the filename.

These are the initializers named by fin256ft_s3/s4's valid-only fine-tuning configs and the teachers named by fbc1k_s5. Their actual creative logical aliases were freshly verified to resolve to the Main run endpoints above. The eight-file [earlier extraction](../tl3_checkpoint_metadata_execution_r1/FINDINGS.md) remains unchanged and was not repeated.

The residual closure dependencies remain: exact historical code/config producer binding, actual `prop_f0` training membership, and the w18 teacher/family target-cache producer/row/preprocessing contracts. The newly read roots also depend on `prop_f0`; empty init/distill does not remove learned-weighting ancestry. All468canonical F0 evaluation groups must be excluded in every newly reconstructed fitted ancestor; production checkpoints cannot be reused as clean F0 controls. No teacher or orgbase branch was dropped, and no clean TL3 fit was admitted.
