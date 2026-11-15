# Valid-regime fine-tune study (H-Q15-VALID-FT, 2026-11-15)

## Trigger: original TL3 members are strongly valid-fitted

Native official WP on the first 300 valid sequences (567,785 scored rows;
[SUBSET_EXPOSURE_R1.json](../tlead_final_r1/SUBSET_EXPOSURE_R1.json), [subset_score.py](../tlead_final_r1/subset_score.py)):
original tl3 TL5 .71487, TL6 .71500, S5 .71545; teammate all-data W18 .70744; our all-data TLead s7 .70138;
train-only PR49 TLead s5 .69726, KD90 .69693. The original public-best lineage sits ≈+.018 above train-only on
valid, ~4× our all-data fits (+.004). Native member correlations ([MEMBER_CORR_R1.json](../tlead_final_r1/MEMBER_CORR_R1.json),
16 seq) split two lineages: originals .990–.994 among themselves, ours/PR49 .990–.993, cross .986–.988.
Plausible cause (unverified): "fttall" = fine-tuned on all data, with all-data parents whose valid-row
targets are in-sample. Valid is a regime-shifted block (adversarial AUC .984), likely closer to test.

## Question

Does extra fitting on valid-regime data improve out-of-sample performance *within the valid regime*? If yes,
stronger valid emphasis in the final all-data fit should transfer to public/private.

## Design ([queue3.sh](queue3.sh), unit `wnn-q15-queue3`)

Init train-only `q14t_tlead_s5@ep3`; PR49 ft recipe (gcorr loss, AdamW lr 1e-4, 2 epochs, fixed ep1, batch 16,
chunk 2000, no KD/since/EMA, inherited preprocessing, aux .3/.3). Chain folds of valid (`chain_folds.npz`,
911/962 sequences), per fold f ∈ {0,1}: **V** = other valid fold only; **T** = train + other valid fold;
**C** = train only. Evaluate on fold f (full). 6 fits (V cheap). Primary: paired T−C and V−C on both folds;
baseline no-ft q14t_tlead_s5 on the same fold rows. Known limit: without the campaign split the 282 train
sequences linked to valid components are included in all arms (equal across arms, not in the baseline).
Falsifier: T−C ≤ 0 on both folds.
