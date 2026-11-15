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

## Results (2026-11-15 ~04:40 UTC)

Fold scores (f0 = 911 seq / 1,711,934 rows, f1 = 962 / 1,816,339), base train-only TLead s5 .69206/.69080:
V (valid-only ft) .69389/.69257 (+.0018/+.0018); T (train+other valid) .69011/.68944 (−.0020/−.0014);
C (train-only ft) .68848/.68827 (−.0036/−.0025). Only pure valid-regime data helps; the ft recipe itself
hurts on train, and train data dilutes the valid effect.

OOF blend ([oof_blend.py](oof_blend.py) → [OOF_BLEND_R1.json](OOF_BLEND_R1.json); V predictions on fold f come
from the model fine-tuned on the other fold): every member gains (tl5 +.00175, tl6 +.00194, fbc +.00140,
kd90 +.00139, all rows). TL3C .692340 → V .694065; TL3C+.25 KD90 .692504 → **V .694195 (+.00169; f0 +.00153,
f1 +.00196)**. Mixing base back in is monotonically worse (a=.25/.5/.75 → .69330/.69385/.69414). V members stay
diverse (corr .991–.994). This is the largest supported lever found (≈10× any blend effect).

Production (ft on all 1,873 valid sequences, no eval): `q15p_V_{tl5,tl6,fbc,kd}` from train-only bases,
`q15p_V_{s7,fkd}` from all-data bases. Packages in `submissions/final_20261115/`:
- `v4_ft.zip`: V_fbc .255 + V_tl5 .2475 + V_tl6 .2475 + V_kd .25 (local OOF twin .694195).
- `mix_orig2_v2.zip`: original tl3 S5 .25 + TL5 .25 + V_tl6 .25 + V_kd .25 (cross-lineage; no local twin).
Recipe variants (lr 3e-4, 4 epochs, wmse loss) queued as `wnn-q15-queue5`.
