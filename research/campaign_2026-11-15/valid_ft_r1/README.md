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

## Recipe variants (TLead s5 base, fold twins; reference V .69389/.69257)

[queue5.sh](queue5.sh): lr 3e-4 .69383/.69070; 4 epochs (ckpt ep3) .69262/.69159 (overfits); wmse loss
.69376/.69259 (≈ gcorr). The PR49 ft recipe (lr 1e-4, 2 epochs, gcorr) stays best. [queue6.sh](queue6.sh) tests
softer settings: lr 5e-5, 1 epoch, since-weight 32.

Softer variants ([queue6.sh](queue6.sh)), mean of folds vs V .69323: since32 .69339, lr 5e-5 .69333, 1 epoch
.69299 — within noise; production not refit.

Valid-like train augmentation ([queue7.sh](queue7.sh), [subset_launch.py](subset_launch.py) keeps only top-K train
sequences by the Oct-29 adversarial OOF score): K=250 .69326/.69298 (mean .69312), K=700 .69319/.69288 (.69304)
vs V .69323 — no gain. Pure valid data remains best. (The adversarial model saw all valid; mild selection leak.)

Base-choice test ([queue8.sh](queue8.sh), running): all-data-analogue TLead from scratch on train + valid fold
(1−f), then V-ft on the same fold; compare with V from the train-only base.

## Base choice: all-data-analogue base beats train-only base (2026-11-15 ~09:00 UTC)

[queue8.sh](queue8.sh): TLead from scratch on train + valid fold (1−f) (PR49 all-data recipe, seed 7), then V-ft on
fold (1−f); score fold f. Base .69345/.69198 (vs train-only base .69206/.69080); **+V .69488/.69276** (vs V from
train-only base .69389/.69257; +.0010/+.0002). V-ft still adds +.0014/+.0008 on top of a base that already saw the
same valid rows. Production implication: fine-tune all-data bases, not train-only ones.

Original tl3.zip members (all-data, strongest public lineage) only exist as NPZs; [npz_to_ckpt.py](npz_to_ckpt.py)
inverts the exporter into same-layout templates (`q15o_{tl5,tl6,s5}`; aux head rows from template). Check: converted
original TL5 scores .71106 on full valid (fitted; train-only TLead .69142, corr .984), consistent with the earlier
.7149 subset readout. [queue9.sh](queue9.sh) V-fine-tunes all three on all valid (`q15p_V_o{tl5,tl6,s5}`).

## Original members are already valid-saturated (2026-11-15 ~09:40 UTC)

Fitted full-valid WP (all fitted, not held-out): original TL5/S5 .71106/.71107 → after V-ft .71101/.71082 (no gain;
corr .993, output std ×.82–.89). Our lineage: all-data TLead s7 .69652 → V .70442; all-data KD90 .69508 → V .70279;
train-only TLead .69142 → V .69867. Reading: the original tl3 members already carry a strong valid-fit stage, which
plausibly explains their public lead; V-ft reproduces that mechanism in our lineage. Re-fine-tuning the originals
only perturbs them, so `otl3_V.zip`/`otl3_V4.zip` are low priority. Cross-lineage correlation stays .986–.990.

New cross-lineage packages (fixed equal weights, no search):
- `orig3_vs7.zip`: original S5/TL5/TL6 .25 each + all-data V_s7 .25.
- `orig2_vall2.zip`: original S5/TL5 .25 each + all-data V_s7 .25 + all-data V_fkd .25.

## Public feedback and runtime correction (user snapshot 2026-11-15)

`tl3_w18swap.zip` .66805 (45m46s); `tl3_s5w18.zip` **SCORE:TIMEOUT at 61m57s** (4 members); kd90_s3_single .66193.
Server time is now ≈15.3 min/member, so every 4-member ZIP is a timeout risk; they are moved to
`submissions/final_20261115/four_member_timeout_risk/` (not for upload). 3-member replacements (TL3 slot weights):
- `o2_vs7.zip`: original S5 .34 + original TL5 .33 + all-data V_s7 .33 (sha256 de5e1641…).
- `o1_vall2.zip`: original S5 .34 + all-data V_s7 .33 + all-data V_fkd .33 (f4db5bf1…).
- `v3_ft.zip`: V_fbc .34 + V_tl5 .33 + V_tl6 .33, train-only bases + V (OOF twin .694065 vs TL3C .692340) (f22a46c1…).
