# Distillation ceiling and plan (2026-10-11 05:50)

## Ensemble ceilings (fold 1, equal-weight blends of banked teachers, `scripts/blend_score_remote.py`, Vast B host)
| Blend | WP |
|---|---|
| muonall_h384 s3 / s4 / s5 singles | .67021 / .67034 / .67150 |
| 2 muonall_h384 (s3+s4) | .67161 |
| **3 muonall_h384 (s3+s4+s5)** | **.67254** |
| 3 muonall + 2 prop0 | .67206 |
| 3 muonall + 2 ctrl h256 | .67137 |
| 3 muonall + gru384 mu3 + ema | .67207 |
| 3 muonall + 6 other diverse teachers (9 total) | .67175 |
| 2 ctrl h256 / 2 prop0 h256 | .66761 / .66872 |
Reading: same-recipe seed ensemble of the BEST recipe is the best blend (+0.0019 over mean single); adding weaker diverse members dilutes. A student that fully absorbed the 3-muonall ensemble would land ~0.6725 (vs 0.67069 mean single teacher, vs best student so far 0.66938).

## What the current student pipeline really is (distill_pipeline.py, read in full)
Warm start from combo parent (h256, ~0.666), 2 epochs (NOT 3) AdamW 1e-4, batch 16, chunk 2000, onecycle; target = 0.8125 * mean of teacher outputs on the train split cache ("homogeneous" strong family; the other 3 cache families are aliases of the same teachers); alpha 0.7. So dsq_none (0.66861) vs dsq_t256 (0.66938) vs parent ~0.666: most of the student gain (+0.0027) is plain extra low-LR training of the parent; teacher signal adds +0.0008. Teachers used were weak h256/h384 plain ones (~0.6660-0.6685) — about as good as the parent, so little to teach. Using the stronger muonall_h384 3-seed ensemble (0.6725) as teacher has not been tried.

## Queued arms (w15_*, Vast B, front of queue; pipeline now has --epochs/--lr/--wd/--hidden/--batch/--sched, defaults unchanged)
dm_t3 (3 muonall teachers, default recipe) ; dm_t3_e6 (6 epochs) ; dm_t3_e6_lr2 (6 ep lr 2e-4) ; dm_none_e6 (no teacher, 6 ep: separates epoch effect) ; dm_t3_a9_e6 (alpha .9) ; dm_h384_t3 (h384 student from muonall h384 parent: upper bound of absorption, serving cost 2x).
Decision rules: student gain over dm_none_e6 = teacher value; gap to .6725 = remaining distillation headroom.
Literature + external ideation: LITERATURE.md, ideation/ (pending).

## Addendum 06:25: weaker partners hurt a strong member (fold 1, equal-weight pairs with muonall_h384 s3 = .67021)
+ a_dualtower (single .66521): .66978 (-0.0004) ; + ctrl_s3 h256 (.66559): .66975 (-0.0005) ; + prop0_e3 (.66818?~.6678): .67081 (+0.0006) ; 3 muonall + dualtower: .67197 (vs 3 muonall .67254, -0.0006).
=> Equal-weight blending only helps when members are within ~0.001-0.002 of each other. The "+0.003 diverse pair" of earlier rounds was between equally-strong (~.666) members. For submission and for teacher sets: use the strongest recipe's seeds; add weaker diversity only with learned/lower weights.
