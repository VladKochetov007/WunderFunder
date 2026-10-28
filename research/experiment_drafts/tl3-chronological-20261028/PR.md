Historical TL3 was trained with validation ancestry, so it cannot be scored as an
independent incumbent on our adopted train/valid boundary. This draft restores its
predictor graph as fresh train-only fits and evaluates the fixed endpoint on all
official validation sequences.

1. **Status and lineage.** Owned by SemyonEpanov/tabdata. One fin256_s3 teacher is
   running; source/correctness/profile receipts are in this packet. This branch
   stacks on PR46; its seven GALAM fits continue unchanged. Full TL3 is not ready.
2. **Mechanism.** Restore the original three prediction roles, diverse full/raw
   GRU teachers, fixed-epoch fine-tuning and mixed-target/four-family distillation.
   Complementary errors may improve the existing clean GRU; this remains a hypothesis.
3. **Compute value.**17 predictor fits /54 configured epochs /257,740 chunk
   updates, with shared ancestors counted once. No propensity fit. Start on free
   L4 GPU0, then verified released cards. No new rental or billing.
4. **Frozen recipe.** Canonical modular trainer, causal event211/tiedlead263,
   original seeds, widths, batches, chunks, heads and fixed epoch endpoints.
   FP32 and neutral propensity replace historical FP16/valid-fitted propensity.
   Valid-only fine-tunes become train-only; fin256's unrecorded optimizer is
   explicitly fresh AdamW. Known Muon x3/x6 semantics are restored separately.
5. **Bounded panel.** This user-authorized reconstruction graph supersedes the
   generic4–6-fit screen only for its17 listed predictors. Per-fit six-hour bounds;
   graph admission closes at17 fits,168 wall hours,256 GPU-hours or12GiB disk
   reserve. No blind retry or hidden recipe shortening.
6. **Validation.**10,325 fitting train sequences /0 fitting valid; all1,873 valid
   sequences /3,528,273 scored rows. Fixed .34/.33/.33 blend. Primary result is
   pooled global_wp_clip2_v1; exposed single-seed benchmark GRU is .6778561232.
   No untouched confirmation, significance or public/private improvement claim.
7. **Leakage.** Whole sequences and feature-connected exclusions; no seq_ix
   chronology. Fresh ancestors, normalization and targets exclude valid. Teacher
   checkpoint epochs are fixed before observing valid. No valid-derived weighting.
8. **Decision.** A complete fixed endpoint exceeding the clean benchmark by .001,
   or a useful separately frozen ensemble contribution, supports further work;
   lower/uncertain results remain evidence about this adapted recipe, not all KD.
9. **Execution.** See README, frozen recipe module and receiving receipts. First
   fin256_s3 is detached and immutable; remaining finite-DAG admission and the
   read-only dashboard bridge are being completed. Never restart the first fit.
10. **Return evidence.** Source/config/data/ancestor identities, every epoch's
    objective/full-valid WP, LR/gradients/resources, resumable state and fixed
    checkpoint, prediction banks, streaming fixed-blend scores and raw negatives.
11. **Current result.** First teacher epoch1: full-valid WP .6744168482
    (t0 .6702838726, t1 .6785498239),3,220 completed updates, no skips/nonfinite
    gradients; fit588.1s + valid226.3s. This is an intermediate teacher, not TL3.
    CPU receiving checks:26 pass/2 native-Muon-reference skips on Torch2.8.
    An actual CUDA Muon engineering profile is pending. No final model promotion.
