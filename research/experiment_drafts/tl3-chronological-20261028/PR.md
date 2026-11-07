Historical TL3 was trained with validation ancestry, so it cannot be scored as an
independent incumbent on our adopted train/valid boundary. This draft restores its
predictor graph as fresh train-only fits and evaluates the fixed endpoint on all
official validation sequences.

1. **Status and lineage.** Owned by SemyonEpanov/tabdata. One fin256_s3 teacher
   completed all4 epochs; source/correctness/profile receipts are in this packet. This branch
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
   fin256_s3 is complete and immutable; the detached R8 finite controller has
   adopted its verified checkpoint, completed train prediction export and
   started fin256_s4. All4
   first-teacher train/val WMSE and WP points are connected to
   the existing local dashboard under tl3c_fin256_s3. The other16 native views
   consume trainer logs without repeating full-valid forward passes. All17
   configurations are visible; one fit is running and15 remain queued. R7's
   pre-update environment failures are preserved in the receiving packet. The
   SemyonEpanov watcher already posted first-teacher results and continues
   hourly/on verified fit completion. Terminal fixed-blend scoring is wired to
   the same dashboard as tl3c_final_fixed_blend without an invented train curve.
10. **Return evidence.** Source/config/data/ancestor identities, every epoch's
    objective/full-valid WP, LR/gradients/resources, resumable state and fixed
    checkpoint, prediction banks, streaming fixed-blend scores and raw negatives.
11. **Current result.** First teacher fixed ep3: full-valid WP .6882596674
    (t0 .6846225432, t1 .6918967916),12,880 completed updates, no skips/nonfinite
    gradients. Epoch scores: .6744168482 / .6852992218 / .6878730780 / .6882596674.
    Delta to the exposed plain-GRU benchmark is+.0104035442; recipes differ and
    this is one teacher, not the final TL3 result or public-transfer evidence.
    Native-Muon CUDA engineering profile:40 finite completed updates,37.11seq/s
    for H128. Initial receiving checks:26pass/2 native-reference skips on
    Torch2.8; finite-controller/recipe/monitor checks now38 receiving CPU passes.
    Read-only dashboard broker additionally passes11 focused CPU tests; actual
    API inspection confirms17 selectable TL3 entries without errors.
