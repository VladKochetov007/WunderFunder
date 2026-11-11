# [Results] Implicit ensembles and diversity augmentations do not beat seeds: TabM-GRU, NorMuon, view masks, blend-value law

1. **Status, lineage and owner.** Completed negative/zero-fit evidence. No new fits in this PR.
   Owner: Vlad / Claude Code. H-TABM29-GRU-R1, H-NORMUON29-R1, E-TABM29-CEILING-R1, view-mask
   screen. Source branch `experiment/next-cycle-recovery-20261021` at `0ea63b982`
   (`src/wnn/tabm_gru.py`, `--ensemble-k/--ensemble-init`, `--muon-update normuon`,
   `--view-mask-p`, `tests/test_tabm_gru.py`).
2. **Purpose.** Several team PRs now test regularization, ensembles and diversity (PR70, PR71,
   PR80 dropout, PR81 card 08). This PR publishes our closed screens so they are not repeated,
   and publishes the zero-fit law that predicts blend value from correlation before any fit.
3. **Blend-value law (zero-fit).** Across all 28 pairs of eight cold-panel models per fold:
   equal-weight pair WP − mean single WP ≈ **0.168·(1 − ρ)**, where ρ is the |y|-weighted
   prediction correlation. For a k-member equal ensemble: gain over the mean member ≈
   0.336·(1 − 1/k)·(1 − ρ̄).
   - Break-even: a candidate member helps only if its deficit to the best member is below
     ≈ 2·0.168·(1 − ρ).
   - At ρ ≈ 0.99 that is only ≈ 0.003. Only seeds and small variants sit below break-even.
   - The law also predicted the Nov 11 external model check: deficit 0.0045 vs break-even
     0.0044, measured gain +0.0001.
4. **Ceiling.** The mean of two Muon seeds is +0.0011 over the mean single. Four siblings give
   +0.0015/+0.0013 over the best single. Any implicit ensemble of this recipe is bounded near
   +0.002.
5. **Fits.** TabM 3 local + Kaggle cells, NorMuon cells, view masks 4 Kaggle T4 fits plus 1 local
   (cold Muon recipe, F0/F2), 12 new fits in all, with reused R3 controls.
6. **Results** (exposed folds, one seed per cell, paired component-bootstrap intervals):
   - TabM-mini GRU k=4 vs single Muon: −0.0012 (F0). Members are diverse (ρ 0.991) but weak
     (−0.0033 each). `screen_failed`.
   - NorMuon vs Muon: no gain. `screen_failed`.
   - Column view masks p 0.15/0.30: standalone −0.0014 [−0.0039, +0.0005] at p 0.15 F0. ρ to
     control 0.994. Pair − control +0.00025, below the declared +0.0010 bar. `screen_failed`.
   - Weight averaging over epochs 3–5 on the stage-6 student: +0.00004/+0.00002 (null). This is
     consistent with PR71's SWA null.
   - Quadratic output defect hypothesis: closed by the reframe review.
7. **Leakage and coverage.** Train-only augmentation, unchanged serving, fold-clean memberships.
   Exposed folds; nulls apply to these recipes, not every possible variant.
8. **Decision.** Use seed averaging and fixed equal-weight blends for ensembling. Before any
   diversity fit, compute ρ to the current members and apply the break-even rule. Spend
   diversity compute only where the predicted deficit is below break-even.
9. **Triage method.** [Idea-triage reflection](../../unified/methods/IDEA_TRIAGE_REFLECTION_20261029_R1.md)
   ranks ideas by ceiling × mechanism presence × transfer multiplier × detectability before
   spending fits.
10. **Evidence.** [TabM/NorMuon](../../campaign_2026-10-29/tabm_gru_r1/README.md) ·
    [view masks](../../campaign_2026-10-29/view_mask_r1/README.md) ·
    [reframe (Opus + Grok)](../../campaign_2026-10-29/reframe_r1/) ·
    [campaign](../../campaign_2026-10-29/README.md).
11. **Trace.** Copied unchanged from `0ea63b982`. No independent review of this PR.
