# [Experiment] Leave-regime-out inner CV: select on shifted folds inside train, confirm on valid

1. **Status, lineage and owner.** Proposal; the audit is complete and the fold builder is not yet
   implemented. Owner: Vlad / Claude Code; any teammate can run phase 1. H-VALSHIFT29-R1;
   E-VALSHIFT29-CHAINS-R1, E-VALSHIFT29-ADV-R1. It replicates E-094/E-110/E-161 (Sep 26). It
   builds on the PR49/ADR0044 train-only split (`0c4de3a2…`).
2. **Why it should work.** The official valid, public and private sets are regime-shifted blocks,
   not random samples of train:
   - Adversarial validation separates valid from train with OOF AUC **0.984** (label shuffle
     0.499).
   - The shift is multivariate: the best single feature gives only AUC 0.70.
   - Only 90 of 9,193 recovered time links cross train↔valid, against ≈ 25% expected under a
     random split. They run in both directions (48/42), so valid is not simply a later period.
   - Random or component folds measure in-regime interpolation. Historically they overstated
     transfer: offsets were −0.006 to −0.010, and local blend gains did not transfer.

   Prediction: a recipe difference measured on held-out *regime clusters* inside train agrees in
   sign and rough size with the same difference on official valid. A random-fold delta agrees
   less well.
3. **New public evidence (Nov 11).** These are train-only ZIPs scored on the public LB:

   | ZIP | Valid | Public |
   |---|---|---|
   | TL3C blend | 0.692336 | 0.66355 |
   | TL3C single | 0.691434 | 0.66246 |

   - The ordering transferred (+0.0009 on valid, +0.0011 public).
   - Both are below the full-corpus TL3 (0.66832): the final train+valid refit matters more than
     selection gains of this size.
   - This gives a usable valid→public calibration pair, but only one pair.
4. **Frozen comparison.**
   - **Phase 0 (zero-fit, CPU).** Cluster the 10,325 fitting sequences into K = 5 regime groups.
     Features: the 84 sequence summaries already computed for adversarial validation (means and
     stds of 28 raw columns). Keep recovered chain components whole. Then report:
     - per-fold adversarial AUC (held-out cluster vs the rest), targeting a separability close
       to valid's 0.98;
     - per-fold market-class mix.
   - **Phase 1 (fits).** Pick two recipes with a known, large valid difference: the PR80 K16
     labels-only control and the PR49-style distilled student. Fit each on folds {1, 2} held out
     (2 recipes × 2 folds = **4 new fits**), at the PR80 contract (L4, K16, E6).
     - Compare each LRO delta with the valid delta: 0.6830 vs ≈ 0.691.
     - A random-fold control reuses existing component-fold evidence; no new fits.
5. **Fitting panel.** 4 fits in phase 1, no teacher fits (reuse PR49 teacher train predictions
   restricted to fitting folds). Only if it validates: phase 2 re-scores already-planned
   screens (for example PR #82) on LRO folds.
6. **Validation and uncertainty.** Official metric on each held-out cluster and on full valid.
   Two folds × one seed cannot estimate fold variance; the claim is sign and order agreement
   only. Valid stays the confirmation set: one look per frozen candidate, and every look logged.
7. **Leakage and coverage.** The clusters use input summaries only, no labels. The 282
   valid-connected train components stay out of fitting, as in ADR0044. Chain components are
   never split. The teacher for phase 1 must be refit without the held-out folds, or the
   distilled arm has to use labels only. This is a known cost: one teacher fit per fold, or
   phase 1 drops to labels-only recipes (K16 vs plain GRU, Δvalid +0.0011–0.0015).
8. **Decision-changing result.**
   - If LRO deltas match valid deltas in sign for both recipe pairs, the team gains several
     shifted selection folds and stops spending valid looks on screens.
   - If they disagree, regime clustering is not the right shift model. Keep valid as the single
     selection set and record the negative.
9. **How to run.**
   - Phase 0: `research/campaign_2026-10-29/validation_r1/adversarial_validation.py`
     regenerates the 84 sequence-summary features (npz 8.8 MB, not committed). A
     ~100-line clustering script and the fold manifest are the deliverable.
   - Phase 1: PR80 runtime with a fold-restricted roster.
10. **Evidence to return.** Fold manifest with SHA-256, per-fold AUC and class mix, four fit
    receipts, paired deltas, and the agreement table.
11. **Results and trace.** Audit evidence:
    [validation audit](../../campaign_2026-10-29/validation_r1/README.md) ·
    [robust validation protocol](../../unified/methods/ROBUST_VALIDATION_20261029_R1.md).
    The stress panel (E-110) stays a guard cell only (forensic W3: fitted to 3 LB points).
    Importance weighting and train filtering were refuted on Sep 26 (n_eff ≈ 1e-5). Nothing in
    phases 0–1 has run.
