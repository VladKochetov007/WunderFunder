# [Experiment] Whole-sequence channel dropout on B300

1. Status: implementation and prospective command only; no local tests, fits,
   inference, profiles or scores. Source-qualified data/splits and B300 remain
   receiver prerequisites. Target is `research-team`.
2. Mechanism: masking a subset of raw input channels throughout each training
   sequence may make the student rely on complementary signals and reduce
   overfitting; an alternative is that corruption destroys predictive information.
   Compare dropout 0 vs fixed 0.05, all else equal. This does not claim
   label-invariance. Related input augmentation was previously suggested and S38
   has a negative symmetry-based result; neither establishes this intervention.
3. Compute: four independent one-GPU fits; spending buys different masks/seeds
   and full fits, not a proven gain. Inference has no dropout cost.
4. Frozen comparison: raw-only GRU256x2, FP32 AdamW .001, batch32, chunk2000,
   six epochs, same order, source, normalization, label loss, and fixed endpoint.
   Original raw input controls loss weights; dropout uses fit-only means.
5. Panel: 2 controls + 2 treatments, F0/1001 and F2/1002, four new fits; zero
   assumed reused controls/teachers, profiles and grading extra.
6. Validation: official global pooled WP on full scored rows and both targets;
   report every paired delta and failure. Two confounded fold/seed cells cannot
   establish significance; fixed-model component intervals omit fit uncertainty.
7. Leakage: frozen component-disjoint membership, fitting-only means, no future
   inputs/labels, sequence reset; inspect actual component aliases and exposure.
   Dropout is only in training; anonymous component IDs do not prove chronology.
8. Decision: +0.0005 WP in paired scores is a provisional follow-up trigger;
   adverse or inconsistent results narrow this exact rate/recipe, not all
   regularization. An interval crossing zero does not by itself close the idea.
9. Run: see README; receiver must bind actual manifests/data and review numerical
   correctness, causal masks, profile memory and GPU environment before long fits.
10. Evidence: source/config/split/input hashes, actual device, losses, LR,
    preclip gradients, throughput/memory, checkpoints and held-out predictions;
    separately bundle evidence and weights. Full CPU serving check is separate.
11. Results: none. No score gain, statistical significance, ready submission or
    first-place result is claimed.
