# Weak-model complementarity: direction map and allocation

This note preserves the fresh-context proposals and the independent tree-mapping review. It does not promote a literature analogy into task evidence, add numerical priors, or authorize fits.

## Where the four proposals belong

1. **Ambiguity-ranked training allocation** maps to H-204, not a new arm. It uses group-out-of-fold ensemble disagreement to allocate a fixed training budget across already labeled examples. This is different from H-207's inference-time output gate. Its first valid screen needs ancestry-clean train-group predictions, frozen sampling probabilities and importance correction, and equal-compute uniform and matched-random controls. E-515's exposed validation association cannot define a training schedule.
2. **Cross-fitted latent residual repair** becomes H-208, a representation-specific child of H-203. It asks whether an independently trained candidate's causal hidden state predicts anchor residuals beyond an anchor-only representation. Candidate fitting, correction fitting and evaluation groups must be separate. The final served endpoint must beat the unchanged anchor and matched input-only/shuffled-representation controls under `global_wp_clip2_v1`.
3. **Causal memory-reset event expert** becomes H-209 under the existing H-183 stationarity/state-clock lineage. It changes recurrent memory rather than gating an output. An offline oracle reset is only an upper-bound diagnostic; it cannot train or select the inference trigger. The actual trigger must use legal causal inputs and beat no-reset and rate-matched sham controls on groups excluded from fitting.
4. **Degree-bounded interaction sidecar** becomes H-210 under H-182's stagewise-interaction family. It feeds a fixed low-degree basis into a recurrent student, unlike H-182's pair-seeded feature selection and linear head. Freeze the basis on fitting groups and compare with matched-parameter input-only and random-basis students.

The random library selection supplied a capacity-control prompt, not evidence that polynomial features help this task. Krogh and Vedelsby discuss ensemble ambiguity under squared error; applying it to labeled-example allocation is a new hypothesis. Neither source supports a predicted gain on the official metric.

## Allocation

The tree keeps recovery of our independently trained baseline/raw-teacher path and the heterogeneous-teacher/student direction ahead of these proposals. H-208 is the most direct next complementary-model descendant once a new candidate has ancestry-clean hidden states. H-209 and H-210 remain lower-cost-to-ideate but higher-cost-to-validate branches with no launch authority. H-204 needs clean train-group OOF disagreements before any student fit.

E-514's fixed pool and E-516's simplex rule stay closed. E-517 has run once on that same exposed pool; its numerical result awaits independent evidence review. No further analysis or tuning on those same groups is authorized by this note. Even a passing E-517 screen would only nominate a materially new, ancestry-audited cohort and genuinely unexposed confirmation. A failure closes that exact rowwise gate, not the broader weak-model research family.

Previously inspected folds remain exposed. Use paired chain/group resampling that recomputes the pooled official endpoint; do not interpret per-row weighted-MSE tails or pairwise correlation as score contributions. Keep expected public gain and all unmeasured numerical priors null. After 10:00 Europe/Kyiv on 2026-10-07, new experiment compute is Kaggle-only under the existing quota and packet controls.

## Provenance

- Proposal cards and literature/library source receipts: [SCOUT_PROPOSALS.md](SCOUT_PROPOSALS.md), [SOURCE.json](SOURCE.json), [PROMPT.md](PROMPT.md).
- Independent mapping: gpt-6-luna max, card-by-card mapping against H-203/H-204/H-205/H-182/H-183; no prediction arrays were inspected and no fits were run.
- E-517 preregistration and current state: `../disagreement_guided_H204/GATED_BLEND_PREREGISTRATION_E517.md`, `../disagreement_guided_H204/E517_POST_RUN_EXECUTION_RECEIPT_20261007.json`.
