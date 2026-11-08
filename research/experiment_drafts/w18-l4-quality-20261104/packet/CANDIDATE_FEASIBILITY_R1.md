# Warm-start candidate feasibility amendment (post-history)

**Recorded:** 2026-11-04 10:52:05 UTC  
**Scope:** feasibility/source amendment after the blind R1 cards were frozen. This note does not alter or backdate `HYPOTHESES_BLIND_R1.json/.md`, and it contains no outcome from the proposed fits.

## Decision

Three one-factor arms are feasible from the same audited FBC initializer and same two-epoch W18 recipe, paired at seeds 1001 and 1002. Keep one common control and these three arms (eight total fits). For every arm, hold the chronological split-manifest hash, exact train membership, train-only teachers/cache and preprocessing, sequence resets, optimizer/sampler/schedule, strict FP32 settings, and checkpoint endpoint fixed. Reuse the per-seed FBC initializer used by the matched W18 control; do not use any full-corpus submitted weights as a validation parent. The archived W18 provenance records a train-only fit over 10,325 source-qualified training groups, `wmse`, chunk 2000, deterministic credit, loss norm `chunk`, cache variant `source`, cache scale 1, `distill-alpha=0.7`, `distill-family-weight=0.1`, and a two-epoch fixed schedule. Keep that exact eligible membership under the official split manifest; do not silently expand it to the 10,607 total groups. Its saved endpoint is epoch 1 (zero-based); root should freeze the exact same endpoint and per-seed parent identifiers for the new cohort before launching.

Run the pre-fit profile on the intended L4 wave and enforce the allocation cap; use FP32 with AMP and TF32 off and quantization disabled. The primary comparison remains the supplied whole-sequence chronological train/valid boundary. Validation is exposed development feedback, not independent evidence. Report the official clipped-label-magnitude-weighted pooled Pearson metric over all official valid scored rows, paired by seed, alongside the predeclared fixed 90/10 blend comparison. Do not tune blend weights on these outcomes.

## Candidate arms

### A. Append causal innovations

- **Change / exact argv delta:** add `--research-features causal_innovations` to the common W18 command; retain `--inherit-init-preprocessing`. The parent has `research_features=none`.
- **Mechanism hypothesis:** append the existing 80 causal moment/innovation channels to the 211-input representation. Slow/fast projected location and scale plus current innovation features may expose within-sequence drift or surprise that is hard to recover from the existing inputs. The child input projection is zero-expanded, so initialization preserves the parent function before training.
- **Predicted observables:** input width becomes 291; provenance records an append-only extension with zero-initialized new input weights and unit/zero normalization for added columns. The feature state resets per sequence and is chunk-carrying. If useful, prediction/residual changes should be concentrated where these channels vary, with positive paired full-valid or fixed-blend deltas.
- **Falsifier:** prior-membership audit includes any non-training group; parity/reset or chunk-carry checks fail; or both paired fixed-blend deltas are nonpositive and the frozen full-valid screen fails. This falsifies the proposed feature recipe, not causal statistics generally.
- **Prerequisite / cheapest test:** before fitting, verify the serialized prior's fit-member list and hash contain only eligible official train groups, and run the existing synthetic feature parity/reset checks. `fit_research_feature_prior` accepts an `extra_valid` argument, so explicitly verify it is empty for this train-only recipe; do not assume the helper itself enforces the split. Confirm the source's 80-column expansion and 291 total input width in setup provenance.
- **Likely failure / cost:** redundancy with existing event/normalization channels, train-prior mismatch on later sequences, or added input-construction cost. No extra trainable recurrent width; expected cost is prior fitting and 80 feature channels.
- **Fixed-ensemble route:** if standalone score is weaker but seed-paired residuals are complementary, evaluate only the predeclared 90% W18 control / 10% candidate blend. Keep only if the official full-valid blend endpoint improves under the already frozen screen.

### B. Reduce main-target teacher mixing

- **Change / exact argv delta:** `--distill-alpha 0.5` instead of the audited W18 value `0.7`; keep cache variant/source, cache scale 1, family weight 0.1, and all other settings fixed.
- **Mechanism hypothesis:** the blended target changes from 30% label / 70% teacher to 50% label / 50% teacher. More label anchoring may counter systematic teacher bias without changing architecture or output shape.
- **Predicted observables:** same model dimensions and training path; lower train-only teacher influence, with improved target-wise residual calibration if teacher error is material. No numerical gain is assumed.
- **Falsifier:** both paired fixed-blend deltas are nonpositive and the candidate fails the frozen full-valid screen; a teacher-label discrepancy audit can also show that the hypothesized bias direction is absent on train.
- **Prerequisite / cheapest test:** confirm the fixed source cache is the same eligible train-only cache for both seeds and verify the loss implementation uses `(1-alpha)*label + alpha*teacher`. This flag is not an inherited architecture field, so the warm initialization remains shape-compatible.
- **Likely failure / cost:** reducing teacher influence may discard useful denoising and increase target noise. Negligible compute or memory change.
- **Fixed-ensemble route:** the student may specialize toward the label-driven residuals that the higher-alpha W18 misses; test it only through the predeclared fixed 90/10 blend if standalone is weaker.

### C. Reduce family-head loss weight

- **Change / exact argv delta:** `--distill-family-weight 0.01` instead of `0.1`; leave `distill-alpha=0.7`, family cache, and other settings fixed.
- **Mechanism hypothesis:** reduce the auxiliary four-family weighted-MSE contribution tenfold while retaining the same positive-weight family head and output shape. This tests whether family-target pressure competes with the two-target main objective.
- **Predicted observables:** same head structure and endpoint interface; lower family-loss gradient contribution to the shared trunk. If the auxiliary signal is over-weighted, main-target pooled Pearson/residual calibration should improve while family fit may weaken.
- **Falsifier:** family gradients are already immaterial or the paired full-valid and fixed-blend deltas fail the frozen screen. Log the main/family loss and gradient contributions when available; do not infer competition from the scalar coefficient alone.
- **Prerequisite / cheapest test:** verify the unchanged family cache is train-only and confirm both weights are strictly positive. The model builder adds the family head for any positive family weight, so 0.01 is shape-compatible with the positive-weight W18 path.
- **Likely failure / cost:** the family task may be useful regularization; lowering it could remove signal or simply change optimization noise. No material runtime increase.
- **Fixed-ensemble route:** if weakening family supervision yields a weaker but more main-target-specific student, evaluate its fixed 90/10 blend and retain only on a paired positive full-valid blend result.

## Exclusions grounded in source and history

- **`gcorr_pool`** is not a valid cached-distillation arm: `setup_lineage.py` rejects cached distillation unless `--loss wmse`. Do not weaken that guard for this allocation.
- **Teacher-cache scale** cannot be changed while `distill-family-weight>0`; validation rejects that combination.
- **Short credit/update windows** enter the controlled macro-loss path, which rejects any positive family-head weight. Setting the weight to zero changes the family output shape relative to the W18 student and introduces a second mechanism, so this is not a clean warm-compatible arm.
- **`--aux-weight`** is inactive with the audited empty `--aux` selection. `aux_next` and `aux_struct` are active inherited shape fields and should not be changed in this matched cohort.
- **TCN** is excluded from warm-start candidates: historical heterogeneous screening recorded 0.683404 for TCN versus 0.688597/0.688761 for two GRU controls, and architecture mismatch prevents a same-function W18 warm start. Treat this only as a historical screen, not a controlled architecture conclusion.

## Source and evidence inspected

- `src/wnn/research_features.py`, `training/inheritance.py`, `training/setup_model.py`, `training/setup_preprocessing.py`, and `training/inputs.py`: causal innovations mode, fitting-only prior path, append-only compatibility, zero-extension behavior.
- `src/wnn/training/losses.py`, `setup_lineage.py`, `validation_inputs.py`, and `validation_training.py`: blend equation, family loss, cached-loss restriction, cache-scale incompatibility, and controlled-macro restriction.
- `baseline-evidence/runtime/launch-20261028/tl3-trainonly-20261028-r1/artifacts/runs/tl3c_w18final_distilled_s5/provenance.json`: historical W18 recipe and membership/split provenance.
- `baseline-evidence/runtime/heterogeneous-20261030-r1/terminal-summary-r2.json`: historical TCN/GRU screen.
- Frozen initial blind cards: `packet/HYPOTHESES_BLIND_R1.json` and `.md`.

The prior W18 and heterogeneous scores are exposed historical results; they do not establish the effect of any proposed arm.
