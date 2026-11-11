# Impact-10 cards

Common to every card (see [README](README.md)): PR80 K16 recipe, seeds 203301/203302, two
treatment fits on L4, reused PR80 K16 controls, fixed-E6 canonical full-valid WP, paired
per-seed and per-target deltas, PR80 decision heuristic (mean ≥ +0.0005, both seeds positive).
Report every outcome. "Changed fields" lists the only intended difference from the control.

---

## 01. Muon on GRU recurrent matrices

**Mechanism.** Under AdamW the GRU update gates collapse early: 45–48% of update-gate
activations saturate and frozen units get memory timescales of ~5×10⁵ rows, already in
epoch 0. Muon orthogonalizes each gate block's update (Newton–Schulz) and keeps gates soft
(3% saturated, q99 timescale ≈ 170 rows). Removing only the orthogonalization returns the
score to AdamW level, so the gain is the orthogonalization itself.

**Evidence.** Our cold panel (H512×1, F0/F2): Muon vs AdamW 1e-3 +0.0038/+0.0043; vs AdamW at
its LR optimum (5e-4) +0.0026/+0.0035, all intervals above zero (E-OPT25-FAST-PANEL-R1).
Reproduced on Kaggle T4 in true FP32 with fresh seeds (+0.0036/+0.0033). It transferred only
+0.00026 to an already-warm distilled student, so the benefit is in *cold* training, which
is exactly the PR80 setting.

**Changed fields.** `optimizer: "muon"` applied to GRU `weight_hh_l*`/`weight_ih_l*` only,
per-gate blocks, momentum 0.95 Nesterov, 5 NS steps, update scaled to
`0.2·sqrt(max(rows, cols))·lr` (the same LR schedule, 3e-4 peak); stem, head and biases stay on
AdamW 3e-4. Port from `src/wnn/optim.py` (`Muon`, `muon_param_groups(scope="rnn")`); the
PR runtime's sequence Muon is Transformer-only and rejects GRU today.

**Zero-fit precheck (recommended).** Measure update-gate saturation (share of
z ∈ [0.02, 0.98] complement) on the K16 control's E6 checkpoint over 32 train sequences.
If it is ≤ 10%, the mechanism is absent and the prior drops sharply.

**Falsifier.** Both paired deltas ≤ 0, or gain with unchanged gate saturation (then the
mechanism is not the one claimed). Our cold evidence used LR 1e-3; if the result is null at
3e-4, a single LR 1e-3 follow-up is the informative next cell.

---

## 02. Distillation from a train-only TL3C teacher

**Mechanism.** Teacher predictions are a lower-variance target than the noisy labels; a GRU
student trained on a mix of labels and a stronger teacher learns smoother, better-calibrated
functions. This is the largest lever measured in this project (it is what separates the TL3
students from labels-only GRUs).

**Evidence.** Our warm child D1 (labels/teacher α 0.9, fbc1k teacher, train-only) reached
0.68997 vs its parent 0.68671 (+0.0033) and labels-only child 0.67562. The PR49 TL3C GRU
students are distilled the same way at α 0.5 and score 0.6914–0.6923 on full valid, all
train-only.

**Changed fields.** `objective_mode` with `kd_weight > 0` (teacher target weight 0.5, labels
0.5, as in PR49), teacher = PR49 `tl3c_fbc1k_s5` epoch-3 checkpoint (train-only, full-valid
0.691355; release `tabdata-pr49-checkpoints-20261029-r1`). One zero-fit export job writes the
teacher's causal predictions on the 10,325 fitting sequences; nothing from valid enters
fitting. Student architecture, schedule and epochs unchanged.

**Falsifier.** Both deltas ≤ +0.0005 would mean the cold K16 student cannot absorb this
teacher in six epochs; report teacher–student prediction correlation on valid alongside.

**Note.** This is not a "standalone labels" screen any more; ancestry gains the teacher
lineage. It answers whether the PR80 runtime should become the student stage of the team chain.

---

## 03. Since-event row weighting

**Mechanism.** Rows right after an i0 price event carry most of the learnable signal; rows
deep inside quiet stretches are dominated by noise. Down-weighting the main-loss row weight by
`max(floor, 1/(1 + since/τ))`, where `since` = rows since the last i0 price change (causal,
input-only), concentrates gradient on informative rows. The official metric is unchanged.

**Evidence.** E-351 (labels-only, two cells): since-weighting on vs off +0.0015/+0.0010; its
gain shrank under distillation (+0.0004/+0.0006), so the labels-only K16 screen is where it
should show. Part of our production parent recipe (τ 32, floor 0.25).

**Changed fields.** Loss row weights multiplied by `max(0.25, 1/(1 + since/32))`, computed
from inputs on the fly; normalization denominator unchanged otherwise. Port from
`--since-weight/--since-floor` in `src/wnn/training/cli_normalization.py`.

**Falsifier.** Both deltas ≤ 0. Report WP separately for rows with since < 32 and ≥ 32.

---

## 04. Auxiliary structure and next-change heads

**Mechanism.** Extra supervised heads on the shared GRU state force it to encode market
structure the main head under-uses: (a) log rows until the next target change (MSE) and its
sign (BCE); (b) cross-entropy heads for spread in ticks, branch and market class. Future
labels enter only the auxiliary loss, never an input or carried state.

**Evidence.** Part of our production parent recipe (`--aux-next 0.3 --aux-struct 0.3
--aux-weight 0.5`), never isolated against a labels-only K16 control. Distinct from PR81 card 07
(future *label values* at fixed offsets) and card 10 (clip events).

**Changed fields.** Add the two head families with total auxiliary weight 0.5 (0.3/0.3 inner
weights); the main WMSE head and its denominator are unchanged; heads are dropped at export.
Port from `src/wnn/training/cli_targets.py` and the matching objective components.

**Falsifier.** Both deltas ≤ 0, or gains only with auxiliary accuracy unchanged.

---

## 05. Longer gradient horizon at matched rows per update

**Mechanism.** With chunk 1000 the gradient never sees dependencies longer than 1,000 rows,
although sequences are 20,000 rows and our blend's largest advantage over other models is at
steps 12,000–20,000 (+0.0058) and 0–1,000 (+0.0065). A 4,000-row horizon lets the GRU learn
slower state; keeping rows per update fixed (8×4000 = 32×1000) keeps the update count
(≈ 6,460/epoch) and noise scale comparable.

**Evidence.** PR72 tested horizons for Transformers only; no GRU horizon screen on this
contract. Cost: same rows per update, longer sequential recurrence (slower wall time).

**Changed fields.** `batch_size: 8`, `chunk_rows: 4000`, `gradient_horizon_rows: 4000`.
Config-only: [configs/horizon4000_k16_s203301.json](configs/horizon4000_k16_s203301.json),
[configs/horizon4000_k16_s203302.json](configs/horizon4000_k16_s203302.json). Two fields move
together by design (rows/update matched); smaller batch is a confound named up front.

**Falsifier.** Both deltas ≤ 0 or gain confined to steps < 4,000. Report WP by step bucket
(0–1k, 1–3k, 3–7k, 7–12k, 12–20k).

---

## 06. Per-update weight EMA endpoint

**Mechanism.** An exponential moving average of the weights over ~1,000 updates averages out
the late-training SGD noise inside the final epoch. PR71 tested equal SWA over *epoch*
checkpoints E3–E6 (null), a much coarser average across very different points of the
cosine schedule; a per-update EMA averages only nearby iterates.

**Evidence.** EMA 0.999 is part of every PR49 TL3C student and of our parent recipe; its
isolated effect on this control is unmeasured.

**Changed fields.** Keep an FP32 EMA shadow (decay 0.999 per optimizer update, initialized at
the first update) and score the shadow weights at E6 as the treatment endpoint. The raw E6
weights of the same fits must reproduce the PR80 controls (determinism check, free).

**Falsifier.** Both deltas ≤ 0. If the raw E6 weights differ from the reused controls beyond
FP32 tolerance, the pairing is invalid and the fit's own raw endpoint is the control.

---

## 07. Causal per-sequence regime channels

**Mechanism.** Valid is a regime-shifted population: an adversarial classifier separates it
from train with OOF AUC 0.984, driven by per-sequence levels/scales (volume-code means, a-column
standard deviations). Global train normalization leaves the GRU to infer the regime implicitly.
Giving it causal running statistics of its own sequence lets it condition on the regime
explicitly and should transfer better across regimes.

**Evidence.** E-094/E-161 and the October 29 audit (adversarial AUC 0.984, max univariate AUC
0.70: the shift is multivariate); no model has been given explicit per-sequence context.

**Changed fields.** Append 2×112 inputs: causal EMA mean and log-EMA-std of each normalized
input (half-life 1,000 rows, initialized at the train-global moments, updated with x[t] before
the prediction at t). Input projection widens 112→336; everything else unchanged.

**Falsifier.** Both deltas ≤ 0. Also report the stress-panel guard cell (least-train-like two
thirds of valid by adversarial score); a gain concentrated there supports the mechanism.

---

## 08. Per-sequence input scale augmentation

**Mechanism.** If the regime shift is mostly level/scale drift of volume-type inputs, training
with random per-sequence rescaling of those inputs teaches invariance to it. The falsifier is
built in: if labels truly depend on absolute volume level, this will hurt.

**Evidence.** Shift drivers are scale/level features (E-094 and the October 29 driver
analysis). Our input-view masks (October 29) were a different, failed augmentation (dropping
columns, not rescaling).

**Changed fields.** During fitting only, multiply all `i0_v*`, `i1_v*`, `i0_dv*`, `i1_dv*`
raw columns of each training sequence by one factor `exp(N(0, 0.3²))` drawn per sequence per
epoch from an independent checkpointed RNG, before the fixed train normalizer. No change at
evaluation.

**Falsifier.** Both deltas ≤ 0. Report the stress-panel guard cell as in card 07.

---

## 09. Causal class-routed readout

**Mechanism.** The four market classes behave very differently (class 0–2 WP 0.74–0.81, class
3 0.44 with 48% of the metric weight). A single linear head shares one readout across all four.
Routing the shared GRU state to a per-class linear head lets class 3 get its own readout without
up-weighting (which failed for us) or separate models.

**Evidence.** Class 3 dominates the residual error budget; class up-weighting and class-3
feature work were negative, but class-conditioned readout was not tested.

**Changed fields.** Four linear heads, selected by the sequence's market class; the class must
be computed causally from inputs already seen (the class is an instrument-level property; use
the repo's causal class rule and verify ≥ 99.9% agreement with the cached labels by step 99 on
train, else the card stops before fitting). Also feed the one-hot class as 4 extra inputs.

**Falsifier.** Both deltas ≤ 0, or gains only on classes 0–2. Report per-class WP.

---

## 10. Seed-ensemble self-distillation (born-again)

**Mechanism.** The mean of two seeds is +0.0011 better than a single seed for this recipe
family. Training a fresh student on the mean of two existing K16 controls' predictions
(mixed 0.5/0.5 with labels) transfers part of that ensemble gain into one network, at zero
teacher-fitting cost. Different from card 02: same architecture and capacity, no stronger
teacher lineage.

**Evidence.** Seed-mean gain +0.0011 (October 29 TabM ceiling analysis); born-again networks
(Furlanello et al., 2018) report student ≥ teacher for same-size students.

**Changed fields.** `kd_weight` 0.5 with teacher = mean of PR80 `k16_s203301` and
`k16_s203302` E6 predictions on the 10,325 fitting sequences (one zero-fit export job).
Seeds 203301/203302 as usual; note that each student shares its init with one of its teachers.

**Falsifier.** Student ≤ its same-seed control on both seeds. Report the student against the
two-seed mean (0.6830/0.6826 singles; the two-seed blend score should be computed once).
