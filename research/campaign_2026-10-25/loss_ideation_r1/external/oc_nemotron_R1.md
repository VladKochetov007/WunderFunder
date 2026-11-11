# Loss/Objective Ideation and Red Team Response

---

## Part A — IDEAS (8 mechanisms)

### 1. **Asymmetric Weighted MSE with Target-Dependent Clipping Thresholds**
**Change**: Replace global clip±2 with per-target adaptive clipping: clip `y` at quantiles `q_low, q_high` (e.g., 0.5%, 99.5%) computed on scored rows only; weight `w = |clip_q(y)|^γ` with `γ ∈ [0.5, 1.5]`.
**Why**: The 43-48% weight from high-|y| regime has WP≈0.43 vs 0.74-0.81 elsewhere. Current clip±2 wastes gradient on extreme outliers while under-weighting the informative high-|y| bulk. Adaptive clipping preserves Bayes alignment (weight ∝ |y|) but reduces variance from tail noise.
**Test**: 4 fits (GRU 256×2, F0, seeds 0/1) × {γ=0.5, 1.0, 1.5} + baseline γ=1.0 clip±2. Measure per-regime WP.
**Falsifier**: If per-regime WP on high-|y| doesn't improve ≥+0.005, discard.
**Risk**: Overfitting clip quantiles to fold; mitigate by fixing quantiles on full train scored rows.

---

### 2. **Propensity-Calibrated Focal Weighting**
**Change**: Weight `w = |y_clip| × propensity^α × (1 - p_correct)^β` where `p_correct = exp(-|p - y_clip|/τ)` with τ≈0.5, α∈[1,2], β∈[0.5,2].
**Why**: Propensity weighting helps (history), but uniform upweighting ignores that easy scored rows (high propensity, low error) contribute little gradient. Focal term down-weights well-predicted rows, concentrating updates on hard scored rows where variance reduction matters.
**Test**: 3 fits × {α=1.5, β=1.0} vs {α=2.0, β=0.5} vs baseline α=1, β=0. F0, seed 0.
**Falsifier**: If validation WP delta < +0.001 vs propensity-only (α=1, β=0), discard.
**Risk**: Focal term may amplify label noise on low-propensity rows; clip `p_correct` at 0.95.

---

### 3. **Two-Stage Target: Residual Distillation on High-|y| Regime**
**Change**: Stage 1: standard GRU teacher (current). Stage 2: train a small "residual head" (2-layer MLP on GRU hidden state) *only* on rows with `|y_clip| > q80`, target = `y - teacher_pred`, loss = WMSE with weight `|y_clip|`. Final prediction = `teacher + residual_head`.
**Why**: Oracle branch readout reaches 0.968 on high-|y| class — the teacher captures structure but misses regime-specific nonlinearity. A dedicated residual head isolates the hard regime without contaminating the bulk.
**Test**: 2 fits: (a) residual head trained jointly, (b) residual head fine-tuned after teacher freeze. F0, seed 0. Compare to teacher-only.
**Falsifier**: If high-|y| regime WP doesn't improve ≥+0.01, discard.
**Risk**: Extra parameters at serving; residual head is <50K params — negligible.

---

### 4. **Batch-Mass Normalization (EMA Denominator)**
**Change**: Replace per-batch `sum(w)` denominator with EMA: `denom_t = 0.99 × denom_{t-1} + 0.01 × sum(w_batch)`. Gradient: `∇ = sum(w * (p-y) * ∇p) / denom_t`.
**Why**: Per-batch weight mass varies 0.1×–9.7× (history), causing gradient scale instability. EMA stabilizes effective LR, especially for low-mass batches (few scored rows) that currently get noisy updates.
**Test**: 2 fits: EMA β=0.99 vs β=0.999 vs baseline. F0, seeds 0/1. Track grad norm stability and final WP.
**Falsifier**: If WP delta < +0.0005 or training diverges, discard.
**Risk**: EMA lag on distribution shift; warm up with first 100 batches' exact mean.

---

### 5. **Beta-NLL with Detached Variance Annealing**
**Change**: Gaussian NLL head: `loss = 0.5 * log(var) + 0.5 * (p-y)^2 / var`. Detach `var` for `β` epochs, then anneal `β: 1.0 → 0.0` over 10 epochs. Clip `var` at `[1e-4, 4.0]`.
**Why**: NLL clipped 100% of updates (mean clip 0.04) because variance head explodes early. Detaching variance lets mean head learn first; annealing re-introduces uncertainty weighting gradually. Beta-NLL interpolates between WMSE (β=0) and NLL (β=1).
**Test**: 3 fits: β-schedule {detach 5ep, anneal 10ep}, {detach 10ep, anneal 5ep}, {fixed β=0.5}. F0, seed 0.
**Falsifier**: If validation WP < baseline WMSE at any epoch, discard.
**Risk**: Extra head at serving; can distill mean-only student after.

---

### 6. **Scored-Row Mixup with Propensity-Aware Interpolation**
**Change**: During training, with probability 0.3, mix two scored rows from *same sequence*: `x_mix = λ x_i + (1-λ) x_j`, `y_mix = λ y_i + (1-λ) y_j`, `w_mix = λ w_i + (1-λ) w_j`, `λ ~ Beta(0.5, 0.5)`. Only apply when both rows scored.
**Why**: Scored rows are sparse (~propensity predictable). Mixup creates synthetic scored rows, increasing effective scored-row density and smoothing the loss landscape in the scored manifold. Preserves causality (same sequence).
**Test**: 2 fits: mixup prob 0.3 vs 0.15. F0, seed 0. Monitor scored-row effective count and WP.
**Falsifier**: If scored-row WP doesn't improve ≥+0.002, discard.
**Risk**: Mixing breaks temporal structure; restrict to same sequence and nearby indices (|i-j|<100).

---

### 7. **Unconventional: Causal Gradient Surgery — Orthogonalize High/Low-|y| Gradients**
**Change**: Compute per-sample gradients `g_i = ∇_θ L_i`. Project `g_high` (|y|>q80) onto orthogonal complement of `g_low` (|y|≤q80): `g_high_orth = g_high - (g_high·g_low/||g_low||^2) g_low`. Update with `g_low + g_high_orth`.
**Why**: High-|y| regime (43-48% weight, WP=0.43) and bulk regime (WP=0.74-0.81) likely have conflicting gradient directions. Orthogonalization prevents high-|y| updates from being cancelled by bulk, without increasing step size.
**Test**: 1 fit: gradient surgery vs baseline. F0, seed 0. Log cosine similarity between `g_high` and `g_low` per batch.
**Falsifier**: If high-|y| WP doesn't improve ≥+0.005 or bulk WP drops >0.002, discard.
**Risk**: Extra gradient computation (~2× backward pass); use first-order approximation (project only final layer gradients).

---

### 8. **High-|y| Class: Quantile Regression Head with Pinball Loss**
**Change**: Add a third head predicting `q_τ(y|x)` for τ∈{0.1, 0.5, 0.9} on high-|y| rows only (|y_clip|>q80). Loss = WMSE (main) + 0.1 × Pinball(τ=0.5) + 0.05 × [Pinball(0.9) + Pinball(0.1)]. At serving: use median head for high-|y|, mean head otherwise.
**Why**: High-|y| regime has asymmetric, heavy-tailed conditional distribution (WP=0.43 vs oracle 0.968). Pinball loss targets conditional quantiles directly, more robust than MSE for skewed targets. Median (τ=0.5) is Bayes-optimal for MAE, which correlates with WP under clipping.
**Test**: 2 fits: quantile head trained jointly vs fine-tuned. F0, seed 0. Compare high-|y| regime WP.
**Falsifier**: If high-|y| WP < baseline by >0.002, discard.
**Risk**: Three extra heads at serving; distill to single head after.

---

## Part B — RED TEAM

### (1) Why NLL, Pearson losses, and distillation variants failed
**Your explanation (objective consistent, gains variance-limited) is incomplete.**  
- **NLL failure**: Gradient clipping at 1.0 clips 100% of NLL updates (mean factor 0.04) because variance head explodes *before* mean head learns. This is an *optimization pathology*, not variance. The mean head never gets clean gradients. Fix: detach variance (beta-NLL) or use two-stage training.
- **Pearson loss failure**: Direct Pearson loss has `∇ ∝ (y-μ_y)/σ_y - ρ(p-μ_p)/σ_p` — the denominator `σ_p` couples all samples in a batch. With sparse scored rows (propensity ~0.01-0.1), batch `σ_p` is dominated by unscored noise, corrupting the gradient signal. WMSE with propensity weighting isolates scored rows.
- **Distillation variants null**: Difference-channel weighting assumes teacher errors are orthogonal to student capacity. But GRU teacher and student share architecture — errors are *correlated*, so reweighting difference channel just amplifies shared bias. Sum-channel-zero (-0.0016) confirms: removing teacher signal hurts.

**Conclusion**: Failures are *optimization and gradient pathology*, not fundamental variance limits. The objective is consistent *in population*, but finite-sample, sparse-scored, clipped-gradient regime breaks the path.

---

### (2) B300 Transformer overfit: what it tells us and redesign
**630M/1.54B Transformers: train WMSE 0.01 → val WMSE 1.55, val WP 0.16-0.41 (declining). GRU students: 0.68.**  
This is **catastrophic overfitting on causal sequence modeling**, not capacity. Key signals:
- Validation WP *declines every epoch* after epoch 15-20 → memorization, not learning.
- Train WMSE 0.01 is impossibly low for this noise level → fitting noise structure.
- GRU students distill *better* than teacher predicts → teacher learns spurious high-frequency patterns that don't generalize.

**Redesign the teacher program**:
1. **Architecture**: Use *recurrent* teachers (GRU/LSTM) with 512-1024 hidden, not Transformers. Recurrence matches causal streaming serving and inductive bias.
2. **Regularization**: Heavy weight decay (0.1-0.3), dropout 0.2-0.3 on recurrent connections, stochastic depth.
3. **Early stopping**: Monitor *validation WP*, not WMSE. Stop at peak WP (epoch 15-20 for 1.54B).
4. **Distillation target**: Use *teacher's hidden states* as auxiliary targets (representation distillation), not just predictions. This transfers inductive bias, not just outputs.
5. **Scale**: 200-500M recurrent params max. 1.54B is 10× over-parameterized for 10K sequences × 20K steps.

**Stop training Transformers on this data**. They are the wrong inductive bias for causal, streaming, sparse-scored sequences.

---

### (3) Most likely wrong assumption in the brief
**"Weighted MSE with weight |y| has exactly the Bayes-optimal minimizer for pooled WP."**  
**This assumes the conditional distribution `p(y|x)` is symmetric and the clipping threshold ±2 doesn't interact with the weighting.**  
But: targets are *clipped* at ±2 *before* WP computation, while the WMSE weight uses `|clip(y)|`. If `p(y|x)` has mass beyond ±2 (it does — financial returns), the Bayes-optimal predictor for *clipped* WP is **not** `E[|y|y|x]/E[|y||x]` — it's a nonlinear function of the tail. The current loss optimizes for unclipped WMSE, then clips predictions at serving. This mismatch explains the high-|y| regime gap (WP 0.43 vs oracle 0.968): the model predicts the *unclipped* mean, but the metric rewards *clipped* correlation.

**Fix**: Train with *clipped targets in the loss denominator too*: `loss = sum w*(p - clip(y))^2 / sum w` where `w = |clip(y)|`. This aligns loss and metric exactly. (Check: brief says "weighted MSE: loss = sum w*(p - y_clip)^2 / sum w" — if `y_clip` is already clipped, this *is* aligned. But if teacher targets use unclipped `y`, distillation breaks alignment.)

---

### (4) What I would stop doing
1. **Stop training Transformer teachers** — wrong inductive bias, overfits catastrophically, wastes B300 compute.
2. **Stop distillation-loss ablations** (difference-channel, Pearson-matching, run-level) — all null, shared architecture makes teacher/student errors correlated.
3. **Stop per-batch weight normalization** — 0.1×–9.7× mass variation destabilizes gradients; switch to EMA denominator (Idea 4).
4. **Stop using global clip±2 for loss weighting** — use adaptive per-target quantile clipping (Idea 1) to match the metric's effective support.
5. **Stop treating high-|y| regime as "just another regime"** — it carries 43-48% weight, has 0.43 WP vs 0.74-0.81 bulk, and oracle 0.968. It needs dedicated capacity (Ideas 3, 8).

---

*Total: ~1,750 words*
