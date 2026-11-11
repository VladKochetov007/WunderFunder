A +0.003 gain has to cancel a common-mode error the current blend shares, or add a member with weighted correlation ρ ≲ 0.982 at the same WP. For two equal-strength forecasters the measured blend law gives ensemble − mean ≈ 0.168·(1−ρ), which clears +0.003 only at that ρ. Same-recipe seeds (ρ ≈ 0.995–0.998) top out near +0.001. Run the two zero-fit checks first.

### 1. Residual-kinematics census
**Family:** measurement.

**Mechanism:** Split saved out-of-fold predictions 50/50 into probe and confirm. On confirm, compute ΔWP for: (i) integer lag τ ∈ [−8, 8] between prediction and label; (ii) scored-row buckets 100–180, 180–400, and 400–end; (iii) three intra-segment maps — causal hold of the prediction at the last target-change row, causal expanding mean since that row, and non-causal broadcast of the end-of-segment prediction; (iv) an additive table fit on probe only, keyed by class × position quintile × target id × sign(p), applied to confirm; (v) per-class weight share s_c and the pooled drop when that class is removed; (vi) member ρ inside each slice.

**Hypothesis:** The leftover error sits in one slice large enough to be worth a model: observer settling, intra-segment jitter, or a shared additive bias.

**Ceiling:** the confirm-set ΔWP of each slice. No training. If the early-bucket weight times (WP_late − WP_early) ≥ 0.003, spend the next fits on initial state. If hold ≥ 0.002, ship hold. If the bin table ≥ 0.002, shared bias is real.

**Transfer:** a lag, a hold, or that table is a few dozen lines around the current blend in the ZIP.

**Test:** zero fits. **Falsifier:** every confirm |ΔWP| < 0.001. **Cost:** CPU on dumps you have.

### 2. Event-triggered observer, zero-order hold
**Family:** hybrid control (distant field).

**Correspondence:** Event-triggered control (Tabuada; Heemels): the plant runs on a fast clock, the controller updates only when a trigger fires, and a zero-order hold keeps the actuation between triggers. The book event is the trigger, the target is the held mark, and the 99 unscored rows are the observer settling window before the score clock starts.

**Hypothesis:** A per-row net spends depth re-estimating a constant mark. Stepping the cell only on events shortens backpropagation through time to the event count and matches the jump structure.

**Ceiling:** item 1(iii). Hold-Δ is a lower bound you can ship with the current weights. The end-broadcast gap is the extra room an event-local summary could still capture. Both near 0 means the ceiling is about 0.

**Transfer:** detect the trigger from columns that flip when train labels flip, run the cell, hold ŷ. Ship the hold on the current blend immediately. Retrain only if hold-Δ ≈ 0 and the end-broadcast gap ≥ 0.003.

**Test:** two fits, one fold, same GRU. (A) loss on event rows only, hold at inference. (B) per-row control, equal step budget.

**Falsifier:** WP(A) ≤ WP(B) − 0.002, or ρ(A, current teacher) > 0.995 with ΔWP < 0.001. **Cost:** two GPU fits; inference gets cheaper.

### 3. Koopman / bilinear defect on blend outputs
**Family:** dynamical systems.

**Mechanism:** Lift signals you already have online, ψ = (p0, p1, p0², p1², p0·p1, lags of p since the last event). Ridge-regress the clipped targets in the |y|-weighted inner product. Serve p + Bψ. This is a Carleman linearization of a state-dependent defect in the blend’s own trajectory.

**Hypothesis:** All three members share a bias that depends on predicted level and on t0/t1 disagreement. That bias lives in the ρ → 1 subspace, so another blended seed cannot see it. Pooled WP is sensitive to a t0↔t1 mix, so the cross term is inside the metric.

**Ceiling:** fit the ridge on probe, ΔWP on confirm. That is the ceiling of this observable set. If the quadratic block adds < 0.0005 on confirm, only a pooled affine is left, and a global scale cancels inside Pearson.

**Transfer:** one small matrix after the three forwards. **Test:** zero neural fits. **Falsifier:** confirm ΔWP < 0.001. **Cost:** minutes.

### 4. Frozen-trunk switching readouts
**Family:** Markov-switching regression (same structure as a speech acoustic model with an enrollment-dependent output transform).

**Mechanism:** One forward of the best GRU. Save penultimate h_t. Fit four |y|-weighted ridge heads and a multinomial gate on h_t. Training row-weights stay the metric weights.

**Hypothesis:** One linear head is a compromise across four regimes whose WP already spans ~0.80 to ~0.40. If h separates regimes, piecewise-linear heads lift the weak regime with the trunk frozen.

**Ceiling:** from that dump, oracle routing by true class is the upper bound; the gate, fit on probe and scored on confirm, is the attainable bound. Also run within-class weighted kNN in a PCA of h. kNN WP near the net’s WP means the residual is aleatoric in this representation.

**Transfer:** trunk plus four small matrices and a gate. One extra GEMM per row. **Test:** zero backbone fits. **Falsifier:** oracle-routed confirm ΔWP < 0.002, or weak-class kNN WP ≤ current class WP + 0.01. **Cost:** one val forward and a CPU ridge.

### 5. Residual-boosting student
**Family:** boosting.

**Mechanism:** Freeze the served blend. Train one causal GRU of hidden size 64 on the residual y − p_blend, same weighted MSE and clip, state reset per sequence. Line-search a scalar α on a fold the student never trained on. Serve p + αf.

**Hypothesis:** Distillation matches p_blend, which is why a student-slot swap moved public score by only ~0.0006. The residual is the complement. A small learner aimed at that complement is one boosting step; the line search sets α = 0 when the residual is unlearnable.

**Ceiling:** confirm Δ from 1(iv) and 3 is the linear floor. The held-out Δ of this fit is the decision.

**Transfer:** a fourth ZIP member, hidden 64, plus a scalar. Check the ~100 s budget for 64 sequences. **Test:** one fit, two seeds only if α > 0. Report WP(p+αf), α, and ρ(f, p_blend). **Falsifier:** best α gives ΔWP < 0.001 on the untouched fold. **Cost:** one small-GRU fit on a 5 GB GPU.

**Order:** do 1 and 3 on CPU, and dump h for 4 in the same forward. Pay for 2A/2B and 5 only on a slice whose confirm Δ cleared 0.002.
