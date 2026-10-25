My leading hypothesis is that **Muon helps by changing the geometry of recurrent updates**, with a possible contribution from their effective size. Head-driven clipping is a plausible interaction, but its frequency does not establish that it suppresses recurrent learning. The best next bets are separating recurrent clipping from the head and extending Muon to the remaining matrices.

These are hypotheses based only on the brief. Another ≈0.004 WP improvement is a reasonable experimental target, not a prediction.

**A. What the measurements establish—and what they do not**

For an ideal polar update, with momentum matrix \(N=U\Sigma V^\top\),

\[
Q(N)=UV^\top,\qquad
\Delta W=-\eta\,0.2\sqrt{\max(m,n)}\,Q(N).
\]

This equalizes nonzero singular values of the **update**. Your finite Newton–Schulz implementation approximates this operation.

The established mathematical consequences are:

- Update spectral norm is controlled; weight spectral norm is not explicitly constrained.
- \(Q(cN)=Q(N)\) for positive \(c\).
- For full-rank blocks, the proposed update has RMS \(0.2\eta\), irrespective of gradient magnitude.

My conjecture is that AdamW repeatedly reinforces a few correlated recurrent directions, especially in the update gate, while Muon distributes movement across more directions. The enormous stable-rank difference supports this interpretation. It does **not** establish whether the benefit comes from improved learning, regularization, or injected movement along noisy directions.

Facts 3–4 require caution:

- Constant gradient scaling approximately cancels in Adam’s \(m/\sqrt v\), too. A mean clip factor of 0.6 does **not** imply a 40% reduction in its effective learning rate.
- Time-varying clipping matters because it reweights gradient history before momentum and second-moment estimation. Polarization does not undo that history.
- Weak coordinate-level signal does not exclude coherent matrix-level signal.
- GRU dynamics depend on \(J_t=\partial h_t/\partial h_{t-1}\), including gates, biases, and states. Small \(\|W_{hz}\|_2\) alone does not establish better dynamics.

**First experiment: four fits separating direction from size.**

On each trajectory, maintain shadow Adam and Muon states using the same gradients. Let \(A_{t,b}\) and \(M_{t,b}\) be their proposed gradient-driven parameter increments for recurrent block \(b\), including their configured learning rates.

Run the four combinations

\[
\Delta_{d,s}
=\|s_{t,b}\|_F
\frac{d_{t,b}}{\|d_{t,b}\|_F},
\qquad d,s\in\{A,M\}.
\]

Skip zero directions. Keep nonrecurrent updates, data order, clipping rule, and decay multiplier identical. Use the existing FP32 Newton–Schulz routine.

- If Adam direction at Muon size closes the gap, effective step size explains much of the gain.
- If Muon direction wins at both sizes, update geometry matters.
- An interaction means neither explanation alone suffices.

Log actual increments, including decay separately. At early, middle, and final checkpoints, measure gate saturation and directional amplification through products of 32 and 128 recurrent Jacobians on fixed validation sequences. These are diagnostics; the clipping and spectral interventions below provide stronger causal tests.

**B. Five concrete next steps**

All rules below preserve the architecture, inference cost, and FP32 policy. Each is a separate small experiment, not a proposed five-way sweep.

1. **Separate recurrent clipping from the head.**

   For each recurrent gate block \(b\), replace the global multiplier with

   \[
   \widetilde G_b
   =G_b\min\left(1,\frac{\tau_b}{\|G_b\|_F+\epsilon}\right).
   \]

   Set \(\tau_b\) to that block’s pre-clip norm q90 from the first 100 steps of a matched Muon baseline; then freeze it. Feed \(\widetilde G_b\) into the existing momentum and orthogonalization. Leave nonrecurrent updates unchanged.

   **Why:** Global clipping currently lets head activity determine which recurrent gradients receive weight in momentum history. Local clipping retains protection against rare large gradients while removing that coupling.

   **Cheapest test:** Two matched fits: existing Muon versus this change. Check recurrent update norms remain comparable and log update-direction cosine against shadow globally clipped momentum.

   **Falsifier:** If directions barely change, head-driven clipping is mechanistically unimportant. If directions change substantially without WP improvement across both cells, removing that coupling is not useful.

   **Expected direction:** Positive if head-driven fluctuations obscure recurrent signal; negative if global clipping usefully suppresses high-weight outliers. High-priority test, uncertain magnitude.

2. **Extend Muon to the input and both head matrices, with matched update norms.**

   For each newly included matrix, compute its shadow Adam increment \(A_t\) and Muon direction \(Q(N_t)\), then apply

   \[
   \Delta W_t
   =-\|A_t\|_F
   \frac{Q(N_t)}{\|Q(N_t)\|_F}.
   \]

   Keep biases on AdamW and preserve existing recurrent updates and decay. Here \(A_t\)’s norm is positive; \(Q(N_t)\) points along the gradient.

   **Why:** “Muon all” is the strongest historical positive lead. The input and head could also benefit from matrix-level conditioning, and the head dominates current gradients. Matching norms prevents a head learning-rate change from masquerading as a geometric benefit.

   **Cheapest test:** Baseline versus this extension. With four fits, separate input-only from head-only and combined.

   **Falsifier:** No repeatable WP gain at matched norms, or deterioration concentrated in the harder target. The two-row output matrix deserves particular scrutiny: equalizing its two singular directions can amplify the noisier target.

   **Expected direction:** Positive prior from historical runs, with substantial uncertainty from the precision change and output geometry.

3. **Test periodic spectral projection of the update-gate recurrent matrix.**

   Every 50 updates, after the normal optimizer step, compute

   \[
   \widetilde W_{hz}=U\operatorname{diag}(\sigma_i)V^\top,\qquad
   W_{hz}\leftarrow
   U\operatorname{diag}(\min(\sigma_i,\tau))V^\top.
   \]

   In PyTorch’s gate order, this is rows \(H:2H\) of \(W_{hh}\). This imposes a cap at projection times; overshoot between them must be measured.

   **Why:** The update gate has the clearest spectral pathology. This intervention tests whether preventing its concentration helps without requiring every recurrent matrix to be orthogonal.

   **Cheapest test:** AdamW versus projected AdamW with \(\tau=3.2\), approximately Muon’s observed endpoint scale. If that recovers much of the gap, test Muon versus projected Muon with \(\tau=2.5\).

   **Falsifier:** Large, verified spectral changes without improved WP undermine weight-spectrum control as the main explanation. Worse WP despite less saturation would show that those dynamics diagnostics are insufficient.

   **Expected direction:** Potentially positive for AdamW; small or negative for Muon. This is primarily a mechanism test. Record projection displacement: it is an additional update.

4. **Try norm-matched Shampoo on recurrent gate blocks.**

   Using the same clipped gradients \(G_t\) as the control, maintain

   \[
   L_t=.99L_{t-1}+.01G_tG_t^\top/n,\qquad
   R_t=.99R_{t-1}+.01G_t^\top G_t/m.
   \]

   Bias-correct these estimates. Recompute inverse fourth roots at step 1 and every 50 steps, flooring eigenvalues at \(10^{-3}\) times the matrix’s mean eigenvalue plus \(10^{-12}\). With the same Nesterov momentum \(N_t\),

   \[
   P_t=L_t^{-1/4}N_tR_t^{-1/4},\qquad
   \Delta W_t=-\|M_t\|_F\,P_t/\|P_t\|_F,
   \]

   where \(\|M_t\|_F\) is the shadow Muon increment norm.

   **Why:** Coordinate-wise adaptation misses correlations; complete singular-value equalization may over-amplify weak directions. Accumulated left/right second moments offer an intermediate geometry. This is a conjectured advantage, not an implication of the reported spectra.

   **Cheapest test:** Two matched fits, recurrent blocks only. Time the eigendecompositions first.

   **Falsifier:** No gain over norm-matched Muon, or enough overhead to erase the improvement at equal wall time.

   **Expected direction:** Uncertain, but potentially substantial if persistent matrix correlations carry signal that per-step orthogonalization handles poorly.

5. **Increase independent sequence averaging while controlling optimizer time scales.**

   Accumulate two microbatches from distinct sequences, retaining 2,000-row TBPTT windows. For equal-sized microbatches, use

   \[
   G=\frac{G_1+G_2}{2}.
   \]

   First verify the loss reduction. To estimate the population weighted-MSE objective without batch-dependent normalization, use

   \[
   \ell_B=\frac{\sum_{i\in B}|y_i|(p_i-y_i)^2}
                 {|B|\,\overline{|y|}_{\mathrm{train}}},
   \]

   with the same fixed denominator convention in every comparison arm.

   Advance schedules by examples seen. Set each EMA coefficient to \(\beta_i'=\beta_i^2\), or the product of the two scheduled coefficients. Match decay over those two original steps.

   **Why:** Weight mass varies almost 100-fold, and within-label-run errors are highly correlated. More rows from one run may add much less information than more independent sequences.

   **Cheapest test:** Three fits: baseline, accumulation ×2 at \(\eta\), and accumulation ×2 at \(2\eta\), all seeing equal data. The two rates distinguish variance reduction from merely reducing cumulative movement; neither is guaranteed to preserve normalized-update dynamics exactly.

   **Falsifier:** Lower gradient variability without improved WP at either rate weakens the case for batch noise as the remaining optimization bottleneck.

   **Expected direction:** Modestly positive; larger gains require current weight-mass fluctuations to be unusually harmful.

**C. Measurements that could mislead, and what I would stop**

The “0.23 noise floor” is an RMS benchmark. For stationary independent zero-mean noise, approximately constant \(v\), and \(\beta_1=.9\),

\[
\operatorname{RMS}(m/\sqrt v)\approx\sqrt{.1/1.9}=.229,
\]

but the corresponding absolute-value median is approximately .155 and q95 approximately .45. Your statistics fit that benchmark reasonably well. Cycled momentum, autocorrelation, clipping, and nonstationarity invalidate a literal comparison. Annealing the learning rate does not directly shrink \(m/\sqrt v\), although it changes the trajectory producing those states.

Endpoint spectra confound cause and consequence. Examine their evolution and actual update norms before explaining WP through final stable rank. Likewise, larger Muon gradients could indicate reduced saturation, changed sensitivity, or different parameter scaling.

Weighted Pearson and weighted MSE share an affine-calibrated, unclipped population optimum under suitable conditions; finite-model rankings need not coincide, especially with prediction clipping. Select on the actual clipped WP metric. The 94% run-offset decomposition identifies where error resides, not whether that component is predictable.

I would stop using synthetic optimizer rankings as a selection gate, extending training beyond the established peak, and trying uncalibrated momentum inflation or additional schedule variants. Avoid hard orthogonality constraints on GRU weights without evidence that they preserve useful memory.

Spend the first four fits on the direction/size experiment. Then select one intervention supported by its result. Advance an apparent ≈0.003–0.004 WP gain to the second fold/seed cell; use sequence-level uncertainty estimates and replication, since row-level resampling would overstate precision.
