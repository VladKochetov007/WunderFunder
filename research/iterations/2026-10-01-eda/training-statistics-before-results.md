# Training and ensemble-statistics deepening, before new outcomes

The previous result supported only a narrow statement: at one initial parent and two early windows, the sum of auxiliary gradients was positively aligned with the main shared-head gradient. It did not support a generic conflict explanation, nor establish that removing auxiliaries is useless. This round tests the reach of that observation and the practical size of an exact ensemble-objective mismatch. Both are diagnostics of fixed, previously selected artifacts, with no new predictor or coefficient selection.

## What the broader gradient study can distinguish

TDIAG2 crosses the two saved h128 seeds with folds0/1 and compares each parent epoch3 to its actual one-epoch head child. Late windows are fixed before results. The same 16 naturally sampled chains, one hash-selected eligible sequence per chain, are used for both seeds/stages of a fold. There is no class balancing; class composition and metric/auxiliary mass are reported. This changes the sampling estimand from the previous balanced diagnostic, not the degree of independence of already exposed data. A chain-uniform sample with one representative sequence is not the original row-weighted training distribution. Shared training chains across model folds are recorded.

The losses evaluated at the parent are the **head-FT objective**, not a replay of the parent's historical wmse training. The head child is h1, not public V1's h2. Feature reuse is conditional on exact equality of every non-head checkpoint tensor, including normalization buffers; merely seeing `freeze=inp,gru` in arguments is insufficient. Full causal sequence replay precedes selection of late windows, avoiding a reset at row8000 or18000. Full future targets are legal auxiliary supervision, never trunk features. CPU float32 and deterministic dropout0 mode do not reconstruct historical AMP or Adam states.

Negative combined shared cosine appearing mainly late or after head FT would narrow the conflict hypothesis to that stage. Positive alignment throughout would weaken a broad conflict explanation further, while leaving finite-update/Adam effects and other data slices unresolved. Component cosines, private-output gradients and global clipping must remain separate: a small negative component does not override a positive combined direction, and auxiliary-private rows affect the common clip norm without receiving any main-loss gradient. A changed clip factor is not a proportional change in Adam's step.

The exact pooled loss is nonlinear in sequence composition through weighted centering and normalization. Therefore the sensitivity calculation recomputes losses and gradients after each chain deletion; it does not add per-sequence correlations or bootstrap frozen per-chain gradients. Sixteen deletion results are not sixteen optimization replications. They answer whether the reported geometry is dominated by a selected chain. No cosine threshold is a veto or promotion rule for the proposed aux-off intervention. The actual objective remains the fixed serving ensemble.

## What member-scale derivatives do and do not identify

For one target, let `e_i = sum_j alpha_j r_ij / 4`, `p_i=clip(e_i,-2,2)`, `y_i=clip(target_i,-2,2)`, and `w_i=|y_i| mask_i`. At `alpha=(1,1,1,1)`, the differentiable-row perturbation is `a_ij=r_ij/4 * 1(|e_i|<2)`. With weighted moments `W, mu_y, mu_p, V_y, V_p, rho`,

`d rho/d alpha_j = [sum(w*y*a_j)-mu_y sum(w*a_j)]/[W sqrt(V_y V_p)] - rho [sum(w*p*a_j)-mu_p sum(w*a_j)]/[W V_p]`.

Targets are averaged with weight1/2 after their own normalizations. The calculation retains their separate masks/weights, centering, and variance denominators. Global derivative contributions by chain can be signed and cancel; their absolute mass is not ordinary metric mass. Near-collinear members can produce tiny derivatives or poorly determined relative directions despite the algebraic objective mismatch. A derivative is local to selected artifacts, not an optimum, a safe step size, or a forecast of finite improvement. No real-data coefficient perturbation is scored.

There is a useful refinement to the proposed affine control. **Without final clipping, a globally constant bias shift of one member shifts the whole ensemble by a constant and remains correlation-invariant.** Relative member scales can matter even without clipping; a global member intercept cannot. The four-parameter affine control (scale and shift for each target) therefore has only two score-active global directions in the fully unclipped idealization. Intercept directions can become active through clipping, or when independently fitted fold/regime shifts are pooled, which is a different mapping. Do not cite the existence of four parameters as four independent calibration opportunities. The sum of the four scale derivatives is zero under exact unclipped common rescaling, but need not vanish after clipping.

Exact rows at the clipping boundary are kinks; the implementation reports their counts and uses the stated zero convention, rather than pretending the ordinary derivative exists there. Degenerate variance scopes are explicitly flagged/stopped. Synthetic tests check finite differences only on synthetic data, including clipped mixtures, all-masked and constant-target/prediction cases. The fixed real C score is reproduced once to bind the calculation to the established baseline.

Earlier negatives remain relevant: class-specific output scaling (journal entry239), class offsets E109, class/density mixtures E127/E138 and running loss normalization E180 do not test this same member-scale derivative. Nevertheless, they lower the prior that another generic calibration treatment will help. The existing per-sequence oracle scaling headroom in `eda_scale.py` depends on unavailable outcomes and is not deployable attainable gain. A nonzero derivative here would justify understanding a local mismatch, not repeating those searches or launching a coefficient fit automatically.

## Pooled geometry and statistical units

The metric averages two target-specific pooled correlations, not class or chain correlations. For each target,

`Cov_w(y,p) = sum_c pi_c Cov_wc(y,p) + Cov_pi(mu_yc,mu_pc)`.

The same decomposition applies to both variances. Its reconstruction is checked numerically. A between-class component is a legitimate part of this objective, not evidence that class prediction is spurious. Structure auxiliaries may alter useful class-relative levels even if their global gradient cosine is positive; the decomposition does not identify their causal contribution.

Member means, standard deviations, raw covariance and residual covariance use these same target-specific weights. Covariance alone cannot tell whether a member supplies useful incremental information under retraining. Member/final clipping fractions are reported both per scored row and by metric weight; macro-chain and pooled scored fractions are kept separate. Chain mass concentration, including `(sum W)^2/sum W^2`, describes concentration only and is not an IID effective sample size.

The uncertainty calculation resamples whole observed chains within fold, recomputing weighted derivative moments for each of1000 fixed draws. It conditions on selected checkpoints, exposed folds, deterministic cache alignment, and the existing four-member choice. It does not include training randomness, adaptive-selection optimism, hidden-market shift or an unknown number of true independent exchange histories. Every member's interval is reported. A favorable sign is not selected as a newly validated coefficient. Leave-one-chain sensitivity identifies concentration without creating extra training seeds.

## Fixed next decisions

If gradient geometry is stable, the aux-off study remains a cheap controlled training intervention with no claimed conflict remedy. If it is stage- or chain-dependent, the interpretation becomes narrower; this round still does not authorize stage-specific loss schedules. If member-scale derivatives are small/uncertain and covariance is nearly singular, the practical motivation for ensemble-aware calibration weakens. If derivatives are coherent across folds and insensitive to individual chains, a separately frozen affine-versus-head comparison becomes better motivated, without an estimated gain or automatic allocation. In every case, one-epoch C evidence remains distinct from the planned two-epoch V1-like comparator, and all GitHub/external uploads stay stopped.
