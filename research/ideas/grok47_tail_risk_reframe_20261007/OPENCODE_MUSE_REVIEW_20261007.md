# Independent review of the Muse Spark reframe

Reviewer: native subagent `resume_critic_luna` (gpt-6-luna, max reasoning), read-only review of the eight-card report and current project contract. This review interprets the proposals; it does not approve a fit or assign numerical priors.

## Blocking correction: metric definition

The prompt and report describe the official objective as global weighted squared error. That is incorrect. The frozen metric is `global_wp_clip2_v1` in `src/wnn/gmetric.py`: clip target and prediction values to `[-2, 2]`, weight by `|clipped y|`, apply the fixed scored-row mask, pool rows globally for each channel, calculate weighted Pearson correlations, and average the two channels. Weighted MSE remains a training surrogate where explicitly declared. It must not be called the official metric, and any proposed “official row loss” derived from MSE is not an official score component.

The raw response is retained unchanged in `OPENCODE_MUSE_OUTPUT.jsonl`. The curated `OPENCODE_MUSE_REPORT.md` now carries a prominent correction. In particular, the paper's CVaR construction is only a possible secondary selection/risk hypothesis here: group-level resampling must recompute pooled weighted Pearson, and it cannot replace the official promotion metric. Per-example gradient clipping is a separate intervention from the paper's CVaR estimator and needs its own hypothesis and checks.

## Proposal disposition and overlap

- **Cards 1–2 (CVaR and tails):** retain as source-grounded hypotheses only. The source's assumptions and reported mechanism do not establish heavy-tailed score/gradient behavior here or improvement to weighted Pearson. Card 1's row “official loss” formulation is invalid as written; group-risk diagnostics must be labeled secondary. A prediction-only cache cannot establish per-step gradient-tail behavior.
- **Cards 3–4 (hybrid architecture and scaling):** eligible after current training/compute work is accounted for. The hybrid must retain its recurrent state path; a conv-only test would not falsify a recurrent-plus-convolution claim. The broad scale grid overlaps ADR0008 and H197, so make the first architecture rung narrow and matched.
- **Cards 5–6 (latent-state teacher and large teacher):** keep as exploratory branches. Card 6 overlaps the active E511 C1 transfer arm except for changing teacher scale/context; defer that child until C1 is evaluated and demonstrates useful student transfer. Any later comparison should change teacher scale/context while holding the student and scoring recipe fixed.
- **Card 7 (bidirectional oracle):** allow only as a research-only, noncausal diagnostic. It is not a predictor or performance ceiling. Fit it on training groups only and label any validation analysis exposed. Training-group-only privileged targets into a causal student are possible as a separate route, subject to explicit leakage checks.
- **Card 8 (weak-model residual/disagreement):** eligible but overlaps earlier blend/residual work (E292, E313, E344), H184's sparse-blend idea, and the new C1 work. The unresolved interaction is a fixed, predeclared cohort of several weak, diverse members versus one weak member at a time; a cohort can reduce idiosyncratic residual noise. It deserves a small, bounded test only after row/group identity and training membership are verified.

## Data and exposure note

Previously inspected validation folds are exposed; reusing them can support exploratory comparison and implementation debugging, not fresh confirmation. The initial review's statement that the current cache has no prediction arrays is not reliable: the controller inventory finds `cache/valid_targets.npz` and many prediction `.npy` files. Some legacy E177 candidate arrays do not have the modern per-prediction sidecars, so their row-order/training-membership provenance still needs a bounded check before any new score is treated as valid. The exact files and any compatibility caveat will be recorded in the weak-cohort experiment packet.

## Allocation consequence

Keep the current C1 fit and Kaggle profile work in flight. No Muse fit was admitted. The cheapest next step for the weak-cohort branch is a CPU-only, metric-correct, group-held-out analysis of existing compatible predictions; it is separate from the C1 teacher/student claim. Do not translate any local delta into a public score forecast.
