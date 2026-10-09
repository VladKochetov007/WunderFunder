# Online learning: independent controller proposal, before provider synthesis

Status: proposed on 2026-10-07. No fitted competition result or public-score forecast. This is a new mechanism family, not another run of the earlier supervised dev-period fine-tuning idea H025 or the unavailable delayed-target refit mentioned in H058. Numerical priors are null.

The observed interface supplies only current features and sequence metadata. All learned adaptation state must reset with the sequence; targets and the scoring mask are unavailable. This was checked against `utils.DataPoint`, the step-by-step scorer and the [official data description](https://wundernn.io/connectome/docs/data_overview). The explicit online text permits causal state updates; it does not separately endorse every gradient-adaptation procedure. Expressing fast weights as resettable state with a fixed, independently trained update rule is the proposed implementation interpretation.

## First branch: learn local feature dynamics, use innovations

Fit a small projection and dynamics prior on permitted fitting groups, then keep a 16- or 32-dimensional recursive dynamics model as sequence state. At row t, compare the now-observed projected input with the forecast made from row t−1, update the dynamics from this completed transition, and forecast the next projected input. A fixed head, trained offline, maps the anchor representation, innovation and next-input forecast to the two task outputs. No task labels enter the online update.

Start with recursive least squares or normalized least-mean-square updates; the [recursive least-squares adaptive-control paper](https://www.tandfonline.com/doi/abs/10.1080/00207177208932257) supplies the adaptive system-identification component. Its original control setting does not establish a financial-prediction gain. Our transfer hypothesis is that local dynamics and their surprising failures reveal conditions useful for predicting the hidden targets.

Compare an unchanged anchor; the same offline-fitted readout with fixed dynamics; warm-up-only adaptation; and continuous causal adaptation. Keep fitting membership, offline inputs, parameter count and target loss fixed. Auxiliary next-input error is a diagnostic; only the complete target endpoint can establish task usefulness. Include zero-learning-rate and stale-update controls. Any new fitting pipeline must train the readout on causally generated states, never full-sequence fitted dynamics. Cheap protocol first: prefix invariance, reset, update-order checks and closed-form ridge equivalence; then a bounded one-fold head fit, followed by serial native serving timing.

The online state scales quadratically with the small latent dimension, independent of sequence length. Every-row low-rank updates are the first candidate; cadence and forgetting become explicit descendants if measured runtime or nonstationarity justifies them. Train-time streaming/head reuse can overlap the existing frozen-trunk H233 work without treating H233's outcome as a gate.

## Second branch: train the learning rule itself

Treat a small fast-weight matrix as recurrent state. Offline training learns keys, values, decay and update rate so that within-sequence feature-based updates help future supervised predictions. This draws from [TTT layers](https://arxiv.org/abs/2407.04620) and the [gated delta rule](https://arxiv.org/abs/2412.06464); their language-model results motivate a component, not an expected competition gain. Compare a parameter-matched recurrent model, no-write fast weights and learned-write fast weights. Measure first-99-row initialization, long-sequence stability and full/chunk/row equivalence. The fast state and all accumulated statistics reset per sequence; no across-sequence adaptation.

A small serial reference should establish semantics before GPU optimization. The authors' [TTT kernel repository](https://github.com/test-time-training/ttt-lm-kernels) is an optimization reference, not a drop-in serving dependency. Chunkwise parallel training and rank-limited CPU inference may use different implementations but must preserve the same update ordering and numerical tests. Larger adaptive teachers remain eligible for compression.

## Third branch: task-aligned self-supervised adapter updates

Train a low-rank adapter with a causal next-feature or masked-current-feature objective and an outer supervised target objective. At inference, use only completed observable transitions for a small update every fixed number of rows; keep the main backbone fixed and shrink adaptation toward its sequence-start state. [Self-supervised test-time training](https://proceedings.mlr.press/v119/sun20b.html) motivates the shared representation; it does not imply that a lower auxiliary loss improves our target.

Compare joint offline training without online updates, fixed unsupervised updates, and meta-trained updates. The distinguishing observation is a gain in the target endpoint over the same joint-training control, not merely a reduction in reconstruction error. Preserve a larger-teacher path if an efficient CPU adapter is weak; distill the adaptive teacher into an ordinary causal student with independent weights and the same evaluated-group exclusions.

## Fourth branch: online reliability for complementary experts

Offline, train each expert to predict both target outputs and a cheap future-input proxy. On the arriving stream, observe completed proxy forecast errors and update a small reliability state. An independently fitted gate uses those states to mix expert target predictions. Update reliability only after its proxy becomes observable, then predict the future targets from the current row. Test an equal mixture, a fixed gate and an adaptive gate with identical experts; a shuffled/stale reliability control tests whether the adaptation signal matters.

The key unproven link is whether proxy forecast skill predicts conditional target usefulness. Test that link on fitting groups and evaluate the complete mixture on excluded exposed groups. This is a new interaction with the complementarity family, not permission to retune the old failed model pool. Standalone weak experts may still be useful. All gate/reliability state resets at sequence boundaries.

## Allocation and descendants

Use the protected exploration allocation; keep the existing leading student and teacher experiments independently owned. First obtain the inexpensive causal dynamics protocol and a matched readout experiment. In parallel, prepare the learned fast-weight teacher reference. Advance according to task endpoint, diversity contribution, measured cost and information value; do not close the family because one auxiliary objective or forgetting factor fails.

Possible level-two children: dynamics × event features, learned reset/forgetting, multi-timescale online states, adaptive expert mixtures, and adaptive-teacher-to-fixed-student distillation. Online supervised optimization on supplied training sequences is another eligible branch, but shuffled sequence IDs cannot define a real market chronology. New provider proposals remain independent until their public outputs are committed.
