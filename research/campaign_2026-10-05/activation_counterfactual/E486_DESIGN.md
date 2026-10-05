# E486: does the bounded candidate shape need retraining?

Design registered before implementation. All model fitting is already complete in E483. This CPU-only diagnostic crosses the shape used to train a checkpoint with the shape used during inference, keeping its parameters fixed. It asks how much of the observed change is an inference transformation and how much requires fitted parameters adapted to that transformation. This is not an optimizer mechanism proof, new-market confirmation, or new public candidate selection.

Use all18 E483 checkpoints: training candidate baseline/sine × recurrent Muon momentum0.8/0.9/0.95 × seeds271831/271832/271833. For each checkpoint independently evaluate baseline and sine candidate equations with the same saved parameter/buffer bytes, frozen fit-only normalizer, causal42 features, fresh zero state for each complete sequence, and the unchanged sixteen exposed test components. This produces36 scored cells from18 existing fits. No new fit, checkpoint selection, amplitude, frequency, feature, schedule, or optimizer choice. Both inference shapes keep the exact reset-after equation and sigmoid gates; sine is u+0.1sin(pi*u), u=tanh(v). Use the independently reconstructed E483 scorer, pinned to its frozen v3 source. Activation swapping produces a different causal predictor and must never be called the same deployed checkpoint behavior.

Recompute all36 inference passes on CPU (one CPU,6GiB RAM, no swap, no GPU,2400s bound), saving selected raw finite predictions/targets, per-component identities and masks, full global/t0/t1 scores, and shared numeric-sorted2000 component bootstrap draws with seed271828. All18 diagonal training-shape=inference-shape outputs must reproduce the immutable E483 raw prediction exports/metrics with exact float32 raw-prediction equality and score maximum-absolute error≤2e-12 before any contrasts are accepted. Require strict checkpoint hashes/keys/shapes, identity buffers and the successful E483fit-proof and officialgrade attempt hashes before test loading. No source mutation or original artifact overwrite.

For each of the nine(seed,momentum) pairs, retain these six contrasts,54 total. Let A=trainBase/evalBase, B=trainBase/evalSine, C=trainSine/evalBase, D=trainSine/evalSine:

1. D−A: the previously measured matched-shape training effect(recomputed, not new evidence).
2. B−A: enabling sine at inference on baseline-fitted parameters.
3. C−D: removing sine at inference on sine-fitted parameters.
4. C−A: training-shape difference under baseline inference.
5. D−B: training-shape difference under sine inference.
6. (D−B)−(C−A): training-by-inference shape interaction.

Always jointly bootstrap all four correlations on the same component multiplicities for each interaction. Keep seeds separate and do not treat dependent contrasts as independent replications. Raw nonfinite predictions/targets fail before clipping; official metric clips both to±2. Inference changes can alter hidden trajectories, output amplitude and clipping, so a swapped activation is not a pure representation-isolation or amplitude control. Report this limit, all36 scores, all54 contrasts and heatmaps rather than choosing a winner. No public forecast follows from this exposed-panel diagnostic. Require independent blind design critique and separate source review before actual inference; preserve failed contracts and rerun only after versioned repair.
