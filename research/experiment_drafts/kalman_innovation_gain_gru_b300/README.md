# Innovation-conditioned latent gain GRU on B300

**Status:** code and experiment draft only. No GPU fit, synthetic model check, B300 profile, student cache, official score, or submission has been run from this branch. The edited Python files pass Ruff lint and format checks. This handoff makes no quality or runtime claim.

**Prefit blockers:** before admitting task fits, run and record a synthetic sequence/chunk/reset/batch-isolation invariant check, obtain the actual H512 B300 profile, and freeze the per-cell teacher-source registry for student caches. The cache-source registry is currently absent; no student cache or fit is runnable until exact source run IDs, epochs, coefficients, checkpoints and clean ancestry are verified. These are handoff requirements, not measurements from this branch.

Hypothesis `H-KALMAN-INNOVATION-GAIN-GRU-20261018-R1`; experiment `E-KALMAN-INNOVATION-GAIN-GRU-B300-20261018-R1`. The branch starts from synchronized `research-team` SHA `104e5461aa862c0d66a44588248385874b800054`; current main source SHA at handoff was `24e9001570de158db9a5031e632dbb0b4477145f`.

## What is implemented

1. [`src/wnn/innovation_gain_gru.py`](../../../src/wnn/innovation_gain_gru.py) implements a causal local-linear latent prior, innovation, bounded learned diagonal gain, and zero-initialized residual injection into the existing fused GRU and readout. It carries GRU state plus the two previous latent measurements across chunks and resets with the ordinary sequence state.
2. `--arch innovation-gain` and the `--innovation-gain-conditioning` control are registered in the shared trainer/model factory. The matched `context_only` arm has identical architecture and parameter count but masks innovation only at the gain network.
3. Checkpoint provenance records the observer and GRU-core source hashes. Prediction export checks those hashes, records them in its sidecar, and the four-family distillation-cache builder verifies them again before accepting an observer teacher.
4. [`recipe.json`](recipe.json) freezes the arms, 30 paired cells, B300 profile gate, training schedules, student cache path, statistics, output archive and no-fit limits.
5. [`PR.md`](PR.md) answers the repository's eleven experiment-draft questions in English.
6. [`evidence/README.md`](evidence/README.md) records the nearby local results and their limits. They motivate this mechanism; none measures it.
7. The literature summary is indexed at [KalmanNet innovation-gain sources](../../unified/sources/KALMANNET_INNOVATION_GAIN_GRU_20261018_R1.md); the experiment hypothesis is indexed in [unified ideas](../../unified/ideas/KALMAN_INNOVATION_GAIN_GRU_20261018_R1.md).
8. The [read-only implementation critique](../../unified/reviews/KALMAN_INNOVATION_GAIN_GRU_DRAFT_REVIEW_20261018_R1.md) found no causal/chunk-state defect by inspection and records the missing student source registry and unrun invariant/profile gates.

## Mechanism

For each chunk, the input encoder produces `z_t`, a measurement projection gives `o_t`, and a local-linear prior uses the preceding two encoded observations:

```text
m^-_t = o_(t-1) + sigmoid(a) * (o_(t-1) - o_(t-2))
nu_t  = o_t - m^-_t
K_t   = sigmoid(b + gain_net([o_t, nu_t]))
 m_t  = m^-_t + K_t * nu_t
```

The `context_only` control zeros only `nu_t` at the `gain_net` input. Both arms still provide the same actual `K_t * nu_t` correction to the recurrent residual path. This is a learned latent observer, not a calibrated Bayesian filter: no covariance or process-noise recursion is claimed. It uses current and past inputs only; model weights never change at inference.

## B300 run order

1. Mount permitted train/valid data, target indexes, source-qualified fold manifests and the code checkout at the frozen commit. Put artifacts/cache/logs on physical storage.
2. Run the actual-shape FP32 profile in `recipe.json` on one B300. Capture GPU/driver/CUDA/PyTorch versions, peak host/GPU memory, warmed update throughput, chunk transfer time, finite gradients, and TF32/autocast settings. Reject or reshape only before freezing the scientific matrix; then profile all paired arms at the same declared shapes.
3. Run ordinary GRU, context-only observer and innovation-conditioned observer for all30 outer-fold/seed cells. Each cell is a clean scratch fit. Use eight concurrent independent fits at most, one per B300; DDP is not required for this model-sized cell parallelism.
4. Build the existing four-family cache for each cell from a frozen clean-fold source registry. Treatment caches replace only the `samefold_baseline` run with the corresponding observer teacher at exactly the original coefficient.
5. Train all three matched H256×2 standard-GRU student conditions for30 cells each, including the no-replacement cache control. Keep the standard student checkpoint convention and identical train targets otherwise.
6. Independently grade frozen outer-fold predictions with `global_wp_clip2_v1`; record fold, seed, component, and target breakdowns. Keep logs/receipts separate from checkpoint/optimizer archives.

The [draft PR body](PR.md) contains exact sample commands and frozen claims. Fit launch is intentionally not part of this documentation handoff.
