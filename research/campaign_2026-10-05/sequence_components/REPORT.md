# Ten completed experiments: linear memory, attention and spline details

All ten GPU fits finished, fixed epoch zero, 474 optimizer updates each. Every
model processed complete 20,000-row sequences with causal state. Training used
24 sequences; the fixed scoring set contained 16 distinct components and
26,990 official scored rows. These are previously exposed diagnostic sequences,
not fresh validation and not a public-score forecast. No model selection or
training-seed uncertainty is included in the component intervals below.

| Model | Parameters | Pooled WP |
|---|---:|---:|
| GRU + MLP | 8,235 | 0.482259 |
| GRU + spline | 8,226 | 0.502267 |
| Fixed diagonal memory + MLP | 2,955 | 0.473689 |
| Learned diagonal memory + MLP | 2,987 | 0.477319 |
| Learned diagonal memory + spline | 2,978 | 0.515935 |
| GRU + attention + MLP | 9,148 | 0.472050 |
| GRU + static branch + MLP | 9,142 | 0.465084 |
| Diagonal memory + attention + MLP | 3,900 | 0.456415 |
| Diagonal memory + static branch + MLP | 3,894 | 0.492819 |
| Diagonal memory + attention + spline | 3,899 | 0.501282 |

The useful pilot lead is the spline component, particularly on the same
learned diagonal memory: **+0.038616**, conditional paired-component interval
**[+0.015913, +0.061933]**. GRU spline-minus-MLP is +0.020008, with interval
[-0.004540, +0.046376]. Learning the decay spectrum adds +0.003630 over its
fixed initialization; interval [+0.002256, +0.005663]. These intervals are
exploratory, conditional on one fitted seed and the exposed panel. The pilot
does not establish that a spline improves the much stronger GRU256 incumbent.

Attention is configuration dependent: GRU attention beats its near-capacity
static branch by +0.006965, interval [-0.003804, +0.018977]. Diagonal-memory
attention loses to its static branch by -0.036404, interval
[-0.053742, -0.021596]. Replacing the attention hybrid's MLP with splines helps
by +0.044867, but the resulting hybrid still trails diagonal memory plus
splines by -0.014653. This tests a small additive positive-kernel memory with
fixed learned scalar decay and current-input query/key/value embeddings. It
does not reject delta-rule memory, query-aware writes, event-ring retrieval,
state-conditioned queries or input-selective forgetting.

The spline is an additive **piecewise-linear** edge function on bounded latent
coordinates. It is KAN-inspired, not the original cubic KAN or a theorem-driven
claim. Its positive result supports testing local nonlinear components; it
does not justify replacing every dense edge in the GRU. Current spline input
range is [-1,1], with seven fixed knots and a shared linear anchor. Effective
observed knot occupancy still needs measurement before attributing the benefit
to fine-grained local response rather than a simple nonlinearity.

The complete experiment cost **478.59 device-GPU seconds (0.133 h)**, including
inference; measured peak device use was **381,681,664 bytes**, worker RAM
**1,465,126,912 bytes**. Exit zero, teardown verified, zero worker and parent
OOM increments. GPU shape probes were separate, about 4.3 seconds each.
Python-loop memory operators are slower than the optimized GRU in this tiny
implementation; these timings do not establish submission callback speed.

Sources and identities: `RESULT.json`, content-bound `ten_arm_fit_v1.sealed.json`,
the existing guarded runner attempt, `runs/ten_arm_v1/` fixed checkpoints and
selected prediction exports. The source and checkpoint/export hashes are
retained in the result. The scorer is unchanged. Human plots are in `eda/` and
the active continuation marimo notebook displays the full table and images.

Next bounded questions:

1. Replicate the near-equal-capacity spline/MLP comparisons across training
   seeds and a separately frozen component cohort; include a simple hinge/ReLU
   edge control to isolate learned knot response from a generic nonlinearity.
2. Add a small zero-initialized spline residual to the strong recurrent
   representation. Compare head learning with a matched MLP and permit a
   separately specified end-to-end continuation. Old post-hoc calibration
   negatives do not answer this nonlinear latent-feature question.
3. Keep separate linear-memory variants: real positive decays, oscillatory
   complex LRU, input-selective dynamics and stable nonlinear write gates.
   Compare fixed memory with learned memory before large hybrids.
4. Change one attention detail at a time: current-input versus state-conditioned
   queries, constant-key versus content-key retrieval, or additive versus
   bounded corrective writes. Then test a spline write gate or decoder. A
   poor large combination does not erase evidence for one useful component.

The current confirmed public best remains **0.6660**. None of these small-model
scores establishes an honest expected public 0.7000.
