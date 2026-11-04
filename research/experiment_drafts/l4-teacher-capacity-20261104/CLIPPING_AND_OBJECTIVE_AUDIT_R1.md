# Clipping, Muon and auxiliary objectives: diagnosis before another threshold sweep

This is an implementation audit and a CPU mechanism illustration, not evidence
of improved model quality. It does not admit additional scientific fits or
change the frozen twelve-fit allocation. Official train-only ancestry and full
official validation under ADR0044 remain mandatory; validation is exposed.

## Findings

1. The frozen GRU source `27e605aa488651a3ab6cd38d3a1965075c24760f`
   records preclip statistics, then clips the combined objective gradient globally
   to1. Recurrent matrices use gatewise Muon; projection, readout and biases use
   AdamW. The1 threshold is inherited, not calibrated for these widths.
2. A constant positive scaling of the entire gradient history approximately
   cancels in both Muon's norm-normalized Newton–Schulz update and AdamW's
   first/second-moment ratio. Epsilon, precision and decoupled weight decay need
   separate accounting. Step-dependent scaling can change the momentum mixture
   and actual update directions. Raw clipping factors are not update-size ratios.
3. `head` gradient statistics group parameter tensors, not loss terms. They mix
   main outputs and auxiliaries, including a shared hidden readout. A67–77%
   squared-norm share in the last epoch does not identify which objective causes it.
4. The scratch recipe has `aux_struct=.3` and `aux_next=.3`. Structural loss sums
   three cross-entropies; next-change loss sums distance MSE and sign BCE. Main
   WMSE divides by target-magnitude weight mass, structural loss by valid rows,
   and next loss by the conditional subset with a later target change. Therefore
   `.3` does not establish small auxiliary gradients. Next-distance targets are
   not clipped. The separate `aux_weight=.5` is inactive without shifted `--aux`
   heads. These are recipe facts, not proof auxiliaries hurt.
5. WMSE is independently normalized per1,000-row chunk. Different chunk weight
   mass does not proportionally change the number of optimizer updates. This
   differs from one pooled weighted loss, and clipping adds another nonlinear
   weighting. Official evaluation must remain unchanged.

The source locations are `src/wnn/training/updates.py`,
`src/wnn/training_dynamics.py`, `src/wnn/optim.py`,
`src/wnn/training/losses.py`, `src/wnn/spread_aux.py`, and
`src/wnn/training/setup_model.py`. Independent read-only optimizer and objective
reviews were performed by separate existing agent sessions.

## Exact-code CPU illustration

[The script](../../../scripts/l4_teacher_capacity_20261104/probe_clipping_mechanism.py)
replays24 prescribed synthetic gradient pairs through the actual `MuonAdamW`.
It uses CPU, two threads, fixed seed1701, fixed LR/momentum and weight decay0.
[Raw output and source hashes](CLIPPING_MECHANISM_PROBE_R1.json) are retained.

| Treatment | Mean raw multiplier | Median Muon update-norm ratio to unclipped | Mean update angle |
|---|---:|---:|---:|
| Constant gradient scaling by0.1 | .100 | .999999 | .008 degrees |
| Global norm clip1 | .325 | 1.002 | 39.1 degrees |
| Global norm clip5 | .768 | .987 | 32.8 degrees |

These angles are synthetic measurements, not measured GRU training angles.
The probe intentionally varies head-gradient magnitude while keeping the same
prescribed recurrent-gradient sequence across treatments. Thus global clipping
can alter recurrent update history through head-dependent factors. It does not
establish any validation effect. CPU Torch2.14 differs from the node's2.8 runtime;
actual OneCycle LR/beta cycling and nonzero decay are omitted.

The raw field `relative_trajectory_difference` measures the relative Frobenius
difference of stacked per-step updates, not cumulative weight trajectories.
The scalar counterexample's `[8,-2]` means two equiprobable stochastic **batch**
gradient outcomes for one scalar parameter at a fixed point. Clipping each
sampled batch gradient to1 gives `[1,-1]`: expected gradient changes from3 to0.
It does not mean two coordinates of a single vector or per-example clipping.
An independent reviewer checked the code/hash joins and these interpretation
limits. No local minimum or real-data clipping harm is established.

## Highest-information next measurements

First, use a predeclared small set of fitting-only batches at frozen checkpoints,
without optimizer steps. Decompose gradients of main WMSE, weighted structural
loss and weighted next loss. Record loss values, normalization masses, valid-row
counts, per-module norms, and cosine of each auxiliary gradient versus the main
gradient on the shared trunk. Split final readout rows into main0:2,
structural2:21 and next21:23. Report the combined norm and clipping factor.

Second, if an optimizer-dynamics probe is needed, replay identical fitting
batches/RNG from the same initialization and compare actual per-gate and head
updates, separating decay. A short replay diagnoses dynamics; it cannot rank
validation quality or reproduce an existing fit's momentum unless that state
is actually available.

Then choose a small prospective paired quality screen using the existing
matched controls: clip5 is the direct threshold arm; auxiliary removal/reduction
is an objective-conflict arm if decomposition supports it; a lower peak LR is
an update-magnitude arm if actual relative updates support it. Do not combine
all changes in the first comparison. Use two seeds per arm and freeze4–6 total
new fits at most, checkpoint/schedule/metric rules and fixed-blend endpoints
before launch. None of these future fits has been launched or allocated here.

For the current series, all six GRU best-observed epochs are2, while the fixed
primary endpoint is3. Both valid WP and WMSE decline in the last epoch. This is
consistent with overfitting/objective mismatch but does not isolate clipping,
auxiliaries or LR. Preserve [the completed results](RESULTS_GRU_20261104_R1.md)
and the predeclared fixed-epoch teacher/student comparison unchanged.

## Primary sources

- [Koloskova et al., ICML2023](https://proceedings.mlr.press/v202/koloskova23a.html)
  establishes stochastic clipping bias under its assumptions; it is not a theorem
  about our nonconvex GRU with hybrid Muon/AdamW.
- [Keller Jordan's Muon description](https://kellerjordan.github.io/posts/muon/)
  describes momentum orthogonalization and scale invariance of the orthogonal
  update. The project uses an explicitly strict-FP32 adaptation.
