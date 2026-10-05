# Independent research critic: recurrent attention, state space, and KAN components

Prepared 2026-10-05, before any new fit. This is a literature-grounded design
review, not a claim that one family should replace the current predictor. No
GPU work or code changes were made.

## What the current evidence says

The updated strong-baseline report is a structural diagnostic, not a clean
generalization estimate: its fitted panel overlaps the trained groups, and its
“excluded” panel is connected by feature-graph components to many fitted rows.
The scores (.7744 and .7428 for the complete blend) do not certify the 0.7000
public expectation. Still, its mechanics are useful: hard output holding loses
about .076 WP on excluded groups, and on stable, no-observed-event class-3
transitions the complete predictor's continued updates reduce weighted squared
error for both outputs. A useful new model should therefore preserve causal
state updates on every row, including quiet rows. The near-perfect scored-row
event/change association is a clock signal, not evidence that the target stays
constant until that clock rings.

The target metric has a relevant property for all candidate heads. For
`Y=clip(T,-2,2)` and `w=|Y|`, the pointwise minimizer of exact weighted MSE is
`E[wY|X]/E[w|X]`, which is also the weighted-Pearson-optimal scalar up to
affine scale/offset before prediction clipping. Thus keep the same exact
target-weighted objective and readout across model arms. A spline, event gate,
attention memory, or state-space layer must improve the *causal information
representation*; a “distributional” or multi-head name is not itself an
objective improvement.

## Primary literature and venue labels

The following are original papers or official proceedings pages. Venue labels
below distinguish actual ICML publications from related work elsewhere.

| Work | Venue | Relevance here | Boundary on the claim |
| --- | --- | --- | --- |
| [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://proceedings.mlr.press/v119/katharopoulos20a.html) | **ICML 2020** | Kernelized causal attention can be evaluated as a streaming recurrence with fixed-size sufficient-statistic state. That attention calculation can be used as a memory branch inside a recurrent model; a Transformer block is not required. | Kernel feature-map attention compresses all history into finite accumulators. It loses exact token-level access, and its retrieval capacity/normalization are modeling choices. The paper's speed results are not evidence for this dataset or a one-vCPU scorer. |
| [Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html) | **ICML 2021** | Makes the online key/value memory interpretation explicit and studies a delta-rule update that can overwrite/correct rather than only accumulate associations. This is a plausible non-Transformer content-addressable state mechanism. | Additive outer-product memory has finite capacity; collisions and state saturation matter. Delta updates add a learned update rule and can be unstable if learning rate/decay are unconstrained. |
| [Resurrecting Recurrent Neural Networks for Long Sequences](https://proceedings.mlr.press/v202/orvieto23a.html) | **ICML 2023** | Introduces the Linear Recurrent Unit (LRU): stable, diagonal linear recurrence with normalization, linear projections and feed-forward gates between recurrent layers. It motivates a compact multiscale state branch that is cheaper at row-wise inference than full attention. | The reported benchmark is Long Range Arena, not noisy, event-driven market data. Fixed linear dynamics may smooth over rare changes unless input-dependent gates or a separate retrieval pathway are learned. |
| [Transformers are SSMs: Structured State Space Duality](https://proceedings.mlr.press/v235/dao24a.html) | **ICML 2024** | Derives connections among structured state-space models and attention using semiseparable matrices; Mamba-2 is one efficient refinement. It argues for testing the computational mechanism, not treating “attention” and “SSM” as unrelated camps. | Its 2–8× result concerns the paper's GPU training/workloads; it does not guarantee a faster/smaller serial CPU callback. It is a conceptual bridge, not a forecast of gain here. |
| [Understanding the Skill Gap in Recurrent Language Models: The Role of the Gather-and-Aggregate Mechanism](https://proceedings.mlr.press/v267/bick25a.html) | **ICML 2025** | Finds a retrieval bottleneck in some SSM language models and reports gains from attention on a small number of gather/aggregate roles and in hybrids. This supports testing a small selective-memory branch alongside a recurrent trunk instead of replacing the whole trunk. | Evidence is mechanistic analysis and language-model experiments. It does not imply our two targets require token retrieval, nor that hybrid gains transfer. |
| [Understanding and Improving Length Generalization in Recurrent Models](https://proceedings.mlr.press/v267/buitrago25a.html) | **ICML 2025** | Shows recurrent SSM and linear-attention models can enter unseen state distributions and degrade beyond training contexts. Test sequence length alone is not a stability/generalization guarantee. | Their long-context gap is not directly our 20k-row task, but recurrence-state coverage and long rollouts remain relevant, especially if training uses short chunks. |
| [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://proceedings.mlr.press/v267/sun25h.html) | **ICML 2025** | Treats the hidden state as a learned model updated online, linking recurrence and memory capacity. It is evidence that recurrent hidden-state design remains active research beyond a fixed vector. | Their TTT layer adapts its hidden model on test inputs and is studied at 125M–1.3B language-model scale. It is too costly and operationally distinct for the first experiment here; do not transfer its claims to supervised market prediction. |
| [Efficiently Modeling Long Sequences with Structured State Spaces (S4)](https://openreview.net/forum?id=uYLFoz1vlAC) | **ICLR 2022, not ICML** | Canonical structured SSM. Its convolutional training form and recurrent inference form motivate stable long-memory filters. | It is more intricate than needed for 20k rows and implementation stability matters. ICLR results are not ICML results. |
| [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://openreview.net/forum?id=tEYskw1VY2) | **COLM 2024, not ICML** | Input-dependent selective SSM parameters make information propagation/forgetting content-sensitive, a plausible fit for event-plus-quiet-state updates. | Sequential inference uses input-dependent recurrence; the specialized hardware-aware scan's throughput claim cannot be assumed on one CPU. COLM venue is distinct from ICML. |
| [KAN: Kolmogorov–Arnold Networks](https://proceedings.iclr.cc/paper_files/paper/2025/file/afaed89642ea100935e39d39a4da602c-Paper-Conference.pdf) | **ICLR 2025, not ICML** | Uses learnable univariate spline functions on edges rather than fixed activations on nodes. A small KAN-like module could test whether low-dimensional event/age/volume interactions benefit from a flexible nonlinear gate. | The theorem is not a guarantee of fewer parameters, interpretability, or better generalization. The original paper itself notes slower training. Spline knot range and boundary behavior create an extrapolation risk; no result here warrants putting KANs across 112 raw features or replacing the temporal state. |

The ICML 2020 and 2021 papers make the key terminology point: “attention
without a Transformer” is a real option. A causal kernel-memory recurrence is
an online associative memory, and a delta-rule fast-weight variant is a
distinct update rule. The ICML 2024 SSD paper also cautions against overly
sharp taxonomies: state-space and attention layers can be alternative matrix
factorizations of closely related causal operators.

## Candidate components, ranked by fit to this task

1. **Small causal fast-weight / linear-attention residual memory.** Preserve
   the complete predictor. At each row, project its causal representation
   `h_t` into query/key/value; update bounded recurrent state and return a
   gated residual. A simple positive-feature recurrence is
   `S_t = rho_t S_{t-1} + phi(k_t) v_t^T`,
   `z_t = rho_t z_{t-1} + phi(k_t)`,
   `r_t = S_t^T phi(q_t) / (z_t^T phi(q_t) + eps)`.
   A learned decay may depend on current observed inputs, but clamp it to a
   documented stable interval. Try a delta-rule correction only as a later
   isolated arm, with a bounded step size and explicit state renormalization.
   This offers content-addressed retrieval and selective forgetting while
   preserving updates in quiet periods. Values must be learned features or
   causal model states; never use previous true targets as deployed memory.

2. **LRU/selective-SSM residual branch.** A compact diagonal recurrence
   `s_t = A(x_t) s_{t-1} + B(x_t) h_t` supplies multiple timescales, followed
   by a small nonlinear projection to a residual. Start with an LRU-like fixed
   stable diagonal recurrence; add selective input-dependent `A/B` only if
   that arm earns the complexity. Use the current observed features (including
   event and quiet-row cues) in every update. This tests whether calibrated
   exponential memory is useful without hard freezing or a Transformer.

3. **KAN-like low-dimensional gate/head only.** Retain the GRU or best temporal
   branch and replace only its small feed-forward residual/gating map with
   cubic B-spline edge functions. Restrict this first test to a compact
   bottleneck and a few causal summary dimensions (learned state, current
   event, log age, and a small volume/activity embedding); do not assert that
   these summaries decode physical columns. A KAN advantage is plausible only
   if its local nonlinear response across these inputs is useful and stable
   across held-out components.

4. **One staged hybrid: stable recurrent state plus a small associative read.**
   If (1) and (2) each help or have complementary error signatures, combine an
   LRU/GRU state with one small fast-weight memory read. Keep KAN out of this
   first hybrid so gains remain attributable. If the hybrid helps, a second
   factorial test may replace only its residual MLP with a small KAN gate.
   Do not launch a three-way attention×SSM×KAN model as the first result; it
   makes capacity, memory, and optimization effects impossible to distinguish.

## Matched controls and bounded evaluation

Use a fixed, component-disjoint fit/evaluation panel and the exact same
training groups, causal feature construction, targets, clipping, absolute
target weights, optimizer/update budget, truncation policy, seeds, and
prediction-blend rule for all arms. The updated strong-baseline panels are
historically exposed and component-linked, so they are useful for debugging
but cannot serve as a promotion estimate. Freeze the evaluation cohort before
reading outcomes. Use more than one seed and report paired component scores,
not row-i.i.d. confidence intervals. Class-routing inputs are forbidden; class
may appear only in posthoc reporting. Report target-specific global WP and
weighted MSE, plus class/event/age slices as diagnostics. Keep clipping and
zero-target weights identical to the scorer; calibrating a gain on these
exposed validation outputs is not valid.

Minimum staged arms:

- Complete recurrent predictor, unchanged: control.
- Same trunk/head with extra projection parameters but no recurrence/memory:
  capacity control.
- Fast-weight attention branch alone, same residual parameter budget.
- LRU branch alone, matched state and residual-head budget.
- Small KAN in the output/gate map replacing an equal-budget MLP, with the
  temporal trunk unchanged.
- Only if the two temporal branches justify it, the fast-weight+LRU hybrid.

For every mechanism arm, include (a) matched parameter count and inference
multiply/latency accounting, (b) an unstructured/shuffled memory-update or
event-gate control preserving the marginal update rate, and (c) the unchanged
best model. A no-extra-trunk residual skip with equal parameters tests whether
any gain is only extra capacity. The strong-baseline report shows a same-width
sum/difference rotation is not evidence of new temporal information; use it
only as a representation/capacity control, not as the proposed mechanism.

## Required falsifiers and implementation hazards

**Causality and state reset.** For each sequence, compare row-wise predictions
to a chunked/parallel training path where available; perturb all future rows
and assert past predictions and hidden states do not change. Assert all state
is reset between `seq_ix` values. Verify warm-up rows update the model state
and predictions continue to be produced on every required row. On scored
transitions, compare event-gated and ungated arms using destination-row masks;
future-event offsets may be used only as oracle diagnostics. Shuffle event
times within sequence while preserving event count to test whether a gate uses
meaningful clock alignment rather than a generic lower update rate.

**State stability / retrieval capacity.** Log state norms, maximum absolute
value, denominator minima, decay distribution, nonfinite counts, and residual
scale through complete 20,000-row rollouts in FP32 and the intended deployment
precision. Repeat with constant inputs, an impulse followed by 19,999 quiet
rows, frequent alternating events, and a sequence boundary reset. For LRU/SSM,
bound/check `|lambda| < 1` (or the equivalent discretized transition spectral
radius), test long-rollout norm drift and impulse response at short and long
lags. For fast weights, test repeated-key overwrites, distinct-key collisions,
constant-key accumulation, denominator floor frequency and rescaling
equivalence; compare additive memory against bounded-decay and delta updates.
Reject any arm with unbounded norms, near-zero normalizers, long quiet-run
prediction drift, or its quality gain explained by state resets/precision
changes. The target sequences are 20k rows; test at that exact length and
longer synthetic rollouts even though extreme length generalization is not the
scoring objective.

**KAN spline coverage and extrapolation.** Freeze knot placement using fit
groups only. Report fraction of evaluation rows beyond each edge function's
outer knots, basis activation at the boundaries, output continuity at knots,
and finite-value rate. Evaluate matched in-range and covariate-tail groups;
then perturb the low-dimensional gate inputs just outside observed knot
ranges while holding the recurrent state fixed. Plot or tabulate change in
gate and final residual. Test the exact inference spline boundary convention
(clamp, linear extension, or zeroed basis) and ensure training/inference use
identical knots and precision. Fail if a few out-of-grid inputs saturate,
explode, or flip residuals materially; a spline gain that disappears outside
fit knots is interpolation, not robust structure. Compare against equal-size
MLP, fixed radial basis/spline features, and a simple monotone/additive spline
gate so the result is not credited to parameter count or smoothness alone.

**Scientific falsifier.** A proposed memory component is supported only if
paired component-level gains survive the equal-capacity head, shuffled-clock,
and unchanged-baseline comparisons for both targets or a predeclared target
tradeoff; the gain must not come only from one label-derived class or event
composition. If attention helps only on rare memory-retrieval slices but
hurts broad WP, reduce its gate/residual scale or reject it. If an LRU smooths
the event transitions and loses the high-frequency mode, reject or add
selectivity in a separately budgeted next test. If the KAN's apparent gain
depends on out-of-range extrapolation, reject. All claims remain local until a
fresh, genuinely component-disjoint evaluation supports them.

## Recommendation

Do not dismiss attention, SSMs, or KANs, but do not replace the strong complete
predictor based on literature reputation or the exposed .74–.77 diagnostic
scores. The most task-aligned first structural comparison is a **small causal
linear-attention/fast-weight residual branch versus a stable LRU residual
branch**, each against an equal-capacity feed-forward residual control. They
test different memories: content-addressed retrieval versus decaying
multi-timescale filtering. If one wins clean component-level evaluation,
combine it with the other only if the error slices indicate complementarity.
Then try a compact KAN solely as the local residual/gating nonlinearity, with
strict knot-extrapolation checks. This sequence retains the viable attention
direction while making every claimed benefit attributable, causal, stable,
and feasible under one-vCPU inference.
