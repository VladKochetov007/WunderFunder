# Autonomous local geometry continuation R3

[R3 plan](plan.json) extended the registered R2 recipe from two to eight
epochs, including the corresponding longer cosine schedule. It compared four
causal feature arms with an H128×2 GRU, three paired seeds, fixed final
checkpoints and FP32 training. All twelve endpoints completed 3,520 optimizer
updates; observed fit times were 25–32 seconds per endpoint.

## Result

| Arm | Mean pooled WP | Per-seed WP (3, 7, 11) | Paired delta vs control |
|---|---:|---|---:|
| Raw/event control | 0.376941 | 0.385845, 0.368473, 0.376505 | — |
| Causal moment/scale | 0.357199 | 0.356145, 0.367251, 0.348201 | −0.019742 |
| Discrete dictionary/occupancy/dwell | 0.377442 | 0.377361, 0.386455, 0.368510 | +0.000501 |
| All 60 added channels | 0.359690 | 0.370623, 0.361801, 0.346646 | −0.017251 |

The moment/scale and combined arms lost to the raw/event control on average.
The dictionary arm averaged almost exactly the control score, with one positive
and two negative seed deltas. Fold means were highly different: control was
0.317557 on fold 0 and 0.450009 on fold 1; the dictionary arm was 0.310083 and
0.467778. This points to substantial slice heterogeneity.

The plan also declared fixed half-weight mixtures. Control + dictionary scored
0.388042, 0.385103 and 0.379688 across the three seeds, a mean +0.007337 over
the matched control predictions. This is a conditional mixture observation,
not evidence that the standalone features improve a model or that a distilled
student benefits.

## Evidence limits and next step

R3 reused the R2 pool of 43 fitting sequences and 46 evaluation sequences
across 16 components and two already exposed evaluation slices. Three fit
seeds do not provide a precise training-variation estimate; no interval here
establishes statistical significance or transfer. Longer training also changed
the schedule horizon, so R3 is not a pure duration ablation. The small screen
does not reject the feature family or its conditional value. Its practical
finding is that concatenating all channels into this tiny H128×2 recipe was
unhelpful, while a predeclared raw/dictionary mixture remains worth checking
with a stronger model and a larger component-complete panel.

- [Frozen R3 plan](plan.json) records the question, paired seeds, controls,
  exposure and limits.
- [Admission receipt](admission.json) binds the unchanged R2 source/data and
  records the resource bounds before fitting.
- [Endpoint results](r3_h128_l2_results.json), [progress trace](r3_h128_l2_progress.jsonl),
  and [artifact hashes](r3_h128_l2_artifacts.json) preserve all twelve fits.
- [Pooled and fold comparisons](summary_r3.json), [paired deltas](paired_deltas_r3.png),
  and [training losses](loss_r3.png) provide the generated report and plots.

The separate Kaggle/team work remains outside this local experiment. R4 is
being prepared as a from-scratch comparison of the strongest locally screened
H512×1 GRU with and without the geometry/discrete channels, using a much larger
component-complete fit and evaluation panel before any submission packaging.
