---
kind: idea
status: proposed
created_utc: 2026-10-18
updated_utc: 2026-10-18
hypothesis_ids:
  - H-CAUSAL-GEOMETRY-DISCRETE-B300-20261018-R1
  - H-MOMENT-GEOMETRY-20261017-R1
  - H-DISCRETE-OCCUPANCY-20261017-R1
experiment_ids:
  - E-CAUSAL-GEOMETRY-DISCRETE-B300-20261018-R1
evidence_paths:
  - research/campaign_2026-10-17/moment_geometry_r1/README.md
  - research/experiment_drafts/causal-geometry-discrete-b300/README.md
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/summary_r3.json
---

# Causal geometry and discrete state features on B300

## Question

Do causal mean/scale/arc cues and frequent-level occupancy/event/dwell features add predictive value to a strong recurrent endpoint, and do the continuous and discrete blocks help independently or interact? The test uses a fresh H512×1 GRU with a matched 60-channel suffix so the model's input width and parameter count remain fixed in all arms.

The H512 fresh comparison is the implemented path. A short F0 warm continuation that root is running under separate two-hour local authority is a one-seed direction probe; it does not provide a cell, initialize another fold, or change this proposed matrix.

## Mechanism and prediction

The earlier raw-input EDA found arcs and near-arcs in some sequence-level mean/variance summaries, with multiple exact value levels and long dwell periods in selected traces. A bounded or discrete-support mixture can produce an arc-like relationship. The observation does not establish a universal circle or physical market process.

The 60 causal channels reuse raw inputs and are an inductive bias, not new target information:

- Indices0–19: prefix mean[4], standard deviation[4], residual to the fitting-only arc[4], anchor occupancy[4], and normalized current value[4].
- Indices20–59: current dictionary indicators[16], prefix row occupancy[16], dominant-event fractions[4], and log dwell[4].

The prior is fit per fold/seed from that cell's fitting rows only. Evaluation groups are excluded. Per sequence, transforms reset at the sequence boundary and use only the current and earlier raw rows. Their statistics are accumulated in FP64 and the final 60 channels are cast to FP32. Neural model inputs, parameters, activations, states, losses, gradients and optimizer state are FP32 with autocast and TF32 disabled. The official grader remains unchanged.

The strongest alternative is that appended input channels perturb optimization or parameterization and their apparent contribution is seed/slice noise, not a stable state representation. The equal-width baseline and the separate 20/40 masks distinguish block contributions from the combined treatment; they do not prove a semantic interpretation.

## Prior evidence

The [registered investigation and visual EDA](../validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md) links the [EDA HTML](../../campaign_2026-10-17/moment_geometry_r1/eda.html) and [arc, support, prefix, and all-column plots](../../campaign_2026-10-17/moment_geometry_r1/scan_r1/README.md). It records the 128/128 sequence scan, all112 feature review and a visually plausible but unverified screenshot mapping. The [R1 moment result](../../campaign_2026-10-17/moment_geometry_r1/README.md) averaged+0.0008833 on a tiny exposed panel with opposing fold effects and conditional component intervals crossing zero.

The [R2 discrete pilot](../../campaign_2026-10-17/moment_geometry_r1/discrete_summary_r2.json) compared four arms at H32×1 and H128×2 on43 fitting/46 exposed evaluation sequences. Results varied by capacity; combinations were not reliably additive and fixed half-mixtures suggested conditional value.

The [R3 H128×2 continuation summary](../../experiment_drafts/causal-geometry-discrete-b300/prior_evidence/summary_r3.json) reports three paired seeds and eight epochs on the same tiny exposed panel. Mean deltas versus control were −0.019742 for moment/scale, +0.000501 for dictionary, and −0.017251 for all features. A fixed control+dictionary half-mixture was+0.007337 across those three seeds. The [small-panel loss and paired-delta plots](../../experiment_drafts/causal-geometry-discrete-b300/prior_evidence/README.md) are descriptive only. None of this establishes statistical significance, teacher/student transfer, serving quality, or public performance.

## Frozen B300 comparison

The [experiment packet](../../experiment_drafts/causal-geometry-discrete-b300/README.md) contains the code, recipe, explicit [30-pair/15-wave matrix](../../../scripts/geometry_b300/matrix.json), profile receipt builder, launch commands, prior evidence and PR recipe.

| Arm | Feature control | Visible indices | Interpretation |
|---|---|---:|---|
| Baseline | all | none | same 60-wide suffix, all channels zero |
| Moments | moments_only | 0–19 | continuous moment/arc/anchor/current block |
| Dictionary | dictionary_only | 20–59 | discrete identity/occupancy/event/dwell block |
| Combined | none | 0–59 | both blocks |

Every arm runs all30 crossed cells: F0/F1 with seeds1001–1008, F2/F3 with seeds1001–1007. Pair identity is explicit in matrix.json. Each wave assigns two pairs to eight cards, one fit per card: four arms for pair A on GPU slots0–3 and four arms for pair B on slots4–7. This is120 fresh fits in15 waves. The profile runs are separate engineering checks and do not count as cells.

All fits use the same H512×1 GRU, train+valid source under the existing component manifests, batch8, 2000-row chunks, three epochs, fixed epoch2, wmse, AdamW, OneCycle, LR0.001, weight decay0.01, gradient clip1 and deterministic seed-bound ordering. The matrix budget inherited from the frozen R4 plan is5,371,320 optimizer updates; the B300 card profiles and completed histories must validate that count. No B300 wall time is guessed. There is no DDP for these independent small GRUs.

## Estimands and uncertainty

Primary contrast: equal-fold mean of combined minus baseline. Secondary preregistered contrasts are moments minus baseline, dictionary minus baseline, and the interaction combined − moments − dictionary + baseline. The family is these four two-sided tests, adjusted with Holm. Compute each endpoint's unchanged official global_wp_clip2_v1 score on its full fixed outer evaluation groups, then pair by fold and seed.

For fit variation, bootstrap10,000 shared seed identities with replacement using the same multiplicities in all folds; renormalize within each fold over available cells and redraw if a fold receives zero weight. F2 and F3 have seven cells, so seed1008 is absent there. Report the percentile95% interval and an approximate two-sided centered-bootstrap p-value; apply Holm to the four p-values. Eight seed identities are few, folds are exposed and overlapping, and the bootstrap is approximate. Do not make a population-of-folds claim or kill a direction because an interval crosses zero.

Separately hold each trained endpoint fixed and resample source-qualified evaluation components within each fold, with the same component multiplicities across every arm and seed in that fold. Recompute the official metric and the equal-fold contrasts. This conditional interval concerns evaluation-unit variation only; it does not replace fitting-seed uncertainty.

A+0.001 absolute WP change is a useful effect reference for follow-up, not a rejection threshold. Preserve per-seed/per-fold deltas, all exclusions and failures, and fixed-ensemble endpoints separately. Never add a local delta to a public score.

## Quality scaling and serving path

Fresh-from-scratch H512×1 is the current test. The zero-initialized append path is implemented, but a warm comparison remains future work and requires a matching fold-clean parent for every one of the30 fold/seed identities. The single F0 seed3 parent in root's short local probe cannot initialize F1–F3.

A larger-GRU quality path should be profiled and preregistered as its own scale/context ladder. Factor teacher and student feature exposure into teacher-only, student-only, and both-feature arms, and compare label-only versus fold-clean teacher distillation under a matched student recipe. Teacher ancestry, feature priors and prediction caches must exclude each evaluated fold. Keep component-level grades and fixed mixtures even when one model is weaker standalone. The current packet does not implement these teacher-cache or factorial-distillation runners. Larger teachers may use DDP only if their measured profiles justify it; independent GRU cells stay one per card.

The branch also includes native feature serving, the single-member callback, a reference checker and packager support for an extra native file. Native build, parity against actual weights, full-context runtime and submission packaging remain pending. Quantization stays out of this research comparison. Do not claim the research-only implementation is submission ready until the native checks and serving measurements complete.

## Registration and next decision

H-CAUSAL-GEOMETRY-DISCRETE-B300-20261018-R1 is a child of the moment-geometry and discrete-occupancy hypotheses. E-CAUSAL-GEOMETRY-DISCRETE-B300-20261018-R1 is a proposal with implemented launch code, zero B300 profiles, zero admitted scientific fits and zero optimizer steps. The [published experiment draft catalogue](../../experiment_drafts/README.md) will record its GitHub URL and exact branch head after push.

A useful positive combined or interaction result would keep the representation active and direct a separate warm/teacher/student follow-up. A negative fresh baseline comparison narrows this H512 recipe but leaves warm continuation, distillation, fixed mixtures and larger-capacity descendants available. The [arXiv helper query](../../experiment_drafts/causal-geometry-discrete-b300/prior_evidence/README.md) returned no directly applicable primary method; this is an internal-Evidence proposal, not literature-backed proof.
