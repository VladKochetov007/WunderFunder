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
  - E-GEOMETRY-QUICK-PROBE-20261018-R1
  - E-GEOMETRY-ADAPTER-PROBE-20261018-R2
  - E-GEOMETRY-ADAPTER-NATIVE-SERVING-20261018-R2
  - E-GEOMETRY-VARIATIONS-20261018-R3
  - E-GEOMETRY-CORRECTION-SCALE-20261018-R3S
  - E-GEOMETRY-ADAPTER-PUBLIC-RESULT-20261018-R2
evidence_paths:
  - research/campaign_2026-10-17/moment_geometry_r1/README.md
  - research/experiment_drafts/causal-geometry-discrete-b300/README.md
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/summary_r3.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_adapter_probe_r2_comparison.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_adapter_probe_r2_frozen_parent_check.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_adapter_probe_r2_recipe.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_adapter_probe_r2_native_q16x/README.md
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_adapter_probe_r2_native_q16x/RELEASE.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_adapter_probe_r2_native_q16x/PUBLIC_RESULT.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_geometry_variations_r3/README.md
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_geometry_variations_r3/geometry_r3_comparison.png
---

# Causal geometry and discrete state features on B300

## Question

Do causal mean/scale/arc cues and frequent-level occupancy/event/dwell features add predictive value to a strong recurrent endpoint, and do the continuous and discrete blocks help independently or interact? The test uses a fresh H512×1 GRU with a matched 60-channel suffix so the model's input width and parameter count remain fixed in all arms.

The H512 fresh comparison is the implemented B300 path. Root separately completed two local F0/seed3 warm-parent probes. R1 feature minus zero-control was−0.0000619 WP (conditional component-bootstrap95% interval [−0.0001217,−0.0000032]); both children finished about−0.0008 below their parent. Adaptive R2 froze all old parameters and trained only the new60 input-weight columns; its score was0.6804499 versus the fixed parent0.6812265, delta−0.0007766 (conditional component-bootstrap95% interval [−0.0013528,−0.0002791]). A separate receipt verifies bitwise preservation of16 parent state keys and211 old input columns, with15,360 nonzero new weights. These are two exact-recipe negative direction probes on one exposed fold/seed, not B300 cells, independent confirmation, or a family rejection. The R2 portable receipt lacks the exact local source revision; additional input guards are now included in the packet's optional adapter path.

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

The [earlier R3 H128×2 continuation summary](../../experiment_drafts/causal-geometry-discrete-b300/prior_evidence/summary_r3.json) reports three paired seeds and eight epochs on the same tiny exposed panel. The later root-owned [F0/seed3 block-fit and alpha-scale follow-up](../../experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_geometry_variations_r3/README.md) is a distinct R3: two matched adapter-only block fits both scored below the fixed parent, while an inference-only alpha=−0.50 alias was +0.0001956 versus parent with a conditional component-bootstrap 95% interval [−0.0000248,+0.0004214] crossing zero. It is adaptive on exposed F0; alpha aliases share the same R2 fit and do not add replications. The [comparison figure](../../experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_geometry_variations_r3/geometry_r3_comparison.png) shows scores relative to the fixed parent. These checks, R1 and R2 all reuse one F0/seed3 lineage and do not establish seed/fold robustness, teacher/student transfer, or a feature-family result.

The user-reported [public result receipt](../../experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_adapter_probe_r2_native_q16x/PUBLIC_RESULT.json) records that the separate R2 q16x ZIP scored 0.66428, below the then-reported team best 0.66832. This endpoint-specific result does not evaluate the R3 aliases and is not a public forecast for the B300 matrix.

The R2 adapted checkpoint's selected native q16x ZIP was subsequently scored over all 468 exposed F0 groups: WP 0.6804519697 versus Torch FP32 at 0.6804499040. This tiny endpoint precision difference does not change the negative adapter-versus-parent training result (0.6804499040 versus0.6812265132). Exact-ZIP Python 3.11 checks and an eight-group signed-zero-sensitive q16x native-cache replay passed; the predeclared FP32 native/Torch tolerance failed (287,989/318,416 values outside threshold), with `Activations::Fast` present and the cause unisolated. The callback's40.43-minute row-cost and 44.41-minute full-replay projections are local estimates, not measured target-host runtime. The [source-indexed receipts](../../experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_adapter_probe_r2_native_q16x/README.md) include the archive hash, tolerance corrections and user-reported public result. No ZIP, predictions, or weights are copied into this packet; the user submitted the separate R2 ZIP and observed no public quality improvement. No R3 variant has a public result.

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

Fresh-from-scratch H512×1 is the current B300 test. The zero-initialized append path and guarded adapter-only training mode are implemented. The single exposed F0/seed3 R2 adapter result is negative for that short recipe and cannot initialize F1–F3. Any broader warm comparison requires a matching fold-clean parent for every one of the30 fold/seed identities. Teacher-only, student-only and both-feature factorials plus geometry-feature distillation remain future work; the packet does not implement their cache/runner path.

A larger-GRU quality path should be profiled and preregistered as its own scale/context ladder. Factor teacher and student feature exposure into teacher-only, student-only, and both-feature arms, and compare label-only versus fold-clean teacher distillation under a matched student recipe. Teacher ancestry, feature priors and prediction caches must exclude each evaluated fold. Keep component-level grades and fixed mixtures even when one model is weaker standalone. The current packet does not implement these teacher-cache or factorial-distillation runners. Larger teachers may use DDP only if their measured profiles justify it; independent GRU cells stay one per card.

The branch also includes native feature serving, the single-member callback, a reference checker and packager support for an extra native file. Native build/ABI and corrected synthetic 20,000-row event/geometry/prediction/reset parity passed. A separate root-owned R2 package has actual full-exposed-F0 q16x scoring, Python 3.11 exact-ZIP checks and same-precision native-cache replay; it is exploratory and its exact training recipe is negative versus parent. FP32 native/Torch equivalence failed its preregistered tolerance. Complete target-host competition runtime remains unmeasured. Quantization stays out of the B300 training comparison. The B300 implementation is not profile-admitted or run.

## Registration and next decision

H-CAUSAL-GEOMETRY-DISCRETE-B300-20261018-R1 is a child of the moment-geometry and discrete-occupancy hypotheses. E-CAUSAL-GEOMETRY-DISCRETE-B300-20261018-R1 is a proposal with implemented launch code, zero B300 profiles, zero admitted scientific fits and zero optimizer steps. E-GEOMETRY-ADAPTER-NATIVE-SERVING-20261018-R2 records separate endpoint-serving evidence, not a B300 fit or positive training result. The [experiment draft catalogue](../../experiment_drafts/README.md) records GitHub draft PR #15 and its original publication source SHA; the live branch head is checked through GitHub.

A useful positive combined or interaction result would keep the representation active and direct a separate warm/teacher/student follow-up. A negative fresh baseline comparison narrows this H512 recipe but leaves warm continuation, distillation, fixed mixtures and larger-capacity descendants available. The [arXiv helper query](../../experiment_drafts/causal-geometry-discrete-b300/prior_evidence/README.md) returned no directly applicable primary method; this is an internal-Evidence proposal, not literature-backed proof.
