# New hypotheses — controller session 2026-10-08

Registered after reflection: same-recipe seed pairs gave ~0 public (H230 pair 0.6653 = single), parent+head-FT finishing gave +0.0013 public, pair runtime 61m22s at the 60-min limit, so servable diversity must come from cheap heads, small boosters or faster engines. User asked for blend analysis (marimo), small MLP/NN-boosting stackers and new framings of old ideas.

## H-274 — prediction_panel_meta_learner
Family `ensembling_stacking`; parents H-206, H-264.

On the fold0 excluded-468 panel, an honestly cross-validated stacker (NNLS/ridge/small MLP, fitted by dependence-component CV) over existing fold0-clean model predictions finds members with positive conditional value that fixed equal blends missed.

Cheapest test / falsifier: Component-CV stacked WP minus best single WP; paired component bootstrap. Falsifier: no stacker beats best single+its best fixed 2-way blend under component CV.

## H-275 — single_trunk_multihead_stacker
Family `serving_compatible_ensembling`; parents H-272, H-056, H-274.

A single served GRU trunk with several cheap heads (parent, GCorr head-FT, multiplicative/hidden readouts, family heads) combined by a learned causal per-row gate or tiny MLP beats the fixed 0.5 parent/head combo at ~zero extra serving cost.

Cheapest test / falsifier: Servable: heads share one trunk (combo ran 37 min vs 34 single). Falsifier: gate/MLP head mix <= fixed-half mix on component CV.

## H-276 — stagewise_residual_network
Family `boosting`; parents H-182, H-206.

NN residual boosting: a small fast causal model (GRU64 or MLP on frozen-trunk hidden state + raw features) fitted stagewise with shrinkage to the residual of a frozen strong trunk adds signal the trunk misses, within the 60-min serving budget.

Cheapest test / falsifier: Measure residual-target WP gain at shrinkage {0.25,0.5,1}; runtime cost of booster. Falsifier: booster residual correlation ~0 / no endpoint gain.

## H-277 — student_headft_combo
Family `endpoint_finishing_x_distillation`; parents H-230, H-272, H-056.

Applying the parent+head-FT equal-mix finishing (+0.0013 public on combo parents, E-457) to distilled students or faithful Muon students yields an endpoint above either alone; finishing and distillation interact rather than overlap.

Cheapest test / falsifier: Local: GCorr head-FT on fold0-clean student, fixed half mix vs student; public: all-data student combo ZIP. Falsifier: head-FT gain on students <= 0 locally.

## H-278 — raw_vs_clipped_family_targets
Family `distillation_target_treatment`; parents H-230.

Training family targets unclipped (raw exported teacher predictions, as in the reference) instead of clamped to +-2 changes student quality; this is the one concrete objective discrepancy vs the reference recipe.

Cheapest test / falsifier: Matched pair same seed/order/cache; only family-target clipping differs. Falsifier: |delta| below seed noise on component CV and no public difference.

## H-279 — organizer_baseline_teacher_slot
Family `teacher_diversity_x_distillation`; parents H-230, H-206.

Replacing our two own raw GRU128 teachers (2x0.03125) with the organizer baseline at 0.0625, as in the reference teacher mix, improves the distilled student because the baseline is a structurally different (more diverse) teacher.

Cheapest test / caveat: Build main-target variant with baseline slot, run A00 recipe with only the cache variant changed; compare on fold0 panel with provenance caveat; public test via all-data student if positive. Needs cache variant built from cache/pred_baseline_{train,valid}.npy; verify baseline fitting provenance before any local held-out claim (organizer may have fit on valid).

## H-280 — diverse_architecture_blend_or_distill (planned; user idea)
Family `architecture_diversity_ensembling`; parents H-206, H-218, H-264, H-274, H-275, H-276, H-230.

A deliberately diverse architecture set (GRU256 trunk plus cheap structurally different members: LSTM/GRU128 on raw or alternative input encodings, small causal TCN, block-memory attention, numerical-embedding encoders), trained to be complementary and blended with component-CV weights or distilled into one served student, beats any same-recipe seed ensemble under the 60-min / 1-vCPU budget.

Prior evidence: Individually weaker members (H218 attention, raw TCN, H232 GRU384 scratch, E540 PLE) and FIXED equal blends lowered the exposed fold0 reference; same-recipe seed pairs gave ~+0.0004 public (H230 pair 0.6653 vs single 0.6649). Untested: learned/CV weights, complementarity-targeted training, diverse-ensemble distillation into the served GRU.

Plan:
- 1. Blend-lab (H-274) component-CV weights over existing fold0-clean diverse members -> conditional value per architecture, with runtime cost per member.
- 2. If any member adds value: train complementary versions (residual/negative-correlation objective vs frozen GRU256 trunk, H-276) at cheap serving size; measure fold0 panel + runtime.
- 3. Serving routes: (a) <=1 extra small trunk within ~20 min headroom (f16 engine ~15% faster); (b) distill the diverse ensemble into the GRU256 student via the teacher cache (H230 family) so serving cost is unchanged.
- 4. Confirm on fold1 before any public package; human upload gate.
