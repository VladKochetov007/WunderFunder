# Modified clean-F0 TL3 counterpart: canonical readiness map

**Status:** readiness mapping only. This is not a recipe, preregistration,
launch command, fit authorization, or exact public-TL3 replica claim.

The canonical trainer can express most ordinary GRU teacher, fine-tune,
propensity, distillation-cache, and fixed-checkpoint stages. It cannot
currently express the archived tied-lead member as recorded, and the existing
propensity trainer does not meet the campaign's strict-FP32 requirement.
Historical W18 cache aliases also lack producer-bound row and source
identities. A later study therefore needs a separately reviewed
**modified clean-F0 counterpart** preregistration with all exact choices
frozen.

This map does not change the frozen rank-1 CP30 campaign or its TL3 mixture
weights. The 468-group F0 evaluation set must remain outside every fitted
learned ancestor, propensity/calibration fit, teacher target, normalization or
other learned preprocessing step, student, and selection decision.

## Fixed comparison boundaries

The current canonical F0 split identity is
**c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7**.
Its membership comparison records 10,535 fit-train groups, 1,405 fit-valid
groups, and 468 evaluation-valid groups. Every fitted stage must bind that
split and preserve ordered group/component identities.

Under the campaign split, **train+valid** means fit.train plus fit.valid
(10,535 + 1,405), excluding the 468 evaluation groups. A valid-only
fine-tune means exactly fit.valid (1,405 groups), never the F0 evaluation
groups. Do not use **--include-heldout-valid** with the campaign split. The
available **--no-eval-during-fit** and fixed checkpoint policy keep evaluation
out of fitting and checkpoint selection.

Keep the incumbent mixture unchanged for later recipe comparisons: 0.34
**w18final_distilled_s5@ep1**, 0.33 **fttall_tlead_s5@ep3**, and 0.33
**fttall_tlead_s6@ep3**. These are recorded public recipe labels, not approved
clean-fit run names or source identities. The local archived ZIP is 12,538,394
bytes with SHA-256
**d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569**; the
portal-uploaded bytes have not been hashed, so the local archive does not
prove byte identity with the submission.

The standing rank-1 comparison requires 30 paired fold/seed cells per arm,
with the existing 8/8/7/7 allocation over four exposed folds. A single F0
diagnostic is not CP30 completion and does not establish a winner. Preserve
the official **global_wp_clip2_v1** metric; do not fit ensemble weights or
tune against public leaderboard feedback.

## Canonical stage map and blockers

| Stage | Canonical components and available path | Readiness gap before a later run |
|---|---|---|
| Propensity model and cache | **scripts/train_propensity.py**, **src/wnn/propensity.py**, **scripts/cache_propensity.py**, training cache-identity and cache-loader components. Campaign split membership can isolate the 1,405 fit-valid groups from the 468 evaluation groups. | Current fitting uses CUDA FP16 autocast and derives input mean/std from the first shuffled training batch; this is not strict FP32. Cache metadata lacks split-manifest SHA and ordered group identity. Creative prop_f0 SHA-256 **9e92f26d1704f84f9170d746b372c2ba67f7fb611831865a1fc689d6b813b21c** differs from restored Main prop_f0 SHA-256 **b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7**; neither name proves clean provenance. A strict-FP32 path and source-qualified cache record are prerequisites. |
| Base teacher families | **scripts/train_rnn.py** delegates to modular training data, input, model, objective, optimizer, checkpoint, and provenance components. The model factory supports ordinary GRUs at the saved widths. | W18 metadata identifies 128×2 GRU fmu128 / frawW endpoints and saved args, but not bound historical source or clean membership. Rebuild or prove each source clean. Exact feature, loss, optimizer execution, EMA, schedule, endpoint, and family-map choices remain open. |
| Full/raw valid-only fine-tunes | The modular trainer has initialization, feature setup, objective, optimizer, and fixed-checkpoint components. Under the campaign split, **--train-source valid** selects fit.valid only. | Rebuild parents from clean ancestors, never production or F0-exposed weights. Valid-only means 1,405 groups. Exact clean parent, optimizer, schedule, loss, precision, endpoint, and preprocessing inheritance need later review. Saved optimizer arguments are not proof of runtime optimizer state. |
| Teacher composition and family targets | **scripts/build_distill_cache.py**, **src/wnn/distillation.py**, **src/wnn/distillation_cache.py**, and training cache-identity checks. | Archived W18 code uses a direct source teacher sum with coefficient total 0.8125 and no division by 0.8125. The two family caches are equal 0.5/0.5 means, each with total weight 1.0. Canonical variants are distinct: main_source is the direct weighted sum; main_normalized divides by the coefficient total; main_homogeneous is the total coefficient scale times one selected family mean. The canonical composition primitive can express a six-member direct sum when each source has a validated one-of-four family tag and matching split, checkpoint, and row bindings. The unresolved issue is historical family assignment and producer provenance (especially orgbase), not the availability of generic weighted-sum arithmetic. A later contract must select and preserve scale explicitly. |
| Distilled member | Canonical target/cache, initializer, model, provenance, and fixed-checkpoint components support preweighted targets and distillation. Saved args describe a 211-input GRU 256×2, initializer label fbc1k_s5@ep3, teacher label w18final_teacher, alpha 0.7, and bootstrap horizon 0. | These are configuration leads, not approved counterpart settings. Initializer, teacher graph, propensity ancestry, family targets, cache scale, feature assembly, and source/config bindings are unresolved or F0-exposed in production. Rebuild the full ancestor/cache graph cleanly. |
| Tied-lead members | Canonical model factory, CLI, and input assembly are reusable for ordinary architectures. | Archived member uses 263 inputs, including 52 lead channels and a shared instrument encoder. The current model factory has no tied-lead option or equivalent guaranteed route. A reviewed canonical adapter or an explicitly different architecture is required before later preregistration. Historical teacher ancestry fin256ft_s3/s4, alpha 0.5, and bootstrap horizon 8 are not a complete source-bound run card. |
| Fixed three-member mixture and evaluation | Source-qualified same-row outputs can be combined with fixed weights and evaluated through **src/wnn/gmetric.py** and **src/wnn/evaluation.py**. | Preserve 0.34/0.33/0.33; do not use blend-optimization utilities. Each member needs a common ordered-row identity. One F0 diagnostic is not the required 30 paired fold/seed cells per arm. |

### Keep the cache scales distinct

The archived W18 source teacher is a direct weighted sum whose coefficients
total 0.8125. Canonical main_source has this same no-normalization meaning;
main_normalized changes the scale by dividing by the total. The archived
family targets are separate equal-weight two-member means with total weight
1.0. Canonical main_homogeneous has a third meaning: it multiplies one
selected family mean by the total source coefficient. It is neither the
six-member W18 teacher nor a family average. No target-array producer binding
currently proves that historical code generated the local cache aliases.

## Minimum preparation before a bounded run can be proposed

1. Freeze the modified counterpart's stage graph and resolved settings in a
   separate preregistration. This readiness map chooses no epochs, optimizer,
   learning rates, seeds, replacement architecture, or target variant.
2. Bind every propensity model, teacher, initializer, cache, and preprocessing
   artifact to source/config/checkpoint hashes, canonical split identity,
   ordered fitting groups, and prediction row order. Rebuild any ancestor
   whose exclusion of all 468 evaluation groups cannot be proven.
3. Provide strict-FP32 propensity fitting and a source-qualified cache sidecar;
   implement and review tied-lead support or explicitly select a different
   architecture.
4. Resolve source-scale 0.8125 versus normalized/homogeneous cache semantics
   without conflating them with unit-sum family means. Bind all targets,
   family assignments, preprocessing, and ancestor identities.
5. Freeze the fixed mixture, same-row export contract, official metric,
   endpoint/checkpoint policy, resource budget, and complete CP30 paired-cell
   design. No public-score blend search is allowed.

## Evidence and source scope

This map reflects metadata and canonical source inspection, not new training
or inference. The [counterpart plan](../TL3_CLEAN_COUNTERPART_PLAN_20261018.md)
and [closure memo](../TL3_CLOSURE_NEXT_STEPS_20261018T1840Z.md) explain why
clean ancestry remains open. The [F0 membership comparison](../export_preparation/F0_MEMBERSHIP_COMPARISON.json)
binds the current ordered membership comparison. The
[W18 metadata findings](../tl3_w18_checkpoint_execution_r1/FINDINGS.md),
[fin256 findings](../tl3_fin256_metadata_execution_r1/FINDINGS.md),
[propensity history trace](../tl3_propensity_history_trace_r1/README.md),
[source closure](../tl3_source_closure_r1/README.md), and
[W18 producer trace](../tl3_producer_trace_r1/README.md) preserve historical
provenance limits.

Canonical implementation references:

- [Training component map](../../../../src/wnn/training/README.md),
  [membership selection](../../../../src/wnn/training/data_membership.py),
  [fit-only preprocessing](../../../../src/wnn/training/setup_preprocessing.py),
  and [training entry point](../../../../scripts/train_rnn.py).
- [Propensity trainer](../../../../scripts/train_propensity.py),
  [propensity cache writer](../../../../scripts/cache_propensity.py), and
  [propensity model](../../../../src/wnn/propensity.py).
- [Distillation cache builder](../../../../scripts/build_distill_cache.py),
  [combination semantics](../../../../src/wnn/distillation.py),
  [cache validation](../../../../src/wnn/distillation_cache.py), and
  [cache identity checks](../../../../src/wnn/training/cache_identity.py).
- [Model factory](../../../../src/wnn/training/models.py),
  [model CLI](../../../../src/wnn/training/cli_model.py),
  [official metric](../../../../src/wnn/gmetric.py), and
  [paired evaluation](../../../../src/wnn/evaluation.py).

Exact evidence and source pins are listed in CONFIG_GAPS.json. No training,
fit, cache generation, inference, remote action, or payload read was performed.
