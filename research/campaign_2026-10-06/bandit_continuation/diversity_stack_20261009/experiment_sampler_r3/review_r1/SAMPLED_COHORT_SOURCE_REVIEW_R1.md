# Sampled cohort R3 source review

The pending R3 packet and its admitted bootstrap/worker path pass this bounded source and consumer review. The review covers the exact pending packet, sealed admission generation, spec loading, and trainer CLI mapping. It does not approve the actor's scientific design, establish a scientific result, or claim an actual provider or GPU run.

The reviewed packet is `4c81fd04cba2e9f5626549ba19389e81cad8ec05746cfdba506794f87d64bf03` from commit `487b6ae4fb0c7fc40232b9aa97ef74c575675ebe`. Its embedded `push_allowed` and `conditional_fit_allowed` fields are both false, so this pending packet must not be pushed. The sealer creates a separate admitted candidate from the root payload, re-embeds the canonical admission bytes in the bootstrap, and regenerates the relevant asset, bootstrap, and packet hashes. The bootstrap and worker both check the exact source, cache, control, overlay, cohort, runtime, and authorization pins before the worker may fit.

`load_spec()` enforces the pinned cohort file and exact six IDs/waves. The worker maps those recipes to the actual R5 trainer CLI. The admitted bootstrap test and focused source/CLI tests passed: 3 tests in 0.38 seconds under the bounded shared support lock. The bootstrap test reaches the fold-0 mount boundary only after admission validation, then calls the extracted worker's actual `load_spec()` and `_validate_admission()`.

One nonblocking metadata caveat remains: the pinned historical `experiment_sampler_r2` candidate pool still labels the selected rows `pending_recipe_aware_runner`. The R3 worker now supplies the selected-row implementation, and R3 does not consume this metadata field. Preserve the historical pool hash and record R3 as its successor.

No provider mutation or model fitting occurred in this review. Root must seal a new admitted candidate and use that candidate's exact hashes. Full actual-device execution, outputs, and scientific grading remain outstanding.

Exact machine-readable pins and checks are in [SAMPLED_COHORT_SOURCE_REVIEW_R1.json](SAMPLED_COHORT_SOURCE_REVIEW_R1.json).
