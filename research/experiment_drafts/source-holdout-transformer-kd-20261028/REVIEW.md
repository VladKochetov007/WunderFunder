# Source review, 28 October 2026

Review covered the new source split, existing data/model/cache/export contracts,
normalized distillation loss and telemetry, packet commands and final comparison.
Separate implementation and review roles inspected source without importing models
or executing tests. Agreement is source-review evidence, not numerical acceptance.

Findings addressed before publication:

1. Canonical additive KD doubles the combined objective scale at weight1. The new
   explicit `weighted_mean` option divides label+KD by `1 + kd_weight`, preserves
   hints, and updates actual loss, window/epoch telemetry and profile totals.
   Legacy omission keeps old config hashes and additive arithmetic. Prepared tests
   cover mixed-target gradients, weight0, hints and reporting.
2. Historical graph/index coverage omits train528. A separate full-source feature
   graph and thin-index producer now includes every physical group, with no manual
   cache aliases or dependency on the unmerged #46 implementation. Missing or
   corrupt data fail closed. Transitive valid-linked train components are excluded.
3. Empty valid fitting membership is supported by canonical SequenceData and
   fit-only normalization. An early packet membership reader expected an absent
   `evaluation.train` field; it was corrected to accept the new valid-only schema.
4. Cache compatibility alone is insufficient: the packet binds each KD run to its
   designated same-seed teacher, config, source, exact FINAL5 checkpoint and fit
   members. Old #43 weights and compatible foreign caches cannot substitute.
5. A finite resource profile is not a numerical correctness check. An explicit
   receiver `accept` stage now requires targeted correctness checks with no skips
   before per-cell profiles/admission. Legacy precision checks run in a fresh
   process with cuBLAS emulation disabled; actual production profiles run separately
   with the frozen BF16/BF16x9 environment. No local execution occurred.
6. Report bootstrapping uses canonical centered component moments, checks shared
   target statistics, preserves all six endpoints and labels its interval as
   conditional on fixed predictions. The library's `n_sequences` field counts
   input component aggregates here; `bootstrap_input_grain` makes that explicit.

Remaining limits: the feature linkage graph is approximate; source separation is
not a proven chronological boundary; full valid is already exposed to research;
two seeds are not significance. Local parquet immutability uses size/mtime/footer
identity; portable copies require the shared full-byte registry. Real B300
correctness, full-data coverage, memory, throughput, final metrics and CPU serving
feasibility are all pending. No model-quality result follows from this review.
