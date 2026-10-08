# H272 fixed epoch-1 direct-label readout result

The multiplicative readout is a positive discovery screen; pursue a bounded independently trained deployment follow-up. It is not yet an expected public-best claim. Previously exposed468groups/141components, one seed3; parent and candidate fitting exclude these groups. No checkpoint or weight selection occurred.

| Readout | Standalone WP | Delta vs parent | Primary delta vs trained hidden-only |
|---|---:|---:|---:|
| hidden_only | 0.6800460241 | -0.0015721181 | +0.0000000000 |
| additive | 0.6798689318 | -0.0017492104 | -0.0001770923 |
| multiplicative | 0.6817322281 | +0.0001140860 | +0.0016862041 |
| conditional_residual | 0.6791437384 | -0.0024744037 | -0.0009022856 |

Parent WP0.681618142181. Multiplicative fixed weights0.1/0.25/0.5/1 all improve it; deltas+0.000029912/+0.000065459/+0.000102804/+0.000114086. These are development measurements, not additions to public0.6660. Standalone parent comparison CI95[-0.000286121,+0.000531892] describes uncertainty; it is not an automatic rejection rule. Primary multiplicative-vs-hidden delta+0.001686204; hidden-only itself is weaker than the original parent.

Fit1933.0165s summed epoch time; controller1941.2489s, 29,840updates/no skips. Export127.3492s/controller131.7702s, exact anchorparity0, all1873groups and evaluation-onlymask808,863rows. No new OOM/kill; fit worker cache-reclaim max events537536 under6.5GB and export under4GB. Parent16tensors bitwise unchanged;24readouttensors finite.

First grade failed before scoring because it requested the stored-NPZ loader from the alignment helper. Original grader, authorization and failure are preserved. V2 changes only loader resolution to the already pinned official scorer; independent source review and exact-shape NPZ fixture passed. No training, metric, masks, bootstrap or comparison change. Direct official reference calculation agrees with parent and multiplicative scores within1e-12.

Next: freeze an all-data/matched-development multiplicative-head recipe with our independently trained strongest suitable parent and verify CPU/native serving parity. Keep faithful heterogeneous teachers→Muon parents→fresh seed9/99 students as leading lineage. Do not use teammate code/weights, choose a blend retrospectively or claim teammate/publicbest parity.

[All fixed comparisons](H272_FIXED_EP1_COMPARISON_RESULT_20261008_R2.json), [independent score calculation](H272_ROOT_DIRECT_REFERENCE_SCORE_CHECK_20261008_R1.json), [fit review](reviews/H272_ROOT_FIXED_EP1_TERMINAL_REVIEW_20261008_R1.json), [grader repair review](H272_GRADE_R1_INDEPENDENT_SOURCE_REVIEW_20261008_R1.json).
