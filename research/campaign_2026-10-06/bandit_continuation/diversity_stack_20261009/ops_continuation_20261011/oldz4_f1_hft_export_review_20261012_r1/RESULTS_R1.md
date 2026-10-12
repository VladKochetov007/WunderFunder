# F1 old-Z4 HFT export and fixed-blend check

The authorized HFT export finished successfully in 70.59 seconds. Its fixed
CP0 prediction is a finite `(1873, 20000, 2)` float32 NPY with SHA-256
`c43bfdeffa02f6dfe84a7b788b84d42854ea5000d9e1676992593184892ffaa8`. The
sidecar binds epoch 0, the same checkpoint, fold-1 split, and the canonical
row-group order. The export used local fp16 inference and stores float32
predictions; this is not a native Q8/Q4 grade.

The unchanged `blend_score_remote.py` scorer evaluated the 457 previously
exposed fold-1 groups with `global_wp_clip2_v1`, the cached official mask and
rows from 99 onward. It clips targets and predictions to ±2, weights by the
absolute clipped target magnitude, and averages the two target correlations.

| Endpoint | WP | T0 | T1 |
|---|---:|---:|---:|
| Old-Z4 student CP1 | 0.67038 | 0.66429 | 0.67646 |
| Old-Z4 HFT CP0 | 0.66818 | 0.66231 | 0.67405 |
| Strong alpha-0.9/e6 student CP5 | 0.67103 | 0.66617 | 0.67589 |
| Fixed 0.5 old student + 0.5 HFT | 0.67009 | 0.66401 | 0.67616 |
| Existing fixed 0.5 old student + 0.5 strong student | 0.67182 | 0.66640 | 0.67724 |
| Prespecified 0.25 old + 0.25 HFT + 0.50 strong | 0.67188 | 0.66644 | 0.67733 |

The only added mixture was frozen in
`ROOT_EXTRA_SHARED_HEAD_INTERACTION_CONTRACT_R1.json` before its score was
viewed. Its WP gain over the existing old/strong half blend is `+0.0000631`.
An 800-resample paired bootstrap over 149 complete components (seed 271828)
gives a 95% interval of `[-0.0004610, +0.0005604]` and `p(delta <= 0) = 0.394`.
This is weak one-fold evidence; it does not establish a repeatable improvement.

All endpoints exclude the same 457 groups during fitting, but fold 1 is
previously exposed and reused in model/blend selection, so these are
robustness results rather than a fresh holdout or cross-fold estimate. The HFT
is a descendant of the old student through the audited core crop; the old and
strong students also share teacher/source ancestry documented in the existing
fixed-half evidence. These are not independent lineages. The HFT resource-only
fit had 877 completed optimizer updates and three protected AMP-skipped updates;
the earlier strict profile gate failed. The fit/export and finite output do not
support a throughput-profile pass or clean-gradient claim. Export resource
receipt reports 4.879 GB worker peak RAM, 13.104 GB parent peak RAM, 1.417 GB
peak VRAM, and zero worker OOM/high/max event deltas.

The mixture scores use local predictions only. No package was built, no public
score was forecast, and no native serving/quantization result follows from
this artifact.
