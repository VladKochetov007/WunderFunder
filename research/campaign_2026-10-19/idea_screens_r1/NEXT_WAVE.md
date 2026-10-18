# Next bounded exploratory question: information versus estimation

User authority for autonomous EDA/simple models remains active. This is the next fixed proposal after R1 outcomes, not an executed result. The current six-component evaluation panel is already exposed and remains discovery-only. Do not call this prospective confirmation or alter the original R1 result.

## Single question

Do i1 and the extra history features look harmful because the small fixed-ridge fit estimates too many weak coefficients? Does EMA's signal come from the signed part, absolute-innovation part, or both?

## Fixed scope before any R2 result

- Reuse the exact R1 source-bound remote sample; no new Parquet scan, component replacement, training GPU, historical weights or public submission. Preserve stride40 after full causal histories.
- Six fixed views: raw112; i0-only52; raw+lags1/16/128; raw+signed-delta EMA; raw+absolute-innovation EMA; raw+both EMA parts. Keep the R1 half-life32 and exactly its current definition of innovation (delta minus the previous signed EMA). No new pole/window search.
- Shared ridge alpha selected from **[1,100,10000]** using only three deterministic component folds inside the14fitting components, grouping seed19020. Normalize on each inner training partition only. Selection criterion is pooled canonical mean WP across held fitting components on their original masks; couple the two targets through one alpha. Tie chooses the larger alpha. Refit the selected alpha on all fit rows, then report the already-exposed outer panel once per view.
- Also retain alpha1 as a fixed diagnostic baseline; report the full inner score table and the selected setting. Do not use outer evaluation to change settings. Count all candidates (6views×3alphas×3folds, plus selected refits) as exploratory analytic fits, not independent replications.
- Report both targets, six component moment tables, fixed-prediction drop-one concentration, coefficient norms, fit/eval counts, elapsedCPU/RSS and all failures. Primary comparison remains each representation versus its own raw/i0 counterpart with the same selection protocol. Do not infer gradient conflict, sparse-factor identification or recurrent-state staleness.

## Execution envelope

Code/review/tiny tests first in a separate R2 packet; no copy of the full recurrent trainer. At most one serial CPU50%/1.5GB/no-swap/600s job on B, CUDA hidden, physical scratch, fresh cgroup/headroom check. Reuse deterministic normal-equation sufficient quantities if useful; no queue manager. Existing F3/support always keeps its resources and ownership. Stop rather than enlarge caps or silently shrink/replace the panel after an error. Preserve R1 and any failed R2 output.

After this comparison, prefer a separately frozen larger component panel to confirm an allocation decision. Neither R1 nor this same-panel R2 can establish a full-recipe or public gain. Remaining cards stay explicitly untested until their defining mechanism is instantiated.
