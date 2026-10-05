# E478: bounded candidate scaling to the two-layer topology

Allsix fixed Muon fits and the independent CPU grader completed with no new OOM. The larger topology retains a positive point-estimate gain at allthree seeds: +0.0027025, +0.0106748 and +0.0044788. Two conditional component intervals exclude zero; the third crosses it. This supports further testing, without proving a public gain.

The modification is n=u+0.1*sin(pi*u), u=tanh(v), in both GRU layers. The alpha-zero custom-backend control shares the same FP32 policy, 61,058 parameters, base initialization, fit-only preprocessing, shuffled order and474 updates. The24 training and16 diagnostic test components are distinct but previously exposed. The submission is a much larger distilled/fine-tuned predictor; these are not its scores.

| Seed | Alpha | Global WP | t0 | t1 |
|---|---:|---:|---:|---:|
| 271828 | 0.0 | 0.4846771 | 0.5123695 | 0.4569847 |
| 271828 | 0.1 | 0.4873796 | 0.5162653 | 0.4584938 |
| 271829 | 0.0 | 0.4823535 | 0.4914203 | 0.4732866 |
| 271829 | 0.1 | 0.4930283 | 0.4949387 | 0.4911179 |
| 271830 | 0.0 | 0.5197423 | 0.5028921 | 0.5365925 |
| 271830 | 0.1 | 0.5242211 | 0.5064412 | 0.5420009 |

| Registered contrast | Global WP delta | Conditional95% interval |
|---|---:|---|
| seed271828:alpha_p10_minus_zero | +0.0027025 | [+0.0005091, +0.0059250] |
| seed271829:alpha_p10_minus_zero | +0.0106748 | [+0.0067304, +0.0150659] |
| seed271830:alpha_p10_minus_zero | +0.0044788 | [-0.0008030, +0.0084615] |

The grader independently reconstructs reset-after hidden-bias dynamics, carries [2,B,64] state across chunks, resets per sequence, checks all sealed sources/inputs, runtime/artifact receipts and raw finite selected targets/predictions. It recomputes external normalization from exactly the24 fit groups and verifies the checkpoint's internal buffers remain identity. A wrong pre-execution buffer assumption was caught and fixed before the scorer seal; no scientific fit was altered.

Allsix official scores and allthree paired comparisons are retained. Component intervals condition on fitted models, not seed or search uncertainty; there is no cross-seed pooled CI. Independent export reproduction passed for allsix scores and allthree 2,000-draw contrasts. See eda/six_v1/RESULT.json and eda/all_scaling_arms.png. Fit wall time285.9948s; independent CPU grade135.1611s. Fit peak0.398GB GPU/1.609GB RAM; CPU grade0.598GB RAM.
