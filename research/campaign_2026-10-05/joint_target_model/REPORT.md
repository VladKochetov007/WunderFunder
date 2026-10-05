# E468: five frozen joint-target heads completed

All five fixed fits and all ten between-arm comparisons completed on the frozen historical fold0. Parent pair WP **0.67781291**; 468 sequences / 141 feature components / 808,863 officially scored rows. The train/evaluation component manifest and checkpoint hashes were checked. This is an exposed historical GRU128 pair, not the current 0.6660 public model.

| Arm | Official WP | Delta versus parent | Conditional 95% component interval |
|---|---:|---:|---|
| joint_tilted | 0.67719482 | -0.00061809 | [-0.00124868, +0.00004680] |
| joint_ordinary | 0.67722888 | -0.00058403 | [-0.00123791, +0.00009917] |
| shuffled_tilted | 0.67726278 | -0.00055013 | [-0.00109821, -0.00000424] |
| independent_tilted | 0.67645905 | -0.00135386 | [-0.00216552, -0.00054843] |
| regression_anchor | 0.67725811 | -0.00055480 | [-0.00119628, +0.00010925] |

No head improves on the parent in this fixed pass. Joint tilted versus ordinary readouts differ only slightly; the independent marginal head degrades more. This does not reject a jointly trained representation, a finer codebook, a different objective, or continuously updating posterior memory. Frozen features, coarse atoms, one fitting pass, and shared auxiliary loss are concrete limits.

The v1 scorer only emitted parent comparisons. A separate versioned v2 scorer retained every candidate and computed all ten between-arm contrasts, as required by preregistration. No weights were retrained or selected; raw v1 results remain. Component intervals are conditional on these fitted models and exclude training-seed/search uncertainty.

Raw fit/evaluation/checkpoint identities and resource receipts are retained in `RESULT.json`; local checkpoint files remain in `runs/frozen_f0_v1/`. Peak observed GPU memory was 432,013,312 bytes; all four attempts had exit0, verified teardown, and zero new worker/parent OOM events. Total charged probe, fit and both scores: 108.68811622298381 seconds.
