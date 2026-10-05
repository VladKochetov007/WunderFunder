# E477: mixed head activations and dropout

All12 full fits and independent CPU grading completed without a new OOM. The half-width GELU/LeakyReLU head changes global WP by −0.017395 and +0.006980 across the two seeds without dropout. The same-channel mixture changes it by −0.005748 and +0.005261. Thus head-mixture effects are seed-dependent at this scale; neither a universal advantage nor a whole-direction rejection is supported.

The mixture-by-dropout interaction is +0.006146 and +0.000753; the two conditional intervals differ. This is an interaction comparison, not necessarily improvement over the matched GELU head. Every arm uses H64 two-layer61058-parameter topology, the same24 fit/16 exposed test components,474 updates and seed-specific common initialization/order. The head mixture is fixed, not learned routing.

| Seed | Activation | Dropout | Global WP | t0 | t1 |
|---|---|---:|---:|---:|---:|
| 271828 | gelu | 0.0 | 0.4441839 | 0.5122195 | 0.3761483 |
| 271828 | gelu | 0.1 | 0.4458977 | 0.5128954 | 0.3789000 |
| 271828 | channel_split | 0.0 | 0.4267884 | 0.4932086 | 0.3603683 |
| 271828 | channel_split | 0.1 | 0.4341564 | 0.4956502 | 0.3726626 |
| 271828 | same_channel_mix | 0.0 | 0.4384356 | 0.5025278 | 0.3743434 |
| 271828 | same_channel_mix | 0.1 | 0.4462958 | 0.5050333 | 0.3875583 |
| 271829 | gelu | 0.0 | 0.4940204 | 0.4912452 | 0.4967955 |
| 271829 | gelu | 0.1 | 0.4926567 | 0.4912450 | 0.4940683 |
| 271829 | channel_split | 0.0 | 0.5010002 | 0.4975464 | 0.5044541 |
| 271829 | channel_split | 0.1 | 0.5003006 | 0.4965470 | 0.5040542 |
| 271829 | same_channel_mix | 0.0 | 0.4992811 | 0.4931462 | 0.5054160 |
| 271829 | same_channel_mix | 0.1 | 0.4986707 | 0.4916803 | 0.5056610 |

| Contrast | Family | Global WP delta | Conditional95% interval |
|---|---|---:|---|
| seed271828:split_p0 - gelu_p0 | preregistered_primary | -0.0173955 | [-0.0261913, -0.0069934] |
| seed271828:mix_p0 - gelu_p0 | preregistered_primary | -0.0057483 | [-0.0101956, +0.0008288] |
| seed271828:split_p0 - mix_p0 | preregistered_primary | -0.0116472 | [-0.0194860, -0.0042007] |
| seed271828:split_p01 - gelu_p01 | preregistered_primary | -0.0117413 | [-0.0179622, -0.0036124] |
| seed271828:mix_p01 - gelu_p01 | preregistered_primary | +0.0003981 | [-0.0043464, +0.0059373] |
| seed271828:split_p01 - mix_p01 | preregistered_primary | -0.0121394 | [-0.0197385, -0.0045415] |
| seed271828:gelu_dropout_effect_secondary | secondary_descriptive | +0.0017138 | [+0.0007762, +0.0026252] |
| seed271828:mixture_x_dropout | preregistered_primary | +0.0061463 | [+0.0033726, +0.0086735] |
| seed271828:split_x_dropout_secondary | secondary_descriptive | +0.0056542 | [+0.0026726, +0.0088163] |
| seed271829:split_p0 - gelu_p0 | preregistered_primary | +0.0069799 | [-0.0003356, +0.0145065] |
| seed271829:mix_p0 - gelu_p0 | preregistered_primary | +0.0052607 | [+0.0006349, +0.0082023] |
| seed271829:split_p0 - mix_p0 | preregistered_primary | +0.0017191 | [-0.0032188, +0.0082649] |
| seed271829:split_p01 - gelu_p01 | preregistered_primary | +0.0076439 | [+0.0011738, +0.0142600] |
| seed271829:mix_p01 - gelu_p01 | preregistered_primary | +0.0060140 | [+0.0013490, +0.0085825] |
| seed271829:split_p01 - mix_p01 | preregistered_primary | +0.0016299 | [-0.0031060, +0.0082203] |
| seed271829:gelu_dropout_effect_secondary | secondary_descriptive | -0.0013637 | [-0.0020969, -0.0005935] |
| seed271829:mixture_x_dropout | preregistered_primary | +0.0007533 | [-0.0001780, +0.0015272] |
| seed271829:split_x_dropout_secondary | secondary_descriptive | +0.0006640 | [-0.0004959, +0.0019034] |

All14 primary and4 secondary contrasts are retained. The joint interactions use identical numeric-sorted component draws across their four arms. Intervals condition on fitted models and do not include seed/search uncertainty or multiplicity correction. No public score is predicted.

The grader independently reconstructs the heads, recomputes training-only normalization and validates actual fit/runner/source/checkpoint receipts. Raw-finite checks precede clipping. Two grader failures (cohort-role comparison and export-path metadata) and unexecuted corrections remain in separate incident notes; no model fit was rerun or discarded. Successful grade attempt: e477-head-mixture-cpu-test-grade-v6b, 167.5934s. Raw result and paired export identity: eda/twelve_v6b/RESULT.json. Independent export reproduction is recorded separately when complete.
