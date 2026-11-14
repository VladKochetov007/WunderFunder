# Quality-first follow-up, November 14

User explicitly resumes research for stronger submissions and more hypotheses, invoking conduct-creative-research. [Contract](CONTRACT_R1.json): bounded eight-local-GPU-hour wave, six new screening fits, two final-data fits, up to two reserved confirmations within the same time cap. Existing train-only alpha .9 is the leading single endpoint; confirmation at seed 4 runs first.

Strict FP32, one local GPU worker, 30 GB RAM / 5 GB VRAM, no paid compute or automatic upload. Full official valid remains exposed. [Frontier](FRONTIER.md).

[Sources and independent ideation](SOURCES_R1.md), [frozen source hashes](SOURCE_PINS_R1.txt), `run_fit.sh` and `thermal_guard.sh` reproduce the current bounded local run. Kaggle preparation is a four-fit paired panel within the six-fit screen cap; source/asset portability precedes launch.

- [Independent recipe critic](CRITIC_R1.md): bootstrap shifts only eligible teacher targets; EMA changes checkpoint selection semantics. Runtime identity is separately verified in [RUNTIME_IDENTITY_R1.json](RUNTIME_IDENTITY_R1.json); final-data validation is fitted and cannot justify quality claims.
- [Four fresh-context cards](IDEAS_R1.md): increment distillation, supervised metric alignment, component-level blend support and capped group weighting. Proposals with zero-fit gates, not demonstrated gains or extra admitted fits. Group weighting requires actual defensible domains; source component IDs alone do not establish them.
- [Bootstrap coverage](BOOTSTRAP_COVERAGE_R1.json): eligible rows cover 64.4% of clipped-label mass in 200 sampled training sequences. The diagnostic omits since weights and does not establish a quality gain.
- [Fixed teacher/student blend](ZEROFIT_TEACHER_STUDENT_R1.json): exposed-valid score .691898 versus .691442/.691420 singles, using unchanged official metric and its independent reference. This tests contribution directly; residual correlation alone would miss the gain.
- [Conditional delivery](deliver_kd90.sh), [frozen admission rule](decide_kd90.py), [final-data launcher](run_final_fit.sh): only after the declared two-seed gate, train s3/s4 on train+valid, export fixed ep3 and test native single/equal-pair ZIPs. Root parity review and artifact handoff remain required afterward.
