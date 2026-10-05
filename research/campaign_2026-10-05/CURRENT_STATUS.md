# Resumed research status — October5

Confirmed public best: **0.6660**, seed4 complete Muon/chunk1000/stopped-bootstrap
combo. Current user milestone: **0.7000 honest expected public**, still unmet.
No fresh public package or score has been claimed from the following pilots.

- **E464:** eight resumed architecture heads, eight FP32 exports and sixteen
  complete-predictor comparisons landed. None beats the seed3 complete combo
  on the exposed holdout. Read `LOCAL_RESUME_REVIEW.md` and retained raw scores.
- **E466:** raw target-mechanics EDA plus stronger fitted/row-excluded probes
  completed. Hard event-freezing predictions hurts badly; difference-mode
  residuals dominate. Row exclusion is not full component independence.
- **E467:** ten full-sequence small models completed. Learned diagonal memory
  plus spline improves its near-capacity MLP control by +0.0386 on sixteen
  exposed diagnostic components; this is not an incumbent/public gain.
- **E468:** five fixed paired-codebook heads and all ten between-arm contrasts
  completed. None improves its historical GRU128 parent pair.
- **E469:** all twenty preregistered replication fits completed, giving thirty
  fits including the original seed. The learned-memory spline advantage
  stayed positive but shrank across seeds; see `sequence_components/THREE_SEED_REPORT.md`. Both retain every original arm, including negative pilot variants.
  Queue-launch receipt handling was repaired without killing/retraining the
  scientific worker; see `checks/E469_CONTROLLER_INCIDENT.md`.
- **Activation × optimizer:** E470 completed all sixteen pointwise-activation × AdamW/Muon
  fits and a preceding sixteen-arm real-GPU update probe. Peak0.384GB GPU
  and1.702GB worker RAM, exit0, noOOM. Official score reproduction and all thirty exploratory paired contrasts
  completed; see `activation_optimizer/REPORT.md`.
  Dropout and scheduling are separately planned to identify their effects.
- **E472 activation replication:** all 32 additional fits finished. Across three seeds (48 total fits), sine-minus-GELU is positive in all six seed/optimizer contrasts. AdamW gains are +0.0007 to +0.0028; Muon gains +0.0027 to +0.0088. Tanh Muon versus AdamW reverses at two seeds. All 90 conditional paired comparisons and official scores are retained in `activation_optimizer/THREE_SEED_ACTIVATION_REPORT.md`. No public prediction follows.
- **E471 recurrent activation:** six-arm bounded candidate perturbation passed independent CPU equation/state/gradient tests and real-data updates. Default CUDA parity failed before updates; the isolated precision diagnostic restored the original tolerances by disabling cuDNN TF32. The common-FP32 real update probe passed all six arms; all six common-FP32 fits finished with official scoring and seven primary/two backend diagnostic comparisons. Candidate-perturbation sign interacts with optimizer; both signs warrant seed replication. See `recurrent_activation/REPORT.md`.
- **E473:** all ten local-slope/scale/GELU-approximation control fits finished. GELU+0.1x exceeds GELU+sine at this seed, particularly with Muon; this challenges periodic curvature as the mechanism. All fourteen contrasts and four exact recipe-reproduction controls are retained in `activation_mechanism/REPORT.md`.
- **W384/T4:** v5 passed two real-device update probes then failed host-RAM
  admission before any full fit. Reviewed v6 was pushed but failed its external-mount/archive
  intersection guard after2.08s, before any probe/fit. Actual-bundle diagnosis
  completed: inventory entries were mistaken for extracted tar members.
  V7 preserves actual-collision rejection, passed an assembled bootstrap
  fixture and independent review, and was privately pushed. Provider status is RUNNING; device-probe
  results have not yet been collected. Fresh probes
  gate its two fits; no width result has been claimed. See dated Kaggle incident audit and v6
  independent review; known hazards must be fixed before launch.
- **TPU:** existing E463 preflight remains provider-QUEUED, without a device
  result. E465 distributed FP32/BF16 full fits remain gated on actual-device
  correctness/memory profiles. CPU sharding and AdamW checks passed.

As of01:36UTC, the private dual-T4 W384 v7 job is provider-RUNNING and the TPU E463 job remains QUEUED. Actual remote fit/probe outputs have not been collected, so neither is credited as a scientific result. Last provider quota observation:20.78GPUh and19.96TPUh remaining, refreshOctober10. Local charged GPU time is11.8161h through E471; unchanged24GPU-job-hour local block and80-device-GPUh campaign ceiling. The first E475 recurrent seed replication is running; all twelve new-seed recipes are registered without pruning.

E474 is ready for final source review after actual shuffled-fit ordering and fixed gradient/update telemetry were added. A directory-versus-file root-index bug was independently caught in the unlaunched GPU probe; it must be corrected before admission. E476 separates peak constant LR, warmup/cosine, and constant LR at the warmup/cosine trace's mean, across two activations and two optimizers. These are prepared studies, not completed experiments. E477 is preparing a matched-width mixed-activation readout and head-dropout factorial.

The full-data E427 volume-gate recovery remains blocked by genuine RAM feasibility: its24GiB worker plus4GiB support reserve exceeds the actual30decimalGB aggregate cap. The previous parent model already peaked near24GiB; shrinking admission metadata alone cannot solve this. The auditable recovery packet is retained in `volume_gate_recovery/`; no recovered E427 fit has run.

Persistent five-minute checker is enabled; notebook discovery/quota refresh
every fifteen minutes. Raw status/alert history are in the gitignored
`checks/monitor_runtime/`; dated incidents and unit/source/config receipts are
tracked. Each new successful Kaggle launch must enter the pinned registry.
Local aggregate cgroup now enforces12CPU/30decimalGB RAM hard/27GB soft/no
swap. One local GPU worker stays below5decimalGB observed VRAM, currently
about0.38GB for the small pilots (seed2 replication peaked0.502GB). Kaggle host/HBM guards remain intact.

Tables and EDA figures are visible in the running marimo continuation notebook.
Research inputs/results retain hashes, all fixed arms, paired component
intervals and historical exposure caveats. Generic reviewer ideas are treated
as hypotheses; independent CLI reviews and their fact checks are retained.
