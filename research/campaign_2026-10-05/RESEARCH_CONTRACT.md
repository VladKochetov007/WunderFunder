# October 5 resumed creative research

The user explicitly resumed local GPU, Kaggle GPU and TPU experiments. The
October 4 user pause is archived in `pause_archive/`; the new controlled runner
is reopened. Older queues and their incident barriers stay blocked. Existing
frozen numerical recipes, graders, resource guards and failure records remain
unchanged. Current public anchor is the fixed seed-4 complete combo, **0.6660**.
This is a milestone, not the end of the authorized fundamental research.
The user subsequently set the new milestone to **0.7000 honest expected public
score**, a +0.0340 gap from the confirmed predictor. This changes the research
target, not the accepted 80-device-GPU-hour budget or resource ceilings.

Method priority now includes a new raw-data target-mechanics EDA by Sol 6.1
at medium effort, blind model ideation, and independent OpenCode/Freebuff
reviews across available model families and reasoning variants. Read
`oct5_ideas.md` as a hypothesis index, not as raw evidence. Analyze joint target
support, run/event semantics, observable conditional distributions and error
modes before designing heads. A coordinate rotation or generic sign/magnitude
factorization is not automatically a new predictive mechanism. The weighted
metric and causal information boundary determine the appropriate readout.
Fitted-validation 0.70 is not evidence of expected public 0.70. Forecasts must
account for public transfer, seed and selection uncertainty; no declaration of
the new milestone follows from a local point score.

The question is which causal representation, training or deployment changes
improve the complete predictor. Primary metric is `global_wp_clip2_v1`.
Fixed complete-predictor comparisons, component-bootstrap uncertainty, seed
replication, and public confirmation are distinct evidence levels. Existing
validation panels are exposed; fitted validation is descriptive. Class-specific
correlations cannot be added to predict pooled gains.

First recover prepared work: eight E464 fixed architecture heads, their FP32
exports and paired comparisons; two matched W256/W384 fold-1 fits on T4;
three aligned/stale/sham volume-gated continuations after review. On TPU,
recover the already-submitted E463 preflight, then production-shape profiling
and the E465 FP32/BF16 comparison when actual-device gates pass. This is a
provisional 15-fit block including two conditional TPU fits, not completed
experiments. Smokes, hardware checks and failed launches are counted separately.
Retain a creative branch on causal volume innovation or an origin-aligned
privileged-information teacher; commit its falsifier before training.

Local limits remain 12 aggregate CPUs, 30 decimal GB aggregate RAM (27 GB soft limit), zero worker swap,
5 decimal GB observed VRAM and one local GPU worker. Allocation starts below
these limits with resource sampling and fail-closed guards. Private Kaggle
allocations target 80% memory use and stop before 90%; full-shape probes must
establish feasibility. Refresh provider quota and actual accounting before
launches. The existing 80 device-GPU-hour ceiling and 24-hour local block
remain in force; no new budget is inferred. Keep a reproduction/recovery
reserve. TPU quota is separate. No paid provisioning or competition upload.

Every attempt keeps code, data, checkpoint and output identity, exact command,
exit status, actual resource peaks and wall/device cost. Append scientific
results to the existing MD/JSONL records; use marimo for the human view and
save EDA images under this campaign's `eda/` directories. Diagnose crashes and
preserve them before a fresh retry. Stop affected work for causal/leakage,
numerical, resource or evaluator failures; historical negatives alone do not
veto a nuanced new experiment.

JAX sharding and memory profiling are implementation tools, not evidence of
distributed-gradient correctness. See the [official sharding guide](https://docs.jax.dev/en/latest/201/sharding.html)
and [profiling guide](https://docs.jax.dev/en/latest/201/profiling.html).
An origin-aligned training-only future teacher has relevant precedent in
[Karlsson et al.](https://proceedings.mlr.press/v151/k-a-karlsson22a.html);
its narrow theoretical assumptions do not prove benefit for this objective.

Task monitoring is mandatory: a persistent five-minute systemd user timer
observes local workers and active Kaggle jobs, refreshes notebook discovery and
GPU/TPU quota every fifteen minutes, and records failure/stale-heartbeat alerts.
It never launches, cancels or retries experiments. User lowered the RAM ceiling
to 30 GB; the actual parent cgroup now enforces that cap with zero worker swap.
