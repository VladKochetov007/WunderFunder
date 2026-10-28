# Compact Transformer: one-factor screens with live train/validation figures

October26 user authorization: eightB300 GPUs, efficient measured execution,
per-epoch validation and train/val graphs. This packet continues PR43 without
rerunning its exact controls. It does not modify any historical live recipe.

| Screen | New arm | New fits | Only changed setting versus PR43 control |
|---|---|---:|---|
| Regularization factors | dropout_only | F0/F2,2 | dropout0→0.1 |
| Regularization factors | decay_only | F0/F2,2 | weight_decay0.01→0.1 |
| Learning rate | lr_low | F0/F2,2 | peakLR1e-4→5e-5 |
| Learning rate | lr_high | F0/F2,2 | peakLR1e-4→2e-4 |

Each screen has four new fits. Total: eight fresh label-only teacher fits,
zero student fits; two reused control fits. The previous joint regularization
fits are additionally shown as historical context, never counted as new work.
Legacy runtime identifiers ctl_f0/reg_f0/ctl_f2/reg_f2 map to the explicit
scientific_arm field within each wave; ctl in a new wave does not mean baseline.

## Frozen comparison

Source0a70b4ab96c1810484544f382df4bcf0c08e8fd4; Transformer113495810 parameters,
hidden768/layers16/heads12/FF4/context1024/raw112; chunk512/accumulate8;
batch128/rank,2DDP ranks/global256; five complete epochs,1150 updates per fit.
Muon/auxiliaryAdamW, clip1, warmup5% of five-epoch updates then cosinefloor0.1.
BF16/BF16x9 with FP32 master weights/gradients/loss/optimizer states, as explicitly
authorized for the parent continuation. This is not strict IEEE-FP32 parity.
Same seeds201001(F0)/201002(F2), splits, fitting-only normalization and objective.
Each changed exact config gets a fresh12-update real-shape profile. Profile
weights are discarded; no scientific checkpoint is reused as initialization.

Whole-component inner holdouts,284/287 sequences, and the identical deterministic
64-sequence fitting probe are reused from PR43. All splits are development-exposed.
No label inputs, future-input access or changed sequence reset semantics.

## Decision and limitations

For each arm, choose one common epoch maximizing mean F0/F2 inner WP over all
five epochs; exact ties choose earlier. Report both paired fold deltas, each
target WP, complete curves and epoch5, not just selected means. A mean gain of
at least0.001 with positive deltas on both folds motivates further confirmation;
smaller/mixed effects remain uncertain rather than universal rejection.
One seed per fold does not establish training-variance significance. Reused
controls remain byte/hash-bound historical comparisons, not fresh confirmation.
The LR arm also changes LR-scaled decay; it tests the configured optimizer as a
whole. Five-epoch lowLR undertraining is not evidence about asymptotic quality.
Architecture/context/batch/optimizer changes are outside this bounded panel.

## Execution and resource estimate

Two sequential waves, four concurrent fits per wave, disjoint GPU pairs0–1,
2–3,4–5,6–7 and32 total pinned host CPUs. Reuse measured batch128: the prior
B256 cohort gained only~1% bottleneck throughput at roughly twice reservedVRAM.
Prior complete fits measured68–79minutes; allow~2.5–3hours plus fresh profile
startup/validation, conditional on current load. Approximately20–24deviceGPUh.
Own rank allocators cap90GiB after fresh exact-cap profiles; observed prior
reserved~80–81GiB/rank. The observer caps12GiB and uses only the firstGPU of each
pair. Admission reserves16GiB perGPU, with an additional live free-memory guard.
Host availableRAM and physical scratch must pass actual admission.
No swap, paid-resource changes or foreign-process cleanup. Each wave deadline
3hours; two waves only; no automatic scientific retry or indefinite queue.

`stage.py --mode prepare --root ROOT` freezes configs/plans and reused result
hashes; `--mode profile` runs two bounded resource cohorts. Root independently
inspects profiles and records exact approvals before fitting. Existing canonical
profile_panel.py, cuda_budget.py and telemetry.py are referenced from the
immutable parent operator deployment. The new run_panel.py copies the existing
controller with only its observer-budget placement corrected; no trainer is copied.
`execute.py --root ROOT` runs both admitted waves under native-cron one-shot
ownership, independently of SSH. `monitor.py/evaluate.py/metrics.py` preserve
canonical source/index/profile admission and add explicit frozen wave recipes.
Any stale/failed observer, invalid index/admission, nonzero fit exit or independent
metric verification failure stops only the owned active cohort and preserves outputs.
The live resource guard also yields our cohort if anyGPU has less than16GiB free.
This is shared-device monitoring, not hard isolation from sudden ARC allocations.

### Resource amendment before scientific startup

ARC-AGI restarted during initial100GiB-cap profiling. All eight resource profiles
passed, but subsequent admission failed because its GPU5 rank grew to~160GiB.
No scientific fit had started. The failed reviews and original profiles are retained.
R2 repeats exact profiles at90GiB cap, keeps the scientific configs/batch unchanged,
charges the observer only to its actual firstGPU, and rechecks allocation at each
wave start. With concurrent ARC load, current wall estimate is3–4hours, not the
initial idle-node2.5–3hours. Actual CPU/source/profile/data admission remains required.

## Curves and delivery

`publish.py --root ROOT` runs from native cron every5minutes. It independently
recomputes metrics from hash-verified prediction arrays before publication.
PNG/SVG figures have separate F0/F2 panels: fixed64 train-probe WMSE dashed,
full inner-validation WMSE solid; official-mask validation WP separately.
Both WMSE curves use the same frozen weights in model.eval and all rows>=99.
Different populations mean their absolute gap is not an unbiased generalization
gap estimate. Raw unsmoothed coordinates are exported as CSV/JSON with result,
checkpoint/config/split/source SHA256 and reuse markers. No invented/interpolated
epochs. Figures and selected-epoch summaries are committed to the experiment
branch, with one updated PR comment as SemyonEpanov. Weights/arrays remain on-node.

Receiver root: `/data/epanov_s/tabdata/runs/compact-factor-sweep-20261026/`.
Source, environment, exact plans/approvals, rank and hardware logs, safe weights,
epoch snapshots and prediction arrays remain separately preserved. Actual startup,
terminal status and publication URL must be recorded from receipts.
