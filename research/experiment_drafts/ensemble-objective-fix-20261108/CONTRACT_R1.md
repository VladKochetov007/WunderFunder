# Generalization repair: receiving contract, 8 November 2026

Authority: user explicitly requests diagnosing, implementing and launching fixes
on both existing clusters, seeking the strongest metric. This supersedes the
previous wave's no-new-fit restriction for this identified follow-up only.
Objective mode competition; exposed empirical development feedback (tierB with
no untouched holdout claim). No guaranteed local/public gain.

Use existing L4 lc-epanov_s@10.61.10.209 and B300 epanov_s@10.14.14.26 only.
ROOT alone owns each existing persistent file-RPC SSH stream, GPU admission and
process signals. No additional SSH, foreign stops, rental/billing, old deletion,
credential changes, competition submissions or external messages. GitHub draft
publication/progress remains authorized under SemyonEpanov/research-team.

## Evidence and rationale

All original eight L4 fits completed12E; GRU/local/Mamba peak around5–6E then
degrade. Early checkpoints exist with verified hashes. EMA plateaus. Original
B300 complement mixtures fell below reference by6E; labels remain near baseline.
Inherited clean TL3 reference WP0.6923774248169172 is not a new gain. Full-valid
has been repeatedly inspected. Old fixed-12E/30E recipes/results remain frozen;
an early-retired B300 attempt must be labelled partial, never a completed30E loss.

Two separately bounded experiments comprise this wave, maximum ten NEW fits:

1. L4 anti-overfit/representation experiment: exactly three arms x two NEW paired
   seeds101101/101102 =6 fits, initially one/GPU, leaving two GPUs for preparation
   and fixed checkpoint readouts. Targeted six-epoch schedule, matched numerical
   recipe except the explicitly declared treatment. Proposed arms: raw112/no
   dropout control, raw112/dropout.2, causal event211/dropout.2. GRU256x2, AdamW
   peak3e-4/decay.01/clip1/cosine5%warmup; full20k sequences, proposedB32/C1000.
   Freeze runnable normalized configs/source after source audit and representative
   native profiles, before admission. Prior source-preserving diagnostic-prefix
   and canonical resume fixes must be applied; all historical sources stay intact.
   Primary readout is fixed SWA of epochs3/4/5/6; secondary fixedepoch6. No
   validation-adaptive checkpoint/scheduler/stop. Grade each standalone and a
   declared fixed90%cleanTL3+10%candidate mixture; no learned valid blend weights.
   Maximum24 exported fixed endpoint grades plus the36 per-epoch full-valid
   monitoring observations (all development-exposed) and reused reference. These
   are not24 total looks; reused exactE6 predictions may avoid duplicate compute.
   If the existing
   feature pathway or training mask is unavailable, report it; never invent masks
   or bypass causality/ancestry. No Mamba rerun in this initial follow-up.

2. B300 ensemble-objective experiment: exactly two arms x same two new seeds =4
   fresh matched4E teachers, four2-GPU pairs. Preserve qualified T768/L24/H12,
   B128/A4/world2/context1024/chunk256, Muon3e-4/decay.01/clip1 and existing
   verified BF16/BF16x9 with FP32 masters/grads/loss/optimizer state. Control fits
   z to labels as before. Treatment is a declared package fitting the fixed
   0.9TL3+0.1z output with unscaled weighted MSE plus weighted-correlation loss,
   proposedequal.5/.5. It removes the old100x curvature scale and changes the
   target objective; do not attribute the package gain to only one factor.
   Correlation uses exact prescribed weights/clipping per target, centered stable
   FP32 moments, globally differentiable DDP aggregation per microbatch. Average
   microbatch correlation is an explicit surrogate, not full-dataset/global
   accumulation-window Pearson. Grade the unchanged complete official metric.
   Pretrial numerical amendment: prediction standard deviation is floored at
   0.01 inside the training-only correlation surrogate (variance floor1e-4),
   giving cov/(std_y*max(std_prediction,.01)). This avoids unstable cancellation
   gradients near constant predictions. The official evaluator and raw Pearson
   helper remain unchanged. The original no-floor CPU/Gloo near-constant failure
   is retained; the amended eight-case oracle passes the original tolerances.
   Add standalone z metric reporting using existing forwards and preserve fixed
   mixture reporting. Primary endpoint fixed4E mixture; separate z diagnostic.
   Two seeds cannot establish significance. No new student fits in this wave.

Original B300 jobs may be retired at a complete safe checkpoint only after
successor source, CPU/gradient/DDP/profile checks and resource plan are ready.
Retire only exact owned identities, preserve original checkpoints/results and
emit explicit allocation/partial-run receipts. Healthy old work continues during
preparation. No old fit is silently reconfigured or resumed.

## Invariants, grading and budgets

ADR0044: official earlier train10325 fitting members only, zero valid ancestry;
full later independent valid1873/3528273 scored rows. Whole sequences, no random
primary folds, no seq_ix time inference. Preprocessing/normalizers/reference
parents fit only fitting members. Split SHA
0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c.
Do not fabricate train is_scored if source lacks it. Keep the declared fitting
mask explicit. Every row advances causal state. Primary evaluation is exposed.

Finite host wave: at most4h wall from its first actual new profile, including
preparation/profile/fit/complete evaluation/collection; fitting admission closes
after3h, leaving1h reserve. Representative profile can be20–50updates, maximum
two identified attempts/cell with a concrete diagnosis; no blind retries. CPU
local support uses at most two aggregate threads; native limits come from live
inventory and measured contention. Preserve foreign services. No new datasets or
large feature cache until actual RAM/disk sizes are verified. Caps are maxima,
not targets; completed work frees GPUs. Independent roles perform no SSH/GPU.

Before GPU admission require exact runnable recipe/source/data/environment,
finite forward/backward/reload/profile at actual precision/shape, clean masks,
causality, finite normalization and state handling. B300 correlation requires
known-answer weighted metric tests, finite/constant/empty-mask behavior and actual
single-vs-DDP gradient aggregation check. SWA must bind exact parent hashes and
preserve normalizer/nonfloating state; no parameter-only warm resume masquerading
as complete model/optimizer/RNG continuation. Actual source/config/runtime/probe
identities must match at receiving. Independent critic never edits the grader.

Dashboard: selectable config, train-eval/full-valid WMSE and WP/target curves,
explicit masks, objective terms, LR, preclip norm/clipping, finite counters,
useful throughput/resources, checkpoint/source/parent identities. Save actual
PNG/SVG each epoch/readout. Fix observation transport errors before admitting
new work; stale data never counts as fresh progress. Durable detached workers,
bounded exact-owned cleanup and publication on terminal/30min remain required.

Decision: report every arm/seed/readout and paired delta. A complete matched
improvement and fixed-mixture contribution justify follow-up, not a public-score
forecast. Small, conflicting or negative evidence stays visible. Promotion is a
separate independent empirical review; successful tests or falling train loss
alone do not promote a model. Deliver checkpoints/predictions, hashes, plots,
bounded execution/exit receipts, comparative results and a resumable handoff.
