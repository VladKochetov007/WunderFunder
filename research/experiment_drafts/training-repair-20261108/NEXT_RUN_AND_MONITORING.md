# Next-run handoff and online interpretation

Scope: the five-point code repair, begun 8 November 2026. Software tests do not
establish a score gain, CUDA throughput, production memory use or a fresh holdout.
No profile, fitting continuation, SSH session or submission was started by this repair.
Historical source archives, recipes, checkpoints and scores remain unchanged.

## Exact starting recipes

| Runtime | Model / precision | Epochs | Per-rank batch | Train chunk / accumulation | Nominal gradient segment / optimizer window |
|---|---|---:|---:|---|---|
| L4 | GRU256x2, FP32, AdamW | 6 | 32, world1 | 1000 / 1 | 1000 / 1000 rows |
| B300 | Transformer768x24, BF16/BF16x9 forward, FP32 masters, Muon | 4 | 128, world2 | 256 / 4 | 256 / 1024 rows |

Both start from the actual completed control recipes, not an inferred hyperparameter
list. The new mode is `standalone_labels_wmse_v2`: only label WMSE, clipped-target
absolute weights, all-fit normalization, no teacher/initializer/reference objective.
B primary `valid/wp` is the model's raw z. Historical B v1 `valid/wp` denotes the
fixed TL3 mixture; compare like outputs. Offline fixed-reference readouts remain
separate, explicit operations. No dataset-global Pearson loss was implemented.

The B profile-only 512/A2 and 1024/A1 candidates preserve a 1024-row nominal
optimizer window and the padded update count. A world1/B256 engineering candidate
preserves world2/B128's global sequence batch. These are unprofiled possibilities,
not speedups or admitted fits. Do not change the GRU's pinned 1000-row recipe to
match the Transformer's window. Transformer 1024 denotes a per-layer attention
window, not a 1024-row backpropagation guarantee or a hard raw receptive-field cap.
GRU state persists over each complete sequence; `model.context_rows` is not its
forward memory limit. Detached state carried across parameter updates is the
existing TBPTT approximation; no cache reset was introduced.

## Before any future GPU allocation

1. Freeze source/config/data/split/normalizer identities and an owned finite
   allocation. Recheck actual host occupancy, CPU/RAM, output filesystem and other
   users. Sum checkpoint and simultaneous CPU-snapshot reserves across workers;
   one fit's free-space check is not a multi-worker reservation.
2. Use the existing receiver/profile path on the exact mode, batch, precision,
   context, grouping and logging configuration. Exercise at least 50 updates:
   sampled updates 1 and 50 must appear, together with actual applied LR, finite
   optimizer state and a successful canonical full-state reload. Measure warm
   throughput and full phase times (data, fitting, evaluation, checkpoint).
3. Compare diagnostics enabled/disabled on a bounded engineering fixture. CPU
   parity is established by tests; GPU numerical agreement and logging overhead
   still need measurement. Weight snapshots are CPU copies taken only at the
   sampled updates and add host memory/transfer cost.
4. Only then freeze a 4-6-fit scientific screen with matched fitting seeds and
   fixed endpoints. First compare standalone models against the same output and
   protocol. Prioritize existing labels control versus one optimizer/LR or
   gradient-horizon change; do not mix several interventions and attribute a gain
   to one. A fixed blend is a separately declared secondary endpoint.

## Online dashboard: observations and actions

| Observation | Healthy contract / target | Action when it fails |
|---|---|---|
| Primary coverage | Train10325 only; complete valid1873 / 3528273 official scored rows, both targets | Stop grading/promoting; diagnose identity, membership and mask |
| Evaluation ancestry | No valid fitting in transforms, teacher, initializer or calibration; valid is development-exposed | Reject the comparison; a new split/agent does not restore fresh data |
| Numerics | Zero NaN/Inf in active loss, parameters, gradients and supported optimizer state | Preserve failure and latest safe state; debug before another fit |
| Gradient presence | No unexplained missing parameter grads after backward/pre-zero; unused output-state hooks explicitly named | Distinguish disconnected returned state from an unused trainable parameter |
| Applied LR | Exact configured warmup/cosine and role/head/body factors, consistent after resume | Diagnose schedule/group construction; do not infer LR from next-step logging |
| Actual updates | Finite absolute delta/RMS and nonzero learning updates when expected; compare role-specific delta/W to the matched control | Inspect head/body scale and state, then test one LR-factor change |
| Zero pre-weight norms | Relative ratio is unavailable where its floor was used; absolute delta and floor flags remain visible | Do not treat initial zero-bias ratios as explosions or replace missing ratios with zero |
| Clipping | Pre/postclip norms and actual updates agree with configured clipping | Clipping frequency alone is not a Muon update-size bound; inspect measured steps |
| Prediction behavior | Per-target WP/WMSE, mean/std, clipping and target weights remain interpretable against their own cohorts | Inspect collapse, saturation, target-specific regression and mask/population shift |
| Plateau | Fixed full-valid WP trajectory and observed peak are recorded; L4 report_only is advisory | Inspect declining WP alongside train loss; retain epochs, preserve the fixed endpoint |
| Checkpoints | E retained immutable epoch snapshots plus latest full safe resume; configured final artifact remains separate from observed best | Fail closed on identity/RNG/index mismatch; never resume a stopped job from a status note |
| Efficiency | Warm rows/sec and total epoch wall time including validation/checkpoints, measured peak GPU/CPU memory and data wait | Choose the profile with better completed-experiment throughput; utilization alone is insufficient |

The L4 advisory preset uses min_epoch2, patience2, min_delta0.0001. These are
operational observation thresholds, not statistical significance or automatic
stop/promotion rules. Historical omitted-action policies retain their old meaning.
Train-eval is a deterministic hash-ranked 128-member fitting diagnostic with a
dense post-warmup mask; official valid has a different population and sparse mask.
Their absolute score/loss gap is not a calibrated overfitting estimate. Inspect
within-cohort trajectories, both targets and the fixed full-valid endpoint.

## Artifacts and decision

Retain every epoch weight/index, latest model/optimizer/scheduler/rank-RNG state,
rank journals, monitor events/evaluations, source/config/data identities and failure
receipts. Keep named RMS activation/output-gradient series distinct. Preserve
actual update samples even when the dense training stream is downsampled.
The native mirror and live emitter use the same Torch-free view adapter; plots
separate actual relative updates, absolute update RMS, pre-weight RMS, actual LR,
gradient norms, optimizer-state norms and denominator flags.

The existing plot exporter accepts `--run-dir` and `--plots-dir` and produces
source-bound PNG/SVG files from journals without model inference. A real old B
journal with all 3280 updates was rendered during this repair; newly added update
fields have only synthetic/CPU evidence until future GPU execution.

Success for the next scientific screen is a useful standalone global-WP gain at
the same primary endpoint and credible behavior across the matched seeds, followed
by its separately specified fixed-mixture/serving contribution. Report all cells,
negative or mixed results, selection exposure and cost. A zero-crossing interval
alone does not close a direction. Do not add a local delta to a public score.
No top-rank or public-gain estimate follows from telemetry fixes or CPU tests.
