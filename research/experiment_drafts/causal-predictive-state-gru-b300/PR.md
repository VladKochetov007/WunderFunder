# [Experiment] Causal predictive-state GRU with next-input self-supervision

**Status: implementation draft; B300 profile and all scientific fits pending.**
This PR is a teacher/representation experiment. It implements one causal
auxiliary objective, a matched cell runner, and an evidence plan. It does not
claim improved WP, B300 throughput, or leaderboard impact.

## 1. Status, lineage and owner

- Hypothesis: `H-2026101801`. Experiment: `E-2026101801`.
- Mechanism family: teacher representation and objective; the learned recurrent
  state may also become useful for a later student connection experiment.
- Parent nodes: `family:teacher`, `family:temporal`. There is no teacher,
  initializer, warm start, or weight ancestry in the primary comparison.
- Implementation source base: `research-team` at
  `a0355ed6b08f87661522744a5be57d265bc45bbc`. Shared-policy Main snapshot:
  `24e9001570de158db9a5031e632dbb0b4477145f`.
- Owner: B300 worker is to be claimed in the cluster issue; root owns the frozen
  comparison, collection, and independent evidence integration.
- Implemented paths: `src/wnn/training/next_input.py` adds the objective;
  `setup_model.py`, `cli_targets.py`, `validation_training.py`,
  `validation_inputs.py`, `losses.py`, and `epoch.py` wire the head, reject
  unsupported use and report fit-only auxiliary loss. `run_cell.sh` profiles
  both arms or executes one paired cell.
- Current status: code and packet drafted; no profile, optimizer update,
  scientific fit, grade, archive, public upload, or leaderboard result exists.

## 2. Why it should work

The supervised task asks a recurrent model to represent current temporal state
from noisy sequential input. The hypothesis is that a small next-input
prediction objective makes the GRU state preserve predictable dynamics and
filters out some irrelevant variation, giving the task readout a more useful
representation. The candidate predicts the normalized 211-column canonical
input at `t+1` from output at `t`, with SmoothL1 loss (`beta=1`) and coefficient
0.10. Normalization is inherited from the cell's fitting-only model
mean/std buffers. The final row of each sequence is skipped.

Both arms have the exact same extra output-head width and initial parameter
shape. The control sets the auxiliary coefficient to zero; only the candidate
receives auxiliary gradients. The head is discarded at inference. It uses the
existing shared readout on `[GRU output, current encoded input]`, so a positive
result supports this joint objective but does not establish that the hidden
state alone carries the benefit. The first discriminating follow-up, if useful,
is a fold-clean hidden-only predictive decoder or OOF hidden-state transfer to
a student, with its own matched endpoint.

Strongest alternative: next-row prediction may mostly fit irreducible noise,
or its extra gradient may pull the representation away from the official
target. The user-supplied contest anecdote about a next-step GRU feature
extractor is unverified. Primary literature supplies precedent in other
domains, not local evidence; the counterevidence and scope are in the
[source note](../../unified/sources/CAUSAL_PREDICTIVE_STATE_SOURCES_20261018_R1.md).
The paired task score is the outcome; lower pretext loss alone is not success.

## 3. Compute advantage

This is a scalable repeated-fit question, not a claim that one GRU needs all
eight cards. Each cell is one independent H512x1 GRU fit on one B300. After a
real-shape profile passes, up to eight distinct fold/seed pairs can run in
parallel on an eight-GPU node, one process per card; the two arms in a cell run
sequentially on the same GPU. DDP adds no value for this small independent
model and is not part of the recipe.

The full comparison is 30 paired cells / 60 six-epoch fits, plus a separate
50-update-per-arm engineering profile. B300 host/device capacity and fit time
have not been measured. Begin with the fixed H512x1 architecture and six-epoch
budget; inspect per-epoch task/auxiliary losses and gradients. If it shows
useful teacher value, a separate experiment can transfer fold-clean hidden
states or teacher targets into the serving GRU. Do not infer a scaling law,
wall time, or student gain from this proposal. No quantization is included.

## 4. Frozen comparison

| Setting | Matched control | Candidate |
|---|---|---|
| Architecture | Cold dense GRU, hidden512, one layer, linear encoder, dropout0 | Same |
| Output layout | Task outputs plus a 211-value next-input head | Same |
| Next-input loss | `0.0` | SmoothL1(`beta=1`), fixed weight `0.10` |
| Input/target | Canonical raw112 + event99; fitting-only mean/std | Same |
| Main task | `wmse`, chunk normalization | Same |
| Fitting data | Full train+valid source qualified by the fold manifest | Same per cell |
| Batch/context | Batch8, 2,000-row update chunks, full sequence reset | Same |
| Training budget | Six epochs; fixed zero-based epoch5 checkpoint | Same |
| Optimizer | AdamW, lr `1e-4`, beta2 `.999`, wd `.01` | Same |
| Schedule / stability | OneCycleLR, 5% warmup, global clip1.0 | Same |
| Numerics | FP32 weights, inputs, state, loss, gradients, optimizer; no autocast, GradScaler, TF32 or quantization | Same |
| Ordering / initialization | Paired seed, seeded batches, cold init, deterministic setting | Same |

The launcher also uses the existing `--sync-free --fused-adam` options in both
arms. B300 actual-shape profiles must establish finite updates and usable
resources before the 60 fits. If those paths fail on the real image, preserve
the profile receipt and freeze an explicit recipe revision before continuing.
No fit-time outer-fold evaluation or best-checkpoint selection is performed.

## 5. Thirty paired runs

Run both arms for each of these exact cells:

- Fold0: seeds1001-1008 (8 paired cells).
- Fold1: seeds1001-1008 (8 paired cells).
- Fold2: seeds1001-1007 (7 paired cells).
- Fold3: seeds1001-1007 (7 paired cells).

That is 30 pairs and 60 quality fits. The shared seed identities across folds
are treated as a crossed fitting factor. The fixed manifests are
`research/campaign_2026-09-29/artifacts/splits/fold{0,1,2,3}.json`; SHA256s and
exposed evaluation sequence/component counts are in `recipe.json`. They are
source-qualified, feature-only successor components. The exact worker command
is `./run_cell.sh pair <fold> <seed>`. Each call trains both arms on the same
GPU and alternates order by fold/seed parity.

Before any fan-out, run `./run_cell.sh profile 2 1001` on one actual B300. It
performs four warmup plus four measured batches per arm; with the frozen
2,000-row chunks this is 50 optimizer windows per arm. It is not a quality
replicate. Do not admit the matrix until the cluster owner records the actual
GPU/software, memory and finite-update result. If it fails, no broad retry:
record the first failure and make the smallest source-bound repair.

## 6. Validation and significance

Use the unchanged official `global_wp_clip2_v1` metric and its original
`need_prediction AND is_scored` mask, target clipping and weighting. Export
each fixed epoch5 endpoint on its matching outer fold only. Independently grade
identical component rows and pair each candidate with its same-cell control.

Primary statistic:

`D = (1/4) * sum_f mean_s[WP(candidate[f,s]) - WP(control[f,s])]`, with equal
fold weights and `n_f = 8, 8, 7, 7`.

Report all four fold means, ranges and every cell. Separate fitting variation
from evaluation-unit uncertainty: 10,000 bootstrap draws of shared seed
identities across folds (seed1001-1008, redraw if any fold has no sampled
available seed) and 10,000 paired whole-component resamples within each fold.
Use RNG seeds `20261018` for fitting-seed draws and `20261019` for component
draws. These intervals are conditional on fixed, exposed folds and source
ancestry; four folds do not estimate unseen-regime uncertainty. The primary
family has one contrast; apply Holm correction to any extra preregistered
arms/endpoints. Do not treat rows, epochs or repeated fits as independent
evaluation units. The first profile is not part of the statistical analysis.

## 7. Leakage and coverage

The [data contract](../../../docs/data_overview.md) and submission API show
that the inference callback receives current state rows and sequence controls,
not future competition labels or scoring masks. The objective uses row `t+1`
as a **training-only input reconstruction target** from the cell's fitting
sequences. It does not add that future row to the input at `t`; it does not
read target labels, label ages, score masks, or validation/test targets during
fitting. The model at inference remains causal and deterministic, and recurrent
state resets per sequence.

The fold manifest excludes outer evaluation components from supervised fitting
and all fitting statistics. The experiment uses only permitted `train.parquet`
and `valid.parquet` input data within that membership. No pretrained model,
teacher cache or initializer is involved. Existing folds have been inspected
before and remain exposed. Cross-fold consistency is robustness evidence, not
a new holdout. This does not claim future-regime or public-transfer
generalization.

The main methodological limitation is that an auxiliary next-row objective
could overfit predictable marginal structure without improving target
forecasting. The equal-weight feature loss and Huber penalty are fixed before
the matrix; no validation-driven coefficient selection is planned. The shared
readout skip means a later hidden-only control is needed to isolate which
representation carries any gain.

## 8. Decision-changing result

A provisional mean `D = +0.0010` absolute WP is a practically useful effect to
prioritize hidden-state/student transfer. It is a priority reference, not a
universal stop rule. Success requires the paired task metric to improve with
finite optimization and no material fold collapse. The next step is a
separate, fold-clean student/hidden-state transfer experiment; it is not
included in these 30 pairs.

A zero-crossing interval, lower standalone mean, or weak early profile score
is unresolved and does not close the family. If the task score is negative but
the predictive representation appears useful, estimate its conditional value
in a matched student endpoint before deciding. A clear negative narrows only
this one-step normalized SmoothL1 head, weight, H512x1 model and six-epoch
schedule. It does not reject multihorizon/coarse state prediction, independent
self-supervised pretraining, input-only online state adaptation, or the
teacher-to-student connection. Public-score effect is unknown until actual
submission feedback; do not add local deltas to the public best.

## 9. How to run

### Environment and profile

On the worker, checkout the published experiment branch and mount the approved
dataset under the private project root. Confirm actual hardware and software
before running:

```bash
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
nvidia-smi topo -m
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'
export WNN_PROJECT_ROOT=/path/to/private/project-root
export WNN_PYTHON=/path/to/environment/bin/python
export WNN_EXPERIMENT_LOG_ROOT=/path/to/physical/scratch/causal-predictive-state-gru-b300
export CUDA_VISIBLE_DEVICES=<one-assigned-B300-UUID-or-index>
./run_cell.sh profile 2 1001
```

`WNN_PROJECT_ROOT` must contain `datasets/train.parquet` and
`datasets/valid.parquet`; it also receives `artifacts/runs/` and cached
prediction outputs. Put logs/checkpoint archives on approved persistent
physical storage. Keep one fit process on each assigned GPU and do not start
parallel fits on one card. The default profile leaves the trainer's 1 GB CUDA
allocator reserve in place. Capture host memory and GPU memory from the worker
monitor as well as the trainer output.

### Scientific fit pair

After the profile is recorded and the issue worker claims a disjoint cell:

```bash
./run_cell.sh pair 0 1001
```

Use the 30-cell matrix above. The helper logs exact stdout to
`$WNN_EXPERIMENT_LOG_ROOT/<run_name>.log`, while trainer artifacts are written
to `$WNN_PROJECT_ROOT/artifacts/runs/<run_name>/`. For an interrupted run,
preserve its directory and resume the same cell with `WNN_RESUME=1`; do not
change the source, arm settings, seed, fold or schedule mid-resume. Stop only
the failed task if it is nonfinite or out of its measured resource bound.

### Predictions and independent grade

For each completed run, export only its declared outer-fold groups with the
source-bound evaluation selector, from the identical input cache and source:

```bash
WNN_PROJECT_ROOT="$WNN_PROJECT_ROOT" "$WNN_PYTHON" scripts/predict_run.py \
  cpsi_candidate_f0_s1001 valid \
  --campaign-split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json \
  --campaign-evaluation-only --inference-precision fp32
```

The script writes hash-bound prediction arrays and sidecars in
`$WNN_PROJECT_ROOT/cache/`. Repeat for the control and all cells. Independently
grade from shared-policy Main snapshot `24e9001570de158db9a5031e632dbb0b4477145f`
using its source-qualified endpoint grader:

```bash
python research/campaign_2026-10-18/priority25_execution_r4/code/grade_endpoint.py \
  --root "$WNN_PROJECT_ROOT" \
  --run cpsi_candidate_f0_s1001 --split research/campaign_2026-09-29/artifacts/splits/fold0.json \
  --epoch 5 --reference cpsi_control_f0_s1001@5 \
  --output /path/to/physical/scratch/grades/cpsi_candidate_f0_s1001.json \
  --scratch-dir /path/to/physical/scratch/grade-cache
```

The grader checks checkpoint provenance, fitting/evaluation membership,
prediction sidecars, exact row IDs and official metric masks. If the exact
grader revision or data mounts differ, stop before scoring and preserve the
first mismatch; do not substitute a row-level score or a fit-time validation
number.

## 10. Evidence to return

For each profile and cell, return:

- immutable config, source commit/dirty patch hash, environment/container lock,
  exact command, run/arm/fold/seed IDs, split and data manifests;
- actual GPU UUID/model/topology, driver/CUDA/PyTorch, host/device memory,
  loader/fit/export/grade timing, power/utilization if available;
- all epoch progress logs: task objective, next-input SmoothL1 and MSE,
  learning rate per group, preclip gradient norms/clip rate, attempted and
  successful optimizer steps, nonfinite/AMP-skip counts, throughput;
- epoch5 checkpoint and resumable optimizer/sampler/RNG state; per-file hashes;
- FP32 evaluation predictions, producer sidecars, exact component/row IDs,
  official grades and paired bootstrap manifests;
- **separate** evidence and weights archives, each with file/archive SHA256,
  relative paths, size/precision/format, and approved private storage location.

Do not include dataset payloads, credentials, or private raw rows in Git or
archives. Run results are not complete until the artifact manifest, root-side
independent grade, exclusions/failures, and unified result note are indexed.

## 11. Results and trace

Current result is `null`: no B300 profile or model-quality fit has run. The
implementation is on the isolated `experiment/causal-predictive-state-gru-b300`
branch, based on the stated research-team SHA. The full packet and source note
are linked from [ideas](../../unified/ideas/README.md) and [sources](../../unified/sources/README.md).
After publication, record the exact PR and cluster issue URLs here. After any
run, append immutable H/E events and link all receipt/archive hashes from the
unified README. An independent reviewer should audit objective-to-code
alignment, fold membership and score provenance before interpreting a result.
