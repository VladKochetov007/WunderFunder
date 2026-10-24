# Large causal Transformer teacher, R2

Implemented for [draft PR #32](https://github.com/VladKochetov007/WunderFunder/pull/32),
targeting `research-team`. **No fit, inference, GPU profile or test has been run for
this packet. Receiving-host acceptance is pending.** This is a new, explicit
four-fit amendment; the historical R1 configs/panels remain unchanged.

## The exact four fits

| Cell | Width × depth | Heads | Parameters, source formula | Fold / seed |
|---|---|---:|---:|---|
| `t1280_f0` | 1280 × 32 | 20 | 629,827,842 | F0 / 101001 |
| `t1792_f0` | 1792 × 40 | 28 | 1,542,548,226 | F0 / 101001 |
| `t1280_f2` | 1280 × 32 | 20 | 629,827,842 | F2 / 101002 |
| `t1792_f2` | 1792 × 40 | 28 | 1,542,548,226 | F2 / 101002 |

The 1.54B model is the primary capacity candidate, the 630M model its matched
reference. Both train from scratch on the complete clean fitting membership for
**six epochs**, selecting final epoch index 5. No checkpoint is selected from
outer-fold quality. Both widths share a cell's seed, data and update budget; this
does not make their initialization tensors identical. Four new teacher fits,
zero reused endpoints, zero student fits; profiles/recovery checks are discarded
engineering work and have separate cost. The two folds have been exposed before.

Each row is one raw112 token, with full-fitting-membership normalization,
pre-LayerNorm, RoPE, 64-dimensional attention heads, GELU FFN expansion 4 and a
two-target linear head. Per-layer rolling attention context is 1024 rows, using
FP32 math SDPA and activation checkpointing. This is bounded attention through
stacked layers, not global attention to all 20,000 rows at each layer. State is
reset per sequence; rows 0–98 warm the model. Detached KV memory is carried after
each 512-row chunk and across optimizer updates: gradient horizon is 512 rows,
with ordinary stale-state TBPTT semantics.

All eight GPUs hold complete model/optimizer replicas. Per-rank batch is two
sequences; two chunks accumulate before an update, giving 16 sequences and up to
16,384 new rows per global update. Final windows/batches retain their actual
weight. AdamW LR 1e-4, weight decay .01, clip norm 1, 5% warmup then cosine to zero,
dropout zero. Full FP32 includes optimizer states and attention; autocast and
TF32 are off. This panel holds optimization fixed across scale and measures equal
data passes, **not equal FLOPs or scale-specific optimal tuning**. Six epochs is
a declared starting budget, not a demonstrated scaling law or optimal duration.

Training WMSE uses all labeled post-warmup rows, including training files without
`is_scored`; evaluation still uses official `is_scored & row>=99`, with clip2 and
pooled weighted Pearson. This relies on canonical source-validated, ordered
20,000-row sequences. Future labels are never used as inputs. Normalization sees
only fitting components, caches FP32 moments under the data root and binds them
to split/source/membership/algorithm identity. Its initial float64 scan is
preprocessing, not mixed-precision model training. The one-GPU profile creates
this small cache; subsequent shapes/ranks reuse it. Nonfinite source features
remain an explicit error; no silent imputation is introduced.

## Files

- `recipe.json`: allocation, cells, fixed endpoints and decision contract.
- `configs/`: four production configs and four separate one-GPU engineering
  configs. The latter can never admit an eight-GPU fit.
- `run.sh`: explicit check/profile/admit/fit/resume/evaluate/cache/bundle stages
  for one cell; never automatically advances to another fit.
- `admission.py`: source-bound acceptance evidence, exact profile identities,
  GPU memory reserve, conservative host resume/scratch estimates and warmed
  update timing. It imports no training code.
- `compare.py`: checks all four fixed endpoints and computes matched official
  deltas with component bootstrap intervals using the canonical evaluator.
- `PR.md`: eleven-section scientific contract.
- `VALIDATION.json`: local static checks and unexecuted remote work.

## Receiving-host commands

Use the allocated Linux B300 node and a clean checkout of this PR's reviewed
commit. `uv sync --frozen` prepares its locked environment **on that host**.
Do not silently substitute a nightly Torch, precision, device or recipe if
acceptance fails. Inspect the captured driver/topology and the failed stage.

```bash
export WNN_EXPECTED_COMMIT='<reviewed full commit SHA from PR #32>'
export WNN_DATA_ROOT='/actual/data/root'
export WNN_SPLIT_F0='/actual/canonical/f0.json'
export WNN_SPLIT_F2='/actual/canonical/f2.json'
export WNN_RUN_ROOT='/actual/nvme/large-teacher-r2'
export CUDA_VISIBLE_DEVICES='<exact eight allocated IDs or GPU UUIDs>'
export WNN_HOST_RAM_BUDGET_GIB='<actual allocated aggregate host RAM>'
export WNN_REVIEWER='<receiving operator>'
PACKET=research/experiment_drafts/b300-transformer-distillation-runtime/large_teacher_r2

# Once for this source/runtime: causality, contracts, data guards, actual CUDA
# interrupted/resumed recovery, then two-GPU accumulated-gradient parity.
bash "$PACKET/run.sh" check t1280_f0

# One explicit cell; replace CELL with each table entry when allocating its fit.
CELL=t1280_f0
bash "$PACKET/run.sh" profile-one "$CELL"
bash "$PACKET/run.sh" profile-eight "$CELL"
bash "$PACKET/run.sh" admit "$CELL"
bash "$PACKET/run.sh" fit "$CELL"
bash "$PACKET/run.sh" evaluate "$CELL"
bash "$PACKET/run.sh" cache "$CELL"
bash "$PACKET/run.sh" bundle "$CELL"

# After all four cells have completed evaluation, on the receiver:
PYTHONPATH=src uv run --no-sync python "$PACKET/compare.py" --run-root "$WNN_RUN_ROOT"
```

Run the pair on F0, then the pair on F2; only one eight-GPU fit occupies the node
at a time. No auto-launch loop is included. Actual F0/F2 manifests, data, storage,
hardware allocation and operator identity remain receiver inputs. Metadata
alone is not evidence that they currently exist on a host available to us.

For an interrupted fit, use `resume CELL` instead of `fit CELL`. This restores
the last completed epoch, model/AdamW/scheduler/rank RNG and safe log offsets;
partial epochs replay. Same source/config/data/world size/device identity is
required. Source updates invalidate prior profiles and exact resumes. Preserved
failed directories are never automatically deleted or retried. After a failed
profile/check/export, use a new attempt root and keep the original evidence;
do not overwrite or splice a partial prediction directory.

## Memory, performance and admission

Each cell gets twelve actual updates first on one B300, then on eight B300s. The
eight-rank profile must match its exact data/config/source/runtime. Admission
requires the prepared pytest cases to run without skips, DDP cases to pass and
20% free GPU memory relative to measured peak reservation. Inspect the rank logs
for stable memory after KV fill; measured profile acceptance still does not prove
full-run stability. `approved_profile.json` records the named receiver review.

The normalizer's first disk scan occurs before profile timing. The report drops
the first two optimizer windows and extrapolates the slowest rank's warmed median
over the exact planned update count. That number excludes scans, saves, recovery,
evaluation and cache export: **it is not an end-to-end ETA**. Do not assume eight
GPUs are faster from model dimensions or compare one/eight timings as strong
scaling: per-rank batch stays two, so global batch changes.

Model+grad+two Adam moments alone require roughly `16 × parameters` bytes per
GPU (about 23 GiB for 1.54B); activations, DDP buckets, allocator reserve and
temporaries add to this. CPU resume can stage `12 × parameters` bytes on each
rank, so host RAM is checked against eight such copies plus measured process RSS
and 20% headroom. This is a conservative staging allowance, not a measured large
checkpoint/recovery peak. Do not use RAM-backed output storage. Scratch admission
reserves `32 × parameters` plus two-float predictions for fit+evaluation rows,
covering native weights/resume and a separate weights archive for that cell.
Retaining several cells requires their combined storage and extra failed-attempt
space. No weight-size estimate is a competition submission-size claim.

The receiver must additionally compare one- and eight-rank prediction exports
of an accepted endpoint before consuming its teacher cache for distillation.
Use the same `predict_sequence.py` command with `--nproc-per-node=1` in a separate
directory and compare source-qualified membership, manifest hashes and arrays
within the agreed FP32 numerical tolerance. This export parity remains a separate
unexecuted gate; the synthetic gradient/recovery tests do not establish it.

## What to return and how to decide

Return all four `score.json` files, per-target WP, matched large-minus-small deltas,
each fit's terminal status/source/config/split/moments hashes, peak memory,
successful updates, LR and gradient logs. Keep the raw profile/check/admission
evidence and prediction/cache manifests with their arrays. `bundle` produces
separate evidence and weights archives for the fit; archive the sibling
`acceptance`, `profile-one`, `profile-eight` and approval files separately, and
keep caches in private artifact storage with hashes. Do not put weights in Git.

The useful teacher signal is mean paired WP gain at least .001 with both cells
nonnegative, considered alongside GPU-hours. Report component-paired bootstrap
intervals from the saved aligned predictions; those describe evaluation-unit
uncertainty conditional on these fits, not training-seed or public-transfer
uncertainty. Two fold/seed cells cannot establish significance. A mixed or small
result motivates LR/context/input-encoding diagnosis; an interval crossing zero
does not close the Transformer direction. No local delta is a public-score claim.

Next transfer panel, separately allocated: pick and freeze one teacher family,
then run matched raw112 GRU256 label-only versus output-KD on the same two folds
(four new student fits, zero new teachers with exact cache reuse). These caches
have **two FP32 outputs and no hidden hints**, avoiding the old 64-hint storage
cost. They feed the PR32 sequence runtime; native211/TL3 training requires its
explicit cache/input adapter and is not claimed plug-compatible. The large
Transformer is a research teacher. One-core CPU/20 MB submission acceptance is
performed on a distilled student later; quantization is outside this panel.

API references checked during preparation:
[DDP](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html),
[SDPA mask/backend behavior](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html),
[CUDA precision](https://docs.pytorch.org/docs/2.14/notes/cuda.html).
These support API semantics, not performance or quality claims for this model.
