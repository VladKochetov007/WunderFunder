# Large causal Transformer teacher, R3 — Muon / 30 epochs

Implemented for [PR #32](https://github.com/VladKochetov007/WunderFunder/pull/32),
targeting `research-team`. **No fit, inference, GPU profile or test has been run for
this packet. Receiving-host acceptance is pending.** This is a new, explicit
four-fit amendment; the historical R1 and R2 configs/panels remain unchanged.

The [24 October high-compute continuation](../HIGH_COMPUTE_PLAN_20261024.md)
plans broader search and replication beyond these four fits. It does not change
this frozen panel or make the proposed follow-up cells executable.

## The exact four fits

| Cell | Width × depth | Heads | Parameters, source formula | Fold / seed |
|---|---|---:|---:|---|
| `t1280_f0` | 1280 × 32 | 20 | 629,827,842 | F0 / 101001 |
| `t1792_f0` | 1792 × 40 | 28 | 1,542,548,226 | F0 / 101001 |
| `t1280_f2` | 1280 × 32 | 20 | 629,827,842 | F2 / 101002 |
| `t1792_f2` | 1792 × 40 | 28 | 1,542,548,226 | F2 / 101002 |

The 1.54B model is the primary capacity candidate, the 630M model its matched
reference. Both train from scratch on the complete clean fitting membership for
**30 epochs**, selecting final epoch index 29. No checkpoint is selected from
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
sequences; eight chunks accumulate before an update, giving 16 sequences and up to
65,536 new rows per global update. Final windows/batches retain their actual
weight. The last window has 3,616 rows per real sequence. Muon uses the existing
AdamW-matched update-RMS convention, with shared peak LR 3e-4, momentum .95,
five FP32 Newton–Schulz rounds, matrix weight decay .01, clip norm 1 and no dropout.
Input/output matrices, norms and biases use auxiliary AdamW, beta=(.9,.95),
with no decay on vectors/biases. Q/K/V are orthogonalized independently.

Warmup is 5% of actual optimizer updates (approximately 1.5 epochs), followed by
cosine decay to 10% of peak, 3e-5, on the final actual update. Every group follows
the same factor. The precise step indexing, all parameters and why these values
were chosen are in [PARAMETERS.md](PARAMETERS.md). Full FP32 includes Muon matrix
intermediates; autocast/TF32 remain off. No model or optimizer state is sharded.

This is an explicit replacement of four **unlaunched** R2 fits, not four extra
fits. Compared with R2, data passes grow 5x and accumulation 2→8 reduces update
count per pass 4x, so total updates grow 1.25x for equal fitting membership.
Muon adds substantial work per update: no 1.25x or 5x wall-time claim follows.
The LR, auxiliary beta2, decay grouping and LR floor also change explicitly.
This compares capacity under one new recipe, not the isolated effect of Muon.
Neither starting hyperparameters nor training duration are empirically optimal;
no LR sweep, extra scientific fits or automatic tuning jobs are allocated.


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

- `PARAMETERS.md`: complete settings, exact warmup/cosine formula, rationale and tuning limits.
- `profile_analysis.py`: separate optimizer/window timing and actual schedule counts.
- `recipe.json`: allocation, cells, fixed endpoints and decision contract.
- `configs/`: four production configs, four one-GPU and four peak-LR engineering
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
export WNN_RUN_ROOT='/actual/nvme/large-teacher-r3-muon30'
export CUDA_VISIBLE_DEVICES='<exact eight allocated IDs or GPU UUIDs>'
export WNN_HOST_RAM_BUDGET_GIB='<actual allocated aggregate host RAM>'
export WNN_REVIEWER='<receiving operator>'
PACKET=research/experiment_drafts/b300-transformer-distillation-runtime/large_teacher_r3_muon30

# Once for this source/runtime: causality, contracts, data guards, actual CUDA
# interrupted/resumed recovery, then two-GPU accumulated-gradient parity.
bash "$PACKET/run.sh" check t1280_f0

# One explicit cell; replace CELL with each table entry when allocating its fit.
CELL=t1280_f0
bash "$PACKET/run.sh" profile-one "$CELL"
bash "$PACKET/run.sh" profile-eight "$CELL"
bash "$PACKET/run.sh" profile-peak "$CELL"
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
the last completed epoch, model/Muon-and-AdamW/scheduler/rank RNG and safe log offsets;
partial epochs replay. Same source/config/data/world size/device identity is
required. Source updates invalidate prior profiles and exact resumes. Preserved
failed directories are never automatically deleted or retried. After a failed
profile/check/export, use a new attempt root and keep the original evidence;
do not overwrite or splice a partial prediction directory.

## Memory, performance and admission

Each cell gets twelve actual updates on one B300, then eight B300s, followed by
a separate twelve-update eight-B300 peak-LR stress profile. The latter disables
warmup only in its engineering config, starts from scratch, and discards weights.
It verifies early peak-rate numerics, not convergence or the post-warmup state. The
eight-rank profile must match its exact data/config/source/runtime. Admission
requires prepared pytest cases (both optimizer recovery paths) without skips,
objective and Muon-update DDP cases to pass, all three profiles to agree, and
20% free GPU memory relative to measured peak reservation. Inspect the rank logs
for stable memory after KV fill; measured profile acceptance still does not prove
full-run stability. `approved_profile.json` records the named receiver review. Per-update logs include actual named group
LRs, preclip norm, clipping flag and CUDA-event `optimizer_seconds`; event timing
is read after the existing finite-parameter synchronization, without extra
per-step `synchronize()` calls. It isolates the optimizer stream interval from forward/backward (including launch gaps).

The normalizer's first disk scan occurs before profile timing. The report drops
the first two optimizer windows and extrapolates the slowest rank's warmed median
over the exact planned update count. That number excludes scans, saves, recovery,
evaluation and cache export: **it is not an end-to-end ETA**. Do not assume eight
GPUs are faster from model dimensions or compare one/eight timings as strong
scaling: per-rank batch stays two, so global batch changes.

Model+gradient+optimizer tensors require `12 * P_muon + 16 * P_adamw` bytes
before additional allocations (about 17.2 GiB for this 1.54B model). Muon keeps
one momentum slot and auxiliary AdamW keeps two. Activations, DDP buckets, KV,
allocator reserve and NS temporaries add to this tensor-accounting floor.
CPU resume conservatively budgets `12 * total_parameters` bytes on each
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
`acceptance`, `profile-one`, `profile-eight`, `profile-peak` and approval files separately, and
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
[CUDA precision](https://docs.pytorch.org/docs/2.14/notes/cuda.html),
[Muon LR conventions](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html),
[Muon author implementation and parameter guidance](https://github.com/KellerJordan/Muon).
These support API semantics, not performance or quality claims for this model.
