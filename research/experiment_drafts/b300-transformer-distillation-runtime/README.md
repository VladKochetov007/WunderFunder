# B300 Transformer teacher and distillation runtime

**24 October implementation:** [Executable high-compute R4](high_compute_r4/README.md)
implements the [expanded plan](HIGH_COMPUTE_PLAN_20261024.md): staged LR/warmup,
context/horizon and 3.15B teacher configs, crossed seed/fold confirmation and
labels/single-teacher/three-teacher GRU distillation. Up to 40 conditional fits
including the original four R3 teachers; exact controls are reused. Source
preparation is complete; B300 checks and all outcomes remain pending.

**24 October precision correction:** [The Blackwell amendment](PRECISION_AMENDMENT_20261024.md)
supersedes strict-FP32 model compute for new R4 cells with explicit BF16/BF16x9.
The four active W2 teacher fits and their measured batches belong to
[PR40](https://github.com/VladKochetov007/WunderFunder/pull/40); they run on a separately
pinned source. This R4 integration does not change them. Its combined runtime
still needs receiving-host checks/profiles; no local tests, model imports,
training, inference or GPU profiles were run.

**Historical 23 October recipe:** [Large teacher R3 — Muon / 30 epochs](large_teacher_r3_muon30/README.md)
records the original strict-FP32 four-fit preparation. Its configs and parameter
tables remain unchanged as history. Use the current amendment and R4 commands
for the continuation; do not treat the old R3 source/receipts as BF16 acceptance.

- [PR.md](PR.md): scientific comparisons, limitations and evidence contract.
- [recipe.json](recipe.json): explicit prospective amendments for PRs 28–30;
  initial and optional follow-up costs. Historical CP30 recipes remain unchanged.
- [configs/](configs/): executable trainer JSON, one file per variant and cell.
  Names contain dimensions, not unmeasured rounded parameter-count claims.
- [Remote acceptance](REMOTE_ACCEPTANCE.md): checks to run on the receiving host,
  including unexecuted unit tests and multi-GPU equivalence requirements.
- Runtime: [trainer](../../../scripts/train_sequence.py),
  [exporter](../../../scripts/predict_sequence.py),
  [official scorer](../../../scripts/score_sequence.py).
  Components live in `src/wnn/training/sequence_*.py`; the legacy trainer is unchanged.

Published [PR #32](https://github.com/VladKochetov007/WunderFunder/pull/32) targets `research-team` and is ready for code review.
The following R1 implementation/history is frozen. The precision amendment above describes the current continuation. R1 executable source
is frozen at `d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6`; later publication/base-sync commits preserve that
implementation. [Validation record](VALIDATION.json) records static checks only.

## Historical R1 implementation

Raw112 causal pre-norm Transformer, RoPE, GELU FFN, per-layer sliding KV state,
math SDPA, activation checkpointing, and canonical `RNNModel` GRU adapters.
All model/input/state/gradient/optimizer arithmetic is FP32, with autocast and
TF32 disabled. No quantization, FlashAttention assumption, Muon or output AR.
The explicit optimizer is AdamW with warmup/cosine and gradient clipping.

Input normalization uses the first 32 source-qualified fitting sequences,
never outer validation. Every 20,000-row sequence is forwarded from row zero;
state resets between sequences. Loss/score mask is `is_scored & row>=99`.
Model/normalizer selection is the fixed final epoch, not outer-fold quality.
Strict component split validation is reused from `load_campaign_split`.

This **amends** old designs: raw112 and sliding KV are neither the anchored
4096-row raw112 proposal in PRs 28–30 nor PR31's anchored211/Muon/dynamics
proposal. Context is 512 rows **per layer** for teachers and 128 for students;
the input receptive field can span `1 + layers * (context_rows - 1)` rows.
TBPTT detaches state every 256 rows, while AdamW updates every four chunks
(the final shorter window uses its true denominator). Carried state across
updates has the ordinary TBPTT stale-state semantics; it is not full BPTT.

DDP pads a shuffled sequence permutation with zero-weight dummy records, not
repeated real examples. Label/KD and hint denominators cover all ranks and every
chunk in each accumulation window. The final shorter batch/window remains part
of the fit. Loss is `WMSE(clipped labels) + kd_weight*WMSE(clipped teacher)
+ hint_weight*MSE(hints)`; label and KD use `abs(clipped labels)` weights.
This additive objective is an explicit amendment, not the old alpha=0.9 mixture.

Teacher hints are a frozen seed-1729 projection of the normalized last hidden
row into 64 dimensions. Student hint projections are learned and removed for
serving. The negative control rotates fitting components, keeps row position,
and leaves output KD targets aligned. Components of unequal size reuse donor
sequences; this control destroys correspondence, not every distributional cue.

Caches bind checkpoint hash, model/context config, source provenance, full
fit membership, split hash, ancestor chain, and every source-qualified output
file hash. Warm-up rows remain present. Export uses disjoint rank shards and
publishes `manifest.json` only on complete coverage. Score streams the canonical
`GlobalAccumulator`; it never averages rank/sequence correlations.
Rank logs contain global work/loss sums and rank-local elapsed time: their
throughput fields are global throughput estimates, not each GPU's local rate.

## Frozen panels and costs

Cells are F0/student seed1001/teacher seed101001 and
F2/student seed1002/teacher seed101002. These names are **not reservations**.
Receiving operator must bind the actual canonical split manifests and claim
the cells before fitting. All existing folds are exposed.

| Panel | New fits | Exact prerequisite reuse | Question |
|---|---:|---|---|
| Transfer / PR28 | 2 T768×24 teachers + 2 GRU256 label controls + 2 GRU256 KD = **6** | None | Does this teacher help a matched GRU? |
| Hints / PR29 | 2 aligned hints + 2 permuted hints = **4** | Two teachers/caches and two KD controls from transfer | Does correct representation alignment help beyond output KD? |
| Streaming / PR30 | 2 T128×2 label controls + 2 T128×2 KD = **4** | Two teachers and four GRU endpoints from transfer | Does KD help the compact Transformer, and how does it compare with GRU? |
| Width follow-up | 2 GRU512 labels + 2 GRU512 KD = **4** | Same teachers/caches | Does student capacity change transfer? |
| Teacher scale follow-up | 2 T1280×32 + 2 T1792×40 teachers = **4** | Two T768×24 teachers | Does teacher quality improve at equal data passes? |

Follow-ups require a separately recorded allocation; none run automatically.
The scale panel is equal data passes, **not equal FLOPs or optimization tuning**,
and measures no student benefit until a separately budgeted transfer comparison.
Running all three initial panels costs 14 new fits, not 4–6 combined.
Reusing a control requires identical source, config, split, seed and endpoint
hash; an arbitrary existing 211-input teacher cannot be inserted by filename.

## Receiving-host setup

Run the following **on the allocated B300 host**, from this branch's checkout.
Use a dedicated disk output root. Do not use RAM-backed scratch. Bind these
environment variables to actual paths; no dataset download or private host
identity is invented by this packet.

```bash
git switch experiment/b300-transformer-distillation-runtime
git rev-parse HEAD
uv sync --frozen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
: "${WNN_DATA_ROOT:?datasets and cache directory required}"
: "${WNN_SPLIT_F0:?canonical F0 split manifest required}"
: "${WNN_RUN_ROOT:?dedicated physical-disk run directory required}"
PACKET=research/experiment_drafts/b300-transformer-distillation-runtime
CONFIG="$PACKET/configs/teacher_t768_l24_f0_s1001.json"
```

The existing lock selects research Torch/CUDA; record the actually installed
build, driver, device names, free memory and topology (`nvidia-smi -q`,
`nvidia-smi topo -m`) before admission. A container/name is not proof of B300
kernel support. For copied datasets, preserve source identity or use the
repository's verified `WNN_PORTABLE_DATA_MANIFEST` path; never bypass split checks.

## Profile, fit, export, grade

First complete [remote acceptance](REMOTE_ACCEPTANCE.md). Each architecture,
batch/world-size and objective/cache configuration needs its own real-shape
profile; these are engineering updates discarded afterwards, not scored fits.

```bash
uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/train_sequence.py --config "$CONFIG" \
  --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT_F0" \
  --output "$WNN_RUN_ROOT/t768_f0_profile" --profile-updates 8
```

Inspect the emitted profile receipt and diagnostics; record the reviewer,
actual memory headroom, finite gradients and successful remote checks. Create
an approved copy with `approved: true`, preserving all identity fields.
A fresh full fit checks that receipt against its config/data/source/runtime
and world size. Approval is a recorded B300 engineering decision, not a
local test result.

```bash
uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/train_sequence.py --config "$CONFIG" \
  --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT_F0" \
  --output "$WNN_RUN_ROOT/t768_f0" \
  --profile-receipt "$WNN_RUN_ROOT/t768_f0_profile/approved_profile.json"

uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/predict_sequence.py --checkpoint "$WNN_RUN_ROOT/t768_f0/model.pt" \
  --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT_F0" \
  --membership fit --output "$WNN_RUN_ROOT/t768_f0_cache"

uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/predict_sequence.py --checkpoint "$WNN_RUN_ROOT/t768_f0/model.pt" \
  --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT_F0" \
  --membership evaluation --output "$WNN_RUN_ROOT/t768_f0_eval"

uv run --no-sync python scripts/score_sequence.py \
  --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT_F0" \
  --predictions "$WNN_RUN_ROOT/t768_f0_eval" \
  --output "$WNN_RUN_ROOT/t768_f0/score.json"
```

For students, use the matching `gru256_labels_f0_s1001.json`,
`gru256_kd_f0_s1001.json`, `gru256_hint_f0_s1001.json`,
`gru256_permuted_hint_f0_s1001.json`, or `student_t128_*` config.
One GPU per student is the starting profile candidate:
`--nproc-per-node=1`. KD/hint profile **and** fit commands additionally require
`--teacher-cache "$WNN_RUN_ROOT/t768_f0_cache"`. Label-only controls reject a
teacher cache. Use a fresh output/profile directory for every arm.
Repeat the same commands with F2 manifest and `_f2_s1002.json` config; never
reuse an F0 teacher on F2.

Resume uses the same arguments and output plus
`--resume "$WNN_RUN_ROOT/t768_f0/training_state.pt"`; it restores only
the last completed epoch boundary, including optimizer/scheduler, rank RNG,
source/data/runtime identity and exact world size. A failed partial epoch is
replayed from that safe boundary, not claimed as continued mid-sequence.
Never change precision, world size or recipe under a resume ID.

Final artifacts include `model.pt`, `training_state.pt`, run/progress/terminal
records, per-sequence prediction files, hash manifests and `score.json`.
Keep weights and resume files out of Git. Return separate weights/evidence
archives with SHA256 and private storage location, retaining failures.
64-dimensional full-row hints need about 5.28 MB per sequence before filesystem
overhead; provision and measure NVMe capacity/read bandwidth before export.
Held-out exports retain only the two predictions, not unused hint vectors.
Use the existing `scripts/research_result_bundle.py` with explicit file lists:

```bash
uv run --no-sync python scripts/research_result_bundle.py \
  --root "$WNN_RUN_ROOT/t768_f0" --output-dir "$WNN_RUN_ROOT/t768_f0_bundle" \
  --evidence run_metadata.json --evidence metrics.jsonl \
  --evidence terminal_status.json --evidence score.json \
  --evidence rank0.jsonl --evidence rank1.jsonl --evidence rank2.jsonl \
  --evidence rank3.jsonl --evidence rank4.jsonl --evidence rank5.jsonl \
  --evidence rank6.jsonl --evidence rank7.jsonl \
  --weight model.pt --weight training_state.pt
```

For one-rank students include only rank0. Include any preserved interrupted log
tails explicitly, and keep prediction/cache manifests with their separately
stored arrays and hashes. The bundler never silently includes the supplied data.

## Limits

No GPU memory/throughput/parity or predictive gain is claimed. Old native211
checkpoints require a separate reviewed input/output/ancestry adapter.
This runtime exports FP32 research endpoints, not a competition ZIP.
CPU one-core runtime, 20 MB package, offline ABI, reset/replay and full-test
limits remain a separate measured submission acceptance step.

Implementation references: [PyTorch DDP](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html),
[SDPA masks/backends](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html),
and [CUDA precision controls](https://docs.pytorch.org/docs/2.14/notes/cuda.html).
These specify API behavior; they are not evidence of our model's B300 performance.
