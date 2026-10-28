# Block GALAM-GRU72 and frozen BatchNorm diagnostic

[Draft PR #50](https://github.com/VladKochetov007/WunderFunder/pull/50), base
`research-team`. Implementation commit:
`980533a97f5736ec0ae4aaab15e20cf607f91f67`.
[Review only this packet above PR46](https://github.com/VladKochetov007/WunderFunder/compare/8482de5c71ce8bdd9974780438d518ea50bfeca8...codex/galam-block-bn-diagnostic).

Prepared for GPU execution, **no numerical checks, profiles, inference or fits
have run for this packet**. Local validation is source-only. The user requested
implementation/publication and explicitly excluded local tests/training and
changes to current workers. PR publication is not a remote launch.

This branch stacks on [PR46](https://github.com/VladKochetov007/WunderFunder/pull/46)
at `8482de5c71ce8bdd9974780438d518ea50bfeca8`; integration target is `research-team`.
The dependency contains the original GALAM model, canonical training integration,
chronological/source split loader and indexed checkpoint provenance. Review this
packet separately from the inherited PR46 changes. Do not switch the live PR46
checkout to this branch or point a running worker at it.

## Two separate questions

**Block execution** changes how the same GRU-full model is computed. One GRU call
returns the hidden trajectory for the chunk; local/global causal attention,
left-context convolutions and the decoder operate on blocks. The existing step
backend stays the default. Opt in with `model.galam.execution_backend = "block"`.
Only `schematic_gru_full` supports this backend. Parameter names and tensor shapes
stay compatible with the original GRU checkpoint; changed execution/config/source
identities must never be presented as an exact resume of an existing fit.

Training BatchNorm still normalizes each timestamp across batch members only.
Its running-stat EMA and counter account for every original time step. Flattening
batch and time would change the model and is explicitly avoided. Historical KV,
convolution tails, causal windows and chunk-state detachment are retained.
Existing sampled diagnostic hooks use the step implementation for that sampled
chunk, preserving the monitor's per-step meaning; normal chunks use block code.
Floating-point reduction order changes, so receiver numerical acceptance is
required. Equal equations do not promise bitwise or long-run training identity.

**BN recalibration** intentionally changes inference statistics, independently of
the block optimization. Copy and pin one original epoch checkpoint and its full
epoch index before observing this diagnostic. A deterministic train-only subset
of 256 eligible sequences (seed1001) supplies all rows, without labels or gradients.
For each BN layer, pool its evaluation-path input moments in a separate pass;
already recalibrated upstream layers stay fixed for downstream passes. Other
parameters and buffers must remain identical. Evaluate original and candidate on
the same complete official valid. The candidate is an explicitly inference-only
artifact, not a resumable training checkpoint or a replacement for the original.

The existing BN momentum0.1 emphasizes recent steps; whether this harms WP is
unmeasured. Pooled train moments can also hurt under distribution shift. This is
one frozen diagnostic, not automatic selection of momentum, subset or checkpoint.

## GPU receiver setup

Use a **separate clean checkout of the reviewed PR commit**, an already provisioned
CUDA Python environment and one explicitly verified free GPU. Both L4 and B300 are
eligible after the same checks. Do not assume GPU0 is free or install a new runtime
into an active worker environment. FP32, autocast/TF32/BF16x9 off for both backends;
the earlier Transformer BF16 amendment is not applied to this equivalence study.

Copy [receiver.env.example](receiver.env.example) outside the repository and fill
every path and SHA. The checkpoint/index must be immutable snapshots, with expected
digests from the producer receipt. Do not bind a mutable `latest` path or an index
still being appended by a live job. `GALAM_OUTPUT` must be a fresh external path.
Preserve the snapshot layout: `epoch_index.json` beside
`epoch_checkpoints/epoch-NNNNNN-SHA256.pt`; the filename and relative path must
match the captured index entry. `GALAM_EPOCH_INDEX_SHA256` is the producer's
canonical `index_sha256` field, not the byte digest of the JSON file. The whole
captured safe prefix is required; only the selected checkpoint needs to be copied.

```bash
source /srv/experiments/galam-block-bn.env
cd "$GALAM_SOURCE"
export PYTHONPATH="$GALAM_SOURCE/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUBLAS_EMULATE_SINGLE_PRECISION=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
PACKET=research/experiment_drafts/galam-block-bn-20261029
test "$(git rev-parse HEAD)" = "$GALAM_COMMIT"
test -z "$(git status --porcelain)"
mkdir "$GALAM_OUTPUT"
```

Do not run these commands on the preparation Mac. Commands below are sequential,
explicit actions, not a queue; preserve every failure under its own attempt path.

## 1. Numerical acceptance on the receiver

```bash
timeout --signal=TERM --kill-after=60s 7500s bash "$PACKET/run.sh" check \
  --config "$PACKET/configs/gru72-step.json" \
  --expected-commit "$GALAM_COMMIT" \
  --device cuda:0 --output "$GALAM_OUTPUT/check.json"
```

Compare step and block outputs, carried state, all parameter/input gradients,
BatchNorm statistics/counters, causal prefixes and differently partitioned chunks.
The receiver checks include the representative B384/chunk64 shape. The initial
check has no real data and cannot establish quality or source-split validity.
No tolerance relaxation or precision/batch fallback is automatic.

## 2. Complete-update speed and memory

```bash
timeout --signal=TERM --kill-after=60s 7500s bash "$PACKET/run.sh" profile \
  --config "$PACKET/configs/gru72-step.json" \
  --expected-commit "$GALAM_COMMIT" \
  --device cuda:0 --acceptance "$GALAM_OUTPUT/check.json" \
  --batch-size 384 --chunk-rows 64 --warmup 4 --updates 12 \
  --output "$GALAM_OUTPUT/profile.json"
```

This uses matched initial weights/data for both backends and includes forward,
backward and AdamW updates. Report synchronized wall time, CUDA time and peak
allocated/reserved memory. Threefold speedup is a target, **not a measurement**.
Synthetic input generation/data transfer, complete epoch validation, checkpoint
I/O and live multi-worker contention are outside this timing comparison.

For real-loader/monitor integration use the canonical trainer, separately for
each backend, on the existing read-only prepared data and original qualified
split. These are disposable **25-update engineering profiles**, reaching the
sampled diagnostic interval; they do not write scientific checkpoints:

```bash
timeout --signal=TERM --kill-after=60s 7500s "$PYTHON_EXECUTABLE" scripts/train_sequence.py \
  --config "$PACKET/configs/gru72-step.json" \
  --data-root "$GALAM_DATA" --split-manifest "$GALAM_SPLIT" \
  --output "$GALAM_OUTPUT/real-profile-step" --profile-updates 25

timeout --signal=TERM --kill-after=60s 7500s "$PYTHON_EXECUTABLE" scripts/train_sequence.py \
  --config "$PACKET/configs/gru72-block.json" \
  --data-root "$GALAM_DATA" --split-manifest "$GALAM_SPLIT" \
  --output "$GALAM_OUTPUT/real-profile-block" --profile-updates 25
```

Compare complete update counts, finite losses/gradients/optimizer, diagnostic
events and data/source identities. Profile initialization/warmup and synthetic
speed do not establish full-epoch acceleration. Neither old PR46 profiles nor the
synthetic receipt admits a long fit under the canonical trainer.

The block config can train through the existing `scripts/train_sequence.py`
entrypoint. A later **fresh** fit requires explicit receiver admission of the
new real-data profile, unchanged source/config/data/runtime, a fresh output root
and separately allocated budget. Its prepared recipe is the PR46 maximum20epoch
GRU72 recipe with the same stopping policy; this packet allocates zero new fits.
Do not use `--resume` with a live PR46 checkpoint to change backend in flight.

## 3. Frozen BatchNorm recalibration and full-valid comparison

This diagnostic is independent of block acceptance and deliberately uses the
original step backend. It can run first if a qualified snapshot is available.

```bash
timeout --signal=TERM --kill-after=60s 7500s bash "$PACKET/run.sh" bn \
  --checkpoint "$GALAM_CHECKPOINT" \
  --checkpoint-sha256 "$GALAM_CHECKPOINT_SHA256" \
  --epoch-index "$GALAM_EPOCH_INDEX" \
  --epoch-index-sha256 "$GALAM_EPOCH_INDEX_SHA256" \
  --expected-training-source-sha256 "$GALAM_TRAINING_SOURCE_SHA256" \
  --data-root "$GALAM_DATA" --split-manifest "$GALAM_SPLIT" \
  --expected-commit "$GALAM_COMMIT" \
  --calibration-sequences 256 --calibration-seed 1001 \
  --batch-size 32 --eval-batch-size 384 --max-wall-seconds 7200 \
  --output "$GALAM_OUTPUT/bn-gru72"
```

Missing checkpoint/index/source/data ancestry fails instead of being guessed.
Calibration membership, ordering, row counts and before/after BN buffers are
recorded. The paired evaluation uses all official valid sequences and the same
`is_scored` mask after warmup99, official pooled clipped WP and both targets.
No valid examples or labels enter calibration. The output cannot overwrite an
input or existing attempt; failures never produce a successful quality receipt.
The outer Linux `timeout` bounds the newly created diagnostic process group even
if a device/I/O operation stalls; it never addresses existing worker processes.
Its expiration may leave partial outputs without a final receipt: preserve those
as incomplete, then use a new attempt directory after diagnosing the cause.

## Decisions and evidence

- Block: numerical acceptance first, then source/runtime-matched resource evidence.
  Keep slower or failed outcomes. Evaluate full-epoch speed only in separately
  admitted future work; no equivalence claim follows from source review alone.
- BN: report original/candidate WP and both target deltas. Gain at least0.001
  motivates a separately frozen checkpoint/seed confirmation; a loss can falsify
  this pooled-statistics method for this checkpoint. No automatic sweep follows.
- Official valid is exposed, and old checkpoints may be selected by its scores.
  This is not an independent holdout or significance claim. Feature-linked
  exclusions are approximate; seq_ix does not prove calendar chronology.

Return acceptance/profile JSON, raw canonical profile directories, BN receipt,
calibration membership, exact original/index/candidate hashes and failure logs.
Archive evidence separately from model weights. No data/model payload belongs in
Git. No submission ZIP or CPU compliance claim is part of this packet.

## Files and sources

- [recipe.json](recipe.json): frozen scope, identities, budgets and interpretation.
- [PR.md](PR.md): eleven-section research handoff; [run.sh](run.sh): explicit actions.
- [configs](configs): unchanged step recipe and block-selector-only variant.
- `check_block.py`, `profile_block.py` and their helpers: receiver numerical and
  synthetic complete-update checks; no data staging or worker management.
- `scripts/galam_bn_diagnostic.py`, `src/wnn/galam_bn/`: indexed ancestry,
  train-only moment collection, independent output and official paired scoring.
- `src/wnn/galam/block_*.py`, `execution.py`: opt-in block execution components.
- Receiver tests are under `tests/test_galam_block*` and `tests/test_galam_bn*`;
  their presence is not a passed-test claim.
- `VALIDATION.json` and `REVIEW.md`: final static checks, source review and limits.

Primary API contracts: [PyTorch2.8 GRU](https://docs.pytorch.org/docs/2.8/generated/torch.nn.GRU.html)
returns the complete hidden output sequence; [BatchNorm1d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BatchNorm1d.html)
distinguishes the biased training variance from unbiased running variance and
documents its EMA update. The receiver records its actual runtime; these links
are not a claim of tested compatibility on every CUDA/PyTorch version.
