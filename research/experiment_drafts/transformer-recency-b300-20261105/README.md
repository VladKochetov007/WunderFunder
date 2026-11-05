# Transformer teacher: causal attention recency bias on B300

Draft PR: [#64](https://github.com/VladKochetov007/WunderFunder/pull/64).

Status: code and prospective launch only; no local tests, model imports, fits,
GPU profiles, score or student transfer. Tests and numerical checks must run on
the receiving host before admission. Four full teacher fits compare the same RoPE
Transformer and window with math SDPA: baseline bias 0 versus fixed bias 1. The
new term is `-bias * (query_position - key_position)/(context_rows - 1)` for
allowed past/current keys. It is added before softmax; forbidden future/outside
window keys stay `-inf`. A context of one has zero distance and no recency effect.
The bias is identical across heads and has no learned weights. Distances are cast
to the query dtype, so under BF16 autocast their values and bias are rounded;
check the numerical effect on the receiver. This is **not ALiBi**:
[Press et al.](https://arxiv.org/abs/2108.12409) motivated a distance penalty but
used different positional settings. No paper result establishes this task's gain.

**Hypothesis:** adding a mild preference for recent inputs may focus a teacher
when the predictive process changes quickly. The alternative is that it suppresses
useful long-range information; slice performance by sequence age, both targets,
and predeclared lag regimes. Compare against identical architecture, data,
normalization, optimizer and training budget. The baseline path preserves the
original boolean mask exactly when bias is zero. Bias >0 is restricted to math
attention: efficient/flex are rejected, not silently run without bias.

**Four initial fits:** F0/101001 and F2/101002, baseline and treatment per cell.
Unreserved cells; verify split manifests, component aliases and that no initializer,
normalizer, data or teacher ancestor includes evaluation components. Raw112;
T768×24, 12 heads, context 1024, 256-row chunks, 4-chunk accumulation, 30 epochs,
Muon + AdamW parameter groups, peak LR 0.0003, warmup 5%, cosine min ratio 0.1,
weight decay .01, gradient clipping 1, batch 1 per GPU, 8 B300 GPUs. Explicit
`bf16_bf16x9` is the user's scoped Transformer precision amendment: FP32 master
weights/optimizer/loss, BF16 activations and BF16x9 GEMMs, not IEEE FP32 parity.
Retain final checkpoint; do not choose with outer held-out outcomes. Every cell is
trained from scratch; new teacher fits 4, no student fits. Student transfer is a
separate follow-up with at least 4 new GRU fits plus qualified caches and profiles.
No promise of speed, leaderboard gain or teacher-to-student transfer.

## On the receiver

Before admission run receiver-side component/source audit, bias-zero parity,
prefix invariance, future masking, full-versus-chunked outputs and gradients,
DDP denominator equivalence, checkpoint/resume and finite BF16 numerical checks
on the exact source. A mathematical linear bias may overemphasize recent rows;
see also [numerical study](https://arxiv.org/abs/2608.03994). Check actual host
and GPU memory, driver, CUDA, Torch build and device topology. Do not run this
against active jobs. Bind source-qualified split/data and distinct physical-disk
output paths. No local profile has been performed.

```bash
cd "$WNN_REPO"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUBLAS_EMULATE_SINGLE_PRECISION=1
: "${WNN_DATA_ROOT:?}" "${WNN_SPLIT_F0:?}" "${WNN_SPLIT_F2:?}" "${WNN_RUN_ROOT:?}"
PACKET=research/experiment_drafts/transformer-recency-b300-20261105
# Set CELL=f0_s101001 and SPLIT="$WNN_SPLIT_F0"; repeat with f2_s101002/F2.
CELL=f0_s101001
SPLIT="$WNN_SPLIT_F0"
for ARM in control recency; do
  CONFIG="$PACKET/configs/${ARM}_${CELL}.json"
  uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    scripts/train_sequence.py --config "$CONFIG" --data-root "$WNN_DATA_ROOT" \
    --split-manifest "$SPLIT" --output "$WNN_RUN_ROOT/recency_${ARM}_${CELL}_profile" \
    --profile-updates 8
done
```

The profile emits a receipt, not a fitted model. Review each receipt including
finite updates and memory headroom, then create an `approved_profile.json` copy
with `approved: true`, preserving identity. Each arm/config/fold/source/data/
world-size needs its own profile. Only after receiver acceptance:

```bash
for ARM in control recency; do
  CONFIG="$PACKET/configs/${ARM}_${CELL}.json"
  RUN="$WNN_RUN_ROOT/recency_${ARM}_${CELL}"
  uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    scripts/train_sequence.py --config "$CONFIG" --data-root "$WNN_DATA_ROOT" \
    --split-manifest "$SPLIT" --output "$RUN" \
    --profile-receipt "$WNN_RUN_ROOT/recency_${ARM}_${CELL}_profile/approved_profile.json"
  uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    scripts/predict_sequence.py --checkpoint "$RUN/model.pt" --data-root "$WNN_DATA_ROOT" \
    --split-manifest "$SPLIT" --membership evaluation --output "${RUN}_eval"
  uv run --no-sync python scripts/score_sequence.py --data-root "$WNN_DATA_ROOT" \
    --split-manifest "$SPLIT" --predictions "${RUN}_eval" --output "$RUN/score.json"
done
```

Repeat with F2. `scripts/predict_sequence.py --membership fit` can produce
source-bound teacher caches for a separately budgeted matched student comparison;
do not export/use outer labels as teacher targets. Freeze the selection before
observing evaluation. Report official `global_wp_clip2_v1` pooled on identical
scored rows, t0/t1, all cells/failures, source/split/ancestor hashes, learning
rates, preclip gradients, device throughput and memory, and distinct hashed
weights/evidence archives. F0/101001 vs F2/101002 confounds seed and fold;
+0.0005 WP is an exploratory budget threshold, not significance. All known
validation/public feedback is exposed, and bootstrap of fixed predictions omits
training/selection uncertainty. A 1-vCPU / 60-minute / 20 MB exact submission
check is separate and no score on the private test or leaderboard is inferred.
