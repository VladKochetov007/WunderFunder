---
kind: experiment
status: implemented_unexecuted
created_utc: 2026-11-06
updated_utc: 2026-11-06
hypothesis_ids:
  - H-B300-SDPA-GATE-20261106
experiment_ids:
  - E-B300-SDPA-GATE-20261106
evidence_paths:
  - research/unified/ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md
  - https://arxiv.org/abs/2505.06708
---

# Query-dependent SDPA head gate for causal Transformer teachers

## Status and lineage

Implementation and receiver packet are prepared on branch
`codex/transformer-sdpa-gate-20261106`, based on `research-team` at
`7e4074314a62d0907c9608f9220cf72958708a2d`. The code adds optional
`attention_gate=constant_half|headwise` to the existing causal Transformer block.
No tests, profiles, fit, inference, cache export, or score ran locally. The
branch is unpublished; PR URL and final implementation SHA remain pending.

The large-dynamics idea already proposed a query-dependent sigmoid gate after
attention as a separate candidate. This packet turns that proposal into a
source-bound implementation and a bounded panel; it does not claim a new
literature mechanism. The paper's results concern large language models, not
this forecasting task.

## Mechanism

Each block computes `a = SDPA(q, k, v)`, then multiplies `a` before its existing
output projection. `constant_half` uses `a * 0.5`. `headwise` uses
`sigmoid(W * norm_attention(x) + b)` with shape `[batch, heads, rows, 1]`.
The new headwise matrix and bias start at zero, so the initial gate is exactly
0.5. The new tensors consume no random draws: same-fold arms retain matched
initialization for every shared backbone tensor. `none` remains the default,
adds no state-dict entries, and an explicit `none` is omitted by `load_config`
so legacy default config identities stay unchanged. The gate never changes Q/K/V,
the bounded KV cache, sequence reset, or output projection.

The new head-gate matrix uses the existing AdamW matrix group with weight decay;
its bias uses AdamW without decay. Existing QKV and other Muon group assignments
remain unchanged. GRU recipes reject an explicit `attention_gate` field.

## Frozen initial panel

Six new teachers form a three-arm, two-fold exploratory screen:

| Arm | F0 | F2 | Role |
|---|---|---|---|
| `none` | seed 101001 | seed 101002 | Matched un-gated reference |
| `constant_half` | seed 101001 | seed 101002 | Fixed attenuation control |
| `headwise` | seed 101001 | seed 101002 | Query-dependent treatment |

Each teacher is width 768, depth 24, 12 heads, FF width 3072, context 1024,
chunk 256, batch 1 per rank, accumulation 4, 8 ranks, 30 epochs, Muon/AdamW,
peak LR `3e-4`, warmup 5%, cosine floor 10%, weight decay `0.01`, and clip norm
1. The checkpoint is always final epoch 29; held-out fold scores never select
an epoch. `normalization_scope=all_fit` and `training_mask=all_after_warmup` are
explicit. Dropout is zero. Configs set `bf16_bf16x9`, FP32 master parameters,
loss and optimizer state, efficient exact-window SDPA, and compiled Transformer
blocks. The receiving profile must establish that the installed B300/PyTorch
build supports this exact mask/backend/compiler path; a failure requires a
reviewed matched amendment, not an arm-specific fallback. No throughput or
memory benefit is assumed.

There are two new fits per arm, six total, zero reused endpoints. F0 and F2 are
previously exposed evaluation folds. Seeds vary with folds, so two paired cells
do not separate fold and seed effects or establish training variance.

## Conditional scale follow-up

An explicitly separate stage prepares four additional teachers: `none` and
`headwise` on F0/F2, seeds 101001/101002, width 1792, depth 40, 28 heads, FF width
7168, context 1024, and the same 30-epoch Muon/BF16 recipe and optimizer settings.
The roughly 1.54B parameter count is arithmetic from the architecture, not a
measured count, profile, or capacity claim. These are four additional fits, not
part of the initial six. Scale profiles/fits have separate run directories and
commands. They require a separate allocation decision after initial results and
do not start automatically. No scale-teacher student transfer is included.

## Conditional student endpoint

After the initial `none` and `headwise` teacher caches pass whole-cache one/eight
rank parity, four separately allocated Transformer students can compare those
two teacher targets on F0/F2. Each fold has one student config (width 128, depth
2, four heads, context 128, seed 1001 for F0 / 1002 for F2); the run stage routes
the exact fold-matched `none` or `headwise` teacher cache. Both profile and fit
consume that same cache path. The student uses output KD weight 1, no hidden hints,
30 epochs, AdamW, world size 1, one B300, and the same explicit BF16/BF16x9
precision policy. Each target arm has two fits; four students are separate from
the six initial teachers and the optional four scale teachers. This compares
student transfer from two teacher recipes, not KD against a labels-only student.
Teacher caches contain fitting-membership predictions from teachers trained on
those same fitting rows, so the cache is in-sample. Outer evaluation membership
is excluded from teacher fits and caches; this does not remove teacher overfit
from the training targets or make F0/F2 fresh holdouts.

## Receiver workflow and gates

See [PR.md](PR.md) for the eleven-section contract, [recipe.json](recipe.json)
for the frozen fit inventory, [configs/](configs/) for all twelve complete
recipes, [VALIDATION.json](VALIDATION.json) for static-only preparation evidence,
and [run.sh](run.sh) for explicit stages. Its sourced
[teacher_stages.sh](teacher_stages.sh) and [student_stages.sh](student_stages.sh)
keep the stage implementations under 300 lines each. The CLI calls canonical
`train_sequence.py`, `predict_sequence.py`, `score_sequence.py`, and
`research_result_bundle.py`; it does not schedule or auto-advance cells.

Receiver order is: clean source/config/data identity; causal, mask, streaming,
reset, finite-precision, optimizer-ownership and two-rank DDP checks; one exact
eight-rank B300 profile per teacher config; human review of each unapproved
profile receipt; then an explicit fit, evaluate, cache, parity and bundle stage
for one cell at a time. The small student profiles/fits use one B300. Profile
receipts are born with `approved=false`; no script changes that field. After
review, the operator preserves `profile_receipt.json`, makes a separate copy
named `approved_profile.json`, sets `approved=true` in the copy, and records the
reviewer and rationale there. A named operator must inspect the source, exact
config/data identity, rank memory, compile/backend behavior and profile logs;
the canonical trainer validates the approved copy's exact identity.

The two-rank gate-aware Muon DDP check, also run by `run.sh check`, is:

```bash
env -u CUBLAS_EMULATE_SINGLE_PRECISION uv run --no-sync torchrun \
  --standalone --nnodes=1 --nproc-per-node=2 \
  scripts/check_sequence_muon_ddp.py --attention-gate headwise \
  --output-receipt "$WNN_RUN_ROOT/gate_ddp.json"
```

Before any student stage, the `none` and `headwise` teacher caches used by the
student contrasts must be exported on eight ranks and re-exported on one rank
from the same checkpoint. The `constant_half` cache is not consumed by students.
The existing R4 `cache_parity.py` checks full fitting membership, producer identity,
FP32 array shape/finite values and `rtol=1e-5, atol=1e-6`, then writes a parity
receipt. Each cell has a unique output/cache namespace; failed or previous
attempts are retained and never overwritten. Set `CUBLAS_EMULATE_SINGLE_PRECISION=1`
for every declared BF16/BF16x9 receiver stage. Do not launch any stage without
the current separately reviewed execution allocation.

## Interpretation limits

The paper [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708)
reports head-specific sigmoid gating after SDPA in large language-model training.
That is a mechanism analogy only: the task, data, model scale and training regime
here differ. A mean paired `headwise - none` or student delta of `+0.0005 WP`
with both fold deltas nonnegative is an exploratory follow-up trigger, not
significance. A query-dependent mechanism claim additionally needs positive
`headwise - constant_half` deltas on both folds. If constant-half matches or
beats headwise, the result supports simple attenuation at most; any attenuation
or larger-model follow-up needs a separate allocation, with no automatic 1.54B
gate promotion. Report every paired fold delta, component sensitivity and all
failures. Evaluation-unit resampling conditions on the two fitted models; it
cannot estimate recipe variance. No score, speedup, memory result, student gain,
or leaderboard effect is currently observed.
