# H093/H094 architecture implementation

This is a code-readiness artifact, not a predictive result. No challenge data
were opened. The actual trainer was exercised on temporary synthetic Parquet
and cached teacher targets; neither architecture has an inference-engine or
package-parity path yet.

The production recipe drafts use `n_in=211,n_out=23`. Trainable parameter
budgets for the specialist/adapter arms and ordinary-GRU capacity controls are:

| Arm | Architecture capacity | Trainable parameters | Matched GRU control | Control parameters | Difference |
|---|---|---:|---|---:|---:|
| H093 learned/hash/universal | shared 96x1 + four experts 32x1 | 154,343 | GRU 96x2 | 152,855 | −0.97% |
| H094 conditional/unconditional/diagonal | GRU 128x2 + rank 8 adapter | 265,755 | GRU 128x2 | 261,143 | −1.77% |

The H093 matched control adds one sequential shared layer where the specialist
arm adds four parallel expert recurrences; this depth/branching difference is
the remaining architectural contrast at the matched budget. H094 and its
control keep hidden width and sequential depth the same.

## H093: prefix-routed specialist GRU

`wnn.specialist.SpecialistGRUModel` uses one shared GRU and four small expert
GRUs. Every expert processes every row in all routing arms. Each scored row's
route is determined by the inclusive mean of normalized input rows 0 through
98; that context is frozen after row 98. The learned router is trained only
through the training objective. The hash route is a fixed seeded projection,
and the universal control averages all expert residuals. The expert recurrent
states therefore see identical row support even when a top-one readout gives
unequal output usage. The trainer's existing warmup mask is 99 rows, so no
target loss is applied while the sequence route is still forming.

Match the original combined recipe across architecture and budget controls:
the same fit source, teachers, propensity model, since weighting, true-mask
weight, EMA, optimizer, seed, schedule, and fixed epoch-3 checkpoint. The
original structural and next-change objectives add 19 and 2 auxiliary outputs,
so `n_out=23` in every model. Use all three H093 routing arms and the ordinary
GRU budget control under one shared argument array; the ordinary GRU's extra
sequential layer is the explicit depth difference from the shared layer plus
four parallel experts.

```sh
common=(
  --train-source train+valid --full-valid --loss wmse
  --aux-struct 0.3 --aux-next 0.3
  --propensity prop_v1 --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03
  --batch 32 --chunk 1000 --epochs 4 --lr 0.001 --wd 0.01 --seed 3
  --optim muon --muon-scope rnn --ema-teacher 1 --ema-decay 0.999 --eval-ema
  --since-weight 32 --since-floor 0.25
  --distill ab67_final_since_s3,ab67_final_since_s4 --distill-alpha 0.5 --distill-boot-h 8
  --fast --checkpoint-policy fixed --checkpoint-epoch 3 --save-training-state
  --eval-source holdout
)

for route in learned hash universal; do
  .venv/bin/python scripts/train_rnn.py \
    --name "h093_${route}" --arch specialist --hidden 96 --layers 1 --dropout 0 \
    --expert-hidden 32 --num-experts 4 --prefix-rows 99 \
    --specialist-route "$route" --hash-seed 1729 \
    "${common[@]}"
done

.venv/bin/python scripts/train_rnn.py \
  --name h093_gru_budget \
  --arch gru --hidden 96 --layers 2 --dropout 0 \
  "${common[@]}"
```

At `n_in=211`, `n_out=23`, H093 has 154,343 trainable parameters in each
routing arm. The plain `RNNModel` control with `hidden=96,layers=2` has 152,855
(0.97% fewer). This includes the input projection, recurrent weights, all
readout heads, router, and expert cells; normalization and fixed hash-projection
buffers are excluded from trainable counts, though all H093 arms store the
same buffers. The concrete settings above mirror the original combined recipe
(`e412-combo-final-s3-v1`); no fit is launched by this note.

All arms instantiate the same tensor shapes and expert cells. The learned
router's MLP is used for routing; it receives no gradient in the fixed-hash and
universal controls. Report trainable parameter counts and expert route counts
on fit and evaluation sequences. This preserves architecture and
recurrent support while disclosing that the control router is inactive.

## H094: prefix-conditioned low-rank adapter

`wnn.specialist.PrefixAdapterGRUModel` applies a rank-8 interaction between
the current normalized input and the inclusive causal mean of the first 99
rows, before a GRU. The `conditional`, `unconditional`, and `diagonal` arms
instantiate exactly the same parameters. The unconditional arm replaces the
observed prefix with its learned global prior. The diagonal control uses the
same rank factors to gate hidden channels elementwise instead of adding the
low-rank cross-feature adapter.

Use the same original combined recipe and capacity-control pattern as H093,
with one fixed seed and identical non-architecture settings across arms. The
structural and next-change objectives again make `n_out=23`; the plain
128-wide, two-layer GRU holds sequential depth and hidden width fixed.

```sh
common=(
  --train-source train+valid --full-valid --loss wmse
  --aux-struct 0.3 --aux-next 0.3
  --propensity prop_v1 --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03
  --batch 32 --chunk 1000 --epochs 4 --lr 0.001 --wd 0.01 --seed 3
  --optim muon --muon-scope rnn --ema-teacher 1 --ema-decay 0.999 --eval-ema
  --since-weight 32 --since-floor 0.25
  --distill ab67_final_since_s3,ab67_final_since_s4 --distill-alpha 0.5 --distill-boot-h 8
  --fast --checkpoint-policy fixed --checkpoint-epoch 3 --save-training-state
  --eval-source holdout
)

for mode in conditional unconditional diagonal; do
  .venv/bin/python scripts/train_rnn.py \
    --name "h094_${mode}" \
    --arch prefix-adapter --hidden 128 --layers 2 --dropout 0 \
    --adapter-rank 8 --adapter-mode "$mode" --prefix-rows 99 \
    "${common[@]}"
done

.venv/bin/python scripts/train_rnn.py \
  --name h094_gru_budget \
  --arch gru --hidden 128 --layers 2 --dropout 0 \
  "${common[@]}"
```

At `n_in=211`, `n_out=23`, each H094 arm has 265,755 trainable parameters.
The plain `RNNModel` control with `hidden=128,layers=2` has 261,143 (1.77%
fewer). Counts include the base input/recurrent/head and adapter parameters;
normalization buffers are excluded.

A CPU shape-and-count check instantiated all six architecture arms and both
budget controls with `n_in=211,n_out=23`; every forward returned 23 outputs and
matched the table. This was a construction check, not a model fit.

The control arms have identical parameter counts by construction; test code
checks this and verifies that only the conditional and diagonal adapter
encoding changes when prefix context changes. The base GRU parameter count is
the same across arms. Counts can be reproduced with
`sum(p.numel() for p in model.parameters())` on the corresponding model.

## Verification boundary

The integration test ran all six arms through the real CPU trainer on five
synthetic 112-row Parquet sequences, using cached distillation, EMA, Muon,
four-row chunks, detach at four rows, and eight-row updates. It compared an
uninterrupted two-epoch fit with epoch-zero save/reload/resume, then reloaded
the exported checkpoint and compared inference using chunk sizes 4 and 13.
This places the first-99 boundary inside chunks in both partitions. Each arm's
saved model weights and optimizer, scheduler, scaler, EMA, loader, RNG, and
auxiliary state matched its uninterrupted reference exactly. Synthetic fits
used `n_in=211,n_out=2,hidden=4`; H093 controls each had 2,148 trainable
parameters and H094 controls each had 2,078.

The training-only route audit counts post-warmup row exposure without reading
targets, with the EMA model excluded. In the synthetic fixture, `is_scored` is
true exactly from row 99 onward, so post-warmup exposure equals that synthetic
official mask. The audit itself filters on the 99-row prefix count and does not
read the mask. Production reporting must distinguish post-warmup rows,
officially scored rows, and post-warmup unscored rows: with the original
`--valid-true-mask 0.03` recipe, the unscored rows in a sequence carrying a true
mask still receive epsilon weight. This synthetic audit does not estimate
production official-mask counts. In this tiny seed, learned H093 routed
all 104 exposures to expert 0; only expert 0 had a nonzero backward chunk (8,0,0).
Hash routed (0,78,26) with backward support (0,8,8). Universal had equal
weighted route mass (34.67,34.67,34.67) and support (8,8,8). These are
implementation diagnostics on four training sequences, not general usage
estimates. They confirm that always-updated hidden states do not imply equal
parameter-gradient support; learned top-one routing can starve experts, so fit
reports must give both route counts and per-expert gradient support.

Synthetic checks cover causal and chunk-exact prefix summaries, prefix freeze,
route freeze after the 99-row prefix, deterministic hash behavior through
chunking, all-expert recurrent forward support, equal adapter-arm parameter
counts, and conditional/unconditional/diagonal context dependence. They do
not establish predictive benefit, equal empirical router usage, native parity,
resource feasibility, or a submission-ready predictor. The latter are
separate prerequisites before a data fit or package promotion.

Top-one supervision can still starve expert output paths if one route dominates;
full data-fit reporting must include route counts by sequence and scored row.
The implementation establishes equal recurrent input support, not equal
gradient support.
