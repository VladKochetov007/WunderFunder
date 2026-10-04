# H093/H094 architecture implementation

This is a code-readiness artifact, not a predictive result. No challenge data
were opened. The actual trainer was exercised on temporary synthetic Parquet
and cached teacher targets; neither architecture has an inference-engine or
package-parity path yet.

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

Use the same split manifest, seed, row set, optimizer, schedule, and fixed
checkpoint for all three arms. The 211-input, 23-output budget control uses a
plain GRU with hidden 96 and two layers: the extra sequential layer is the
explicit depth difference against H093's one shared layer plus four parallel
32-wide experts. The ten shift heads plus sign output below make `n_out=23` in
all four models.

```sh
for route in learned hash universal; do
  .venv/bin/python scripts/train_rnn.py \
    --name "h093_${route}" --split-manifest "$manifest" \
    --arch specialist --hidden 96 --layers 1 --dropout 0 \
    --expert-hidden 32 --num-experts 4 --prefix-rows 99 \
    --specialist-route "$route" --hash-seed 1729 \
    --aux 1,2,3,4,5,6,7,8,9,10 --aux-sign 0.1 \
    --epochs 4 --checkpoint-policy fixed --checkpoint-epoch 3
done
```

Run the matched ordinary-GRU arm with the same fit data and non-architecture
options, changing its name and architecture dimensions:

```sh
.venv/bin/python scripts/train_rnn.py \
  --name h093_gru_budget --split-manifest "$manifest" \
  --arch gru --hidden 96 --layers 2 --dropout 0 \
  --aux 1,2,3,4,5,6,7,8,9,10 --aux-sign 0.1 \
  --epochs 4 --checkpoint-policy fixed --checkpoint-epoch 3
```

At `n_in=211`, `n_out=23`, H093 has 154,343 trainable parameters in each
routing arm. The plain `RNNModel` control with `hidden=96,layers=2` has 152,855
(0.97% fewer). This includes the input projection, recurrent weights, all
readout heads, router, and expert cells; normalization and fixed hash-projection
buffers are excluded from trainable counts, though all H093 arms store the
same buffers.

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

Use one fixed seed and identical fit settings for every arm. Compare against
the same plain 128-wide, two-layer GRU to hold sequential depth and hidden width
fixed. The ten shift heads plus sign output below make `n_out=23` in all four
models:

```sh
for mode in conditional unconditional diagonal; do
  .venv/bin/python scripts/train_rnn.py \
    --name "h094_${mode}" --split-manifest "$manifest" \
    --arch prefix-adapter --hidden 128 --layers 2 --dropout 0 \
    --adapter-rank 8 --adapter-mode "$mode" --prefix-rows 99 \
    --aux 1,2,3,4,5,6,7,8,9,10 --aux-sign 0.1 \
    --epochs 4 --checkpoint-policy fixed --checkpoint-epoch 3
done
```

Run the matched ordinary-GRU arm with the same fit data and non-architecture
options:

```sh
.venv/bin/python scripts/train_rnn.py \
  --name h094_gru_budget --split-manifest "$manifest" \
  --arch gru --hidden 128 --layers 2 --dropout 0 \
  --aux 1,2,3,4,5,6,7,8,9,10 --aux-sign 0.1 \
  --epochs 4 --checkpoint-policy fixed --checkpoint-epoch 3
```

At `n_in=211`, `n_out=23`, each H094 arm has 265,755 trainable parameters.
The plain `RNNModel` control with `hidden=128,layers=2` has 261,143 (1.77%
fewer). Counts include the base input/recurrent/head and adapter parameters;
normalization buffers are excluded.

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

The training-only route audit counts scored-row exposure without reading
targets, with the EMA model excluded. In this tiny seed, learned H093 routed
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
