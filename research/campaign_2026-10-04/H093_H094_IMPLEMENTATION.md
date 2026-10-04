# H093/H094 architecture implementation

This is a code-readiness artifact only. No data were opened, no model was fit,
and neither architecture has an inference-engine or package-parity path yet.

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
checkpoint for all three arms:

```sh
for route in learned hash universal; do
  .venv/bin/python scripts/train_rnn.py \
    --name "h093_${route}" --split-manifest "$manifest" \
    --arch specialist --hidden 96 --layers 1 --dropout 0 \
    --expert-hidden 32 --num-experts 4 --prefix-rows 99 \
    --specialist-route "$route" --hash-seed 1729 \
    --epochs 4 --checkpoint-policy fixed --checkpoint-epoch 3
done
```

All arms instantiate the same tensor shapes and expert cells. The learned
router's MLP is used for routing; it receives no gradient in the fixed-hash and
universal controls. Report both total stored parameters and expert route
counts on fit and evaluation sequences. This preserves architecture and
recurrent support while disclosing that the control router is inactive.

## H094: prefix-conditioned low-rank adapter

`wnn.specialist.PrefixAdapterGRUModel` applies a rank-8 interaction between
the current normalized input and the inclusive causal mean of the first 99
rows, before a GRU. The `conditional`, `unconditional`, and `diagonal` arms
instantiate exactly the same parameters. The unconditional arm replaces the
observed prefix with its learned global prior. The diagonal control uses the
same rank factors to gate hidden channels elementwise instead of adding the
low-rank cross-feature adapter.

Use one fixed seed and identical fit settings for every arm:

```sh
for mode in conditional unconditional diagonal; do
  .venv/bin/python scripts/train_rnn.py \
    --name "h094_${mode}" --split-manifest "$manifest" \
    --arch prefix-adapter --hidden 128 --layers 2 --dropout 0 \
    --adapter-rank 8 --adapter-mode "$mode" --prefix-rows 99 \
    --epochs 4 --checkpoint-policy fixed --checkpoint-epoch 3
done
```

The control arms have identical parameter counts by construction; test code
checks this and verifies that only the conditional and diagonal adapter
encoding changes when prefix context changes. The base GRU parameter count is
the same across arms. Counts can be reproduced with
`sum(p.numel() for p in model.parameters())` on the corresponding model.

## Verification boundary

Synthetic checks cover causal and chunk-exact prefix summaries, prefix freeze,
route freeze after the 99-row prefix, deterministic hash behavior through
chunking, all-expert recurrent forward support, equal adapter-arm parameter counts,
and conditional/unconditional/diagonal context dependence. They do not
establish predictive benefit, equal empirical router usage, native parity,
resource feasibility, or a submission-ready predictor. The latter are
separate prerequisites before a data fit or package promotion.

Top-one supervision can still starve expert output paths if one route dominates;
full data-fit reporting must include route counts by sequence and scored row.
The implementation establishes equal recurrent input support, not equal
gradient support.
