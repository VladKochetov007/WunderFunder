# Surprise-volume candidate

This isolated candidate tests whether endpoint-relative volume *surprises* help
the recurrent update gate after controlling for model capacity. It leaves all
2026-10-04 E427 artifacts and source untouched.

`surprise_volume_features` uses only raw inputs through the current row. For
each row it differences the 22 volume channels from the prior row, scales the
difference by the RMS of earlier changed rows in that sequence (unit scale
until one such row exists), accumulates standardized changes since the latest
endpoint, then divides by the square root of the number of changed rows in that
run. Endpoint rows emit zero and reset the run accumulator. The scale history
continues across endpoint resets. The result is `[batch,time,22]`.

`SurpriseVolumeGRU` preserves the parent input encoder and readout, appending a
rank-8 projection of those 22 features to first-layer GRU inputs. Added input
weights start at zero; only update-gate rows may learn from the extra columns.
All arms have identical parameter layouts: `aligned` receives current causal
features, `stale` receives the prior completed run's terminal feature, and
`noop` receives zero context. Each arm must strict-copy the same parent
checkpoint and use identical rows, order, teachers, optimizer, and schedule.
Project reset/candidate extension weights back to zero after every optimizer
step and EMA update.

The main falsifier is that aligned fails to beat both stale and no-op on the
predeclared paired component metric, or violates the existing per-class harm
guard. A performance difference appearing only for aligned versus the parent
but not versus the same-topology no-op is a capacity effect, not support for
the surprise mechanism. This candidate has only synthetic CPU evidence; it has
not been integrated with the competition trainer or scored on market data.

## Bounded CPU check

From the repository root, run at most two CPU cores and 2 GiB RAM:

```bash
PYTHONPATH=research/campaign_2026-10-05/volume_revision/src:src \
WNN_CPU_QUOTA=200% WNN_THREADS=2 WNN_MEM_MAX=2G \
scripts/limited .venv/bin/python -m unittest discover \
  -s research/campaign_2026-10-05/volume_revision/tests -v
```

This verifies causal prefix invariance, endpoint resets, stale alignment,
zero-context parent parity, equal arm topology, update-gate-only gradients,
and full-sequence feature use across uneven chunks. It does not measure GPU
headroom or train a model.

## Controller's next commands

The source is ready for a trainer overlay. Before fitting, wire
`surprise_volume_features` once on each original float32 full sequence and
pass that complete context to every TBPTT chunk via `forward_chunk(...,
full_context=context)`. Include `model.project_constraints_()` immediately
after optimizer steps and EMA updates. A source-only synthetic preflight can
then use the bounded command above. Once the controller has added the model
hook to a private trainer copy, the matched one-epoch command shape is:

```bash
scripts/limited .venv/bin/python <isolated-trainer>/scripts/train_surprise_volume.py \
  --arm aligned --name surprise_volume_aligned_s3 --seed 3 --epochs 1 \
  --checkpoint-policy fixed --checkpoint-epoch 0 --no-eval-during-fit \
  --train-source train+valid --valid-fold -1 --batch 32 --chunk 1000 \
  --init <exact-parent-ep3-checkpoint> --lr 0.0001 --wd 0.01 \
  --optim muon --muon-scope rnn --precision fp16
```

Repeat with `--arm stale` and `--arm noop`, changing only the arm and run
name. No such competition trainer exists in this candidate directory yet, so
the command is a controller integration recipe rather than a runnable command
today. No GPU fit, data read, or Kaggle action was performed here.
