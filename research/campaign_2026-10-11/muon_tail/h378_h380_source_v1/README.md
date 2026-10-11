# H378 / H380 full-state Muon tails

This is an independently owned child implementation of the frozen trainer
`research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h230_source_v6_lead`.
The parent source tree is frozen. `PARENT_SOURCE_BINDING_V1.json` records its
50-file source map and canonical digest; a branch refuses a parent training state
whose producer map, source ref, or digest differs. The child's source files are
bound separately in each generated training contract.
`H378_H380_SOURCE_MANIFEST_V1.json` freezes the child package/trainer source map,
its canonical digest, and the supporting recipe/test hashes.

Both arms start from the same completed seed-3 full training state:

`artifacts/runs/r3_muonall256prop0_f1_s3_e3/training_state.pt`

The inspected parent state is an epoch-boundary `wnn.epoch-training-state`
version 1 artifact at `next_epoch=3`, with 22,380 optimizer attempts, all-
parameter Muon, fp16, WD 0.01, auxiliary coefficients 0.3/0.3, and the original
loader and RNG state. The branch loader validates the full parent/child contract;
only the declared horizon and scheduler fields can change. Model, optimizer,
scaler, loader, auxiliary dynamics, and RNG are restored before the child-local
tail scheduler is constructed. The parent's OneCycle scheduler is intentionally
not restored. Restored per-group beta1 remains fixed, and weight decay is kept
at 0.01.

H378 keeps both auxiliary coefficients fixed at their parent values. H380 uses
the same prefix and primary objective while linearly decaying the effective
`aux_struct` and `aux_next` coefficients from 0.3 to 0 over the three tail
epochs. Serialized model arguments remain at 0.3/0.3, preserving parent head
width and export reconstruction. Actual effective coefficients and tail learning
rates are written to `tail_schedule.jsonl`, and to `trace.jsonl` when the
inherited trace interval is enabled.

## Runnable recipes

These recipes use the project data and the unchanged fold-1 membership. They
inherit the rest of the recipe from the verified parent state. They prepare
artifacts only; root controls live resource admission and execution.

```sh
PROJECT=/home/vlad/development/wunder/wnn_connectome_starterpack
SOURCE=/home/vlad/development/wunder/wnn_muon_tail_20261011/research/campaign_2026-10-11/muon_tail/h378_h380_source_v1
PARENT="$PROJECT/artifacts/runs/r3_muonall256prop0_f1_s3_e3/training_state.pt"
SPLIT="$PROJECT/research/campaign_2026-09-29/artifacts/splits/fold1.json"

WNN_PROJECT_ROOT="$PROJECT" PYTHONPATH="$SOURCE/src" \
  "$PROJECT/.venv/bin/python" "$SOURCE/scripts/train_rnn.py" \
  --name r3_muonall256prop0_f1_s3_h378_e6 --split-manifest "$SPLIT" \
  --branch-state "$PARENT" --tail-epochs 3 --tail-schedule cosine \
  --tail-peak-lr 3e-5 --tail-final-lr 3e-7 --tail-aux-schedule none

WNN_PROJECT_ROOT="$PROJECT" PYTHONPATH="$SOURCE/src" \
  "$PROJECT/.venv/bin/python" "$SOURCE/scripts/train_rnn.py" \
  --name r3_muonall256prop0_f1_s3_h380_e6 --split-manifest "$SPLIT" \
  --branch-state "$PARENT" --tail-epochs 3 --tail-schedule cosine \
  --tail-peak-lr 3e-5 --tail-final-lr 3e-7 --tail-aux-schedule linear-to-zero
```

For an interrupted tail, restart with `--resume-state <run>/training_state.pt`
and the same output name plus `--split-manifest`. The child contract restores
the complete saved recipe and tail parameters; explicitly supplied conflicting
options are rejected. Its data-derived manifest hashes are recomputed and then
checked by the ordinary exact resume-contract comparison. Ordinary
`--resume-state` checkpoints without `branch_tail` retain the original strict
identical-contract scheduler restore behavior.

## Bounded implementation checks

`tests/test_training_branch.py` covers allowed and rejected contract changes,
tail LR endpoints and fixed beta1, auxiliary decay without mutating model
arguments, branch-resume option rehydration, and full restoration of the actual
seed-3 producer state on CPU. No fit, profile, GPU launch, prediction, or grade
was run for this source version.
