# E540 V9: reuse the completed linear control

V9 preserves the V8 linear FP32 epoch-0 checkpoint and trains only the PLE candidate. The V8 linear fit completed 72,996 updates with zero AMP skips and 72,996 finite GradStats observations. V8 stopped after training because the worker looked for checkpoint hashes inside checkpoint-embedded provenance, while `train_rnn.py` writes those hashes to sibling `provenance.json`. The downloaded checkpoint files were verified against that sibling file and the worker's artifact manifest.

Before training-data hashing, the V9 worker validates the private control assets, V8 source/recipe/authorization receipts, exact trainer arguments, split/subset and fit/evaluation memberships, update diagnostics, and source hashes. It then loads both control snapshots on Kaggle, checks fixed epoch 0, tensor finiteness and snapshot equality, and exports the control prediction from the original checkpoint. The PLE arm is profiled and fitted only if the fresh FP32 profile passes every existing strict gate and the measured projection fits the worker cap. PLE checkpoint hashes are read from sibling `provenance.json`, matching the actual trainer serialization contract.

Both predictions cover exactly the 468 previously exposed evaluation groups and request features only. V8 provenance records that same membership in a different row order; V9 verifies unique-set equality and writes both prediction arrays in the frozen subset order. No grade runs in the worker. The selected 512 complete fitting components contain 1,597 train groups and 252 valid groups. With batch size two and `drop_last=True`, one of 1,849 selected sequences is omitted per epoch; its identity is not recorded. V9 requires the PLE sampler-order and normalization receipts to match the V8 control before making a paired comparison.

This is a bounded mechanism screen, not a public forecast. The linear allocation and PLE map have matching parameter shapes but different effective capacity, which remains an explicit comparison caveat. No V9 fit, prediction, provider upload or grade has been run; the source and packet are pending independent root review and conditional authorization.

To build the private packet from the downloaded V8 control without opening full data files, pass its downloaded result directory explicitly:

```sh
python scripts/build_e540_kaggle_v9_packet.py \
  --control-run-dir /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/kaggle_v8_terminal_20261008_r1/downloads/results/e540-v8-fp32-20261008 \
  --output-dir research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/kaggle_v9_source_review_20261008
```

The builder hashes only the small control artifacts and existing source/metadata inputs; it does not import Torch, open Parquet footers, or load target arrays. Its output remains pending until root reviews the exact archive and issues a matching authorization.
