# E483 independent scorer source note

The scorer is implemented in `score_momentum.py`; synthetic-only tests are in `test_score_momentum.py`. It does not import `fit_momentum.py`, E470 fit helpers, or any actor scoring code. It imports only the official runner-spec identity helpers, PyTorch's frozen `RNNModel` only to reconstruct initial-state hashes, and official `wnn.gmetric` functions for `global_wp_clip2_v1` and centered weighted-Pearson moments.

The scorer validates the sealed E483 spec and completed runner attempt; current runtime hashes; worker and parent OOM-counter deltas; all actor/parent/input hashes; the 18-cell fit-only manifest; exact seed initial-state hashes; fit-role-only external normalization; each checkpoint digest, strict state-dict key map and identity buffers; 474 updates and whole-run clipping counts; all-step telemetry receipts; the optimizer group names/effective betas; and disjoint full parameter coverage. It independently reconstructs the 42→64→two-layer reset-after GRU with the registered baseline/sine candidate formulas and H64 GELU concat head.

The CLI defaults to fit-proof validation, without reading test rows. Reading the registered 16 exposed components requires the explicit `--score-exposed-test` option. That path validates raw label finiteness before clipping, reconstructs fit-only normalization from the 24 canonical fit row groups, recomputes all official scores, and writes paired selected raw prediction exports with IDs, masks, row positions, and the common 2,000 sorted-component bootstrap draws. It retains all 18 arms, 27 seed-specific contrasts, target-wise arm metrics, and does not pool fit seeds.

Synthetic tests cover baseline full-model output/state/input-gradient/parameter-gradient parity against the frozen native GRU on CPU, chunk carry, raw nonfinite rejection, official metric versus component-moment pooling, deterministic sorted shared bootstrap counts, all nine contrast formulas per seed including difference-in-differences, and optimizer-group isolation. No exposed test rows, GPU scoring, or training was used to implement or run these checks.

Source-only checks were run under one CPU, 2 GiB memory, and the bounded runtime helper:

```bash
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_RUNTIME_MAX=300 \
  scripts/limited bash -lc 'cd /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-05/activation_momentum && python -m unittest -v test_score_momentum && python score_momentum.py --help'
```

A source/preflight identity check also passed without loading parquet rows. Actual fit-proof validation and exposed-panel scoring remain for the completed fit attempt and explicit root authorization after independent source review.
