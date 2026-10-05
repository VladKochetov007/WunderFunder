# E480 preregistration: two-seed replication of activation × optimizer × schedule

## Question and admission boundary

E480 independently repeats the full E476 `2 × 2 × 3` factorial at two additional fixed initializations, seeds 271829 and 271830. E476's seed-271828 result motivated replication of its warm-cosine schedule contrast; E480 does not remove or tune any cell based on that result. Its primary comparison is warm-cosine versus constant-mean at matched arithmetic learning-rate sum. All 12 recipes are retained for each seed, for 24 fits total.

This is a source-only packet. No E480 full fit, CUDA probe, test scoring, ledger update, or provider action is authorized by this packet. A fit requires root source review, content-bound admission, the callable actual-data CUDA preflight, and an independent scorer. The fit actor never loads test row groups or test targets.

## Fixed factorial

| Factor | Levels |
|---|---|
| Activation | `base` (GELU embedding and head); `sine_embed` (GELU + 0.1 sin(x) embedding, GELU head) |
| Optimizer | AdamW; MuonAdamW with Muon only on the four two-layer GRU matrices in three gate blocks and AdamW on all other parameters |
| LR protocol | constant peak `0.001`; 47-step linear warmup then 427-step cosine decay; constant at the arithmetic mean of that exact 474-step warm-cosine trace |
| Seed | 271829; 271830 |

Every arm retains 61,058 parameters, the E474 H64 two-layer GRU topology, common seed-specific `RNNModel` initialization across all matched cells, and the exact E476 activation and optimizer recipes. E480 recomputes each initialization from an explicit seed using the same seed-then-construct order as E474 `make_seed_states`; the actor has no mutable global default seed. It checks seed 271829 bitwise against E474's existing seeded state receipt and builds seed 271830 using that same E474 procedure.

## Frozen data, objective, and cadence

Load exactly the 24 E470 fit components and no test components. Verify the frozen dataset SHA-256 and cohort SHA-256 before updates. For each seed, shuffle those 24 fit-group indices with `np.random.default_rng(seed)` once; reuse that exact order in all 12 cells for that seed. Train one epoch with B=4 sequences, 256-row chunks, 79 carried-state chunks per sequence batch, reset state at each B4 batch, and exactly 474 optimizer updates. Keep E470 causal 42-feature inputs, fit-only normalization, target-absolute-weighted MSE, `need_prediction` mask, global gradient norm clip 1.0, no dropout, and full IEEE FP32 math unchanged.

AdamW uses lr values from the registered trace, weight decay 0.01 and betas (0.9, 0.999). MuonAdamW uses the same global LR at every update, weight decay, betas, gradient clipping and E476 exact partition. There is no schedule adaptation, validation during training, checkpoint selection, or exposed-score selection. At zero-based updates 0, 46, 236 and 473, record the same E476 pre-clip global gradient norm and per-family parameter-update RMS for input projection, recurrent core, hidden head, and output head.

The actor generates all 474 values for each schedule before any update and checks their float64 hashes and values against E476 `preflight_cpu_v5/lr_traces.json`. The fixed E474 source trace remains constant 0.001. The constant-mean E476 trace is 0.00055 per step (floating-point exact mean 0.0005500000000000001 in its receipt); warm-cosine and constant-mean have the same arithmetic LR sum. This matches LR arithmetic area, not Adam/Muon effective update area or parameter-delta RMS.

## Predeclared analysis

The primary within-seed contrasts are warm-cosine minus constant-mean for each of the four activation-by-optimizer settings. Report all four effects separately at both new seeds, plus the complete 24-cell panel and the remaining registered contrasts. A replication screen is directionally supported only if all four primary effects are positive at both new seeds; intervals are reported without pruning or reselecting arms. A mixed result remains a mixed replication. Constant-peak versus constant-mean is a separate LR-amplitude/budget comparison and must not be described as isolating schedule shape.

The independent scorer will evaluate the exact six registered outputs per seed on the previously exposed 16 test components with official `global_wp_clip2_v1`, exact raw-value finite checks before clip, component-paired bootstrap and fixed seeds. These components have prior exposure. Any positive contrast is conditional on these components and the tested initializations; it is not a new-fold estimate, an unseen-market claim, or a public-score forecast.

## Source checks and commands

The CPU preflight command loads the actual frozen fit cohort and applies one fit-only B4×256 update to each of the 24 cells. It records finite losses, changed parameter tensors, the exact first four fit groups for each explicit seed, and parameter counts. Run only under the root's one-CPU/two-GiB helper:

```bash
WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_SLICE=wnn-campaign.slice \
  scripts/limited .venv/bin/python \
  research/campaign_2026-10-05/activation_schedule_replication/fit_activation_schedule_replication.py \
  --device cpu --preflight-only --seed 271829 --seed 271830 \
  --output research/campaign_2026-10-05/activation_schedule_replication/preflight_cpu_v1
```

The source offers the matching actual-data CUDA shape/resource preflight without entering full-fit mode:

```bash
python research/campaign_2026-10-05/activation_schedule_replication/fit_activation_schedule_replication.py \
  --device cuda --preflight-only --seed 271829 --seed 271830 \
  --output research/campaign_2026-10-05/activation_schedule_replication/gpu_probe_v1
```

That CUDA command is callable source only; it has not been run. A pass checks one actual-data optimizer update per cell, finiteness, parameter change, and per-cell allocated/reserved CUDA peaks. It is a gate for root admission, not a fit, result, or full-run resource guarantee.
