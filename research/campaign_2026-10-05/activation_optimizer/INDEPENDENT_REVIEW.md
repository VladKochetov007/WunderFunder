# Independent review: activation × optimizer screen

Reviewed 2026-10-05. The v1 source had a fail-closed defect: it clipped targets
and predictions before checking finiteness, so infinities could be converted to
valid boundary values. The full-sequence evaluation path also lacked an explicit
finite-prediction guard.

The final v2 trainer fixes those cases narrowly. It validates raw metric arrays
before clipping, rejects nonfinite raw labels before the unchanged target clip,
and checks every full-sequence prediction before selection and scoring. The v1
trainer/test SHA-256 values are `3372619801e7e17ff4e37dde9f2f6ac1d018deab9b592efca2bc550de2e6cc1d`
and `583c57cdecc1af8ccd09fbadfef6e0d83b84fad53a7bcb082be62f338096c5ee`.
The sealed v2 trainer/test hashes are `871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9`
and `cfb5d94abcf732dd632943177cec2a404134240b45873329168c969795b963db`.
The v2 diff preserves the model, fixed 16 arms, data order, mask, optimizer
settings, loss, clipping bounds, and 474-update schedule.

The regression tests reject NaN and both infinities for raw metric inputs,
labels, and prediction guards, while confirming finite out-of-range labels keep
the same clipping behavior. The guarded test command passed:

```bash
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=2G WNN_RUNTIME_MAX=600 scripts/limited \
  .venv/bin/python research/campaign_2026-10-05/activation_optimizer/tests.py
```

The root's baseline-parity receipt records exact tanh initialization, forward,
and AdamW-step parity with E467, at 8,235 parameters. My decision is **GO for
the guarded B4 × 256 real-GPU probe**. That does not itself authorize a full
matrix fit; that launch remains controller-owned and conditional on probe
success. The fixed 16-arm screen must retain every declared activation and
optimizer arm, including both sine directions, sinc, and split activation.

## Scoring receipt

`score_activation.py` reads the completed `runs/sixteen_v2` exports without
changing them. It checks the sealed trainer/test hashes, frozen cohort identity,
the ordered 16 test components, exact per-sequence row counts, raw finite values,
and selected targets against the source parquet mask and labels. It recomputes
all arm scores with `global_wp` and refuses partial or misaligned exports.

The bounded scoring command completed under 1 CPU / 2 GiB and produced the 8 × 2
WP heatmap, optimizer contrasts by activation, and 30 paired component
contrasts: the preregistered 14 activation-minus-tanh contrasts within optimizer
plus 8 MuonAdamW-minus-AdamW contrasts, and 8 clearly secondary GELU-relative
controls for sine, centered-sine/cosine, sinc, and split GELU/LeakyReLU. The
secondary comparisons were added after the point scores showed plain GELU
outperforming tanh for AdamW; they prevent a general GELU change from being
misread as evidence specific to periodic terms. All contrasts use 2,000 paired
component resamples with seed 271828. They are conditional on one fit seed and
16 historically exposed components, with no multiplicity correction; they are
exploratory and are not public-score evidence.

The scorer SHA-256 is
`b85eb93af079d1b003b1fcca2e474d39c5bb8942d95b90a699e13d8da4ea3011`. It ran
under the same CPU/RAM guard:

```bash
WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_RUNTIME_MAX=600 \
WNN_SLICE=wnn-campaign.slice scripts/limited \
  .venv/bin/python research/campaign_2026-10-05/activation_optimizer/score_activation.py
```

It accepted all 16 sealed arms and 26,990 aligned scored rows, and wrote
`paired_components.json` plus three plots under `eda/sixteen_v2`. The plot uses
independent CI line segments and point markers, so a percentile CI that does
not contain the point estimate is represented without changing the interval or
causing negative error-bar lengths. The earlier rendering-only attempt is
preserved in `eda/sixteen_v2_incomplete_plot_attempt`; no fit was rerun.

## Separate follow-up design

Keep E470's 16 fixed arms and all 30 scorer contrasts intact. A separate
preregistered 34-recipe mechanism screen can isolate placement, dropout,
scheduling, and optimizer learning-rate effects without retroactively changing
the completed fit:

1. For placement, compare `tanh` at both sites with plain GELU and GELU+0.1
   sine at both sites, embedding only, and residual-hidden only. At an inactive
   site retain tanh as the reference. Cross those seven unique placements with
   dropout off/on (`p=0.1` after each active pointwise site), using AdamW, LR
   0.001, and a constant schedule: 14 recipes.
2. For dropout and schedule effects, use tanh, GELU, and GELU+0.1 sine at both
   sites; cross dropout off/on with constant LR and a fixed warmup-stable-decay
   schedule (`wnn.optim.wsd_lambda`, 5% warmup, final 20% decay to 1% LR). Fix
   AdamW and base LR 0.001: 12 recipes. Three are shared controls with the
   placement block, leaving nine new recipes.
3. For optimizer-by-LR interaction, use those three activation controls at both
   sites without dropout. Cross AdamW and MuonAdamW with base LR 0.0005, 0.001,
   and 0.002 under both constant and the same fixed warmup-stable-decay
   schedule: 36 recipes. Six match the AdamW recipes in the preceding block;
   after deduplication, the three blocks contain 53 unique recipes.

For the optimizer comparison, report MuonAdamW-minus-AdamW at each matched
activation, LR, and schedule, alongside the LR response within each optimizer.
Do not call the two families equally tuned or rank the optimizer family from a
single shared LR: Muon's update geometry differs from AdamW. Keep steps, seed,
component split, batch/chunk shape, clipping, and scoring fixed; save every
recipe. Run this as a new component-level diagnostic with at least two fixed
training seeds before making claims beyond a mechanism screen. The existing
16 exposed test components may motivate hypotheses but must not select or
confirm the follow-up.
