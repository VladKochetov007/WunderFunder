# WNN-REFLECT-LOOP — 2026-10-06T07Z

Read-only fire. No upload, no fit, no kernel, no new experiment. Supplied-valid numbers are not a public forecast. Confirmed public best stays **0.6660** (seed-4 Muon/chunk1000/stopped-bootstrap GRU256 combo). Do not add any local delta to 0.6660. Honest expected public 0.7000 is unmet and is not claimed here.

Subagent `01a10fef-324d-7941-baac-800bbf3598a8` was not found in this session.

## Inputs and mtimes (local +0300, as `stat`)

| Path | mtime | vs LATEST T06 |
|---|---|---|
| `research/campaign_2026-10-06/reflections/LATEST.md` (prior T06) | 2026-10-06 09:39:23 +0300 | self |
| `research/campaign_2026-10-06/wave1_tiny_gru_all_s0.json` | 2026-10-06 08:58:35 +0300 | older |
| `research/campaign_2026-10-06/wave1_tiny_gru_mixed_s0.json` | 2026-10-06 10:04:16 +0300 | **newer** |
| `research/campaign_2026-10-06/INDEX.md` | 2026-10-06 10:04:34 +0300 | **newer** |
| `research/campaign_2026-10-06/TINY_GRU.md` | 2026-10-06 10:04:34 +0300 | **newer** |
| `research/campaign_2026-10-06/kaggle_inbox/NOTE.md` | 2026-10-06 08:41:04 +0300 | older |

New since T06: mixed-mask tiny GRU seed 0. Cap-3G capsules still have no score.

## Kernel

One call, no poll: `KaggleApi().kernels_status("vladkochetov07/wnn-night-e463-tpu-20261005-v3")` → **ERROR**. Unchanged from T06. Not COMPLETE, so no download. Inbox note already has the empty `[]` log.

## Ranked measured gains versus the incumbent blend

Incumbent on the full supplied validation file: **0.7007712917864051** `global_wp_clip2_v1` (`wave1_anatomy.json` `scores.blend_full_valid`). Not a public forecast.

Full supplied validation, gain versus 0.700771. Least-bad first. Every full-file probe loses. Tiny GRU is not in this table: it is a 250-seq toy, not a blend replacement.

| Rank | Idea | Score | Gain | Source |
|---|---|---:|---:|---|
| 1 | Sum snapped onto the quantum lattice | 0.6994421743165101 | **−0.001329117** | `wave1_anatomy.json` `scores.sum_lattice_snap` |
| 2 | Each output snapped to the class quantum | 0.6951621753753043 | **−0.005609116** | `wave1_anatomy.json` `scores.independent_lattice_snap` |
| 3 | Causal EMA of the blend, no reset, decay `e^{-1/8}` | 0.6932738925517488 | **−0.007497399** | `wave1_online.json` `full_valid.global_ema8` |
| 4 | Hold last-trade output globally | 0.6889819432861753 | **−0.011789349** | `wave1_tradehold.json` `scores.global_last_trade` |
| 5 | Causal EMA, reset on each joint touch, decay `e^{-1/8}` | 0.6849220949730942 | **−0.015849197** | `wave1_online.json` `full_valid.ema8` |
| 6 | Hold every row at last touch (`global_age1`) | 0.6190401445388347 | **−0.081731147** | `wave1_online.json` `full_valid.global_age1` |
| 7 | Copy opening-touch prediction onto age ≥ 32 | 0.6160922297495068 | **−0.084679062** | `wave1_stretch.json` `full_valid.global_hold_quiet` |

Public scores stay separate: confirmed public best **0.6660**. Nothing here is an expected public 0.7000.

### Stretch result (required)

Holding the touch prediction on age ≥ 32 loses about **0.085** full-file WP: blend 0.700771 → `global_hold_quiet` 0.616092 (**−0.084679**) in `wave1_stretch.json`. Far rows WP **0.481842**. Near rows WP **0.858814**.

### Protocol: tiny GRU episode vs row

Hidden 32, 250/250 sequences, seed 0 unless noted. Kill bar **+0.005** episode minus matched row.

| Seed | Loss mask | Row far WP | Episode/mixed far WP | Delta | Source |
|---:|---|---:|---:|---:|---|
| 0 | far no-trade only | 0.211144 | 0.222179 | **+0.011035** | `wave1_tiny_gru_s0.json` |
| 1 | far no-trade only | 0.206981 | 0.223924 | **+0.016943** | `wave1_tiny_gru_s1.json` |
| 0 | all scored rows | 0.286498 | 0.239161 | **−0.047338** | `wave1_tiny_gru_all_s0.json` |
| 0 | mixed (row elsewhere + episode on far) | 0.286498 | 0.263025 | **−0.023473** | `wave1_tiny_gru_mixed_s0.json` |

Far-only mean protocol delta about **+0.014**. Both far-only seeds clear +0.005. Mixed loss **fails** the protocol bar versus row-on-all. All-row episode is worse. The +0.011 / +0.017 result is two specialists trained only on the hole, not a drop-in extra term on a model that also fits easy rows. Absolute far no-trade WP stays ~0.22–0.29 versus the frozen blend **0.423749** (`wave1_tradehold.json`). Protocol paid is not campaign paid.

Linear episode vs row remains **−0.016997** (`wave1_episode.json`). Affine +0.000072, excluded.

### Far-cell trackers versus far blend 0.481842 (unchanged)

Last-trade hold on no-trade far: **−0.104368**. EMA8 reset **−0.003934**. Age-1 hold **−0.190638**. Quiet-cell residual MLP still **+0.000162365** (`wave1_quiet.json`), wrong cut and below the bar.

## Did a new task setting pay versus the champion?

**No.** Mixed episode+row lost **−0.023** on the hole versus matched row-on-all. All-row episode **−0.047**. Far-only tiny GRU still protocol-only (seed 0 **+0.011**, seed 1 **+0.017**). Linear episode **−0.017**. Stretch hold **−0.085**. No +0.005 versus the frozen blend on the far quiet cell, and no full-valid gain versus 0.700771. TPU kernel is ERROR with no score. The loop stays up.

Scheduler left running. Workspace present. Champion unbeaten.
