# Campaign 2026-10-06 index

## Active bandit continuation

See [live frontier](bandit_continuation/FRONTIER.md), [contract](bandit_continuation/CONTRACT.md), [marimo EDA and plots](bandit_continuation/EDA_NOTES.md), and [independent literature tasks](bandit_continuation/ideation/README.md). The current local limits are 10 CPU cores, 20 decimal GB RAM and 5 decimal GB observed VRAM. Earlier sections below preserve their historical limits and status.

Supplied-validation numbers below are not public forecasts. Confirmed public best is 0.6660. No upload in this pass. Local cap for the stretch probe: 5 cores, 10 GB, no GPU.

| Note | What it is |
|---|---|
| `REPORT.md` | Wave 1. Feature, book, and tiny residual probes do not beat the blend. |
| `wave1_anatomy.json`, `wave1_quiet.json`, `wave1_book.json` | Those measurements. |
| `STRETCH.md`, `wave1_stretch.json` | Age ≥ 32 is two populations. Far rows are the hole. Holding the touch prediction loses 0.085 WP globally. |
| `ONLINE.md`, `wave1_online.json`, `wave1_slices.json`, `wave1_tradehold.json` | Output-space online trackers lose. Last-trade hold on no-trade far rows −0.104. Far+trade 0.549, far without trade 0.424, near 0.859. |
| `EPISODE.md`, `wave1_episode.json` | Linear episode-loss vs row-loss on far no-trade: −0.017. Did not pay. |
| `TINY_GRU.md`, `wave1_tiny_gru_s0.json`, `wave1_tiny_gru_s1.json`, `wave1_tiny_gru_all_s0.json`, `wave1_tiny_gru_mixed_s0.json` | Far-only episode vs row: +0.011 / +0.017. All-row episode: −0.047. Mixed (row elsewhere + episode on far): **−0.023**. |
| `cap3g_transfer/`, `cap3g_transfer_v2/`, `cap3g_transfer_ram12/` | Codex E493/E494 drafts. No profile, fit, or score. |
| `cap3g_cadence_ram12/` | Codex E495 B16 cadence drafts. Receipt status `prepared_unsealed_no_profiles_or_fits_launched`. |
| `reflections/` | Hourly read-only reflection. |

## Codex and Kaggle, checked after the OOM

Codex sessions from 2026-10-06 prepared the capsules above and started `night-e496-calibration-s3-f0-smallcap-v1`. That attempt exited −15 after 752 seconds. Its log is empty. Its own cgroup recorded no OOM kill; the parent cgroup already had `oom_kill` 3 when the attempt started. There is no calibration score.

Live kernel status on 2026-10-06:

| Kernel | Status |
|---|---|
| `vladkochetov07/wnn-e482-t4-sixfit-20261005-v2-runner` | COMPLETE. Already graded in the Oct 5 activation notes. Not a new result. |
| `vladkochetov07/wnn-night-e463-tpu-20261005-v3` | ERROR. Empty log `[]`. E463 TPU preflight, not a scorer. `kaggle_inbox/NOTE.md`. |
| `vladkochetov07/wnn-night-w384-serial-20261005-v2` | ERROR. |
| `vladkochetov07/wnn-w384-fold1-dualt4-20261005-v7-runner` | ERROR. |
| `wnn-night-w384-serial-20261006-v3` and `v3r1` | Not uploaded. Local packets say `prepared_private_not_uploaded`. |

TPU `wnn-night-e463-tpu-20261005-v3` is **ERROR** (empty log `[]`). See `kaggle_inbox/NOTE.md`. The hourly loop stays up. No kernel was launched from this pass.
