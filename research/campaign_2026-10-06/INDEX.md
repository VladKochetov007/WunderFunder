# Campaign 2026-10-06 index

[All proposed directions](bandit_continuation/IDEA_REGISTER.md) includes warm-session H248, causal-feature H249–H251 and teacher-interaction H252, alongside all existing curated arms. Registered proposals are not completed experiments.

## Active bandit continuation

See [live frontier](bandit_continuation/FRONTIER.md), [contract](bandit_continuation/CONTRACT.md), [marimo EDA and plots](bandit_continuation/EDA_NOTES.md), and [independent literature tasks](bandit_continuation/ideation/README.md). The current local limits are 6 CPU cores, 20 decimal GB RAM and 5 decimal GB observed VRAM. Earlier sections below preserve their historical limits and status.

Latest continuation: [E528 measured input reuse and plots](bandit_continuation/decoded_feature_efficiency_E528/E528_REPORT_A2.md), [E529 saved-plan implementation](bandit_continuation/continuous_execution_E529/README.md), and [E530 online-learning family](bandit_continuation/online_learning_E530/README.md). User-reported [C1 public scores](bandit_continuation/submission_C1_E524/E524_PUBLIC_RESULTS_20261007_V1.json) are 0.6647/0.6654, below incumbent 0.6660. Do not reupload these identical singles. No full Maxim-inspired replication has landed: H230 cache/profile are complete but its student fit remains unrun. Local GPU was explicitly reopened after the historical cutoff, and bounded campaign-owned Vast compute is authorized under [ADR0013](../../specs/adr/0013-owned-paid-compute-and-continuous-queue.md); actual access/admission evidence is required. No Vast instance is rented.

Supplied-validation numbers below are not public forecasts. Confirmed public best is 0.6660. No upload in this pass. Local cap for the stretch probe: 5 cores, 10 GB, no GPU.

Current continuation status (2026-10-07 10:55 Europe/Kyiv): E-508-v5 passed actual input identity/membership checks and completed its profile-only T4 run, but terminally stopped before fitting because projected pair fit/export time8524s exceeded the frozen6300s cap; no fit or OOM. E-463-v2 remains `QUEUED` on TPU with the existing poller active. E-520 is a plan only and awaits independent review plus fresh cloud preflight. The local experiment cutoff passed at10:00; subsequent new fits use Kaggle only. See the [live frontier](bandit_continuation/FRONTIER.md), [v5 terminal result](bandit_continuation/raw_teacher_pair/E508_V5_TERMINAL_PROFILE_RESULT_20261007.md), [v5 admission](bandit_continuation/raw_teacher_pair/E508_V5_REPAIR_PREREGISTRATION_20261007.md), and [Maxim replication addendum](bandit_continuation/reviews/MAXIM_STATUS_ADDENDUM_20261007.md).

E-514, E-516, E-517 and E-518 all used the same exposed weak-model pool. E-517's WLS gate selected zero in every fold. E-518's exact-metric fit selected gate betas0,0.01,0.02; the pooled gate scored0.6832229 versus anchor0.6832265 and global control0.6832257. Independent recomputation verified arrays, hashes, endpoints and bootstrap; the registered screen failed. Close all further blending on this pool because calibration repeatedly selected the anchor and the exact-metric gate trailed both controls; interval overlap describes uncertainty and is not the stop reason. Preserve broader complementarity for a materially new, ancestry-audited cohort. Own-baseline/raw-teacher recovery and heterogeneous-teacher compression rank ahead. See [ADR-0009](../../specs/adr/0009-uncertainty-is-not-a-kill-switch.md), the [E-514 result](bandit_continuation/conditional_weak_portfolio_E514/REPORT.md), [E-518 report](bandit_continuation/disagreement_guided_H204/E518_REPORT.md), [independent review](bandit_continuation/disagreement_guided_H204/E518_INDEPENDENT_EVIDENCE_REVIEW_20261007.md), and [delta plot](bandit_continuation/disagreement_guided_H204/plots/E518_DELTA_DETAIL.png). The next read-only bandit reflection is scheduled for11:16:43 Kyiv.

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
