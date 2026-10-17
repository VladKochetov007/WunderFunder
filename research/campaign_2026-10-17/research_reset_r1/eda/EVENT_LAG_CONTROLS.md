# Fixed-lag input event controls (no fit)

This audit checks whether simple event indicators from the two instrument inputs retain short event-time dependence within sequences, and whether cross-instrument alignment differs from a fixed phase-shift control. It is descriptive input-stream evidence only; it does not test target predictability or forecast value.

## Scope and method

- Used the first eight row-group-sorted sequences from each existing seeded `summary.json` split/class stratum: 8 × 4 × 2 = 64 complete sequences, 20,000 rows each (1,280,000 rows total). The manifest and summary hash are embedded in the JSON result. This is a deterministic subset of the prior seeded sample, not a fresh representative draw.
- Read only the official `i0_p*`, `i1_p*`, `i0_v*`, and `i1_v*` components. The official data overview identifies indices 0–10 as bid-side and 11–21 as ask-side, while warning that indices are identifiers rather than physical depth order. The paired event here means only that `i0_p0` and `i0_p11` both changed from the preceding row. The price-any events are any of the 22 price-like components changing; volume-any events are defined analogously.
- Excluded trade-price, trade-volume, additional features, targets, scoring masks, and `seq_ix` as predictors. No target or score column was read; no model, fitted parameter, label-selected lag, or row-level significance calculation was used.
- Split every sequence into two 10,000-row halves. The first event flag in each half is false, so no change is carried across the half boundary. For fixed lags 1, 4, 16, and 64 rows, computed the binary phi coefficient between `i1_price_any22[t]` and `i0_price_p0_and_p11[t+lag]`; negative direction pairs the events in reverse order. Also measured within-half autocorrelation for each event series.
- As a negative control, circularly shifted the `i1_price_any22` series by 1,000 rows separately within each half and recomputed the same fixed-lag statistics. This preserves the event count and cyclic autocorrelation of that series while changing its phase against i0. It is a phase-association control, not a causal or exchangeability test.
- First averaged a sequence’s two halves, then summarized across the eight sequences in each split/class stratum. The halves and rows are not treated as independent samples. Class strata are inherited from retrospective labels used by the existing seeded sampler; the labels were not loaded or used in this calculation, but this conditioning and the first-eight selection limit generalization.

## What the control shows

The input event series have substantial short-run persistence in this subset. For example, the median sequence-level autocorrelation of `i0_price_p0_and_p11` at lag 64 is 0.041–0.431 across the four validation strata; for `i1_price_any22` it is 0.016–0.106. This supports testing compact causal event-age or burst-state summaries as input representations. It does not establish that they help predict either target.

At lag 1, the median `i1`-lead minus reverse-direction phi is positive in all four validation strata (0.014–0.048, with all eight sequence averages positive in each stratum). The matching training medians are positive in three strata and slightly negative in class 1. At lag 4, the median directional difference is positive in all eight strata, but the sample still contains only eight sequences per stratum and the comparison is exploratory.

The absolute cross-instrument association against the shifted control is strongly heterogeneous. At validation lag 1, observed lead phi minus shifted-control lead phi is -0.027, -0.009, +0.036, and -0.134 for classes 0–3, respectively. Class 3 has a negative raw lead phi (-0.140) and class 2 a positive one (+0.030), while classes 0–1 are smaller. Thus a single universal positive “i1 leads i0” story is not supported; the fixed phase alignment carries different association by retrospective stratum. These signs are not causal and do not imply score gains.

`i0_volume_any22` fires on most rows: its median half-level event rate ranges from 0.87 to 0.99 across the eight strata. That indicator is therefore often dense rather than a sparse event clock; its autocorrelation should not be read as evidence for a useful volume-event representation by itself. Per-stratum lag tables, event counts, and per-sequence/half values are in the JSON.

## Cheapest follow-up and falsifier

Repeat the same pre-fixed event definitions, lag grid, half split, and 1,000-row control on the remaining 24 members of each already-seeded 32-sequence stratum, still aggregating by sequence and without loading labels. Treat the coupling hypothesis as unsupported if the lag-1/4 lead-minus-reverse signs or observed-versus-shift-control differences become mixed or collapse toward the shifted control across the additional sequences. Keep any event-age feature proposal only as a cheap diagnostic candidate until a separate causal validation demonstrates value.

## Reproducibility and resource use

The executable is [`event_lag_controls.py`](event_lag_controls.py). It and the exact manifest were copied to the new append-only Beleriand directory `/home/grinchenkoeg/wunder_research_reset_20261017_r1/event_lag_controls/`; the remote `result.json` was copied here unchanged as [`EVENT_LAG_CONTROLS.json`](EVENT_LAG_CONTROLS.json). The pre-existing remote files `eda_summary.json`, `event_exceptions.json`, `full_event_audit.json`, and files under `patch_staging/` were inventoried and left untouched. The run used one Arrow CPU thread, a 120-second CPU limit and 2-GiB virtual-memory cap; measured wall time was 7.98 s, user+system CPU 7.25 s, and peak RSS 1,207,048 KiB (about 1.15 GiB). Script SHA-256: `3f7d6bab51c4977d0882ee0e502614a2653ce97fca61d28a9afceaa08b78e718`; result SHA-256: `2a1c425afe924a7aca23ac408ee1b19fb206a12959861d9ed3abef06bf5ca109`.
