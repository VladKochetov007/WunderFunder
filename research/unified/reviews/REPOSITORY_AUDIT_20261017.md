---
kind: review
status: complete
created_utc: 2026-10-17
updated_utc: 2026-10-17
hypothesis_ids: []
experiment_ids: []
evidence_paths: [audit-2026-10-17/manifest.json]
---

# Repository audit: established state and evidence boundaries

The October 17 audit reconciled local packages, beleriand, saved Git histories and live GitHub. It examined existing experiments and code; it introduced no model hypotheses, fits, new prediction exports or submissions. This note is the portable repository summary. Large raw artifacts stay at their documented sources.

## Read the structured maps first

| Question | Canonical page |
|---|---|
| What is running or authorized now? | [Current state](../../../specs/current-state.md) |
| What are the current models and packages? | [Model catalog](../models/CURRENT_MODEL_CATALOG.md) |
| Which scores can be compared? | [Validation protocols](../validation/VALIDATION_PROTOCOLS.md) |
| Where are sources, weights and datasets? | [Storage map](../operations/STORAGE_MAP.md) |
| Which exact metadata and public records were retained? | [Hash-bound audit evidence](audit-2026-10-17/README.md) |
| How is the maintained implementation organized? | [Code architecture](../../../docs/ARCHITECTURE.md) |

## Verified public state and artifact identities

The [official leaderboard](https://wundernn.io/connectome/leaderboard) read on October 17 showed **kotiki .66832, rank22, 97 submissions**. The displayed leader was .67243: gap .00411. This is a dated public observation; no private score or future rank was inferred.

The latest creative packages are tl3/tl4, mixing distilled and tiedlead GRU256×2 lineages with fixed coefficients. Server journals record rounded .6683 and 39:12/50:07 runtimes. Their local/server hashes matched. Their own supplied-valid grades are fitted diagnostics because final models used train+valid.

Additional GitHub records contain 81 observations of 70 distinct portal IDs: 65 scored, four timeouts and one check error. This does not reconstruct all 97 submissions or bind the exact current best .66832 to a private portal payload hash. Main E later scored .6675; cross-team seed6 scored .6679; QAT scored .6680. The [model catalog](../models/CURRENT_MODEL_CATALOG.md) separates these from creative production packages and CPU engine variants.

## What the history supports

| Existing comparison | Result | Limit |
|---|---|---|
| Historical heterogeneous distilled pair vs teacher | strict425 .7181503 vs .7153250 | Specific creative protocol; not a public delta |
| FullPLE on older w15/composite | About +.000465 strict425 | Not the same baseline as w19; w19 matched effect was about +.00024 |
| Late low-LR continuation | About +.0009 strict425, but public .6680 vs .6682 | Demonstrated local/public disagreement |
| Full GDN R3 | Pair .7118311 vs matched GRU .7124463 | Finished despite stale local status; delta negative, interval crosses zero |
| Main short architecture screen | H512×1 .673237761 vs H256×2 .669777642 | One seed, exposed F0, fixed short budget; not an overall champion |
| Main warm-parent label-only vs alpha.9 KD | CP5 .6668425394 vs .6811553603 | Parent already had KD ancestry; not a cold supervised baseline |
| Main group-age G vs continuation C | +.00002956; interval crosses zero | Both completed; other related arms only reached engineering profiles |
| Main trajectory averaging | Two paired deltas +.00004351 / +.00005746 | Later receipts supersede stale pending text; intervals cross zero |
| Main QAT and CPU D | QAT .6680/54:26; CPU D .6681/47:27 vs original .6681/50:16 | ZIP compression did not imply acceleration; CPU timings are single noncontemporaneous observations |

Main sources: [architecture frontier](../../campaign_2026-10-14/cpu_student_frontier_r1/FRONTIER.md), [warm-parent quality tradeoff](../../campaign_2026-10-15/student_quality_tradeoff_r1/README.md), [growth/group-age campaign](../../campaign_2026-10-15/gru_parent_growth_r1/README.md), [trajectory/amplitude grades](../../campaign_2026-10-13/strategy_execution_r1/first_results_analysis_r1/RESULTS_R1.json), [QAT frontier](../../campaign_2026-10-13/qat_renewable_r1/FRONTIER.md). Creative history is retained in the server/top15 sources described in the storage map; its numbers are not silently reassigned to Main panels.

## Corrections to the initial reading

1. GitHub was accessible through local SSH. Connector/server failures were route-specific. The live and saved Git graphs are complementary; 174 saved top15 commits are not reachable from current live refs.
2. Main12Oct was an intermediate snapshot. Live Main adds 93 experiment-ledger rows (891 total / 813 IDs) and 27 hypothesis rows (506 total). Repeated IDs and operational entries mean neither rows nor unique IDs count independent fits.
3. Older metric text using per-sequence averaging is superseded by the pooled official metric. Exposed folds, fitted diagnostics, FP32 caches, native grades and public scores remain distinct.
4. A zero-i1 cache was produced by an external prediction script that did not restore its training feature mask. That cache cannot reject the ablation; no corrected inference was run in this audit.
5. Train row group528 now decodes and has finite floating columns. Existing recipes still retain their historical exclusion; no dataset or fitting membership was changed.
6. Fourteen new draft PRs are implementation handoffs with `launch_ready=false`, not fourteen completed experiments. The modular trainer's October17 receipt originally established static checks, not numerical/runtime equivalence.

## Coverage and remaining unknowns

The audit checked all 7,646 files in the primary local package against its manifest, inventoried server trees, reviewed 77 late training logs and 42 cloud run logs, and checked 11 server ZIP manifests/hashes. Nine materialized local ZIPs matched; two cloud-backed local placeholders were not read. Data work covered all Parquet footer counts, full mask counts, small membership arrays and current row-group528 decoding, not a full feature-page corruption scan.

Live Git was inspected at `0edd432271991250f4718f9b68f33340eff46214`: 28 branches, 14 PR head/merge pairs and 2,281 reachable commits. Main text/source extraction covered 35,504 selected files under1MB; inventory is not a claim of line-by-line review of every file. Large historical checkpoint archive payloads were not downloaded and rehashed again.

Unverified items remain explicit: complete private submission history, best ZIP-to-exact-score receipt, private results, live GitHub discussion/approval state, current provider states beyond saved terminal receipts, and explanations for partially logged runs. Existing observations establish neither the task's achievable ceiling nor a unique cause of the public gap.
