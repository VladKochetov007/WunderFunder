# E535 fixed-panel stream report — 2026-10-08

## Finding

The full frozen 468-sequence fold0 evaluation panel was streamed from the supplied validation Parquet in one pass, projecting six columns and reading one complete 20,000-row group at a time. The scan covered 9,360,000 rows, including 808,863 scored rows. It computed descriptive target/activity summaries only: no model fit, predictions, correlation, or public-score estimate.

Target-pair transitions (either t0 or t1 differs from the preceding row) coincided with simultaneous changes in i0_p0 and i0_p11 on 1,566,082 of 1,566,172 transition rows (99.9943%). Every remaining transition had such a paired price-code event within one row. This is consistent with the existing event definition in src/wnn/features.py:44-56, which defines ev_i0px from those same bid/ask price-like columns. The stream therefore validates the data pattern on this entire fixed panel, but it is not independent evidence that a newly added price-event clock supplies new information beyond ev_i0px.

The target and paired-price event ages are also strongly aligned. Rows at least 21 steps after both event types account for 357,279/808,863 scored rows (44.17%) and 49.15% of t0 and 49.39% of t1 official clipped-|target| weight mass. These are substantial stale-regime exposures. The model still must update causal state every row; these summaries do not support skipping row processing.

## Exposure and comparator impact

The current reconstruction of the archived reference panel is a 432-group subset of H230's 468 groups; the historical 425-group membership remains unrecoverable. The 36 H230-only groups are 7.69% of the 468 sequences, but contribute 82,612/808,863 scored rows (10.21%), 10.70% of t0 clipped-|target| mass, and 11.26% of t1 mass. They average 2,294.8 scored rows per sequence versus 1,681.1 in the 432-group reconstruction. Thus an unpaired grade over these panels has different row and metric-weight exposure. This is a concrete comparison limitation, not proof that either implementation is faulty.

The previously reported 0.6674 all-data two-seed teammate endpoint and 0.6649 H230 single fold-trained-member endpoint remain unmatched in fit membership, model count/averaging, teacher identities, and warm-start ancestry. Public values alone cannot identify a data bug or method effect. The appropriate causal follow-up is a fresh-own-weight, fold-safe comparison with the same fit groups, 468 evaluation members, target/cache, initialization ancestry, fixed checkpoints, and serving path; compare single members and their frozen equal mean on that same panel. Historical 425 membership should not be imputed from the current 432 reconstruction.

## Metric and target observations

The scan uses official prediction rows (sequence rows at indices 99 onward) intersected with is_scored and reports only the official clipped target magnitude weights, not a score. See docs/data_overview.md:35-52, 137-177 and src/wnn/gmetric.py:14-38, 173-215. Across scored rows, roughly 22% of each target is zero and roughly 3.3–3.4% has raw absolute magnitude above 2; official scoring clips both target and prediction to ±2. These are exposed validation-label distributions, not a claim about an undisclosed target law. The exact target law remains unknown.

The six-column diagnostic does not establish quote rank or “best price”: docs/data_overview.md:108-131 says p indices identify columns, not depth rank. A paired i0_p0/i0_p11 change is only an event proxy. The fact that all but 90 target transitions co-occur on-row, and all occur within one row, should be treated as a potentially mechanical relationship because the input event flag uses the same two fields.

## Revised scientific test

Keep the numerical prior for E535-HYP-01 null. Narrow the proposed mechanism: do not add a second target-event clock as if it were independent of ev_i0px. A useful unconventional descendant is a feature-equivalent semi-Markov GRU: give the baseline and candidate identical current causal event features, then change only the candidate's hidden-state decay/update gate as a function of event age (including the existing since_i0px signal) while still processing every row. The sizable clipped-weight mass in long stale intervals makes this an interpretable test of memory persistence and decay rather than a transition-only fit.

Discriminating test: on identical fold-safe groups and own-weight ancestry, compare a standard row-wise GRU with a candidate whose update/decay gate is conditioned on the same ev_i0px and since_i0px features; keep seed, optimizer, updates, targets, cache, sampled rows, fixed epoch and served member count identical. Include a predeclared gate-shuffle or age-ablation control if resources permit. A null result on the complete fixed-panel endpoint, or a gain that disappears with feature-equivalent baseline inputs, would falsify the architectural contribution. The current fold0 is exposed, so even a positive result remains development evidence and needs an appropriately fresh confirmation.

## Reproduction and bounds

The full machine-readable result is E535_FIXED468_VALID_ROW_GROUP_STREAM_V1.json; its membership hashes bind fold0.json, the current reference crosswalk, and both group lists. The six-column helper is E535_FIXED468_SCAN_V1.py. The plotted summary is in plots/E535_FIXED468_event_target_weight_diagnostic_V1.png and its SVG companion.

The actual scan used PyArrow ParquetFile.read_row_group with use_threads=False, one group per iteration, from the active Marimo kernel. The appended notebook cell at model_data_forensics.py:187-194 invokes the helper with runpy; it starts no fit. The approved outer limit was one CPU, 1,500,000,000 bytes and 900 seconds; the helper also had an in-loop cutoff at 780 seconds. The enclosing identified Marimo scope was limited to one CPU, 1,500,000,000 bytes and 20 minutes. At post-run observation its cgroup MemoryPeak was 464,908,288 bytes, CPUUsageNSec was 9,889,891,000 for the whole server lifetime, and memory.events recorded zero max, OOM, or OOM-kill events. The helper's scan elapsed time in the result artifact is 1.82 seconds.

The old kernel's 30-minute RuntimeMax timeout, a replacement-launch attempt from a worktree without its own .venv, and a first notebook data-path attempt against an absent worktree-relative dataset are preserved as harness failures. None read a Parquet row group. The successful run used the existing supplied file at /home/vlad/development/wunder/wnn_connectome_starterpack/datasets/valid.parquet. No full-table read, fitting-set scan, prediction array, fit, or grade occurred.
