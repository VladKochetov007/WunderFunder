# Isolated external encoding study

The primary bounded pass completed without reconstruction failures on the sampled valid intervals. It establishes useful distinctions between representations, not the competition generator or a public-score gain.

- Inputs: contiguous ten-minute prefixes of BTCUSDT, ETHUSDT and USDCUSDT recordings (274,714 messages total), plus 1,200 MD snapshots and 13,487 received trades. Exact consumed-prefix SHA256 hashes, source sizes and mtimes are in `results_primary.json`; they are not full-file hashes.
- Reconstruction: 16,673 eligible depth updates; zero sequence gaps; one pre-snapshot invalid update in each file was excluded. Recorded snapshots were installed at receipt, covered updates were ignored, quantities replaced by price, zero quantities deleted, and top-25 coverage was checked against initial snapshot bounds.
- Consistency: 2,806 comparisons at exactly matching depth/bookTicker update IDs, zero mismatches. Asynchronous receive-time equality was not required.
- Resource use: 16.17 CPU seconds, 16.30 wall seconds, peak RSS 92,280 KiB, exit code 0. Derived output is a compact JSON summary; no examples, models or numerical parameters were exported to the competition pipeline.

## Representation findings

**Price-keyed edits separate rank transport from quantity edits.** In the chronological held-out thirds, 48 BTC, 20 ETH and 11 MD transitions had nonzero quantity changes at occupied rank slots and zero quantity change at shared prices. Entry and exit at the top-25 boundary were accounted for separately. This falsifies treating all rank-slot differences as interior order flow. It does not identify gross additions, cancellations or executions within a depth batch; MD snapshot net edits remain inferences.

**Normalized liquidity geometry obeyed its declared symmetries.** Across 179 measured-book probes (60 BTC, 60 ETH, 47 USDC, 12 MD), translation, common price-unit scaling, quantity-unit scaling and bid/ask reflection changed descriptors by at most 2.85e-14. The toy rank insertion generated 13 units of rank-slot L1 change and zero shared-price edit. These checks establish the representation contract, not predictive usefulness. Arbitrary per-column normalization does not preserve physical distance or quantity ratios; clipping can collapse genuinely different books.

**Publication policies produce distinguishable signatures.** Holding trade state, clearing it on other messages, coalescing five messages and delaying trades by five messages generated different price/quantity/trade co-change histograms in both chronological partitions. Different histograms can reject a specified publication model, but multiple generators can share the same observed signature. No generator was selected. Exchange event-to-receive lags were roughly 137–138 ms at the median and reached approximately 1.17 seconds, supporting the need to preserve receive order separately from exchange time.

## Validation and boundaries

Seven independent toy checks passed: absent snapshot, continuous bridge and replacement/deletion against a manually constructed book, persistent invalidity after a sequence gap, snapshot receipt causality, unknown coverage after boundary loss, exact decimal identity and conservation, and publication information loss. Ruff format/check passed after formatting and import/explicit-zip fixes. Independent replay is pending.

The first two thirds and final third are chronological partitions of eligible book summaries or publication messages, respectively. The study did not fit a prediction model, evaluate competition targets, compare symbols/timestamps with competition data, or infer competition column mappings. All actual study artifacts remain in this directory.

## Reproduction

The durable primary run used `wnn-external-20260929.service`, the campaign slice, and the command below. `run_primary.log` contains completion/resource output; `results_primary.json` contains the full machine report and exact CLI arguments.

```bash
WNN_SLICE=wnn-campaign.slice WNN_CPU_PCT=7 WNN_THREADS=1 \
WNN_CPU_QUOTA=100% WNN_MEM_MAX=4G OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 \
ARROW_NUM_THREADS=1 scripts/limited .venv/bin/python \
scripts/study_external_encoding.py \
--output research/campaign_2026-09-29/external/results_primary.json
```

Implementation: `book_replay.py`, `encoding_study.py`, and `scripts/study_external_encoding.py`. Checks: `test_external_encoding.py`. Binance stream semantics were checked against its official `binance-spot-api-docs/web-socket-streams.md`; no external learned state is permitted in a supplied-data follow-up.
