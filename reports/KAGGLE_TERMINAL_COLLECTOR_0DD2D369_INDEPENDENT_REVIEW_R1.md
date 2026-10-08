# Kaggle terminal collector correction — independent review R1

**Verdict: PASS for the hash-verification correction and bounded read-only collector.** This is a successor review to `KAGGLE_TERMINAL_COLLECTOR_CDB594DF_INDEPENDENT_REVIEW_R1.md`; that earlier HOLD remains preserved as historical evidence. The correction closes the same-size corruption gap in both the collector retry gate and provider poller completion gate.

## Frozen inputs

- Commit: `0dd2d36983eaa42ad63278b81dca559b40413c91`
- `collect_terminal_outputs.py`: `a8073a9404d08560990ecd1dca69cfcd56038b903b4a6d2016b6520c33a98e6f`
- `poll_provider.py`: `d48369d8ff8c3756a9baf37eac269000c319fea5128308b12d4cbdbc2bbb3cd0`
- tests: `3373dd87da4351180b44a47a7000d1b9404d3779a9ac3e7d593a2cf38adcf61c`

## Findings

`_receipt_files_match()` now requires a non-empty receipt inventory, relative in-root files, exact integer sizes, lowercase SHA-256 values, and recomputed file hashes. `collection_is_complete()` uses that check before treating a prior collection as reusable, and `_terminal_evidence()` uses the same check before the poller stops status monitoring. A changed or hashless file therefore returns incomplete and remains eligible for a bounded new collection attempt.

The regression test uses `b"same-size"` and replaces it with `b"tampered!"` (both 9 bytes). It verifies both consumer gates reject the changed bytes; it also verifies that a missing digest is rejected. The output download command now monitors the whole attempt directory, so its 8 GiB cap includes provider logs, outputs, and command streams. The aggregate-cap regression passed.

I applied the new consumer to the actual smoke3 R2 and smoke4 R1 producer receipts and independently recomputed each listed file's size and SHA-256 from disk. Both receipts list 37 files with zero mismatches. The collector accepted both; the poller recognized both as downloaded evidence. Smoke3's provider status is `COMPLETE` while its recorded application outcome says both fits failed before training. Smoke4's provider status is `ERROR`. These receipts establish artifact compatibility only, not successful fitting. A TPUv7 terminal output receipt was not present, so its live-output compatibility remains unobserved.

## Validation

- Test command, run under the shared support lock and 1 CPU / 1 GB / 180-second wrapper: `flock -w 10 /tmp/wnn-codex-support.lock env WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=180s /home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python -m pytest -q tests/test_kaggle_terminal_collector.py` — **15 passed**.
- Ruff check and format check passed for the collector, poller, and tests.
- Python 3.11 `py_compile` passed for those three files.
- `git diff --check` passed. The source changes are limited to collector/poller artifact verification and focused tests; no scientific recipe or model source changed.
- Review was read-only with respect to Kaggle: no CLI/API calls, provider writes, pushes, launches, installation, or worker execution were made.

The shared campaign slice was at 12.39 GB of its 20 GB maximum before the bounded test run, below the 14 GB deferral threshold. No individual verification exceeded the 1 GB support cap.

## Remaining limitations

The independent actual-receipt check covers the smoke3 R2 and smoke4 R1 outputs, not TPUv7's yet-unavailable terminal output. The existing terminal receipt filename remains the current pointer if a later retry succeeds; attempt directories are preserved, while the prior pointer contents are replaced by the latest receipt. Hash verification is recomputed on each poll, so work scales with the total bytes listed in a target receipt.
