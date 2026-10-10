# q8vnni packaging test (local only, not for upload)

Member: bd26_h230_f1_s3_ep3@ep2 (fold-1 held out), solution `submission/solution_native.py`, builds from
`scripts/build_native.sh` (docker `wnn-native-build`, Rust 1.94, Debian bookworm, levels v2/avx2/v3).
Gates: `gate_replay.py` in docker `wnn-python311-submission-check` (Python 3.11.16, numpy 2.2.6, offline,
one pinned core, 1 GB), 30 fold-1 evaluation sequences (600,000 rows), via `run_gates.sh`; metric via `compare.py`.
Results: `gates/*.json`, `gates/metric_agreement.json`, `packages/*/ABI_STATIC.json`.
Reproduce: `scripts/package_submission.py NAME bd26_h230_f1_s3_ep3@ep2:1.0 --native-from packaging_q8vnni_20261010/native
--solution submission/solution_native.py --precision {i16,q8vnni} --runs-dir <main>/artifacts/runs --out-root packaging_q8vnni_20261010/packages`.
