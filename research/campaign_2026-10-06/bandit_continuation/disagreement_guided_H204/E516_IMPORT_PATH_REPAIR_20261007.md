# E-516 import-path repair

At 2026-10-07 08:28 Europe/Kyiv, the first Marimo execution completed source identity checks and E-515, then E-516 stopped at its first import of `wnn.gmetric`: the Marimo kernel had the research helper directory on `sys.path` but not the project `src/` directory. No E-516 coefficient fit or endpoint statistic ran in that attempt; the failure is recorded in `E516_IMPORT_PATH_FAILURE_20261007.json`.

The E-516 cell was repaired through the live Marimo code-mode API (not by editing the running notebook on disk) to prepend `project_root / "src"` to `sys.path` immediately before invoking the frozen E-516 function. The only change was Python package discovery. Candidate arrays, group membership, target masks, objective, simplex constraints, intercept, ridge, folds, bootstrap seed, and decision rule were unchanged. The exact failed E-516 cell was then rerun. No neural model was trained and no new public submission was made.
