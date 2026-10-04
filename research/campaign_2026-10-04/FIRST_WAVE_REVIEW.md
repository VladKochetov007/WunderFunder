# Controller admission: first scientific wave

The implementation-before-fit rule applies to every admitted experiment's full
code, controls, data lineage and result path. Open research cards are not claimed
implemented and cannot enter the queue until their own checks pass. Subsequent
branches can be prepared in isolated worktrees while this checked wave runs.

## H-065 / E-410: complete-predictor comparison

Fit the historical one-epoch head procedure on the four fixed combo parents.
Primary comparison is frozen in `specs/combo_merged_screen.json`: equal parent/head
merges against the incumbent recipe's existing equal parent/head merges. No
mixture search, alpha search, checkpoint selection or holdout tuning is allowed.
Evaluation uses the canonical component-excluded folds, which are historically
exposed. Report fold and pooled WP, paired component bootstrap and class guards;
do not treat these intervals as seed/search uncertainty.

The head uses AdamW, seed3, chunk2000, batch16, fixed epoch0, inherited input
preprocessing and structural/next auxiliaries. Training includes only the
parent fold's allowed validation groups. The parent's trunk and input projection
are frozen. Fresh CTRL-head runs are transport checks if backend changes; the
primary deployed-recipe comparison remains predeclared.

The historical-pool full-data combo is separately authorized (H-065). It uses
train plus validation development rows, excludes the old holdout, retains the
original teachers and fixed epoch3, and then receives the same head procedure.
The old holdout is not used to tune this run. Preserve legacy teacher cache bytes
and histories; two old valid prediction caches have no contemporaneous sidecar,
and this limitation must not be hidden by manufacturing one now.

## Evidence supporting admission

- Local CPU and CUDA GRU update/chunk-inference compatibility passed.
- Actual trainer synthetic parquet replay passed on CPU and CUDA, including
  fused Adam, pinned loaders, auxiliaries, EMA, optimizer-switch state and
  no-update final artifact recovery. CUDA retry v2: three tests passed.
- Exact inference-state parity with/without state saving is checked.
- Training, split, portability and recipe tests passed. Source and every
  consumed data/model/cache file are sealed before each attempt.
- Real parent and worker cgroup limits were inspected. OOMPolicy=kill was
  confirmed to set memory.oom.group=1. GPU allocator and independent observed
  total-VRAM monitor use separate byte ceilings. No OOM occurred in checks.

Runtime changes are frozen during each admitted attempt. Readiness does not
authorize unsupported future entrypoints. New representation/teacher work
continues separately; historical negatives are controls, not vetoes.

Submission packaging follows measured native parity/runtime checks. Competition
uploads remain the user's action. No public improvement is inferred in advance.
