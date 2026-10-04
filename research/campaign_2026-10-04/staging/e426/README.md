# E426 staged implementation state (partial, not launchable)

The active root `scripts/train_rnn.py` and `src/wnn/training_state.py` remain
untouched. Staged modified copies and reviewable unified diffs are under this
directory. No imports, compilation, tests, fits, or launches were run because
E424 remains active.

Implemented in the staging copies:

- `src/wnn/event_ema.py` counts only raw input `a6` endpoint markers (parquet
  column only), groups them by optimizer update window in the exact seeded
  batch order, and builds constant, aligned, or deterministic within-epoch
  shuffled count schedules. Aligned retention follows the preregistered
  `base_decay ** (events_t * updates_per_epoch / total_train_events_epoch)`
  formula, so the product of all teacher decays matches the constant arm.
- The staged trainer opts into clock modes with `--teacher-ema-clock` and
  `--teacher-ema-shuffle-seed`; rejects resampling, partial epochs, row-clock
  EMA, origin replay, or other configurations that would invalidate count
  alignment. It keeps the consistency teacher separate from the inference EMA,
  updates each on its own clock, chooses only evaluation EMA for inference,
  and reconstructs per-epoch schedules from the seeded sampler on resume.
- The staged training-state module has separate teacher/evaluation EMA fields
  while retaining the legacy single-EMA field for old training-state files.
- CPU test drafts cover raw-column event counts, exact schedule normalization
  and deterministic count shuffling, independent EMA replay/resume, and
  evaluation-EMA identity.

This remains **partial and unverified**. Do not launch E426 from it. Required
before executable status: review the patch for trainer path/order assumptions;
add a tiny actual-trainer integration test proving separate teacher/eval EMA
updates in the training loop and fixed-eval checkpoint export; run py_compile,
unit/replay tests, actual synthetic-parquet CPU fit/resume, and checkpoint reload
under the authorized resource envelope. The tests drafted so far exercise the
clock, state helper, and EMA update behavior, but do not yet run `train_rnn.py`
end-to-end. `endpoint_events_by_update` currently pre-reads one input-only
column for fit groups; quantify its I/O cost before sealing recipes.
