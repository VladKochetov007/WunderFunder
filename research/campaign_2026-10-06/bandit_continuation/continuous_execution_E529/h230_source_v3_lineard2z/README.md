# H230 family-head source snapshot V2

This is an owned portable copy of the trainer pinned at
`dde32d0f5e39efe65c0497fec06449d38e18b369`. It contains the full `src/wnn`
Python package and `scripts/train_rnn.py`; it does not copy any teammate code,
data, cache, checkpoint, or weights. See
`H230_SOURCE_SNAPSHOT_MANIFEST_V1.json` for original and resulting hashes.

The family output/cache structure now has an explicit `--distill-family-head`
switch. The effective structural gate remains enabled when the family-loss
weight is positive, preserving old candidate behavior. When the explicit flag
is present, it also keeps the 8 output rows, warm-start extension, raw family
targets, and cache loading at weight zero. For this pair, the explicit flag
holds structure constant and the weight changes only the family-loss
multiplier. Both commands use the same seed, initializer, `source` main target,
family arrays, recipe, and resource request; only the family-loss weight
changes scientifically.

`WNN_PROJECT_ROOT` points data, cache, and initializer resolution at the
project. Code provenance hashes remain rooted at this snapshot, so a later fit
would record the copied implementation rather than the root checkout's
different source. Candidate and control argv are preserved in
`H230_CANDIDATE_CONTROL_COMMANDS_V1.json` and normalized to prove their only
scientific argv difference is `--distill-family-weight` (`0.1` vs `0.0`). The
different `--name` values only isolate output directories.

Focused tests are in `tests/test_family_head_control.py`. They read the
authorized own initializer checkpoint by its frozen hash and check 31-output
and exact candidate/control initial-state parity. They use the actual H230
cache metadata and the first payload row from each real `main_source` and
`families` array to verify the copied reader's target columns for both weights,
then verify that weight zero removes the family-loss gradient while preserving
the main objective. To stay under the 1GB RSS budget, tests copy only those
first NPY payload rows into temporary small caches and provide a tiny source
sequence; they do not map the multi-GB arrays or open the full Parquet datasets.
No initializer weights are written into this source snapshot or its outputs.
The bounded test attempts and resource-limited failures are preserved in
`H230_SOURCE_TEST_ATTEMPTS_V1.json`.

Status: **source and tests only; no profile, fit, export, grade, or GPU launch**.
The command pair is prepared but requires fresh root resource/device admission
before any scientific execution. The resource successor requests one 3-CPU,
9GB worker, 2,400-second fit timeout, 4.3GB allocator cap, and 5GB observed
VRAM ceiling; these values do not constitute live admission.
