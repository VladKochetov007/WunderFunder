# FG19 Native Serving Preparation

This receipt records the native EMA serving preparation for the FG19 467-input
GRU path. The implementation is integrated on main as `7a46eedd4`; this
worktree holds the bounded v2 build and the remaining small delivery changes.
The serving path supports the control suffix and causal EMA suffix, preserves
the existing 211- and 323-input modes, and fails closed on unsupported sparse
representations. The producer source was frozen at
`68ff682977b822f38351aac0f621146f5f13b440`.

## Evidence and status

- `receipt.json` records source and artifact hashes, ABI details, build command,
  resource measurements, the synthetic producer EMA comparison, and outstanding
  checks.
- The bounded v2 build passed `scripts/check_submission_abi.py`. The resulting
  extension is outside the repository at
  `/home/vlad/development/wunder/research_tmp/overnight-ema-native-20261020-artifacts/native-built/v2/wnn_native.abi3.so`.
- The synthetic EMA comparison used three independent sequences, 1,025 rows,
  112 features, and seed 20261020. The float32 online recurrence starts at
  zero for each sequence and agreed with the frozen producer helper to max
  absolute error `3.54e-8` (mean absolute error `5.86e-9`). This checks the
  mathematical recurrence; it is not a loaded-extension callback or checkpoint
  parity claim.
- `solution_multi.py` now reads the package's existing `precision.txt` and
  passes its value to `Predictor`; old packages without that file use `f32`.
  This restores the already-supported `q16x` and `q8vnni` package choices along
  with `f32`, `f16`, and `i16`. Package build defaults are unchanged, including
  the legacy v2/avx2/v3 set.
- The first bounded Docker attempt was stopped after about 26 seconds because I
  initially misread the dependency-install restriction. It was downloading
  Cargo.lock-pinned crates, which the parent then explicitly authorized. The
  transient container was removed, its partial target scratch was cleaned, and
  no Python environment was changed. The successful v2 build used the pinned
  lockfile.

## Remaining release checks

The v2 shared library still needs an import and scorer-callback check in the
packaging runtime, including reset behavior across independent sequences. A
real checkpoint is not available in the staged source area, so its appended
normalizer values, exported metadata, and predictions remain unverified. In
particular, verify that the exported 467-wide checkpoint has the expected
shared normalizers (appended mean 0 and standard deviation 1) before packaging.
No FG19 score, checkpoint parity, or serving latency is claimed here. A v3 build
is held pending the parent's two short local GPU profiles and decision that it
is still needed.

The JSON receipt is the compact reproducibility record. The parent owns the
campaign index and will add the navigation link.
