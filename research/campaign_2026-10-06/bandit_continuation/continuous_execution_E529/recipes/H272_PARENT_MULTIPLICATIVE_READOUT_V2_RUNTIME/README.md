# H272 parent multiplicative readout V2 runtime

This isolated package defines two fixed seed-4 parent recipes. `cleanfold` is a
source-qualified fold-0 discovery screen with a matched hidden-only control;
`full_data` is a deployment-only head recipe and has no prediction or scoring
command. Both fit only two readouts from the frozen H272 four-control
initialization, preserving the original constructor order and rank-32
multiplicative equation. The trainer never changes either parent.

The full-data parent uses the exact `prop_v1` checkpoint and model-bound cache
files recorded in its recipe. Its E-031 record reports 741 fit-devA and 764
monitored-devB groups, but the original runtime source bytes and membership
provenance are unattested. This uncertainty stays attached to the recipe; it is
not converted into a held-out claim or an exact reproduction claim.

## Validate source and data identities

From the repository root, use the isolated runtime script and the exact recipe
path. Validation loads checkpoint/provenance metadata, split membership, cache
headers and hashes. It does not load training sequences, predict, score, or fit.

```sh
.venv/bin/python research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/recipes/H272_PARENT_MULTIPLICATIVE_READOUT_V2_RUNTIME/scripts/train_parent_readout.py \
  validate --recipe research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/recipes/H272_PARENT_MULTIPLICATIVE_READOUT_V2_RUNTIME/recipes/H272_PARENT_READOUT_V2_CLEANFOLD_20261008.json \
  --data-root .
```

The same command accepts the fixed `H272_PARENT_READOUT_V2_FULL_DATA_20261008.json`
recipe. Before any profile or fit, a root-issued authorization receipt must bind
the final recipe SHA, the complete source lock, and its source map. The runtime
rejects absent, stale, or mismatched receipts. The package itself grants no
profile, fit, export, provider, or grader authority.

## Profile, fit, and export interface

The exact commands are shown by `--help` and pinned by the source lock. Root
must supply separate `profile`, `fit`, and export receipts after source review,
actual-device profile, fresh quota, memory and whole-worker admission. Profile
uses eight timed 2,000-row updates after two warmup updates. Fit preserves both
full epochs and fixed epoch-1 selection, with 14,920 updates per head per epoch
for cleanfold or 15,130 for full data. It writes no prediction arrays.

Only cleanfold has the fixed-order FP32 prediction export path. The full-data
recipe refuses that command and supports only the multiplicative head parameter
export for later serving review. Neither path chooses a blend or checkpoint by
score. Fold-0 grading remains separately authorized and is limited to the
frozen 468 evaluation groups and official `global_wp_clip2_v1` mask/metric.

## Local native bundle path

The E416 native scorer now consumes its existing causal event context and
512-wide `[recurrent, projected]` state to compute the rank-32 multiplicative
correction in FP32. `tests/test_native_parent_readout_parity.py` checks full
sequence parity against PyTorch, reset-on-sequence behavior, the 99-row warmup,
causality, and deterministic repeated calls. It uses synthetic input sequences;
no candidate checkpoint or score is read by that test.

After a fixed parent/head pair is actually admitted, fitted, reviewed, and
exported, build the native extension and invoke `serving/build_bundle.py` with
the recipe-bound parent checkpoint, the recipe's exact head-export directory,
and the extension path. The builder verifies the original parent and fixed
epoch-1 head identities, tensor names/shapes/finiteness, writes a local
`submission.zip` plus JSON receipt, checks the exact ZIP member closure and
SHA-256 values, and rejects archives above 20,000,000 bytes. It has no upload
code. Example path arguments are intentionally absent until the fit checkpoint
and platform-built extension exist.

The bundle is only a packaging and native parity step. A real fitted bundle
still needs full public-runtime parity, evaluator RSS, inference-time and
complete served-endpoint review before any submission decision. No real bundle,
ZIP size, RSS, or runtime measurement is claimed by this source-only handoff.
