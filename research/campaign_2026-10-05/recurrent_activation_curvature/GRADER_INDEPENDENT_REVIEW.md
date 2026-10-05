# E481 independent grader source review

**Source decision: GO for the full receipt gate after the fit attempt finishes; do not score yet.** The current full fit is still running and the run directory has only partial checkpoint artifacts, so no complete manifest or successful attempt receipt is available to validate. I ran no scorer, opened no test-role rows or labels, and made no predictions.

## Exact scorer and synthetic checks

- Scorer `score_recurrent_curvature.py`: SHA-256 `52b91f9a3eaead30c71ef824feed8370b5a1e339ad7f8005c509b115b903c4d5`.
- Synthetic tests `test_score_recurrent_curvature.py`: SHA-256 `b233bf363ce111d96ffb81194dd54249e1d4528a8eeecdbc889fe56932b67379`.
- The scorer's bound actor, test, probe, preflight-v2, E470 and E474 identities match the reviewed E481 packet. It validates the sealed spec, runner/runtime/OOM/teardown receipt, fit-only cohort, exact 18-cell grid, updates, seed-specific row order, common B=4/chunk=256 cadence, telemetry family values, checkpoint hashes, strict key layout and identity normalization buffers before it calls the test-inclusive loader.
- Raw test targets are checked for finiteness before metric clipping; the scorer checks source masks, exports selected raw targets/predictions and component provenance, recomputes official pooled WP moments for shared component-bootstrap draws, retains all 30 registered contrasts, and keeps the three seeds separate.
- `ruff check`, `py_compile`, and all six synthetic scorer tests passed under the bounded CPU helper.

## Current partial fit artifact inspection

At inspection time the E481 runner attempt had status `running`, with no `fit_manifest.json` yet. The partial directory held four seed-271828 checkpoints: AdamW baseline, sine and cubic, plus MuonAdamW baseline. I checked each file's digest, tensor-key set against the independent model, finite values, identity buffers and 61,058-parameter topology. Their SHA-256 values were:

- `seed271828_adamw_baseline.pt`: `373f064ccfb20909ad861f6fd6adbc341ca57c0b187250ad559b6895829c77c3`
- `seed271828_adamw_cubic.pt`: `176c535229dcc3edb7682b3f2d504164365dc84fef2a09c0c50877fbfc47c8f8`
- `seed271828_adamw_sine.pt`: `a3250498a8e44f063211964fc1140b7acfff516e0595c1fd7ee966de94602e4f`
- `seed271828_muon_adamw_baseline.pt`: `a0646b501e578c56d81eead33e0a6424dd5943242552dfad1b4483007a617304`

These are partial outputs without an attempt artifact receipt; they do not establish complete or successful fits. The scorer has no fit-only verification mode. Its regular path will validate all 18 checkpoint records and the successful attempt before the first test-role load once the manifest is written.

## Receipt-check limitation

The actual actor records `chunk_start` and `loss` for telemetry samples, but the scorer currently checks sample steps, expected B4 fit-row groups and finite gradient/update RMS without checking those two recorded values. The source identities and fixed training code are bound, so this omission does not alter the prediction formula; it is still a useful receipt validation improvement before scoring. The scorer correctly checks B=4 and chunk_rows=256 at manifest scope; the E481 actor's individual arm records do not contain those fields, so per-arm fields should not be invented or required.

This review is conditional on the eventual fit manifest and final attempt passing the scorer's full validation. A failed or incomplete attempt must remain unscored.
