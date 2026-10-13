# QAT deployment consumer review R1

This is a read-only consumer review of MAIN's `scripts/evaluate_qat_deployment.py`
against the Python packed archive and native Q8 import interfaces. It is not an
approval of the quantizer or native implementation; those require the separate
critic assigned by the campaign controller. The review covers the source state
identified below and records the dependencies resolved during the review.

## Finding

The fixed evaluation path uses the intended full sequence groups, verifies each
evaluation sequence ID and `(20,000, 112)` feature shape, runs the official
`global_wp_clip2_v1` accumulator, and masks rows using `is_scored & need_prediction`.
Warm-up NaNs therefore remain outside the metric. `EnsemblePredictor.predict`
resets state on sequence changes; the first full sequence is explicitly reset
and replayed bitwise. Fit-only profiling selects two distinct hashed fitting
components, separate from `evaluation_groups`.

The native Q8 case and packed archives also line up at the representation
boundary: archive reads return exact signed code and float32 scale maps; Q4 is
expanded into signed byte codes while retaining the Q4 scales; the native
dynamic-Q8 importer routes those arrays through `DynQ8::from_rows`. Head block
0/1 correspond to hidden/projection columns. The evaluator's recorded serving
semantics correctly distinguish native I16 input projection, A8 matvec inputs,
and FP32 recurrent state from training's FP32 input projection and state.

Two integration blockers found in the first source state were resolved before
this review receipt was written:

1. Preparation initially supplied original FP32 weights alongside native Q8
   codes. The archive writer correctly rejects that mismatch because selected
   matrices must equal their exact code-times-scale representation. MAIN now
   reconstructs the selected blocks with `represented_rows` and supplies those
   effective tensors with the untouched native codes/scales.
2. The state diagnostic initially requested `Predictor.hidden_state` before
   that property existed. The native worktree now exposes a detached, read-only
   `(layers, hidden)` snapshot; the native actor reports that the new library
   test passes. The native source changes are still uncommitted in the reviewed
   worktree, so the source hashes below identify the API contract rather than a
   released native revision.

No remaining mask, sequence-transition reset, manual reset/replay, code/scale
import, or Q4 transport mismatch was found in the reviewed path. The following
small guards would make later runs harder to mislabel or reproduce incorrectly:

- `case_weights()` currently discards archive metadata. Check that `packed_q8`
  declares `recurrent_head`, and each W4 case declares its matching preset,
  before recording a result. Preparation currently writes the intended
  preset/file pair, so this is a defense against a mismatched frozen artifact.
- Add `scripts/package_submission.py` to `source_files`: `prepare()` calls its
  `load_checkpoint_member()` when materializing the parent. The exported parent
  tensors and checkpoint are already hash-bound, so this is a provenance and
  reproducibility gap rather than a transport-integrity bypass.
- Apply the evaluation path's sequence-ID and shape assertions to the fit-only
  state diagnostic too. It currently trusts the frozen row-group list there.
- The packed-Q8 archive policy statistics are calculated from the already
  materialized Q8 weights passed to the writer. Do not interpret those fields
  as the original FP32-to-Q8 error; use the parent tensors and state diagnostic
  for that comparison, or preserve a separate source-weight error report.

This was a static consumer audit, not a fit or an independent full evaluator
run. The root controller reports that the updated native library passed its
getter test. No quality, runtime, or submission claim follows from this review.

## Reviewed source identities

Hashes were captured from the named MAIN and native worktree files on
2026-10-13. The native worktree was at base commit `df93b75b` with the listed
modified Rust sources uncommitted.

| Source | SHA-256 |
|---|---|
| MAIN `scripts/evaluate_qat_deployment.py` | `8da58ca23358cacbda25455f486ed8601927d91695ef2cd82f55fd261f6cba4f` |
| MAIN `src/wnn/quantization.py` | `fbca8fff8af5ae488eea2c2b53e304fd3620fdaf32f1f213ac499378b0261298` |
| MAIN `src/wnn/quantization_export.py` | `06347fec1955af937285c2751694dfc395d4f7c48d673803190ce85e3f8dce90` |
| MAIN `src/wnn/gmetric.py` | `2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa` |
| MAIN `src/wnn/splits.py` | `b62797867647cc2244349700bdcf3b2e5c44fd2317b80b39a340edbbed77fec4` |
| MAIN `scripts/engine_fp16_check.py` | `ebb160667d09193b58255691f0a2433bc47f2d285e34deb199b1758d728582b9` |
| MAIN `scripts/package_submission.py` | `1e8ae603ae0bcc8d0d85c5a1ffb95674aa5e3f0b640e8df267e29b5c0751807c` |
| MAIN `utils.py` | `e8e1034d61be359ebe943015e4e6bd2f76716303eff520857b7d07eeeb6144a6` |
| Native worktree `native/src/python.rs` | `b336ceee2e170c9b6176c5008183a4194c5b05e0ab933d893aec0b01a8be54a5` |
| Native worktree `native/src/rnn_fast.rs` | `315fa95cebc67a805d217456f439cd39b98d3f2946981e30610945b187e8b841` |
| Native worktree `native/src/vnni.rs` | `7c27e84fef4820dca2e3478b6fb6a6a2715dbc6937673511e616deaa9af75273` |
