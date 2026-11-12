# Static review — 10 November 2026

Scope: source inspection, source/archive identities and packaging only. Reviewers:
root packager, independent `submission_dedup_audit`, CPU callback author
`cpu_lstm_candidate` and receiver author `cpu_receiver_checks`. No reviewer ran
local tests, constructed/executed the model, loaded native code, built Docker,
started a remote job or submitted to the portal.

## Findings and fixes

- Preserve exact PR53 checkpoint identities and its train-only W18 canonical
  baseline. The previously uploaded PR49 mixture instead contains fbc1k. PR54's
  full-corpus W18 is a different checkpoint and is not substituted here.
- Legacy fit metadata uses source-keyed row lists, whereas its evaluation list
  belongs to official valid via `eval_source=vfold`; the builder now validates
  that actual schema. LSTM metadata separately uses source-qualified pairs.
- Pin source archives, producer manifest/training SHA, fixed-mixture report,
  all checkpoint bytes, native libraries and vendored threadpoolctl/license.
- Keep exact analytic GELU and LSTM i/f/g/o equations, saved normalizers,
  FP32 recurrent state, unmodified first-two output heads and FP64 blend/cast.
  Both recurrent branches advance on warm-up and reset on sequence changes.
- Align receiver INPUTS/recipe field names with the actual packed schema.
  Resolve utils.py before isolated-Python imports and package sibling imports.
- Stream bounded ZIP extraction without a pre-bound `testzip` expansion. Fix
  footer identity, dedicated lstm.npz reference selection, one-time Torch
  interop configuration and the IPv6 loopback route check.
- Use persistent `/outputs/scratch` under the read-only container. Bind the
  complete profile receipt to package/inputs/recipe and receiver/scorer source
  identities. Full validation rejects duplicate seq_ix and mismatched coverage.
- Record process CPU time, RSS, cgroup peak memory, versions and actual callback
  timing. Do not turn missing hidden-test cardinality into a fabricated runtime
  pass, or a positive exposed point difference into a significance claim.

Ruff0.16.7 formatting/lint, Python3.11 AST parsing, JSON parsing, `bash -n` and
`git diff --check` pass. Hash/CRC reads verify the final ZIP and readback matches
its frozen source. None of these checks executes its predictor. Original failed
packaging attempts are preserved outside Git; only final build-r4 is delivered.

## Outstanding receiver evidence

Linux ABI compatibility, numerical parity, complete state checks, actual served
WP/delta, CPU duration and peak memory remain unmeasured. No Docker image was
built locally. The strict receiver may expose environmental issues when first
run. Its installed dependencies are pinned within the official documented
ranges, but the exact portal wheel versions and hidden-test size are unknown.

PR53's +0.0001246378 is an exposed research-predictor difference, not a score for
this ZIP. Quantized native GRUs and continuous tiedlead feature state may change
that difference. Historical consumed teacher-cache ancestry is not completely
recorded. The packet remains **PACKAGED_NOT_EXECUTED**, `ready_to_submit:false`.

## Receiver R2 review

The scored-row retention change was independently reviewed by
`submission_dedup_audit`. Full replay still advances every raw row; target,
base, candidate and seq/step arrays use the same ordered official mask and
check the exact selected-row count. A diagnostic mask-independence regression
was caught statically and fixed before publication: diagnostics retain raw
mixture values on unrequested rows, while full scoring uses actual returned
outputs. Ruff/AST/shell/source-hash checks pass; no numerical equivalence or
throughput result is claimed. See [RECEIVER_R2.md](RECEIVER_R2.md).
