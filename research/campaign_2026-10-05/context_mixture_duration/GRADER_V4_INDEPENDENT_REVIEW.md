# E488 independent grader v4 source review

Reviewed v4 after the v3 actual fit-only attempt stopped at the telemetry
schema check. The v3 attempt failed before test-row loading; this review does
not read evaluation labels or calculate scores.

## Exact reviewed identities

- `score_duration_independent_v4.py`: SHA-256
  `ff36d4c5e1672940f000f88b1637fd933efb96631a3521ea32cb068cdb85bd80`
- `test_score_duration_independent_v4.py`: SHA-256
  `2e704845ac961b5d11e16559ba8d743984f3fc2d30dc2f52ecefb14ba9696897`

## Findings

The v4 telemetry check now matches the frozen actor schema: `head_update_rms`
must be exactly a one-key `{"head": finite_value}` family summary, and that
value must equal `parameter_update_rms["head"]`. The actual fit-only manifest
uses this mapping. The duplicate total gradient-norm telemetry is also
required to agree with the recorded preclip global norm.

The new whole-fit-manifest fixture validates all 12 actual fit-only
trajectories and all 24 saved snapshots/checkpoints from the existing run,
while patching the test-row loader to raise if reached. It reads the fit
manifest, checkpoint files and fit-role cohort metadata only. It confirms the
schema correction across all six sampled telemetry steps per trajectory. The
separate source validators still require exact fit/probe code and input role
inventories; `evaluation.py` remains pinned as grader-only dependency. The
verify-only control flow calls the complete evidence validator, writes its
proof, and returns before the test-row loader. Actual attempt/spec/runner
verification must still pass in root's next verify-only execution.

I ran all 11 v4 tests under `scripts/limited` at 1 CPU / 2 GiB; they passed,
including the actual fit-only manifest/checkpoint fixture. Ruff,
`py_compile`, and a clean-`/tmp` `--help` invocation passed under the same
limits. I did not invoke the complete actual-attempt gate, load test groups,
or score.

## Disposition

**GO for root to retry actual E488 `--mode verify-only` using this exact v4
source hash.** Score mode remains gated on that actual attempt succeeding.
