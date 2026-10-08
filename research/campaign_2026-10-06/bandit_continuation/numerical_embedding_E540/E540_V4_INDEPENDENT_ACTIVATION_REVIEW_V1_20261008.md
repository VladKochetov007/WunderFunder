# E540 V4 independent activation review

**Decision: FAIL activation pending one runtime correction.** The scientific packet, source seal, membership, timing projection, archive and focused tests pass static review. The Kaggle runner watchdog keeps a timeout computed before runner preflight, so it can exceed the actual-start worker cap and consume the 300-second finalization reserve. Do not push the V4 kernel on this review.

## Scope and provenance

Reviewed actor commit `d15bbf358e21024f97d01ff96c1c59a7b91487f5`, branch `research/e540-kaggle-warm-cohort-20261008`, at `/tmp/wnn-e540-kaggle-warm-cohort`. The actor worktree was clean before and after review. The review was performed from a separate review worktree, without changing actor files.

Applied the `research-evidence-review` and `research-recipe-audit` criteria. This was a static/code-only review plus synthetic focused tests. No provider calls, source uploads, Parquet reads, target-array reads, model fits or scoring were performed. Controller reports that the private V4 source dataset was uploaded; the kernel was never pushed and no science quota was spent.

## Blocking finding

In `scripts/kaggle_kernel.py`, `write_runtime_spec()` computes `limits.timeout_seconds` during bootstrap (lines 92–108; called at lines 191–196), before the Kaggle runner performs code and input identity preflight. The kernel passes `WNN_KERNEL_DEADLINE_MONOTONIC` to the runner (lines 197–200). The E540 worker recomputes its own budget from its actual start as `min(5200, deadline - start - 300)` (`scripts/e540_kaggle_worker.py`, lines 495–511), but the runner does not recompute the watchdog at launch: `scripts/kaggle_gpu_runner.py` starts the worker at lines 253–278 and enforces the timeout copied from the earlier spec at lines 288–298.

For example, if the runtime spec is written at t=200s, the runner preflight takes 500s, and the worker starts at t=700s, the spec and runner watchdog remain 5,200s while the correct actual-start cap is 5,000s. The runner can therefore use 200s of the 300s reserved for finalization. The kernel-level 6,000s wait still bounds runner supervision and kills descendants, but it does not make the promised runner watchdog equal to the actual-start worker budget. The worker's internal checks happen at stage boundaries and do not replace the runner's process watchdog during a long or blocked stage.

**Smallest correction:** immediately before each worker `Popen` in `scripts/kaggle_gpu_runner.py`, derive the watchdog from inherited `WNN_KERNEL_DEADLINE_MONOTONIC` and current monotonic time, capped by the current spec timeout and 5,200s, with the 300s finalization reserve. Fail closed if less than one second remains, and persist the resolved limit in the worker spec/receipt. Add one synthetic regression where spec creation is at t=200 and worker launch is at t=700; assert the runner receives 5,000s, not 5,200s. Reseal and rehash the packet after that runtime-only change.

## Confirmed passes

- **Projection:** `elapsed_projection_seconds()` adds current elapsed time, projected fit/prediction work and one 300-second projection margin. Because gate-time elapsed already includes both completed profiles, it does not add profile wall time a second time. The boundary test confirms 70s elapsed plus the 300s margin gives 370s.
- **Kernel watchdog and cleanup:** the generated kernel starts its monotonic clock at module initialization, checks remaining budget during bootstrap, and waits for the runner only until the 6,000-second deadline. `terminate_process_tree()` identifies descendants with `psutil`, including children in separate sessions. The synthetic separate-session cleanup test passed. This deadline is from module start, not provider push; provider queue/cold start and output transport are explicitly outside the bound.
- **Recipe lineage:** V4 changes execution projection/deadline handling and packet/run IDs; the trainer/model implementation and V3 scientific settings remain fixed: paired `linear` and `ple`, AMP initial scale 4096, clip-2 WMSE with chunk denominator and absolute clipped-target weights, fixed epoch 0, one epoch, 512 fitting components, and the strict 316-update finite/zero-skip profile gate. The code computes 79 chunk updates per outer batch; with 1,597 train and 252 valid fit groups and `drop_last=True`, the expected fit count is 72,996 updates per arm. These are recipe expectations, not observed training results.
- **Membership:** using only the frozen split JSON and subset JSON, the seed-540 permutation reconstructs the exact 512 selected eligible components and exact selected train/valid groups. Counts are 1,597 train groups and 252 valid groups (1,849 total; 1,848 consumed with one dropped sequence of unknown identity). The exact 468 evaluation groups match the frozen split and have no component overlap with fitting groups. These evaluation groups remain exposed; they are not a fresh holdout.
- **Source and package identity:** the archive is 516,082 bytes, SHA-256 `2e4a1b4e6ccb2145326c813624dcbd73fda71fa099c1a8577f6541bf8ef39885`, with 72 unique members matching the manifest payloads plus its declared spec path. The bundle manifest hash `7905dfbd096092c6bc46afecd2196742e31ce7386dbefb6086a618d0a8e66d6a`, kernel metadata hash `595cc24b61c0559a474a8bc3bc11cd2abfe314b843493279ff603934125ffe0d`, and generated `bundle/kernel/kernel.py` hash `c8d137f070408c93243e03e9c15746cba63d6f3b0a8c41183f4748d6a5d68bde` match the packet. All 58 source hashes and 13 local-input descriptors match. The archived spec parses identically to the local spec; its bytes are covered by the whole-archive hash. `dataset-metadata.json` is tracked at the actor commit (observed SHA-256 `b36f64fc870daffe2c1f5955c741c2bae113567cc44721741ac67455ce3f842e`). External Parquet and target-array identities were not opened or independently checked locally.
- **Comparison limitation disclosed by packet:** both arms allocate 274,050 parameters, but their effective capacity differs because the linear control's 896 ramp coordinates are zero. The packet correctly limits a PLE result to this screen and does not claim it isolates spline geometry alone. It also records that realized paired sampler order and the dropped sequence identity were not captured.

## Focused validation

Under the stated 1-CPU, 1.5-GB, 120-second local scope, ran all 12 tests in `tests/test_e540_kaggle_packet.py` plus these three synthetic `tests/test_numeric_ple.py` cases:

```text
test_quantile_fit_is_bounded_membership_bound_and_handles_ties
test_quantile_sequence_reader_uses_only_explicit_fit_membership
test_ple_slots_are_continuous_and_keep_unbounded_raw_bypass
```

Result: **15 passed in 1.09s.** The tests exercise projection accounting, worker budget helpers, finite/zero-skip profile gates, expected update count, archive/input identity helpers, exact component membership, PLE provenance and separate-session descendant cleanup. They do not exercise the stale Kaggle runner-watchdog timeout described above.

No actual T4 profile, fit, prediction, independent grade, delivered-submission check or public-score claim is established by this review. The controller must still perform fresh quota/admission, uploaded source verification and actual execution after the runtime correction.
