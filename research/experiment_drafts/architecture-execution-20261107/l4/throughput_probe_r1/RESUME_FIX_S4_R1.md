Both original GRU fits reached the safeE8 boundary at51680 updates. Both S3
checkpoint copies migrated and reported archive readiness, but the monitored
resume path failed before any new successful update. The native traceback in
[099](native_receipts/099-transition-live-status.json) identifies
`sequence_fit_resume.py:82`: coordinated_call rejected the monitor restoration
callback's None return. This is a canonical resume plumbing error, not a changed
tensor, optimizer state, source admission failure or GPU memory error.

[100](native_receipts/100-stop-exact-bugged-supervisor.json) and
[101](native_receipts/101-confirm-gru-safe-stop.json) preserve the actual stopped
parent counters, migrated checkpoint/proof identities and archive readiness.
[102](native_receipts/102-cleanup-own-s3-groups.json) preserves ROOT's exact owned
cleanup receipts. [103](native_receipts/103-native-gpu-safe-state.json) reports
the old supervisor absent, no S3 relay children and zero reported allocated GPU
memory on GPUs0/1. These are point-in-time observations, not a claim about later
process liveness. The old S1 and failed S3 artifacts remain unchanged.

The stopped original safe checkpoint pins are:

| Seed | Epoch / updates | Original S1 training_state.pt SHA256 |
| --- | --- | --- |
| 101001 | 8 / 51680 | 06a1bc3ce96a2033576dfc53085ebbfcff06663d390a9b5e3c39a0a7e978d93d |
| 101002 | 8 / 51680 | 909654a77629a17dd99c7a615d8d1997a115c7a7aad50f372ad3f22b61e5cc9a |

The bounded S4 repair changes only the monitor restoration callback relative to
the qualified S3 source. It calls the same monitor.restore_offsets with the
same arguments, then returns True to acknowledge the side effect. A raised
monitor error still reaches the unchanged canonical coordinator. Model,
optimizer, scheduler, RNG restoration and rank/metric log restoration code are
unchanged. Rank truncation already uses an integer-or-True result; metric
truncation returns an integer, including zero, which coordinated_call accepts.
The None-returning monitor callback is the identified inconsistent branch.

The [S4 builder](prepare_gru_resume_source.py) independently verifies all620
files in frozen S1 and native S3 before creating a new destination. It retains
the three S3 source amendments byte-for-byte and adds only the resume callback
fix. The cumulative whitelist relative to original S1 is exactly:

- src/wnn/training/sequence_monitor_gates.py
- src/wnn/training/sequence_monitor_diagnostics.py
- src/wnn/training/sequence_config_validation.py
- src/wnn/training/sequence_fit_resume.py

No existing capsule is edited. Native preparation is ROOT-owned:

```sh
R=/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1
py=/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python
Q=$R/control/gru-monitor-transition-r2
"$py" "$Q/prepare_gru_resume_source.py" \
  --source "$R/source-r1" --inventory "$R/manifests/l4-source-r1-manifest.json" \
  --s3-source "$R/source-gru-monitor-archive-r1" \
  --s3-inventory "$R/source-gru-monitor-archive-r1-manifest.json" \
  --output "$R/source-gru-monitor-resume-r1"
```

The builder pins S1 inventory
`7d3324230effb050ff083b939e26ad638e5df5c5b72ea70b0f8aed169997db4a`
and native S3 inventory
`a28f86a74c6d8617c6b4df37fbb9c972dd9148e2ea1e520a27888109a744e8f6`.
Its output manifest and patch receipt retain S1 as the checkpoint ancestry
parent, record S3 as the diagnostic parent, and attest one changed file relative
to S3. S4's native hashes must be recorded after creation; none is invented here.

New receiving adapters preserve the S3 versions:

- [migrate_monitor_checkpoint_s4.py](migrate_monitor_checkpoint_s4.py)
- [transition_support_s4.py](transition_support_s4.py)
- [archive_safe_state_s4.py](archive_safe_state_s4.py)

They use the existing request schema and strict canonical resume validator.
The migration whitelist is exactly four source files and the same normalized
`all_epochs -> final_only` config amendment. The new destination is a fresh
`R/runs/gru_s<seed>_monitor_opt_r2/fit` sibling. Recovery copies each original S1
checkpoint directly, rather than treating failed S3 execution as a new fit or
continuing from unverified updates. Every other model/optimizer/scheduler/RNG,
data/runtime and offset field remains exact. Historical inference indexes are
not relabelled or copied into S4.

Fresh source-qualified20-update S4 profiles are required for both original
seed/GPU pairs, followed by separate ROOT approval. A measured S3 profile cannot
be relabelled as S4. The independent recovery actor supplies an actual canonical
CPU load_resume probe using the real model/optimizer/monitor and scratch native
prefixes before GPU recovery. That probe may stub only the CUDA RNG setter,
explicitly verifying the serialized bytes supplied to it; CPU RNG and state
restoration must be real. A stubbed setter leaves actual CUDA RNG restoration
untested until the native GPU resume receipt. Callback fixture tests are not a
substitute for that real-state probe.

The focused packet verification passed26 CPU tests: the previous17 state and
admission fixtures plus nine S4 callback/AST/whitelist tests. The original None
failure is reproduced, the new callback returns the unchangedE8/51680 frontier,
genuine monitor and optimizer failures remain visible, disabled monitoring is
unchanged, numerical restoration statements are AST-identical, and missing or
additional source edits are rejected. These tests construct no model or forward
pass; the real-state CPU probe is separately identified above.

Independent reviewer monitor_execution_20261107 inspected the exact source
callback, S1/S3 inventory guards and cumulative four-file whitelist, then ran all
nine S4 tests independently:9 passed in6.46s. The reviewer found no static
blocker and explicitly limited that result to callback/semantic tests with
state/RNG stubs. It does not approve an actual checkpoint or CUDA resume.

Original deadline1794154066.717658 and the two original seeds remain binding.
The total scientific cohort stays eight L4 fits with the same12E endpoint.
External immutable safe-state archival must be ready and supervised before any
recovery trainer proceeds. No automatic third fit, source broadening, LR change,
new batch, checkpoint selection or quality claim is admitted. At this document's
receipt cutoff103, S4 source/profile/GPU-resume evidence is pending.
