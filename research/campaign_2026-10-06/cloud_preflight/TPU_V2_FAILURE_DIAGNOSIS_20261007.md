# E-463 TPU v2 terminal diagnosis

Collected 2026-10-07 after Kaggle reported
`vladkochetov07/wnn-e463-real-tpu-fullshape-preflight-20261007-v2` as terminal
`ERROR`. The complete output set and provider log are retained under
[`tpu_results_v2/`](tpu_results_v2/); hashes and command statuses are in
[`tpu_terminal_v2.json`](tpu_terminal_v2.json).

The kernel started Python and passed its pre-JAX host RAM check, then failed in
`host_admission` after 0.434 seconds. `materialize_verified_source()` found the
frozen manifest but required `source_bundle.zip` in the manifest directory.
Kaggle's live dataset file listing contains the manifest plus 17 files already
expanded beneath `source_bundle/`; it contains no `source_bundle.zip`. The
failure therefore came from a stale packaging assumption in the packet, before
JAX import or TPU initialization. It is execution evidence only and says
nothing about model quality or TPU numerical compatibility.

The downloaded provider dataset exactly matches the current Kaggle file list.
Its manifest SHA-256 is
`851898475dfa701f2089ec76cfe9592538aa2e7f9b70ebcb68c0a614e332d6fb`, and all
17 direct source files match their manifest sizes and SHA-256 values. See
[`source_dataset_identity.json`](tpu_results_v2/source_dataset_identity.json)
and [`provider_dataset_files.json`](tpu_results_v2/provider_dataset_files.json).
The retry will verify those mounted files in place, including rejection of
missing, changed, extra, unsafe or symbolic-link members, then run the same
actual-eight-device TPU, full-shape synthetic update, finite-gradient,
checkpoint and export/reload checks. It will not depend on the absent ZIP.

The fresh quota snapshot after v2 terminal shows 176.083 TPU seconds used,
zero reserved, and 72,000 allowed. V2 consumed 23.03 seconds from its
prelaunch snapshot. The controller preserved baseline 134.623 seconds and set
the cumulative campaign cap to 7,200 seconds, so 41.46 seconds have been
consumed and 7,158.54 remain for the corrected attempt and any retry. The live
account still has 57,459.1336 spendable seconds after holding back 20 percent
of its current unspent quota. The updated allocation and authority are recorded
in [`continuous_recovery_allocation_20261007.json`](continuous_recovery_allocation_20261007.json)
and [ADR-0010](../../specs/adr/0010-continuous-experiment-delivery.md).
