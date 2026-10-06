# Independent E489 v2 continuation probe review

Disposition: GO for the newly sealed fit-only actual-shape probe after the
complete frozen dependency check and the new runner's independent review pass.
This is not admission for the six full trajectories or test-role evaluation.
No CUDA work was executed by this reviewer.

The snapshot v2 entrypoint and original both hash to
`7ee1bf79dac158d7aaa6530f55173126ad5e8f8df1c4e50832d83e9b0ca19b20`.
The trainer hashes to
`af2ad34d8bac617fe5f53e28a2120c5d59a303888b54937e35e27edd1b925cdc`.
The snapshot nn module hashes to the trainer's original pinned identity
`388af04570f0a3944159bf86c1a234dc55c78b951804c38ef59fa314f92f3db3`.
Do not relax this identity to accept the live readout-mixture module.

Relocation preserves the relative ROOT and sys.path relationships in the copied
tree. The probe checks independent E487 saved-delta proof and parent identities
before configuring CUDA. Dataset hashing streams one-MiB chunks. The cohort
loader reads only the 24 fit row groups with Arrow threading disabled; it does
not materialize the entire 5.6-GB parquet. Numeric arrays include 24x20,000x42
float32 input entries, approximately 80.6 MB before temporary normalization
arrays; the actual Python/Arrow/Torch total must still be measured.

The three native-alpha0 parity checks and six seed/alpha timing cells preserve
the strict original output/state/gradient/per-family update tolerances, seeds,
shared initialization, IEEE context, continuous two-chunk optimizer state,
changed/finite updates, and fit-only exposure. Timing includes transfers and
synchronization. Physical device admission and outer monitor retain the
3,000,000,000-byte stop and two-GiB allocator allowance.

The proposed 3,000,000,000-byte host cap is an explicitly documented operational
amendment to the old six-GiB reservation, motivated by prior small-pilot peaks
below 1.8 GB and the bounded loader. It does not establish a measured peak for
this CUDA probe. Any cap/guard failure is execution-censored and requires a
recorded mitigation and new admission; no automatic retry or numerical gate
change. Keep the 600-second probe execution ceiling and count actual charged
probe/failed-attempt time toward the original 7,200-second study allowance.

The sealed spec must include all executable copied modules and all transitive
design/proof/exposure receipt inputs checked by parent_source_hashes(), plus
the frozen parquet/cohort identity and a fresh output directory outside prior
raw artifacts. Root's reported missing ACTUAL_PROBE_V3_FAILURE.md must be
resolved through the complete parent hash check before launch. The same applies
to any further missing dependency: do not edit expected hashes to pass.

Full-fit admission remains conditional on three-seed probe success, complete
resource/teardown/OOM records, and maximum measured update time multiplied by
11,376 and 1.5 plus fixed overhead and actual charged probe time fitting the
remaining study budget. Use the new local start cutoff at actual child launch.
