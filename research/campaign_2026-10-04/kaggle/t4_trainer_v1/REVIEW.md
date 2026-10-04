# Actual trainer compatibility follow-up

Both sealed jobs passed all 17 tests: six actual-main integration checks and
eleven training-state checks. These include exact epoch resume, controlled
optimizer/update clocks, completed export, volume28/current normalization and
resume, propensity layout, and the initialization thread-budget regression.
The CUDA main tests exercise stream prefetch, two loader workers and fused AdamW.
The fixtures are tiny synthetic parquet data, not a scientific fit.

Root source aggregate SHA-256:
`238c5d41564c9e45130de811c1264cbf7ed4d8a0c937d553389bd50c3063d07d`.
Both source snapshots, sealed specs, JUnit reports and complete logs are retained.

Peak process-tree RSS was 2,916,057,088 and 2,927,546,368 bytes, within 4-GiB
reservations. Observed GPU peak was 382,730,240 bytes each. Jobs finished in
25.97 and 16.57 seconds. Because the first job completed before the mandatory
30-second launch stagger, both used physical GPU index0 sequentially. This
does not test concurrent trainer host memory; the earlier E-408 compatibility
did exercise both devices concurrently. The loader emitted Python fork warnings
but no failing tests or resource incidents.

No data/cache dataset was attached and no competition artifact was uploaded.
Provider quota after execution is preserved separately; TPU remains queued.
