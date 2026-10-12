# R7 independent export-packet review

Decision: offline packet and actual-producer consumer checks pass; launch remains blocked pending root release and fresh runtime admission.

I reviewed the sealed R7 packet from submission-candidates commit cd9ff50b385471a244e11ac7fd7ad23c281dcb07. Its file-manifest SHA256 is ed5533b7869f1fbb326f896c4e413db919a73741082fcbdc7987b99653b2be6a. The packet is an export-only successor. This review made no provider calls, launched no workers, and changed no packet or source files.

## Evidence checked

The R7 validator passed offline against MAIN actual producer artifacts in 2.066 seconds. I independently checked the packet’s nine manifest entries against their exact file sizes and SHA256 values. The sealed R5 bundle is 246,396 bytes with SHA256 88641b3c9fc629a7a2ce8b11cfd98ecf8b53fe943ffb2d7f632e1dfb997fa549; R5 driver/source bindings are unchanged.

All six active export specs match the R5 preseal hashes and the R7 binding: seed 3/4/5, each with FIT_TRAIN and FIT_VALID. They select the epoch-2 model, use the same R5 driver and prediction procedure, emit feature-only predictions, and retain the existing FP32/CUDA and 2-CPU, 8,000,000,000-byte worker, 5 GB GPU, 6 GB stop, and 1,800-second limits. The unused FULL_VALID catalog specs are not selected by R7.

The three actual R6 epoch-2 checkpoints and their adjacent provenance files match the R7 bindings and loaded successfully through the unchanged R5 consumer during validation. Their checkpoint SHA256 values are:

- seed 3: 201221c6b4140dda70b078db2a9d7a7246ec40b813f054570d195cecefc61582
- seed 4: 714ac9b04cd1be0f63fb5f9f60cce2059e5e9284848470cae9d7198794767df7
- seed 5: 763fac7f0aa24507167425e9905f1514a0a28ace48036b1c1c5819c7eb02c8d1

The actual source handoff exists at the bound path and matches SHA256 0eabad07a2bea44f0d4691575ccc7105526c39049c56dbf32ba948d7efb717c8. It binds source-qualified membership: 10,535 train fit groups, 1,405 valid fit groups and 468 valid evaluation groups, with source-qualified disjointness and frozen split/provenance checks recorded. The packet preserves train and valid row-group namespaces separately.

The metadata targets vladkochetov07/wnn-clean-t3-f0-h384-exp-20261012-r7. Its source is the intended R5 runner slug via kernel_sources; it does not rely on an unsupported metadata version pin. R7 requires a fresh version-2 readback and verifies mounted contents by hashes. Kernel source, bindings, release file and metadata total 79,895 bytes, below the 1 MB envelope.

Output accounting is internally consistent: each selected output is float32 C-order with shape [10535, 20000, 2] for FIT_TRAIN or [1405, 20000, 2] for FIT_VALID. With 128-byte NPY headers per file, six arrays total 5,731,200,768 bytes. The packet reserves an 8 GiB collection cap, 4 GiB free-space reserve, 256 MiB non-array reserve and at most a 2,000,000,000-byte temporary copy. Runtime validation checks the actual NPY header, shape, order, finite values, sidecar group order, split/checkpoint identity and full digest. No R7 output arrays exist yet; the finite-value check here is only a validator fixture, not evidence about future outputs.

The R7 code performs a fresh inventory and admission check for each single-worker stage and serializes the six exports. The offline actual-producer validator confirmed that the pending root release fails before worker creation. No worker can start while the release remains pending.

## Remaining gates

Keep ROOT_EXPORT_RELEASE_R7 unauthorized. Before any worker starts, root must perform the fresh version-2 provider readback, verify the mounted source and checkpoint/provenance hashes, and issue a release bound to those exact facts. Root must also recheck current host and disk headroom for each worker without weakening the 8 GB reservation or 20% host reserve. The earlier observed host availability of 3.868 GB was below the 6.442 GB reserve, so it does not establish present admission.

After any authorized exports, validate all six actual arrays and sidecars and bind their terminal receipts. This review establishes packet/consumer readiness only; it establishes no Kaggle runtime success, new score, or submission-quality claim.

## Validation command note

The raw validator output is committed alongside this review. Its support_envelope.memory_max_bytes field is hard-coded to 1,500,000,000 bytes in the validator and is not the limit used for this review. The actual command was run through MAIN scripts/limited with WNN_CPU_QUOTA=50%, WNN_THREADS=1, WNN_MEM_MAX=1000000000 and WNN_RUNTIME_MAX=120s. The machine receipt records that invocation and the raw validator-output hash.
