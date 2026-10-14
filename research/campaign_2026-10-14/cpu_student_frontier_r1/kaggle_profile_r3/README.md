# Kaggle profile R3: contained distillation cache

This successor repairs the diagnosed R2 failure before fitting: content-addressed mount symlinks resolved outside the registered clean-T3 cache root. Producer metadata, target bytes, scientific recipe and strict consumer ancestry/containment checks stay unchanged. The bootstrap stream-copies only the three verified cache files (1,910,723,660 bytes total) into ordinary files under the registered root, checks exact length/SHA, reserves disk, hands off file cache incrementally and removes staging copies before provider output collection. No scientific outcome is claimed.

- [Builder](build_packet_r3.py), [sealed packet](PACKET_R3.json), and [binding](BUNDLE_BINDING_R3.json) identify the private stage at `/home/vlad/development/wunder/qk14r3b/profile_upload`. Original source is recovered from the hash-verified R2 archive; only its bootstrap is updated. The first offline build against mutable MAIN correctly rejected the changed `scripts/train_rnn.py` dependency before provider activity. The corrected builder does not silently rebind that trainer.
- [Implementation](../../../../scripts/kaggle_qat_kernel_r1.py) and [focused tests](../../../../tests/test_qat_cache_materialization.py) preserve source files, reject changed bytes/receipt mismatches, enforce deadlines and remove partial copies. Together with existing preflight tests,9 checks passed. The fixture uses actual producer schema/logical paths with deliberately reduced payloads; it does not establish a completed Kaggle copy or GPU profile.
- The final bundle retains three100-update profile jobs,8GiB worker reservations,1,800second kernel deadline and300second finalization reserve. Independent review and fresh root admission precede any push; no provider write has occurred at this note's creation.

Predecessor: [R2 collected failure](../../../campaign_2026-10-13/qat_renewable_r1/kaggle_f0_s4_r2/README.md). Root owns review, dispatch, phase observation and terminal collection.

## Root execution

Independent [R3 review](../../../unified/reviews/KAGGLE_CACHE_MATERIALIZATION_R3_REVIEW_20261014_R1.md) supports one bounded profile. [Admission](ROOT_ADMISSION_R3.json), [fresh quota](QUOTA_PRE_CREATE_R3.json), [private dataset creation](DATASET_CREATE_R3.json), [READY/version1/downloaded-manifest verification](DATASET_READY_R3.json), and [kernel push](KERNEL_PUSH_R3.json) bind the actual provider operation. Kernelversion1 was accepted at23:58UTC and observedRUNNING shortly afterward; no worker-update or quality result yet. Root owns terminal collection (100MB maximum), diagnosis and useful successor admission. The shared durable poller includes this exact target; the old terminal collector is pinned to unrelated historical targets, so it is not claimed as this job's collection owner.

[Sealed specifications](specs/README.md) retain the original preflight and three full-shape FP32/W4/mixed profiles.

[Final bounded offline test receipt](OFFLINE_TESTS_R3.xml) records9passing preflight/materialization checks on physical Btrfs. New builder/tests pass Ruff. The reused bootstrap has5pre-existing Ruff diagnostics in unrelated process cleanup/error chaining; this repair preserves those sections and avoids wholesale formatting of frozen source.
