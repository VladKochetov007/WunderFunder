# Clean F0 matched student pair — cache-validated preparation R2

R2 preserves the R1 manifest and preflight unchanged. It binds the completed homogeneous teacher cache to the owner’s actual-input and frozen-consumer validation receipts. The paired science remains the same seed-3 comparison: own combo parent versus own Muon-all/prop0 parent, with the fixed six-epoch H256 student recipe and alpha-0.9 target.

The cache uses the unweighted float32 mean of the F0 T3 seed-3/4/5 epoch-2 predictions, scaled by 0.8125. Its train and FIT_VALID arrays are respectively `[10535, 20000, 2]` and `[1405, 20000, 2]`, both float32. The bound validation receipt reports full array SHA and finite-value checks passing, exact target parity samples, and zero overlap between the 1,405 FIT_VALID groups and the 468 exposed evaluation groups. Teacher checkpoint, provenance, train-prediction, full-valid prediction, and sidecar hashes are bound through the receipts copied under `cache_evidence_r2/`.

R2 also corrects one manifest-only fingerprint typo without rewriting R1: R1’s top-level FIT_VALID set hash was truncated to 62 hex characters. The R2 value uses the authoritative 64-character FIT_VALID order fingerprint from the cache-validation receipt; all teacher membership and split records remain bound.

The R2 preflight checks current cache metadata SHA and array path/byte sizes, verifies the receipt chain and all source/checkpoint bindings, and confirms that profile and fit authorization remain false. It does not reopen or rehash the 1.9 GB cache arrays; it relies on the attached frozen-consumer receipt for their full hashes and finite scan. The preflight passed with 27 immutable bindings. No model, torch import, profile, score, fit, upload, or provider action was performed.

## Remaining gates

- Bind exact private source/cache dataset versions, READY receipt, nested mount paths, and project-root aliases.
- Run and review the already declared one-device, serial B16/C2000 FP16 full-cache profile on actual Kaggle hardware.
- Reverify both parent checkpoints/provenance under the staged trainer aliases, check output collisions, and obtain root’s separate profile and student-fit admissions.

The focused R2 tests exercise receipt binding, evaluation-overlap refusal, output-hash refusal, and the no-authorization state. Run the static preflight from the repository root with the campaign limit:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=650000000 WNN_RUNTIME_MAX=120s scripts/limited python3 \
  research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r6_pair_20261012/validate_pair_preparation_r2.py \
  --receipt research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r6_pair_20261012/CACHE_VALIDATED_PREFLIGHT_R2.json
```

This is a preparation and receipt-binding check, not permission to start the profile or either fit.
