<!-- https://github.com/VladKochetov007/WunderFunder/pull/25#issuecomment-6256106017 updated 2026-10-19T13:50:39Z -->
<!-- tabdata-b300-20261018:E-B300-R01:status: -->

### tabdata E-B300-R01: status update — 2026-10-19 13:50 UTC

**Status:** Partial / in progress. Scored 0/60; in-flight cells (including delivery) 16; unstarted 20.

H512x1 wide vs H384x2 deep; six-epoch strict-FP32 fixed-epoch-5 labels-only recipe; near-parameter comparison, not matched FLOPs; all four folds are exposed.

**Frozen comparison / provenance:** strict FP32, autocast and TF32 off; unchanged `global_wp_clip2_v1`; fixed checkpoint; original panel, folds and fitting seeds. Profiles, inference aliases and recovered exports are not additional training replications.

```json
{
  "actor_bundle_sha256": "e6a155ee13a7e907bb7fca2c1ebed07879e8466422015546e20360af9b404220",
  "canonical_source_identity_sha256": "fce5257e02b8d4fa701f987af1b662bf4190050f9167b0e66a9ed935da375189",
  "canonical_source_manifest_sha256": "a072293668dc8d0dd1201ea4641a1f5e9b99da718744bf33e89639abff3b8b7c",
  "commit": "a266b036b02e8e012c6eb31072291dad618a169d",
  "grader_sha256": "56ffc693da52668726bab1f7008e31753a76aed3ff5d79089167785079d9eb31",
  "source_tree_archive_sha256": "420395e9fd1206ec0466c8304b80dc095e9d416b97752ebfe39ac53a78e5b2b1",
  "source_tree_identity_sha256": "f4ef9ca8450a3e76d6b6f8295259e70aec5024d3a4f3296558bb0323add74130",
  "source_tree_manifest_sha256": "6f2d3188ae3c547976d6948044bfd97fe9036b12092ce675c5c779ea033791bd"
}
```

| Arm | Scored / planned | Mean WP¹ | Paired Δ¹ | Matched cells |
|---|---:|---:|---:|---:|
| No fixed-checkpoint official endpoints yet | 0 | — | — | 0 |

¹ Equal mean of the available fold means. A delta uses only matching fold/seed cells. Until all four folds and the complete frozen matrix are present, it is a partial statistic. This table is not a confidence interval, significance test or public-score forecast.

**Validation and decision:** these are exposed development folds. Training-seed uncertainty, component uncertainty and selection/public-transfer uncertainty are distinct. Complete CP30 fits do not by themselves establish statistical significance. Preserve negative and inconclusive results; no automatic merge, pruning, checkpoint selection or model promotion.

**Evidence / contract trace:** https://github.com/VladKochetov007/WunderFunder/blob/experiment/tabdata-b300-r01-cp30/research/experiment_drafts/tabdata-b300-r01/PR.md. Large weights and supplied data remain in the approved private server storage; this reporter does not upload them.

**Retained failures / rejected evidence:**

- tabdata_b300_r01_v1_deep_f0_s1003: fit_failed: [Errno 2] No such file or directory: '/data/epanov_s/tabdata/runs/tabdata_b300_r01_v1_deep_f0_s1003/producer_receipt.json'

- tabdata_b300_r01_v1_deep_f0_s1004: fit_failed: fit actor exited with status 1

- tabdata_b300_r01_v1_deep_f0_s1005: fit_failed: fit actor exited with status 1

- tabdata_b300_r01_v1_deep_f1_s1003: fit_failed: [Errno 2] No such file or directory: '/data/epanov_s/tabdata/runs/tabdata_b300_r01_v1_deep_f1_s1003/producer_receipt.json'

- tabdata_b300_r01_v1_deep_f1_s1004: fit_failed: fit actor exited with status 1

- tabdata_b300_r01_v1_deep_f1_s1005: fit_failed: fit actor exited with status 1

- tabdata_b300_r01_v1_deep_f2_s1003: fit_failed: [Errno 2] No such file or directory: '/data/epanov_s/tabdata/runs/tabdata_b300_r01_v1_deep_f2_s1003/producer_receipt.json'

- tabdata_b300_r01_v1_deep_f2_s1004: fit_failed: fit actor exited with status 1

- tabdata_b300_r01_v1_deep_f2_s1005: fit_failed: fit actor exited with status 1

- tabdata_b300_r01_v1_deep_f3_s1003: fit_failed: [Errno 2] No such file or directory: '/data/epanov_s/tabdata/runs/tabdata_b300_r01_v1_deep_f3_s1003/producer_receipt.json'

- tabdata_b300_r01_v1_deep_f3_s1004: fit_failed: fit actor exited with status 1

- tabdata_b300_r01_v1_deep_f3_s1005: fit_failed: fit actor exited with status 1

- tabdata_b300_r01_v1_wide_f0_s1003: fit_failed: [Errno 2] No such file or directory: '/data/epanov_s/tabdata/runs/tabdata_b300_r01_v1_wide_f0_s1003/producer_receipt.json'

- tabdata_b300_r01_v1_wide_f0_s1004: fit_failed: fit actor exited with status 1

- tabdata_b300_r01_v1_wide_f0_s1005: fit_failed: fit actor exited with status 1

- tabdata_b300_r01_v1_wide_f1_s1003: fit_failed: [Errno 2] No such file or directory: '/data/epanov_s/tabdata/runs/tabdata_b300_r01_v1_wide_f1_s1003/producer_receipt.json'

- tabdata_b300_r01_v1_wide_f1_s1004: fit_failed: fit actor exited with status 1

- tabdata_b300_r01_v1_wide_f1_s1005: fit_failed: fit actor exited with status 1

- tabdata_b300_r01_v1_wide_f2_s1003: fit_failed: [Errno 2] No such file or directory: '/data/epanov_s/tabdata/runs/tabdata_b300_r01_v1_wide_f2_s1003/producer_receipt.json'

- tabdata_b300_r01_v1_wide_f2_s1004: fit_failed: fit actor exited with status 1

Automated reporting only: no fitting, retries of scientific fits, export or grading was launched.

<!-- https://github.com/VladKochetov007/WunderFunder/pull/25#issuecomment-6257196087 updated 2026-10-19T13:02:58Z -->
<!-- tabdata-b300-20261018:E-B300-R01-DELIVERY-RECOVERY-V2:status: -->

### E-B300-R01-DELIVERY-RECOVERY-V2: status update

**Status:** partial recovery delivery evidence; paired official grades 0/30; delivery-pending 18; in-progress 0.

Distinct delivery-recovery scope for already completed original E-B300-R01 fits. No refits were performed by this reporter. This source-qualified recovery status does not replace or edit the existing full-panel R01 status comment on PR25.

This report uses a separate recovery marker; it does not replace or edit the existing full-panel R01 status comment on PR25. It records delivery-only recovery of already trained fixed-epoch-5 checkpoints and does not count additional fits.

Official metric `global_wp_clip2_v1`; equal-fold mean of observed paired deep-minus-wide deltas: `not available`. All four folds are exposed; this is descriptive evidence, not a pristine holdout or a significance claim.

Source pins: `{"actor_bundle_sha256": "e6a155ee13a7e907bb7fca2c1ebed07879e8466422015546e20360af9b404220", "grader_sha256": "56ffc693da52668726bab1f7008e31753a76aed3ff5d79089167785079d9eb31", "helper_manifest_sha256": "d5ac193fc3241b7e54d985e71077167f1170fb0c2300967631226937ab22ae8f", "historical_helper_manifest_sha256": "1af1ac94958f134ce87aa644b9fdb55b56c413b34f8649a67ab05038ccd3b6be", "panel_sha256": "471f465cc1d13fc6e5153a4c998390bb14e06823fed4c6721f929ce95fee47fb", "source_tree_identity_sha256": "f4ef9ca8450a3e76d6b6f8295259e70aec5024d3a4f3296558bb0323add74130"}`.

Contract and recovery evidence: https://github.com/VladKochetov007/WunderFunder/blob/experiment/tabdata-b300-r01-cp30/research/campaign_2026-10-18/b300_execution_r1/parallel_expansion_v1/r01_delivery_recovery_v2/README.md

**Rejected or incomplete receipt chains retained:**

- fold0/seed1003: ValueError: expected a regular nonsymlink file: /data/epanov_s/tabdata/state/r01-delivery-recovery-v2/grades/seed1003/fold0/grade.stdout.txt

- fold0/seed1004: ValueError: expected a regular nonsymlink file: /data/epanov_s/tabdata/state/r01-delivery-recovery-v2/grades/seed1004/fold0/grade.stdout.txt

- fold0/seed1005: ValueError: expected a regular nonsymlink file: /data/epanov_s/tabdata/state/r01-delivery-recovery-v2/grades/seed1005/fold0/grade.stdout.txt

- fold1/seed1003: ValueError: expected a regular nonsymlink file: /data/epanov_s/tabdata/state/r01-delivery-recovery-v2/grades/seed1003/fold1/grade.stdout.txt

- fold1/seed1004: ValueError: expected a regular nonsymlink file: /data/epanov_s/tabdata/state/r01-delivery-recovery-v2/grades/seed1004/fold1/grade.stdout.txt

- fold1/seed1005: ValueError: expected a regular nonsymlink file: /data/epanov_s/tabdata/state/r01-delivery-recovery-v2/grades/seed1005/fold1/grade.stdout.txt

- fold2/seed1003: ValueError: expected a regular nonsymlink file: /data/epanov_s/tabdata/state/r01-delivery-recovery-v2/grades/seed1003/fold2/grade.stdout.txt

- fold2/seed1004: ValueError: expected a regular nonsymlink file: /data/epanov_s/tabdata/state/r01-delivery-recovery-v2/grades/seed1004/fold2/grade.stdout.txt

- fold2/seed1005: ValueError: expected a regular nonsymlink file: /data/epanov_s/tabdata/state/r01-delivery-recovery-v2/grades/seed1005/fold2/grade.stdout.txt

- fold3/seed1003: ValueError: expected a regular nonsymlink file: /data/epanov_s/tabdata/state/r01-delivery-recovery-v2/grades/seed1003/fold3/grade.stdout.txt

- fold3/seed1004: ValueError: expected a regular nonsymlink file: /data/epanov_s/tabdata/state/r01-delivery-recovery-v2/grades/seed1004/fold3/grade.stdout.txt

- fold3/seed1005: ValueError: expected a regular nonsymlink file: /data/epanov_s/tabdata/state/r01-delivery-recovery-v2/grades/seed1005/fold3/grade.stdout.txt

The reporting process reads receipts and archives evidence/weights privately; it never trains, exports, or grades.
