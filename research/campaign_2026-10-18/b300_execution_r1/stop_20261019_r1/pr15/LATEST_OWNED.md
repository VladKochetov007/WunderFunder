<!-- https://github.com/VladKochetov007/WunderFunder/pull/15#issuecomment-6256107437 updated 2026-10-19T14:30:46Z -->
<!-- tabdata-b300-20261018:PR15:status: -->

### tabdata PR15: status update — 2026-10-19 14:30 UTC

**Status:** Partial / in progress. Scored 64/120; in-flight cells (including delivery) 16; unstarted 40.

Recovered original fits retain the failed worker receipt and actor-returned envelope; their missing fit receipt/timing is not reconstructed. New fits are independently revalidated.

**Frozen comparison / provenance:** strict FP32, autocast and TF32 off; unchanged `global_wp_clip2_v1`; fixed checkpoint; original panel, folds and fitting seeds. Profiles, inference aliases and recovered exports are not additional training replications.

```json
{
  "archive_sha256": "29fb624f73a4d6ca8dc38a428fb73fa7fec2c410617b18781c4e0a8d30506985",
  "commit": "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e",
  "file_count": 172,
  "manifest_sha256": "873662aded939afd64c17c80b684011de023c04a31be5b2c74e0b9d97a18efc0",
  "source_commit": "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e",
  "source_tree_sha256": "a7aff9a98f45012e78c7a51251399754f4320bf271cb8639c957e2f372c4a265",
  "total_uncompressed_bytes": 7146485
}
```

| Arm | Scored / planned | Mean WP¹ | Paired Δ¹ | Matched cells |
|---|---:|---:|---:|---:|
| baseline | 16 / 30 | 0.67824425 | — | 16 |
| combined | 16 / 30 | 0.67702747 | -0.00121678 | 16 |
| dictionary | 16 / 30 | 0.67756978 | -0.00067447 | 16 |
| moments | 16 / 30 | 0.67809108 | -0.00015317 | 16 |

¹ Equal mean of the available fold means. A delta uses only matching fold/seed cells. Until all four folds and the complete frozen matrix are present, it is a partial statistic. This table is not a confidence interval, significance test or public-score forecast.

**Validation and decision:** these are exposed development folds. Training-seed uncertainty, component uncertainty and selection/public-transfer uncertainty are distinct. Complete CP30 fits do not by themselves establish statistical significance. Preserve negative and inconclusive results; no automatic merge, pruning, checkpoint selection or model promotion.

**Evidence / contract trace:** https://github.com/VladKochetov007/WunderFunder/pull/15. Large weights and supplied data remain in the approved private server storage; this reporter does not upload them.

Automated reporting only: no fitting, retries of scientific fits, export or grading was launched.

<!-- https://github.com/VladKochetov007/WunderFunder/pull/15#issuecomment-6268922664 updated 2026-10-19T14:16:26Z -->
<!-- tabdata-user-stop-request-20261019-r1 -->
### User-requested tabdata stop; remote stop not yet confirmed

The user requests preserving results and removing all owned training from the B300 node. **No new fits, profiles or scientific retries are authorized.**

**Remote stop is NOT confirmed:** earlier SSH sessions closed; the latest explicitly requested plain SSH attempt (14:12 UTC / 17:12 Moscow) reached the password prompt. After password input, the server closed the connection without providing a shell. The exact cause is unknown; there was no Permission denied response. No remote command or stop signal could be executed. Existing GPU workloads may still be running; no empty-node or freed-resource claim is made.

Latest preserved PR15 snapshot (14:10 UTC / 17:10 Moscow): 64/120 scored fits, 16/30 cells per arm. Equal-fold means: baseline 0.67824425; moments 0.67809108 (delta -0.00015317); dictionary 0.67756978 (-0.00067447); combined 0.67702747 (-0.00121678). Partial development-fold evidence, not a significance or public-transfer claim.

R01 recovery has no published validated paired scores: twelve receipt chains report a stdout filename mismatch. Raw grade JSON/checkpoints require server access before numerical recovery can be claimed. Age/innovation each report 0/30 scored pairs.

Checkpoints/predictions/logs must remain intact. With a working node shell, stop the owned `tabdata-*` units and verify their cgroups/project CUDA processes have drained; preserve foreign workloads.

[Frozen result/API snapshots and access evidence](https://github.com/VladKochetov007/WunderFunder/blob/6b2f92e2fa72cf43b37b73ead9dba9dcb7989ac8/research/campaign_2026-10-18/b300_execution_r1/stop_20261019_r1/README.md).

<!-- https://github.com/VladKochetov007/WunderFunder/pull/15#issuecomment-6267762794 updated 2026-10-19T12:58:07Z -->
<!-- tabdata-b300-20261018:PR15:results:f30a83740f265102a7f3550648f274a5fccdae965a8a9727b00cd784c016923a -->

### tabdata PR15: results update — 2026-10-19 12:58 UTC

**Status:** Partial / in progress. Scored 64/120; in-flight cells (including delivery) 16; unstarted 40.

Recovered original fits retain the failed worker receipt and actor-returned envelope; their missing fit receipt/timing is not reconstructed. New fits are independently revalidated.

**Frozen comparison / provenance:** strict FP32, autocast and TF32 off; unchanged `global_wp_clip2_v1`; fixed checkpoint; original panel, folds and fitting seeds. Profiles, inference aliases and recovered exports are not additional training replications.

```json
{
  "archive_sha256": "29fb624f73a4d6ca8dc38a428fb73fa7fec2c410617b18781c4e0a8d30506985",
  "commit": "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e",
  "file_count": 172,
  "manifest_sha256": "873662aded939afd64c17c80b684011de023c04a31be5b2c74e0b9d97a18efc0",
  "source_commit": "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e",
  "source_tree_sha256": "a7aff9a98f45012e78c7a51251399754f4320bf271cb8639c957e2f372c4a265",
  "total_uncompressed_bytes": 7146485
}
```

| Arm | Scored / planned | Mean WP¹ | Paired Δ¹ | Matched cells |
|---|---:|---:|---:|---:|
| baseline | 16 / 30 | 0.67824425 | — | 16 |
| combined | 16 / 30 | 0.67702747 | -0.00121678 | 16 |
| dictionary | 16 / 30 | 0.67756978 | -0.00067447 | 16 |
| moments | 16 / 30 | 0.67809108 | -0.00015317 | 16 |

¹ Equal mean of the available fold means. A delta uses only matching fold/seed cells. Until all four folds and the complete frozen matrix are present, it is a partial statistic. This table is not a confidence interval, significance test or public-score forecast.

**Validation and decision:** these are exposed development folds. Training-seed uncertainty, component uncertainty and selection/public-transfer uncertainty are distinct. Complete CP30 fits do not by themselves establish statistical significance. Preserve negative and inconclusive results; no automatic merge, pruning, checkpoint selection or model promotion.

**Evidence / contract trace:** https://github.com/VladKochetov007/WunderFunder/pull/15. Large weights and supplied data remain in the approved private server storage; this reporter does not upload them.

<!-- tabdata-b300-20261018:PR15:endpoint:227c176d6a86202bbf4729b66eac409862b1c73cc06dd943adb4ff35dd9421e3 -->

**New or corrected scored endpoints in this update:** 1.

<details><summary>Exact endpoint identities and hashes</summary>

```json
[
  {
    "archive_manifest_path": "/data/epanov_s/tabdata/artifacts/github-reporting-v1/227c176d6a86202bbf4729b66eac409862b1c73cc06dd943adb4ff35dd9421e3/MANIFEST.json",
    "archive_manifest_sha256": "dc3d1e18c8d66817074762c2ab98835c6f4c134c61d8a333207a44a4e667015a",
    "arm": "moments",
    "checkpoint_payload_sha256": "7279cf17687e3f5fcadbd1d5ef00c5778272b8f11aa9fe314393cf5458469ca7",
    "checkpoint_sha256": "8e6edcfc704293bbc86534a781f6fd3478a87087fdc6c9a9ccdaf9bc6abe29db",
    "evidence_archive": {
      "bytes": 624640,
      "path": "/data/epanov_s/tabdata/artifacts/github-reporting-v1/227c176d6a86202bbf4729b66eac409862b1c73cc06dd943adb4ff35dd9421e3/evidence.tar",
      "sha256": "13f296eb236018a6657f17c667625890f46d9a17b48d8cae93d86646d0537385"
    },
    "fit_receipt_sha256": "4add4d77720e3075d1a110defba806bd229b2b9a494d06bd492199793a7574b1",
    "fit_recovery_sha256": null,
    "fit_timing_note": "verified worker receipts",
    "fit_timing_seconds": 18738.83508849144,
    "fold": 1,
    "id": "pr15_b300_v1_moments_f1_s1004",
    "prediction_metadata_sha256": "3a04143e4ad2e9c8b74b2aad26d5d67209d9eb908fe2adfc22cc1378d0abc023",
    "prediction_sha256": "c6191787162bdc456c349ae73442253c50f65695793a09eef775addf54cf6f20",
    "receipt_digests": {
      "endpoint_sha256": "d66506ad68fe4a5751e34779c176421884a842c512cc506d45e8110dcabe1d9d",
      "export_receipt_sha256": "6eed2285fc39e836a83fe591b6a3e6889f517a17f2eb9b141f74e6532fb638bf",
      "fit_receipt_sha256": "4add4d77720e3075d1a110defba806bd229b2b9a494d06bd492199793a7574b1",
      "request_sha256": "f9fa10db192ad18996f228dce4d545526d26a5e6e8aeff9b5b9abcfb9feb2937",
      "score_receipt_sha256": "1be6ee6de0b0c59a67c5887607055b956d253c3cc62fc1e6568ea1d64d357fa8",
      "split_manifest_sha256": "cbff6351074dff6244d8a0a6895de498dbf11df562798c575384918090c8a855",
      "worker_receipt_sha256": "1e59c6ca65a0956aa37b8761f676000c8b6fee7617497f3770d6f8bf7f7499bf"
    },
    "receipt_sha256": "d66506ad68fe4a5751e34779c176421884a842c512cc506d45e8110dcabe1d9d",
    "score_sha256": "76c9256b092e2598812b2fb59ed3cd05beabbdeb137320e35d6b3f2b5afbc00b",
    "seed": 1004,
    "source_commit": "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e",
    "source_sha256": "873662aded939afd64c17c80b684011de023c04a31be5b2c74e0b9d97a18efc0",
    "weights_archive": {
      "bytes": 45025280,
      "path": "/data/epanov_s/tabdata/artifacts/github-reporting-v1/227c176d6a86202bbf4729b66eac409862b1c73cc06dd943adb4ff35dd9421e3/weights.tar",
      "sha256": "85633a0796f97cbf1c706c7553f4172314c6d8cdf3ccd42e82d559f4c2600daa"
    },
    "wp": 0.6543461899724955
  }
]
```
</details>

Automated reporting only: no fitting, retries of scientific fits, export or grading was launched.

<!-- https://github.com/VladKochetov007/WunderFunder/pull/15#issuecomment-6267736493 updated 2026-10-19T12:56:45Z -->
<!-- tabdata-b300-20261018:PR15:results:6850b85d00d5224924507c0c538ab98928a3ed67af1ee6e26fab770ed95247cf -->

### tabdata PR15: results update — 2026-10-19 12:56 UTC

**Status:** Partial / in progress. Scored 63/120; in-flight cells (including delivery) 16; unstarted 41.

Recovered original fits retain the failed worker receipt and actor-returned envelope; their missing fit receipt/timing is not reconstructed. New fits are independently revalidated.

**Frozen comparison / provenance:** strict FP32, autocast and TF32 off; unchanged `global_wp_clip2_v1`; fixed checkpoint; original panel, folds and fitting seeds. Profiles, inference aliases and recovered exports are not additional training replications.

```json
{
  "archive_sha256": "29fb624f73a4d6ca8dc38a428fb73fa7fec2c410617b18781c4e0a8d30506985",
  "commit": "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e",
  "file_count": 172,
  "manifest_sha256": "873662aded939afd64c17c80b684011de023c04a31be5b2c74e0b9d97a18efc0",
  "source_commit": "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e",
  "source_tree_sha256": "a7aff9a98f45012e78c7a51251399754f4320bf271cb8639c957e2f372c4a265",
  "total_uncompressed_bytes": 7146485
}
```

| Arm | Scored / planned | Mean WP¹ | Paired Δ¹ | Matched cells |
|---|---:|---:|---:|---:|
| baseline | 16 / 30 | 0.67824425 | — | 16 |
| combined | 16 / 30 | 0.67702747 | -0.00121678 | 16 |
| dictionary | 16 / 30 | 0.67756978 | -0.00067447 | 16 |
| moments | 15 / 30 | 0.67819097 | -0.00001977 | 15 |

¹ Equal mean of the available fold means. A delta uses only matching fold/seed cells. Until all four folds and the complete frozen matrix are present, it is a partial statistic. This table is not a confidence interval, significance test or public-score forecast.

**Validation and decision:** these are exposed development folds. Training-seed uncertainty, component uncertainty and selection/public-transfer uncertainty are distinct. Complete CP30 fits do not by themselves establish statistical significance. Preserve negative and inconclusive results; no automatic merge, pruning, checkpoint selection or model promotion.

**Evidence / contract trace:** https://github.com/VladKochetov007/WunderFunder/pull/15. Large weights and supplied data remain in the approved private server storage; this reporter does not upload them.

<!-- tabdata-b300-20261018:PR15:endpoint:52430fbb4f14b9b07e8930fbf6ff6c7d17571e79eaf2091e79769c3f8ed3dee0 -->

**New or corrected scored endpoints in this update:** 1.

<details><summary>Exact endpoint identities and hashes</summary>

```json
[
  {
    "archive_manifest_path": "/data/epanov_s/tabdata/artifacts/github-reporting-v1/52430fbb4f14b9b07e8930fbf6ff6c7d17571e79eaf2091e79769c3f8ed3dee0/MANIFEST.json",
    "archive_manifest_sha256": "be118862f2f6e9327eed42d7b0dee3995e830d0f0dbbf2b617025f9c68eeb84a",
    "arm": "moments",
    "checkpoint_payload_sha256": "6e57f849dcaaccc0f07aa04c0b4097d13485812c9ce2a4e4457a3f47055b5499",
    "checkpoint_sha256": "15f57dab90db610425c7a8c0960e5e054a1d3763d7da0b4b8b1e2c8ec2691dbc",
    "evidence_archive": {
      "bytes": 624640,
      "path": "/data/epanov_s/tabdata/artifacts/github-reporting-v1/52430fbb4f14b9b07e8930fbf6ff6c7d17571e79eaf2091e79769c3f8ed3dee0/evidence.tar",
      "sha256": "ae1a7f65821c10c15f87a9e2a8eba5d64d3f09b57ed1129ae6bf4a36071c0ad9"
    },
    "fit_receipt_sha256": "a76a50b31f46b649d8f56b0e523996b2ef478752fd3cd15156229fa2a7e5b420",
    "fit_recovery_sha256": null,
    "fit_timing_note": "verified worker receipts",
    "fit_timing_seconds": 18697.82963848114,
    "fold": 3,
    "id": "pr15_b300_v1_moments_f3_s1004",
    "prediction_metadata_sha256": "a5ceab030d0445fe7749b28cba7dac4dc7b163a0d1bfe32deff6e89eb131a026",
    "prediction_sha256": "86db0eeecc140418b3a6d22d90d7f0d13745f5e7020ccc66409b20c696ad19fb",
    "receipt_digests": {
      "endpoint_sha256": "b8b6a2511289140a48dbf1d46a22b8abbff380d7cc95774de4211455013995ec",
      "export_receipt_sha256": "0cac3a532d1d3a7b9a9d6624decc3a258cdb5f58f21ca65f8a6982ca249d54cc",
      "fit_receipt_sha256": "a76a50b31f46b649d8f56b0e523996b2ef478752fd3cd15156229fa2a7e5b420",
      "request_sha256": "6b26f2f00e42703bdae484b4e283919964b6287eaceea8606a2a67830ce65c94",
      "score_receipt_sha256": "c75eba533c030e7c9811086a50a891417537768a69ffde1513c01cb9fbfe73a6",
      "split_manifest_sha256": "d833febfda87b13a806ee862b0fcfd50f04f9184a19f6d5bdd8676e9c3f901be",
      "worker_receipt_sha256": "5c71d98e4ca34f581a99d12fa191fdc92debdb5e10a4800f223cbfd4b7615966"
    },
    "receipt_sha256": "b8b6a2511289140a48dbf1d46a22b8abbff380d7cc95774de4211455013995ec",
    "score_sha256": "700784ebfb72bc9069c589465b0948855303a116f0ffceb54b88bbf41420618c",
    "seed": 1004,
    "source_commit": "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e",
    "source_sha256": "873662aded939afd64c17c80b684011de023c04a31be5b2c74e0b9d97a18efc0",
    "weights_archive": {
      "bytes": 45025280,
      "path": "/data/epanov_s/tabdata/artifacts/github-reporting-v1/52430fbb4f14b9b07e8930fbf6ff6c7d17571e79eaf2091e79769c3f8ed3dee0/weights.tar",
      "sha256": "354150cd17db2bf7d73a1a58edad367219d10e47b8b9865bbb8eeeb7e0e52a59"
    },
    "wp": 0.6981059736486326
  }
]
```
</details>

Automated reporting only: no fitting, retries of scientific fits, export or grading was launched.

<!-- https://github.com/VladKochetov007/WunderFunder/pull/15#issuecomment-6267605322 updated 2026-10-19T12:49:44Z -->
<!-- tabdata-b300-20261018:PR15:results:ea4d00299778f56d264fc24c5ce08290408799e200cb3b69d4c1322a78a165c9 -->

### tabdata PR15: results update — 2026-10-19 12:49 UTC

**Status:** Partial / in progress. Scored 62/120; in-flight cells (including delivery) 16; unstarted 42.

Recovered original fits retain the failed worker receipt and actor-returned envelope; their missing fit receipt/timing is not reconstructed. New fits are independently revalidated.

**Frozen comparison / provenance:** strict FP32, autocast and TF32 off; unchanged `global_wp_clip2_v1`; fixed checkpoint; original panel, folds and fitting seeds. Profiles, inference aliases and recovered exports are not additional training replications.

```json
{
  "archive_sha256": "29fb624f73a4d6ca8dc38a428fb73fa7fec2c410617b18781c4e0a8d30506985",
  "commit": "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e",
  "file_count": 172,
  "manifest_sha256": "873662aded939afd64c17c80b684011de023c04a31be5b2c74e0b9d97a18efc0",
  "source_commit": "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e",
  "source_tree_sha256": "a7aff9a98f45012e78c7a51251399754f4320bf271cb8639c957e2f372c4a265",
  "total_uncompressed_bytes": 7146485
}
```

| Arm | Scored / planned | Mean WP¹ | Paired Δ¹ | Matched cells |
|---|---:|---:|---:|---:|
| baseline | 16 / 30 | 0.67824425 | — | 16 |
| combined | 16 / 30 | 0.67702747 | -0.00121678 | 16 |
| dictionary | 16 / 30 | 0.67756978 | -0.00067447 | 16 |
| moments | 14 / 30 | 0.67815283 | -0.00001667 | 14 |

¹ Equal mean of the available fold means. A delta uses only matching fold/seed cells. Until all four folds and the complete frozen matrix are present, it is a partial statistic. This table is not a confidence interval, significance test or public-score forecast.

**Validation and decision:** these are exposed development folds. Training-seed uncertainty, component uncertainty and selection/public-transfer uncertainty are distinct. Complete CP30 fits do not by themselves establish statistical significance. Preserve negative and inconclusive results; no automatic merge, pruning, checkpoint selection or model promotion.

**Evidence / contract trace:** https://github.com/VladKochetov007/WunderFunder/pull/15. Large weights and supplied data remain in the approved private server storage; this reporter does not upload them.

<!-- tabdata-b300-20261018:PR15:endpoint:bcebec34a31f7a2c27eb6d4cd022a26692f077b4cfd49c6d03f6ee0ecd0ba536 -->

**New or corrected scored endpoints in this update:** 1.

<details><summary>Exact endpoint identities and hashes</summary>

```json
[
  {
    "archive_manifest_path": "/data/epanov_s/tabdata/artifacts/github-reporting-v1/bcebec34a31f7a2c27eb6d4cd022a26692f077b4cfd49c6d03f6ee0ecd0ba536/MANIFEST.json",
    "archive_manifest_sha256": "58ab1c7e0103db6346ad97b5fcec7ee2be49363f80e08618cc68de2f31d79d0e",
    "arm": "dictionary",
    "checkpoint_payload_sha256": "eaa95a12314f2c9f9f54be9821a5cb46a8a07a231f539940d3882a23dbf5b4f8",
    "checkpoint_sha256": "a61b84f0e3f6830d396e29390627660d1d4444d0c685df2b0c74ee403366d6d7",
    "evidence_archive": {
      "bytes": 624640,
      "path": "/data/epanov_s/tabdata/artifacts/github-reporting-v1/bcebec34a31f7a2c27eb6d4cd022a26692f077b4cfd49c6d03f6ee0ecd0ba536/evidence.tar",
      "sha256": "7a1a74d254e64a3070c8cd9199bb3d2922dc3c66fe240e32dcb4ec7e1f2f462f"
    },
    "fit_receipt_sha256": "aaf4e9e28238db3e2c167ba30fb1b85731038b25f7917c57209f033b1b6083f4",
    "fit_recovery_sha256": null,
    "fit_timing_note": "verified worker receipts",
    "fit_timing_seconds": 19159.70765376091,
    "fold": 1,
    "id": "pr15_b300_v1_dictionary_f1_s1004",
    "prediction_metadata_sha256": "1c9d91ffcfde41d28da051b7d8caeef022059329615eaaebf9f7de74cb1a83c2",
    "prediction_sha256": "b24dec0cea071136414d2ecd0ff81feacc9a72d4c23a3b257a05dfb3fbc1ca55",
    "receipt_digests": {
      "endpoint_sha256": "ed906b1667cd3b2be0eb430c8da78b7c093dab5ac760b64d5144f686a962fa95",
      "export_receipt_sha256": "602d29d9e63ff91573a95aa414a5b3f6039db0889fbc6cf3b53c41f67cbf7daf",
      "fit_receipt_sha256": "aaf4e9e28238db3e2c167ba30fb1b85731038b25f7917c57209f033b1b6083f4",
      "request_sha256": "dc9f724c91739a7bd91c1841e30b50bcbf84fc1fcfc17ba166d47a21438d9a02",
      "score_receipt_sha256": "2ef244208bb563227329c63caaf2df82f1204db806c67d1f2d05f87eb3cfa727",
      "split_manifest_sha256": "cbff6351074dff6244d8a0a6895de498dbf11df562798c575384918090c8a855",
      "worker_receipt_sha256": "1dd37f4cf4045f5e306e68ae3512f33cef0ad56e9a2c01e61eba07944ad5521d"
    },
    "receipt_sha256": "ed906b1667cd3b2be0eb430c8da78b7c093dab5ac760b64d5144f686a962fa95",
    "score_sha256": "192add7ca49fb2eb16ff660fed85ade9504242705ad9c56d35a21cb64bb89497",
    "seed": 1004,
    "source_commit": "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e",
    "source_sha256": "873662aded939afd64c17c80b684011de023c04a31be5b2c74e0b9d97a18efc0",
    "weights_archive": {
      "bytes": 45025280,
      "path": "/data/epanov_s/tabdata/artifacts/github-reporting-v1/bcebec34a31f7a2c27eb6d4cd022a26692f077b4cfd49c6d03f6ee0ecd0ba536/weights.tar",
      "sha256": "cfa41d276c70bd0c4e57cc4129b163bcd9f15b50c3eefa98bfcaab1bad6fb8e3"
    },
    "wp": 0.6563233820297291
  }
]
```
</details>

Automated reporting only: no fitting, retries of scientific fits, export or grading was launched.

<!-- https://github.com/VladKochetov007/WunderFunder/pull/15#issuecomment-6267516641 updated 2026-10-19T12:44:58Z -->
<!-- tabdata-b300-20261018:PR15:results:6e44a28c7a27e9d7c690a8e07291639d951a50f622f13bc9a2c970ea8c90ce92 -->

### tabdata PR15: results update — 2026-10-19 12:44 UTC

**Status:** Partial / in progress. Scored 61/120; in-flight cells (including delivery) 16; unstarted 43.

Recovered original fits retain the failed worker receipt and actor-returned envelope; their missing fit receipt/timing is not reconstructed. New fits are independently revalidated.

**Frozen comparison / provenance:** strict FP32, autocast and TF32 off; unchanged `global_wp_clip2_v1`; fixed checkpoint; original panel, folds and fitting seeds. Profiles, inference aliases and recovered exports are not additional training replications.

```json
{
  "archive_sha256": "29fb624f73a4d6ca8dc38a428fb73fa7fec2c410617b18781c4e0a8d30506985",
  "commit": "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e",
  "file_count": 172,
  "manifest_sha256": "873662aded939afd64c17c80b684011de023c04a31be5b2c74e0b9d97a18efc0",
  "source_commit": "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e",
  "source_tree_sha256": "a7aff9a98f45012e78c7a51251399754f4320bf271cb8639c957e2f372c4a265",
  "total_uncompressed_bytes": 7146485
}
```

| Arm | Scored / planned | Mean WP¹ | Paired Δ¹ | Matched cells |
|---|---:|---:|---:|---:|
| baseline | 16 / 30 | 0.67824425 | — | 16 |
| combined | 16 / 30 | 0.67702747 | -0.00121678 | 16 |
| dictionary | 15 / 30 | 0.67754674 | -0.00066399 | 15 |
| moments | 14 / 30 | 0.67815283 | -0.00001667 | 14 |

¹ Equal mean of the available fold means. A delta uses only matching fold/seed cells. Until all four folds and the complete frozen matrix are present, it is a partial statistic. This table is not a confidence interval, significance test or public-score forecast.

**Validation and decision:** these are exposed development folds. Training-seed uncertainty, component uncertainty and selection/public-transfer uncertainty are distinct. Complete CP30 fits do not by themselves establish statistical significance. Preserve negative and inconclusive results; no automatic merge, pruning, checkpoint selection or model promotion.

**Evidence / contract trace:** https://github.com/VladKochetov007/WunderFunder/pull/15. Large weights and supplied data remain in the approved private server storage; this reporter does not upload them.

<!-- tabdata-b300-20261018:PR15:endpoint:09edd3cca761e8e9da18d779e0e51b0662ec5740478282a347fbe9fa3646b8e5 -->

**New or corrected scored endpoints in this update:** 1.

<details><summary>Exact endpoint identities and hashes</summary>

```json
[
  {
    "archive_manifest_path": "/data/epanov_s/tabdata/artifacts/github-reporting-v1/09edd3cca761e8e9da18d779e0e51b0662ec5740478282a347fbe9fa3646b8e5/MANIFEST.json",
    "archive_manifest_sha256": "1b1df4f64886c2c30f87036cda5c82eed7225d0d96f95aabe5183221632fcf49",
    "arm": "dictionary",
    "checkpoint_payload_sha256": "5cd94ce9f2dd78fa6173442034ec5b1dd1266711fa9a33945aa83978b01dfc19",
    "checkpoint_sha256": "c07856b1e28f0aa5f1ddd4204a9be976931d5536e8047182fa647a63045df9e4",
    "evidence_archive": {
      "bytes": 624640,
      "path": "/data/epanov_s/tabdata/artifacts/github-reporting-v1/09edd3cca761e8e9da18d779e0e51b0662ec5740478282a347fbe9fa3646b8e5/evidence.tar",
      "sha256": "41502fc591efa66db31661780e0248bef139639ccb19d997508a2d192275e996"
    },
    "fit_receipt_sha256": "63133f6d12594ffa19d83b09b84a32c3aee5e4d3c6451b075d78c4c6868891d2",
    "fit_recovery_sha256": null,
    "fit_timing_note": "verified worker receipts",
    "fit_timing_seconds": 18879.55322408676,
    "fold": 3,
    "id": "pr15_b300_v1_dictionary_f3_s1004",
    "prediction_metadata_sha256": "3bf9763bd7286d15ca458eed0288c6a1b970636372c6ba8ea3c11270522c5bd5",
    "prediction_sha256": "b112ceaec2d9f4fbbd5a26b9c20bb43559da0cc2d1eaf215b1dedb0350495dd5",
    "receipt_digests": {
      "endpoint_sha256": "ab7fcbae28aad5a031df54d51702c3f2b5bf6cb97a5c79521931165fb877afde",
      "export_receipt_sha256": "f52e7dd4e53fd683d5d1cf277598835f63513508542e4c0a0dfd9a8579335cfe",
      "fit_receipt_sha256": "63133f6d12594ffa19d83b09b84a32c3aee5e4d3c6451b075d78c4c6868891d2",
      "request_sha256": "3eafa64d2c9b18608b9575f53510d54098fc8a94939e17155d63e34a7ec729b6",
      "score_receipt_sha256": "1f17979ca42ebb7dbfaca4f977ed426d312a9e7f89b01d4d8fd54267bfb133fd",
      "split_manifest_sha256": "d833febfda87b13a806ee862b0fcfd50f04f9184a19f6d5bdd8676e9c3f901be",
      "worker_receipt_sha256": "b1150e24df19781ffcb25927690e728141b4b81e0d25e77d3e2713e5d0ddc10f"
    },
    "receipt_sha256": "ab7fcbae28aad5a031df54d51702c3f2b5bf6cb97a5c79521931165fb877afde",
    "score_sha256": "6f21cf2776600fc9e891bf189bf7889411e6ecf700fffa08b2bc94c8de6ae56a",
    "seed": 1004,
    "source_commit": "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e",
    "source_sha256": "873662aded939afd64c17c80b684011de023c04a31be5b2c74e0b9d97a18efc0",
    "weights_archive": {
      "bytes": 45025280,
      "path": "/data/epanov_s/tabdata/artifacts/github-reporting-v1/09edd3cca761e8e9da18d779e0e51b0662ec5740478282a347fbe9fa3646b8e5/weights.tar",
      "sha256": "463bdd97f27e7f7610f52034b4cd0b6c3141c96fbd177ed9480c8a282ca35013"
    },
    "wp": 0.6992602638697119
  }
]
```
</details>

Automated reporting only: no fitting, retries of scientific fits, export or grading was launched.
