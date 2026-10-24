# First complete intermediate validation results

The user identified missing intermediate validation on October24. The live
trainer logged online training loss and evaluated only after its fixed final30
endpoint. An external observer now evaluates immutable epoch snapshots while
the original four fits continue.

These are **epoch28 diagnostics, not completed epoch30 experiment endpoints**.
Each train loss below uses a frozen 64-sequence fitting probe evaluated at the
same weights as the entire original held-out fold. Both WMSE values cover all
rows after the 99-row warmup; official WP uses the held-out scored mask.

| Model | Fold | Epoch | Train-probe WMSE | Full val WMSE | Official val WP | Val sequences |
|---|---:|---:|---:|---:|---:|---:|
| 630M | 0 | 28 | 0.0112677 | 1.5501325 | 0.1729164 | 468 |
| 630M | 2 | 28 | 0.0120255 | 1.5983214 | 0.1684288 | 443 |

The very large fitting-to-validation gap is consistent with severe overfitting
and poor generalization at this checkpoint. A fitting probe is a subset estimate,
not a full-training-set metric. These first points do not identify the epoch when
overfitting began or establish a rising validation-loss curve. Earlier weights
were overwritten, so earlier validation cannot be reconstructed. The original
fixed30 endpoint selection remains unchanged; these exposed outer folds are not
used for early stopping.

The 1.54B epoch15 diagnostics were still being evaluated when this note was
written. New points update the persistent
[PR32 validation comment](https://github.com/VladKochetov007/WunderFunder/pull/32#issuecomment-6388105507)
and [PR40 comment](https://github.com/VladKochetov007/WunderFunder/pull/40#issuecomment-6388105675).

Both completed points were independently recomputed from saved prediction,
target and mask arrays, with artifact hash verification. The first immutable
[evidence manifest](https://github.com/VladKochetov007/WunderFunder/blob/feae6821e23e7268ad636c646778422894e108cf/research/experiment_drafts/b300-transformer-distillation-runtime/execution_20261024/validation-1792867505.json)
records every archived file hash. Its 65,331-byte archive has SHA256
`bc2fa3f9fd64038cacbb68aabed75502277ab1a9692a8234ffbee1ff5b37d37fb`.
The archive contains metric/provenance/code receipts, not arrays or weights.
Actual node operator bytes are preserved there; the published implementation
also adds explicit descriptions of the two masks.

Seven numerical metric tests, checkpoint counter/contract rejection checks,
atomic-inode capture and frozen-job JSON roundtrip checks passed. Source
structure checks passed. Inference workers use less than their 16-GiB CUDA
allocator cap, and the monitor runs in its own cron PAM session. See the
[monitor implementation and limitations](validation_monitor/README.md).
