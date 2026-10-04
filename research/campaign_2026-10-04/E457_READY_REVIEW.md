# E457 fixed parent/head ZIPs ready

Both prescribed endpoints passed actual i16-v3/i16-v2/f32 callbacks on 20 full streams, unchanged numerical thresholds, bitwise portable/reset replay and causal suffix checks. Source parent/head trunks are bitwise identical; checkpoint, exported weight, dataset and archive hashes are bound to the readiness reports. This is a completed CPU packaging experiment, not a new GPU fit.

| Endpoint | Manual-upload ZIP | SHA256 | Runtime projection anchored to the measured E412 public runtime |
|---|---|---|---:|
| Parent only | `submissions/c20261004_combo_parent_s3_i16/submission.zip` | `a908efd5903396070b7e0b2aef98454b91fea31316e3621ce5c6ec5f1b62ae46` | 34.2–34.6 min |
| Fine-tuned head only | `submissions/c20261004_combo_head_s3_i16/submission.zip` | `715958e25a12cf75b105bfe80668798aa3a4a1153902e0dd28bfa3b53e049a73` | 33.8 min |

Three alternating fresh-process timing pairs per endpoint were measured in a verified GPU-idle window. Runtime projections are cross-hardware estimates, not official runtimes. Both satisfy the 52-minute projected margin.

The measured equal-mixture public score is 0.6655. The endpoints test whether the final head procedure transfers and whether the mixture remains useful. Endpoint public ordering and gains are unmeasured; do not present narrow forecasts or promise 0.6660. The 20-stream callback subset is an implementation check, not validation confirmation. Uploading these candidates is a manual user action.

Evidence: `E457_HEAD_TRANSFER_PREREG.md`, `E457_head_transfer/readiness.json`, `E457_head_transfer/runtime_review.json`, each endpoint's callback/readiness logs, and the sealed runner record `e457-head-transfer-v1`.
