# E419 v3 FP32 control-score recovery

Copied v3's two verified fold0 full-valid FP32 control predictions and sidecars into the existing local screen path via hard links. Fixed single-arm score specs to use the actual t4v3 names. Guarded `screen_score.py` completed successfully in34.72s under a2GiB/1CPU unit. Both score outputs are0.681495977 WP. Sidecars validate the fold0 split hash and prediction bytes.

The two run containers differ by run-name metadata only; saved model tensors are exactly equal. Both controls have identical prediction SHA `c0d9f8187049e3d386da24e04171a65c6c0b4cc4dfb2319ad7dbf458b5362db6`; different checkpoint-file hashes include their run arguments. Thus the recovered score verifies the single-arm score path and expected duplicate FP32 baseline. It is not a paired model comparison and provides no QAT/precision effect estimate.

All copied data are hardlinks to immutable captured v3 artifacts. E419 v3's failure report remains authoritative: after these predictions were produced, the remote score step requested the stale t4v2 names. No QAT or PTQ model fit occurred.
