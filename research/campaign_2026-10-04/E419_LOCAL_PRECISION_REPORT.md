# E419 matched local training-precision continuation

Both arms start from the same fixed combo fold0 epoch3 parent, same data/preprocessing/teachers and loader RNG, seed3, batch16, chunk1000, AdamW1e-4, one epoch, fixed epoch0. Exports for both use common FP32 inference. No evaluation-based checkpoint selection.

FP16 WP0.6815120231527334; FP32 WP0.6814959774104583. FP32 minus FP16−0.0000160457423, paired component95% interval[−0.0001502265235,+0.0000863879555],468 sequences/141 components. One exposed fold and training seed; no public confirmation or universal precision claim.

FP16:14920 attempted updates,6 AMP skips,14914 completed optimizer updates,14920 EMA updates; train856.3s, elapsed919.08s, host peak5,513,555,968 bytes, VRAM2,245,001,216 bytes. FP32:14920 completed updates,0 skips; train1432s, elapsed1522.887s, host peak25,753,325,568 bytes, VRAM2,297,430,016 bytes. FP32 approached its24GiB host cap and incurred memory pressure/reclaim; no OOM. CUDA FP32 may permit TF32; this comparison does not isolate TF32 or skip-cadence effects.

Both gain-cache GPU extractions completed subsequently. Earlier interrupted pre-epoch runs remain preserved and are not scientific results. Kaggle QATv2 failed preflight because a helper was omitted from the portable bundle; no fits took place. A new v3 closure was prepared with explicit PTQ controls and independent same-T4 float continuations.

Evidence: `checks/e419_precision_report.json`, `checks/e419_precision_screen.json`, `checks/precision_sequence_v1.json`, sealed specs and runner resource records.


Remote v3 follow-up: both matched FP32 control workers completed full-valid FP32 predictions but score lookup used stale t4v2 names. The saved t4v3 predictions were recovered and scored locally with corrected single-arm specs; both score0.681496158. Their checkpoint model tensors and prediction arrays are exactly equal; this confirms the scoring path and duplicate control, not a scientific comparison. No PTQ/QAT fitting ran. See E419_FROZEN_CONTROL_RECOVERY.md and the preserved Kaggle failure report.
