# E419 dual-T4 QAT v4 preparation

This private retry reuses the exact v3 FP32 fixed-epoch-0 checkpoints, provenance, validation arrays and sidecars. The v3 source archive is staged byte-for-byte alongside the v4 bundle; the code closure SHA remains `36acf3c456209e195e5c8ba8d813ae436fe63adeb0a1ee1eb9b3ca6425db2796`. Cached checkpoint/prediction identities and the fold-0 split hash are recorded in `control_reuse_provenance.json`. The two control checkpoints have identical tensor state; the recovered null contrast and the v4 control screens are path checks only.

The corrected v4 screens use only t4v4 identifiers. Each T4 worker first validates its cached FP32 score path, then retains zero-update parent PTQ and FP32-control PTQ exports/predictions/scores for both quantization presets, followed by its prespecified fixed one-epoch QAT fit, fixed epoch-0 full-valid prediction, and single-arm score. The v3 production-shape QAT probes already succeeded on both physical T4s; no repeat probes or FP32 control fits are scheduled.

Validation before upload: 53 CPU tests passed, 2 native-only tests skipped, 0 failed; both cached-control score smokes returned WP 0.681496158054268 on the frozen fold0 split. `validation_preupload.json` binds the archive/code hashes and screen/spec checks.

Resource contract: one dual-T4 kernel, worker cap 14,400 seconds, provider timeout 15,300 seconds, 8.0 device-GPU-hour worker maximum plus 0.5 GPU-hour provider-overhead reserve, hard attempt cap 8.5 GPU-hours. At the observed 26 GPU-hours available, a full charge leaves 17.5h, above the 6h weekly reserve plus 9h staged width-pair budget. Worker stop limits remain 13 GiB VRAM and 13 GiB host RSS. No TPU retry or competition upload is included.

No dataset upload or kernel push has occurred yet.
