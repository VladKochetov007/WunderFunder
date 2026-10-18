# PR15 checkpoint and export recovery

The original PR15 worker completed all 16 admitted seed1001 fits before delivery validation failed. The validator compared the raw Torch archive hashes for `model_ep2.pt` and `model.pt`; the trainer had serialized identical checkpoint payloads into separately named files, so their container bytes differed. This was a delivery-validation defect, not a failed training run.

The original frozen packet and its failed receipts remain unchanged. The isolated [`host_v2` packet](host_v2/FROZEN.json) compares complete loaded payload identity while preserving each file's actual SHA-256 and checking each against trainer provenance. Its export wrapper also adds the pinned source's `scripts` directory so the unchanged `predict_run.py` can import `train_rnn`. See the [repair analysis](REPAIR_PLAN.md), [fit contract](host_v2/FIT_CONTRACT.json), and [execution manifest](host_v2/EXECUTION_MANIFEST.json).

Root's delivery-only recovery validated and exported/scored all 16 existing endpoints without refitting. The first export-only attempt failed at import before prediction; that failure is preserved, and the retry used the corrected import path. Root reports 12 deployed acceptance checks passed for the corrected packet. The local Windows environment lacks Torch: the two local packet tests pass and the ten Torch-dependent identity tests skip here; the deployed CPU Torch identity suite and F0 full-payload/provenance check passed in root-reported verification.

This packet repairs delivery and validates existing checkpoints. The first-wave grades are only one seed across the four exposed folds (4 of 30 paired cells per arm); they are preliminary screening evidence, not a completed CP30 comparison or an independent holdout result. The [PR15 v2 queue](../pr15_queue_v2/README.md) records those scores and the separate 104-cell continuation.

Historical delivery_resume.py is a rejected exploratory helper and was not used to recover scientific fits. Successful recovery uses immutable host-v2 validation and retained recovery receipts; do not run the historical helper or infer execution from its presence.
