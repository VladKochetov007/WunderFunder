# Submission review status

An independent Luna/max reviewer was requested to inspect endpoint ranking, since×KD interaction, final-data ancestry and membership. It did not return a report within the collection window and was closed. **No independent scientific approval is claimed.**

The delivering agent completed the checks recorded in `submission_checks/`: official full-valid scores cross-checked with the separate reference implementation; member NPZ tensors exactly equal fixed-ep3 checkpoint tensors; original runtime identity retained; native ABI, ZIP CRC, finite outputs and A-B-A bitwise reset; native INT8/FP32 parity on 64 sequences. These are reproducible actor checks, not an independent review.

Final-data logs record 12,479 fitting sequences, versus 12,480 row groups in the raw train+valid count. Exact attribution of the one omitted sequence was not independently verified in this collection. The manifest preserves actual source/checkpoint/provenance identities; no held-out quality is claimed for these final-data models. The broad final launcher bypasses the historical train-only guard; source flags in the saved checkpoints show train+valid and evaluation withheld.

The alpha-.9 single is an adaptive, one-seed discovery endpoint. The fixed alpha-.5 pair improves on the same exposed official validation; its final-data refit has unknown public transfer. Package timing is a 64-sequence local measurement with a ratio projection to the prior same-engine public single, not a host guarantee.

[Handoff](SUBMISSIONS_R1.md) and [machine-readable checks](submission_checks/README.md) describe the exact release scope and limitations.
