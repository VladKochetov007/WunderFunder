---
kind: review
status: complete
created_utc: 2026-10-15
updated_utc: 2026-10-15
hypothesis_ids: [H-FROZEN-PARENT-ANCHOR-20261015]
experiment_ids: []
evidence_paths: []
---

# Root review of queued frozen-parent profile

GO for the existing50-attempt resource profile after C fit and export release the GPU, conditional on fresh admission. This is not full-S-fit admission. Root reviewed the actor's recipe independently.

Recipe SHA c772f15cfee219ff48dbfc7b99ecf956a51efc7442c671900ab1465e17d7d63d; profile SHA334a0b4fbc1f26fd8ad06526912e1684436117e3145d0e734244030bc88e1316. Direct normalized option comparison with the admitted C fit finds only: replace homogeneous T3 cache with the single direct parent teacher; change run name; omit `stop-after-epoch=1`. The last omission does not change two-epoch exposure or fixed CP1. All other explicit optimizer, sampler, normalization, auxiliary and architecture options match.

Read actual `train_rnn.py` consumer: `Sequences.__getitem__` retrieves legacy predictions through mmap and aligns them by target-cache row-group IDs. A single teacher has averaging divisor1. `bind_fold_teachers` checks sidecar run/split/split-manifest, complete canonical target row order and whole prediction SHA. Parent fitting/ancestry pins remain necessary because matching a manifest string alone is not ancestry verification. The ordinary `teach is not None` loss branch is common to direct and cached teachers: clipped-label magnitude weights, mixed clipped teacher/label targets, same denominator and auxiliaries. Controlled-loss settings remain off in both vectors.

The new profile must remove `save-training-state`, as the trainer forbids continuation saves on profile exits. Five full B16 batches mean50 C2000 optimizer attempts; median step wall is per full sequence batch. Existing6.5GB host reserve,1CPU,4.3GB allocator,300s timeout and5GB observed VRAM threshold apply within the two-core/17GB aggregate admission.

S tests a target package, not isolated new information: parent raw targets differ from T3's0.8125 scale, train exports declare FP16, valid precision is missing, and the initializer itself has older KD ancestry. Those limitations do not prevent a bounded empirical screen. No payload read, GPU run or quality claim was made in this review.

Canonical [recipe/profile packet](../../campaign_2026-10-15/student_quality_tradeoff_r1/frozen_parent_anchor_r1/README.md).
