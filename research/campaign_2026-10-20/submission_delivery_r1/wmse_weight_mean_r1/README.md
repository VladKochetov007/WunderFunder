# Mean of parent and WMSE continuation checkpoints

**Proposed compatibility review only. No new fit, average, export or quality grade.**

The [contract](CONTRACT_R1.json) fixes one candidate: the equal parameter mean of the existing T3 parent CP5 and the completed WMSE continuation CP0/CP1. It reuses the canonical averaging helper and requires exact topology and normalization-buffer compatibility. Primary comparison is its fixed 75/25 oldZ4 blend against the existing 75/25 oldZ4/WMSE CP1 blend. There is no coefficient search or additional training replication.

The possible benefit is a better position along a shared optimization trajectory with the same recurrent inference architecture. This is an inference to test, not a measured gain. The two-epoch AdamW continuation differs from the cyclical/constant-SGD setting of the original SWA paper and from the vision/language fine-tuning examples in model soups. A parent and its children share ancestry and are not independent models or seeds. All F0 selection is exposed development evidence; a final complete native endpoint must earn its place.

References: [SWA](https://arxiv.org/abs/1803.05407), [model soups](https://arxiv.org/abs/2203.05482). Root verified these primary abstracts before proposing the local adaptation. They motivate weight averaging but do not establish benefit for this dataset. Source runs and actual metrics remain in the [WMSE/NLL packet](../nll_adaptation/README.md).

The [independent critique](INDEPENDENT_CRITIQUE_R1.json) accepts a bounded compatibility probe but notes the mean may pull the child back toward its weaker initializer. It confirms metadata links parent CP5 to the child initialization and locates both child snapshots. Tensor/buffer equality remains unverified. The canonical tensor mean is reusable; its epochs-3/4/5 derivation wrapper is unsuitable for this cross-run lineage. No score or shipping approval.
