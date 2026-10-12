# Frozen R3 source overlay, recipe review R2

This tree starts from the complete `clean_t3_fold0_h390_r3_teacher_cohort/source_snapshot`, preserving its `SOURCE_MANIFEST_H390.json`. `SOURCE_MANIFEST_R2.json` binds all 63 original source entries and identifies exactly two intentional overrides; the root controller and resource state remain in the main project.

`scripts/train_rnn.py` adds only the validated target-scale treatment described in [SOURCE_DELTA_R2.patch](../SOURCE_DELTA_R2.patch). `src/wnn/resources.py` is replaced byte-for-byte by main commit `ac6a577f3f209ae70b8dd0830c9d90175d3c1b88` (SHA256 `0b6280130a480f5547552df83cc4a15e7cfe88ad04498ca11c2ab2d54bb428e9`). This is an operational allocator-policy change: it uses live free VRAM and the configured 1,000,000,000-byte safety reserve, with default allocator fraction 1.0. No explicit `WNN_GPU_MEMORY_BYTES` cap is passed. The resource implementation was independently reviewed by the root's resource critic.

The trainer's model, target equation other than the requested scale, losses, label weights, optimizer, schedule, sample order and checkpoint behavior remain the R3 control recipe. The source delta deliberately separates the scientific target change from this operational resource-policy overlay. This source tree is immutable once an admitted run starts.

- [R2 source manifest](SOURCE_MANIFEST_R2.json) — SHA-bound inventory of the complete R3-derived source and both explicit overrides.
