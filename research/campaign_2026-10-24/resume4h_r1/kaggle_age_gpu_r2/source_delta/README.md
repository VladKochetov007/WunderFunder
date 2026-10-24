# CUDA source delta

This folder preserves candidate source bytes copied from the verified R2 overlay. The age_readout_runtime.py file is the single allocator-cap adaptation; its SHA256 is 866ea97fae1ecd1598243e56fc23c112a01b9cf2b3cb70e73009f7ae3126eb6e. It sets the allocator constant to exactly 4_000_000_000, matching the recipe limit. age_readout_cli.py and age_readout_runner.py retain the reviewed profile-only CUDA guards from R1.

The combined archive also carries the two latest membership-repair modules; they remain at their canonical project paths and are pinned in [SOURCE_SHA256_GPU_R2.json](../SOURCE_SHA256_GPU_R2.json). No source was imported or executed here.
