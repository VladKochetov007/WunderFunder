# H382 local clean crossfold R3 recovery

This packet is a uniquely identified continuation of the already reviewed H382 R2 sequence. It resumes at fold 2, seed 3, epoch-4 fit after R2 step 2 was recorded not_started/admission_failed when the aggregate CPU guard refused a 3-core reservation. No worker began for that R2 step; the failure and all R2 files remain preserved.

The six remaining fit/export commands keep the previously reviewed source, data, folds, optimizer, batch, loader count (zero), model run IDs, fixed checkpoints, and exact worker argv. Only attempt/contract IDs and sequence position are new. Every export contract binds its R3 fit predecessor. The completed fold-2 e3 fit/export remains the immutable predecessor and its score is recorded separately.

No arm is launched by packet construction. Launch still requires current bandit_resources.py CPU/RAM/GPU/output checks. The cache handoff is file-only, swapless, held under the GPU lock, and changes no limits.
