# E488 controller clarifications before implementation

Accept the independent blind critic. Epoch indices are zero-based0,1,2;
shuffle fresh `np.arange(24)` with `default_rng(seed+epoch_index)` in canonical
fit-role order, independently per epoch but identically between paired arms.
Keep optimizer moments across all epochs and reset sequence state between
batches/epochs. Use E484's exact fit-only mean/scale(minimum0.001), causal
feature transform and clipped normalized features, with identity model mean
and scale buffers. No evaluation-role rows may affect normalization.

Record fixed telemetry separately for each seed/arm, at global zero-based
updates0,46,236,473,947,1421. These indices refer to the continuous trajectory,
not a restarted optimizer. Receipt fields must identify epoch/batch/chunk and
finite coefficient RMS, gate min/max, preclip gradient RMS and head update
RMS. Save checkpoints/progress immediately at epochs1/3 and preserve partial
receipts on any failure; partial completion is not a complete matrix.

The actual-shape probe must exercise all12 seed/arm cells for finite loss,
clipping, changed parameters and correct state shape, plus exact same-device
native E484 GELU/fixed readout parity with clipping-active AdamW steps. Check
memory immediately after parity allocations and after each update. Source
review must cover learned-gate paths and second/third-epoch optimizer-state
continuity. Keep the900GPU-second study cap and require measured probe timing
projection plus initialization/checkpoint/teardown margin before full fits.
