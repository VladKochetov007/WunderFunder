# Native adapter prepared, generated checks passed

One immutable library and one state-packing implementation, no kernel or precision search. Generated20k-row test passed7 checks: O23→O2 head trim/layout matches the original packer, outputs/states match the original wrapper byte-for-byte, raw-feature/recurrent state persists across chunks, resets and causal prefixes are exact, invalid dtype/shape/nonfinite input is rejected. Check itself1.994seconds; bounded owned CPU35 scope3.703seconds, released.

This removes checkpoint-format/stream-wrapper work from the eventual trained native comparison. The actual checkpoint loader was reviewed but no trained checkpoint was loaded and no real features, targets, OOF, GPU, forecasting fit or metric were read. Trained quantization drift, full native ensemble quality and package/runtime qualification remain untested. Native state() returns only recurrent state; it cannot restore raw event-feature history. Actual trained native inference requires a separate allocation after the fixed forecast screen is assessed.

Current training/evaluation sources and incumbent ZIP are unchanged. All sources/check outputs preserved locally and SSH; terminal campaign collector will include them.
