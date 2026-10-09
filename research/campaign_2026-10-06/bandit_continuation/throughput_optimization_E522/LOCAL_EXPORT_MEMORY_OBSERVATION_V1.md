# Local export memory observation and next bounded optimization

Both wide-teacher training exports finished exit0 in about555.5s with1.192GB observed VRAM and no OOM increments. Worker memory high-water approached its6.5GB cap. Seed3 worker max counter increased89989; the hierarchical parent counter reflects child pressure too, so this cannot be described as parent20GB exhaustion or anonymous tensor demand. Original memory.stat/utilization/attempt receipts remain the measurement basis.

Do not increase the cap from this observation. A short20.4s sampling window showed GPUutilization68/2/98/99/98percent and CPUaverage1.084cores: no whole-run efficiency estimate or causal bottleneck proof. Preserve zero loader workers for current delivery.

One future memory/throughput test is an exact memmap target cache rather than per-process compressedNPZ target materialization. First inspect actual data-loader ownership and source arrays; measure anonymous versus filecache plus setup/complete-epoch cost. A producer must bind originalNPZ SHA, exact row-group order, shape/dtype/missingness/masks, use new-only files and preserve training/evaluation equations. Compare source/materialized bytes and model inputs before a bounded profile. This is an unadmitted engineering hypothesis, not an asserted speedup, and has lower priority than landing the current heterogeneous student.
