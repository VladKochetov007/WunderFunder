# E522 throughput probe

User authorized pipeline/resource efficiency optimization. No scientific fit, checkpoint,
score or production recipe changes. Root owns bounded actual-data GRU128x2 throughput
profiling with the E521 scientific flags, hidden128 and profile-only early exit.
Start batch16,4warmup24timed full20k sequences/chunk2000. Only consider batch32 after
actual measured GPU/RAM headroom passes; a batch change is a separate profile, not a
scientificly equivalent production fit. Normalization/sample order/optimizer update count
must be audited before changing a production recipe. Existing fp16/fusedAdam/pinnedCUDA
prefetch/cuDNN recurrent path retained. No framework rewrite in this probe.

3CPU,5GB scopedRAM; parent6CPU18GBhigh20GBhard,noswap. OneGPUlock,4.3GBallocator,
5GBobservedVRAM,600s worker wall,zero loaderworkers. Source hashes frozen by controller.
Preserve existing frozen E520 cloud packet. Compare actual sequence throughput and
GPU/device footprint; do not promise cloud transfer from laptop timings.

Separate profile adds oneCPUDataLoader worker underexisting3CPU/5GB cgroup,loader tensorbudget1.5GiB. Sharedmemory33GBfree checked, prefetch from existingbudgethelper. Earlier measured B16 profile noOOM; parentcurrently11.78GB, enough5GBreservation under18GB. No production recipe admission: worker count can alter RNG/sample-order and must be separately audited before fits. Onlyresource/throughputclaim.

Second loaderprobe reserves4CPU withinaggregate6CPU and5GBRAMcap,2workersprefetch2 bounded1.5GiBtensorbudget. Prior1workerprobe passed withzeroOOM2.096GBVRAM. EachworkersCPUthread1,sharedmemory33GBfree. Profileonly, no scientific fit changes orpromotion.
