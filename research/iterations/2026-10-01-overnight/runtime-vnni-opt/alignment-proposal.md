# Packedweight64bytealignment, before timing (01:32 UTC)

Guarded block128 stillabove50minute target. Static inspection found packedweights use default std::vector<int8_t>, whereas originaldense kernels use64bytealignedallocator. A default16byte-aligned allocation can make each64byte VNNIweightload crosscachelines. Change only packedvectorallocation toexisting kernels::Int8Vec (64bytealignment), retaining identical packedbytecontents and all operations. This is an exact memorylayout optimization, no arithmetic/quantization or model changes.

Preserve native_block128 source and allreceipts. New native_aligned derivedfromit; verify integeroracle/tails, actualpackeddataaddressmod64==0, and full100koutputhashequalinitialquant. BenchmarkCPU18 paired with block128, record sibling58; ≤76.14µs remains target and≥85µs remainsabort. Stageoverall deadline01:48UTC. NoWP.
