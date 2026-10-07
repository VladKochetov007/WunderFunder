# ADR0011: measure complete-cohort utilization and amortize verified inputs

Status: accepted for E522 resource optimization on2026-10-07. User explicitly requested
more ideas per compute hour through pipeline/framework/kernel improvements.

E520v6 spent984.22s on initial rawhashes, then748.68s repeating trainhash in the portable
loader, before its1800s cap stopped it preCUDA. NoGPU fit orOOM. Therefore eliminate
redundantIO before changing model/framework. Preserve the first complete byte check.

Authorize worker-local digest reuse only for declared canonical paths after actualbytes
match expectedsource and frozenportable registry hashes, with unchanged device/inode,
size/mtime/ctime and open-descriptor identity. Changed orunverified identities fail or
use the original verifier; no permanent shortcut across independent sessions. Bind new
worker code and all scientific invariants in a newly identified recipe and independent
review. Frozen predecessors and negative runs remain unchanged.

Optimize completecohort cost: startup, verification, profile, fit, export, evaluation,
compilation and allallocateddevices. Amortize unchanged verified data across admitted
jobs. Reuse existing oneprocess/deviceKaggle runner for independentseed/arm scheduling
only after realinventory and combinedCPU/RAM/cache profiling. Noassumedhardware or2xgain.

Largerbatch/loaderworker changes remain distinct recipe variants until update counts,
RNG/sample-order/precision and endpoints are audited. E522localbatch32 gave1.94%throughput
but81.94%moreVRAM;1loaderworker gave4.42%. These laptop shortprofiles arenotcloudgains.
Two-worker scoped5GB probeOOMed because itsfullworkerfootprintwasunderestimated; preserve
it anddo notretry/promote without safermeasuredresources. Aggregate20GB cap held.

Use proportionalindependentreview: verify changed mechanisms and immutablehash bindings,
reuse establishedtransport/grader whenunchanged. Do notwaitforunrelatedfamily caches or
newframework construction to run a valid boundedrepair. Durablepolling logs everyminute;
agentmessages report meaningfulphase/results/failures rather thanidenticalqueue snapshots.

Evidence: [E522 report](../../research/campaign_2026-10-06/bandit_continuation/throughput_optimization_E522/UTILIZATION_REPORT.md).
