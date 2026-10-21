# AllThree startup metadata snapshot

This is one early, read-only snapshot of `fg19_child_allthree_f0_s1001_fit_v1`, captured at 2026-10-21 02:00:36.378025 UTC for invocation `94d3a7132a434d448345b0206bd3daa3`. The native unit was still active and running with `MainPID=1875143`; this is startup telemetry, not an exit, terminal verification, acceptance, or stop.

The capture shows the unit process and its own cgroup populated, with approximately 711.7 MB current and 712.0 MB peak memory at that point. The parent campaign cgroup is a separate measurement. The selected GPU snapshot lists three other owners, and no process attributed to the AllThree unit was visible yet. This single early snapshot does not show an owned VRAM allocation and is not evidence of fit failure. The stat-only run-directory observation found no output directory; the joined `unit.log` was empty, so no update count, batch, epoch, or quality result is available here. The 44,760 update number in admission is an arithmetic expectation, not observed progress.

The persisted raw response is [STDOUT.raw.json](STDOUT.raw.json), 38,779 bytes, SHA-256 `a1d5e8c9e53323bdd63f3dc3c03eb8fa0ad057c33a47c109de3823c8487109d6`. It was persisted before parsing; transport returned 0, stderr was empty, and no retry occurred. The three receipt files were strict-base64 extracted from that raw response using create-only writes, fsynced, and byte-compared against their decoded source members:

- [ADMISSION.json](ADMISSION.json) — 12,912 bytes, SHA-256 `5312ae5feffc67d7a33051379afabb9c134dd946f1f97315d839c7d6f10d5540`.
- [LAUNCH.json](LAUNCH.json) — 9,594 bytes, SHA-256 `ec9d8dc819894e148753838ba33ba95080c192bacd5ac53413bd6847568fd538`.
- [unit.log](unit.log) — empty at capture.

The exact reader was [REMOTE.py](REMOTE.py), 13,018 bytes, SHA-256 `0df25d26ba9e3c1eda9a2a82cf6b7940577fb8b25b81002fdfd7c27f15341b9b`. [AST_SUBSTITUTION_RECEIPT.json](AST_SUBSTITUTION_RECEIPT.json) records that its syntax tree matches the NLL startup reader after only identity, receipt, schema, and prose-constant substitutions. The source identity remains the fresh admission authority; this capture does not rehash the source tree.

For the original NLL reader lineage, see [NLL startup reader](../../child_nll_fit_v1/startup_health_v1/REMOTE.py). The NLL predecessor acceptance and exact stop are [TERMINAL_ACCEPTANCE.json](../../child_nll_fit_v1/TERMINAL_ACCEPTANCE.json) and [EXACT_STOP.json](../../child_nll_fit_v1/exact_stop/EXACT_STOP.json). No model, checkpoint, optimizer, cache, dictionary, dataset, source, or trainer payload was read or hashed for this startup index.

The structured summary is [OBSERVATION.json](OBSERVATION.json). Later observations and terminal evidence must be recorded separately; this snapshot does not establish training completion or quality.
