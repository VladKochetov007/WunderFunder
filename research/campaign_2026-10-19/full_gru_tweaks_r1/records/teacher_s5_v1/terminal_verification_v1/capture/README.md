# S5 terminal verification capture

This capture records the terminal state of the native seed-5 unit and the bounded verifier after the verifier had finished. The capture transport succeeded once at 21:39:11 UTC with return code 0, wrote raw stdout before parsing, and did not retry. The [raw capture output](STDOUT.raw.json) is 19,972 bytes with SHA-256 fb4811b7131bde684e9576ae97afb68a4dff088b7540084f36a8bd892cfd2b80; see the [transport receipt](TRANSPORT.json).

The capture embeds the PASS [verification output](VERIFICATION.json), the verifier's [admission snapshot](VERIFIER_ADMISSION.json), its launch request [result](VERIFIER_LAUNCH.json), and the exact native [terminal evidence](TERMINAL_EVIDENCE.json). The latter is byte-identical to the root [TERMINAL_EVIDENCE.json](../../TERMINAL_EVIDENCE.json), SHA-256 e284787effbb9ad1a64be763c6bc5eb9c805bc1715c37b08aa032b2673b433b4, which binds the successful S5 exit and the 30,350-byte native health capture 9bdc3861961d19ed6454007376bb69f94f4429eb2eba5dce99497a5335d98421.

At capture, the native unit was active/exited, MainPID 0, ExecMainCode 1, ExecMainStatus 0, invocation 4317408cdc69442c8438c74aaf5809eb. The verifier unit was also active/exited, MainPID 0, ExecMainCode 1, ExecMainStatus 0, invocation 6594891a6fea423f907e7c478da0af31. Its initial launch receipt had recorded only that the systemd-run request returned and the unit was running; this later capture establishes completion.

The verifier launch and admission point to the checked wnn-campaign.slice, and the captured parent memory.events.local counters are all zero. Both terminal unit ControlGroup fields are empty and MemoryCurrent is not set after exit. This means no terminal unit cgroup path was available to inspect at capture; the records use the reported unit MemoryPeak fields and parent-cgroup evidence without substituting host RAM statistics. See the [resource reconciliation](../../RESOURCE_RECONCILIATION.json) and [later exact stop](../../terminal_exact_stop/README.md).

The captured unit.log is empty; the durable PASS result is in VERIFICATION.json. This record contains compact status and metadata, not checkpoint, optimizer, or dataset payloads.
