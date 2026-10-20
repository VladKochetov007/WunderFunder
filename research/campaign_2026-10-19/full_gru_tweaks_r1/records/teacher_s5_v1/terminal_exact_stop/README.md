# S5 exact stop

After the root accepted the durable verifier result at 21:39:26.091965 UTC, the exact-stop request targeted only the native S5 teacher and its S5 verifier. The single transport completed at 21:39:26.660438 UTC with return code 0; both systemctl stop calls returned 0 and produced no stderr. The [raw stop response](STDOUT.raw.json) is 1,798 bytes, SHA-256 f272fa2aed81477dd37b9e046bb8d9c3691798c98af0655f539d16d19610cab0; see the [transport receipt](TRANSPORT.json).

Immediately before stop, both units were loaded active/exited with MainPID 0, ExecMainCode 1, and ExecMainStatus 0, with their recorded invocations: native 4317408cdc69442c8438c74aaf5809eb, verifier 6594891a6fea423f907e7c478da0af31. After the stop, both were LoadState not-found, ActiveState inactive, SubState dead, with empty control-group and invocation fields. No S5 unit remained populated.

The parent campaign cgroup's populated map was empty. Its memory.events.local counters remained zero, and the global memory.events counts were unchanged from the preceding terminal capture. The stop response binds to the accepted [TERMINAL_ACCEPTANCE.json](../TERMINAL_ACCEPTANCE.json) SHA-256 8d22151187a5d64ecd13cab15d10dc8fc320faddf02cdc5dac23ac3cba64411a. That acceptance record was created just before the stop request and still says exact stop is pending; this later stop receipt supersedes that timestamped status text. See the [resource reconciliation](../RESOURCE_RECONCILIATION.json).

This stop closed the native and verifier services only. The next cache step is **NOT RUN**; no inference, grading, or model-quality result is claimed.
