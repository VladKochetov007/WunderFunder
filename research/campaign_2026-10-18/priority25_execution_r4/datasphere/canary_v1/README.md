# Runtime canary after DataSphere pre-start failures

This exact no-training probe uses the Oct8 successful manual Python3.12/cu126 requirements, four dataset IDs, g2.1,100GB SSD and SIGTERM/60s. It includes its real entrypoint and a nonempty SDK local module, writes one tiny runtime JSON, and reads no supplied data. The frozen profile/trainer are unchanged. Success establishes only this runtime envelope; a repeat no-start error does not identify a global outage or exclude all input/environment causes.

Root reviewed [executed probe.py snapshot](probe.py.snapshot), [config.yaml](config.yaml), byte-identical historical [requirements.txt](requirements.txt) and [runtime_marker.py](runtime_marker.py). The private controller is the existing profile controller with exactly two constant changes: the canary job name and a600-second submit-relative cancel trigger. Actual cancellation depends on the durable host/network/provider. A10-minute cap may be insufficient for cold dependency installation; classify deadline exhaustion separately from provider failure. No full fit follows automatically from a marker alone.

The [terminal receipt](../../receipts/DATASPHERE_CANARY_V1_TERMINAL.json) records ERROR/INTERNAL13 before start with no files, removed credential and unchanged observed balance. No runtime result was obtained.

See [historical admission receipt](../../receipts/DATASPHERE_CANARY_V1.json) for live quota/balance, exact files, job/service ownership and output verification. SDK collection-command success is not result verification: compare provider output size/SHA and marker contents. The current source packet alone does not prove admission or execution.

The probe is archived byte-for-byte as `probe.py.snapshot`; the private admitted job input remains `probe.py`. A post-admission style check reported import ordering (I001) and the timezone alias (UP017). These are recorded style findings, not runtime failures; the frozen executed bytes are preserved. New attempts should format/check before sealing.
