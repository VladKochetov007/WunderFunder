# Fold-0 dictionary preparation

This record folder contains admission evidence for one CPU-only, train-fold dictionary preparation from immutable source commit `51a9b168584e86ca10ebdc0532465d6158ab19d3`. At 2026-10-19 14:53:57 UTC, systemd accepted invocation `4e6d90ff7d4344df92d61d0d1c3f01fb` as `computer-vision-homework-fg19-dictionary-f0-v1.service`. The source tree was staged on B with 162 manifest-pinned files; its archive SHA-256 is `43070da5ec5a8d47e751c85f35bd01263422f20ed207504ab8029b46e800d173`.

The unit is limited to 0.5 CPU, 1.4 GB memory high, 1.5 GB memory max, zero swap, and 600 seconds; CUDA is hidden and CPU math libraries use one thread. Admission found the existing propensity fit occupying one CPU with an 8 GB memory reservation. Together with this unit, the reserved CPU total was 1.5 cores. The conservative projected campaign memory was 13,487,632,128 bytes against the 17 GB admission target. These are admission-time resource checks, not measured dictionary-fit usage.

The frozen recipe selects 64 fit-only components, samples 31,808 training rows, and performs 756 FP32 CPU dictionary updates. It reads canonical split/index and graph metadata but does not use outer-fold feature or target values, score a model, or fit a downstream model. The cache and receipt paths are create-only.

At admission the job was submitted successfully. That fact does not establish completion or a valid artifact. Use the bounded read-only capture below for current unit state, native log, possible fit receipt, cache metadata, source pins, and campaign cgroup occupancy.

```sh
./.venv/bin/python research/campaign_2026-10-19/full_gru_tweaks_r1/records/dictionary_f0_v1/capture_preparation/capture_dictionary_f0_v1.py
```

The capture helper writes each snapshot to a new directory under `captures/`. It does not stop or change the unit and records cache metadata without reading the cache contents.

Actual native exit0 was14:58:42UTC. [Artifact acceptance](ACCEPTANCE.json) binds64train sequences,31808rows,756successfulupdates, all162source pins and cache stat22557B. [Raw capture](captures/capture_20261019T145934562310Z/CAPTURE.json) precedes the [15:01:38UTC exact stop](exact_stop/STDOUT.raw.json); parent events remain historicalhigh2179/max561/OOM0, local0, no populated descendants. Cache producer SHA9cf76503b579d2a88e7342b5f727e1ad5b3945c0938f1ae62d9738442a68b659 is not an independent payload rehash in capture. Retained systemdMemoryPeak2461696B is a separate observed field, not asserted as full runtime RAM; terminal own-cgroup files were unavailable. The capture helper received a seek-before-tail boundedness correction after the already-saved actual REMOTE_CAPTURE.py; original executed bytes/raw results remain authoritative.
