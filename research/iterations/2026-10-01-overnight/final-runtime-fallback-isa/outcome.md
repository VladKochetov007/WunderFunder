# Baseline3 fallback ISA runtime supplement

Both prospectively fixed single traces completed correctly and quietly, with no retry. AVX2 ran first, then v2, through the unchanged baseline3 ZIP's actual PredictionModel.predict. Every trace processed100,000 callbacks across five complete sequences and emitted99,505 finite prediction pairs. Exact-byte comparison passed for2,065 archived emitted prefix pairs and495 warmup zero placeholders per ISA. All100k outputs were checked for finiteness and hashed;100k archived parity is not claimed.

| Forced existing ISA |100k callback wall|39.4M callback projection|50min /60min thresholds|
|---|---:|---:|---|
|AVX2|12.6969s|83.3761min|Both exceeded|
|v2|16.0463s|105.3706min|Both exceeded|

Both callback and whole-trace SMT58 occupancy measures were below the frozen5% quiet threshold. Each process retained one observed thread on CPU18 under100%CPU quota,4GiB memory, zero swap and the180s hard limit. Peak final getrusage RSS was104,360KiB(AVX2) and103,464KiB(v2). Pre/post package, raw fixture, archived reference and source hashes matched. The actual outer elapsed times were15.36s and18.73s respectively.

These measurements establish a timing limitation for the existing fallback paths on this machine. They are single short warm-input traces, with no uncertainty estimate, other-hardware guarantee or complete-process39.4M extrapolation. The earlier full-valid nativev3 audit remains a separate45.40min full-process projection. No artifact or model was changed, and no targets, scoring masks, WP query or training were used.

AVX2 ended04:45:04.518UTC and v2 ended04:45:23.251UTC, each exit0. Both workerPIDs were absent and both owned scopes inactive/dead at04:45:59UTC. CPU18/58 was explicitly released. All launch argv, timestamps, return codes, immutable attempt markers, progress, logs and worker receipts are preserved in execution/. Independent critic review is separate.
