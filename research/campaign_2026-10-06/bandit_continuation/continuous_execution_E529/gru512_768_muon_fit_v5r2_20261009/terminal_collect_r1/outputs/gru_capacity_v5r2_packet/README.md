# GRU512/768 Muon capacity fit V5R2

This identified V5R2 successor preserves the V5 scientific recipe and reuses two actual V4 B8 profiles; it performs no profile rerun. It fits the frozen seed-9, clean-fold, FP16 (GradScaler scale 4096) Muon GRU512/768 pair for four epochs with fixed epoch 3, then exports FP32 feature-only predictions for exactly 468 frozen evaluation groups. The worker computes no score. V4 profiles completed 230 updates each with zero skips; V4 itself did not fit because its precision metadata consumer read the wrong JSON nesting.

Two simultaneous workers are required. The measured dual-T4 Kaggle host exposes four CPUs and 30 GiB RAM. Each independently initialized arm is allocated two CPU threads, for four aggregate, and receives a distinct physical Tesla T4 UUID. Startup checks that the visible CUDA device is the assigned T4 and reports that device's memory. The preflight verifies actual host CPU/RAM and distinct GPUs before raw/cache identity work. The V4 wall-time projections were measured at four CPU threads per arm; under this two-thread-per-arm shared-host allocation they are reference estimates, not timing guarantees.

Mounted input bytes are never copied into the worker directory. The worker verifies the six exact mounted inputs once, reusing the V4 content digest only when the exact path/stat identity matches; each arm rechecks the frozen stat and symlink binding. The loader is not shared; the host may reuse ordinary OS page cache.

This is a clean-fold capacity screen, not a public forecast and not the missing all-data heterogeneous teacher-family replication. No fit is authorized until root writes an independent review and exact conditional authorization into this packet.
