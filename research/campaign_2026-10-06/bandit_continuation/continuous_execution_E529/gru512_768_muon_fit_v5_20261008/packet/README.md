# GRU512/768 Muon capacity fit V5

This packet reuses two actual V4 B8 profiles and performs no profile rerun. It fits the frozen seed-9, clean-fold, FP16(GradScaler scale 4096) Muon GRU512/768 pair for four epochs with fixed epoch 3, then exports FP32 feature-only predictions for exactly 468 frozen evaluation groups. The worker computes no score. V4 profiles completed 230 updates each with zero skips; V4 itself did not fit because its precision metadata consumer read the wrong JSON nesting.

Two simultaneous workers are required for the primary pair. Each worker receives a distinct physical Tesla T4 UUID, independently initializes its model and optimizer, validates that the visible CUDA device is the assigned T4, and reports that device's memory. A preflight checks actual host CPU/RAM and distinct GPUs before raw/cache identity work. If concurrency two is unavailable, only a separately root-authorized Muon512-only fallback is accepted.

Mounted input bytes are never copied into the worker directory. The worker verifies the six exact mounted inputs once, reusing the V4 content digest only when the exact path/stat identity matches; each arm rechecks the frozen stat and symlink binding. The loader is not shared; the host may reuse ordinary OS page cache.

This is a clean-fold capacity screen, not a public forecast and not the missing all-data heterogeneous teacher-family replication. No fit is authorized until root writes an independent review and exact conditional authorization into this packet.
