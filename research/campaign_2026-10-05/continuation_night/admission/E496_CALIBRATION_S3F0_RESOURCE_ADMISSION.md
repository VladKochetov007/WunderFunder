# E496 seed3/fold0 calibration resource admission

Controller review, October 6 03:29 Kyiv. Independent source preparation GO is bound to E496_HANDOFF_PACKET.json; this admission is only the CPU fit-only denominator calibration. No optimizer, fit, prediction, scoring or evaluation rows are used by the calibration pass.

The loader has zero workers and a 3-GiB loader budget. SequenceFile reads one 20,000-row Parquet group with Arrow threads disabled. Teacher/propensity arrays are mmap-backed; only the row-group index is eagerly decompressed from target archives. Batch32 raw features are about 287 MB; normalization holds two subsampled feature batches. An 18-decimal-GB job bound leaves room for transient copies and mapped pages beneath parent36/40GB pressure/hard caps. Measured MemAvailable at admission review was 28.4 GB, above job18 GB plus required8 GB reserve. Parent CPU quota is500%, job1CPU, CUDA hidden and VRAM0. Timeout14400seconds; the new-local-start cutoff remains07:00UTC.

The runner must recheck aggregate headroom and source/input bytes before launch. Actual peak memory and terminal/OOM receipts determine downstream admission; this estimate does not clear any GPU fit.

The original 18GB draft was refused before reservation/launch by the frozen CPU-helper6GiB ceiling. Original sealed input is preserved. A separate smallcap draft uses6GiB; zero-worker batching bounds resident tensors well below this, and read-only teacher/propensity mmaps can be reclaimed. This may increase IO/runtime; the actual cgroup memory receipt remains required. No worker has been started under the rejected18GB spec.
