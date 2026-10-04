# E426 real-data event-clock input audit

The input-only audit passed on the actual fold-0 fitting membership. No model was fitted or scored. Three matched teacher-clock fits remain pending.

The campaign manifest admits 10,535 TRAIN and 1,405 validation sequences, with 468 validation sequences excluded for evaluation. Injecting TRAIN row group 528 was rejected before reading event counts; eligible row group 529 remains included. This independently covers the production exclusion that the earlier synthetic trainer fixture did not cover.

The counter reads only input column `a6`, one Parquet row group at a time. It does not read targets or scored-row masks. The exact float32 endpoint marker is -5.199337482452393. At seed 3, epoch 0, batch 32 and chunk 1,000, the sampler retains 11,936 of 11,940 fitting sequences and produces 7,460 updates. Those batches contain 31,685,622 endpoint events. The observed mean per update is 4,247.4, range 1,858–7,997, with coefficient of variation 17.59%.

Constant, aligned and within-epoch shuffled clocks preserve the event-count multiset and epoch-total log-retention. Their computed decay products are respectively 0.0005735115463124044, 0.0005735115463124025 and 0.0005735115463123999. This establishes a nontrivial clock intervention with matched total retention; it does not establish that event alignment improves a teacher or student. The teacher's initialization-copy convention and fixed evaluation EMA remain part of the staged implementation.

The bounded CPU worker finished in 39.83 seconds, exit 0, verified teardown, peak RAM 1,441,067,008 bytes, no GPU and no worker OOM events. The runner seals complete input hashes; the report additionally records Parquet size, modification time and footer hashes. The raw report and log are preserved.

- Result: `checks/e426_event_count_audit_f0_s3_v1.json`, SHA-256 `3d1b28ff6e3e3067a77bfbd09ed250ee836fa94bb48bc6e5e3564040248f8b2b`.
- Worker: `runner/attempts/e426-real-event-count-audit-v1.json`.
- Source/spec: `staging/e426/scripts/audit_event_counts.py`, `specs/e426_real_event_count_audit_v1.sealed.json`.
- Marimo figure: `eda/event_teacher_clock/event_clock_schedule.png` and SVG; report hash and descriptive statistics in `eda/event_teacher_clock/source.json`.

The scientific next step is the fixed constant/aligned/shuffled comparison with separate teacher and evaluation EMAs. A full-sequence GPU memory smoke precedes admission. Historical coupled-EMA results do not close this separated-clock question.
