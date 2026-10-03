# Exact training-state measurement round

By18:47 Kyiv, the synthetic state instrument passes but actual training replay is unavailable from EMA-selected exports. This round materializes missing evidence instead of extrapolating from surrogate gradients. Goal remains broad research through23:00 today, public best reported0.6641, no0.667 claim.

Halley (01a101e5-122c-7e90-87f6-41f343a9d353), gpt-6.1-sol medium, writes only live_capture/. Implement isolated source-bound state capture/replay and run CPU/schema tests. Genuine incoming raw/EMA hidden states, raw/lagged weights, final weights/labels/cachedteachers/aux masks, RNG, optimizer moments/ordering, scaler/scheduler and pre/post clocks are necessary. Capture-disabled online semantics must match original. Exact online/default replay parity precedes aligned/permuted-main observations. Unsupported fast/Muon/fp16 paths must fail explicitly. Budget30min builder pass.

Averroes (01a1019d-b660-75c1-bcfe-56fc8a15f897), same model/effort, writes only capture_critic/. Blind pre-mortem before builder implementation reads; independently executable corrupt-state/control fixtures, actual update-order and skip semantics. Freeze before reviewing builder. Budget20min initial pass.

Other retained threads continue observable_simulator, washout_usage and usage_review. Five open contexts are permitted, but only FOUR CPU compute slots may execute concurrently; wrapper admission queues excess work. Each body1CPU/2GiB/900s. GPU gets8CPU/24GiB, aggregate36GiB=38.65decimalGB/12CPU/zero swap, observedVRAM cap5.5decimalGB. CPU-only builder/critic do not acquire GPU.

No current trainer, grader, watcher, budget or queue mutation by actors. Candidate code stays isolated. No independent GPU smoke launches: controller first reviews frozen artifacts and independent critic. Existing combo folds2/3 finish first. A later <=10min measured GPU smoke using batch32/chunk1000 may run only through campaign admission, fresh job tag, strict source/checkpoint/sample provenance and graceful completion before23:00. It is a measurement, not an optimization fit or validation-score candidate. Synthetic actual-architecture smoke and real-data capture remain separately labeled. No future/state invented from EMA-only exports.

Verification tierA for explicit toy invariants, B for reproducible actual-backend step parity, C for any training/public-gain inference. Preserve raw failures, captures, hashes and costs. AMP skipped-step and scheduler/EMA clocks must be measured, not normalized away. No submission upload.
