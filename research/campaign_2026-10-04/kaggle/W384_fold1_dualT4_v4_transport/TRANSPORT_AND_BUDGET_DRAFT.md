# W384 fold 1 dual-T4 v4 transport and budget draft

This is a source-only launch proposal. It has not been uploaded, admitted, or run. The v3 bundle, v7 receipt-overlay packet, and provider inputs remain unchanged.

## Scientific scope

Run the already reviewed W256 and W384 fold-1 AdamW continuations from the v3 protocol under fresh `t4v4` run identities. Keep the model, data, teachers, objective, seed, optimizer, training precision, and fixed checkpoint rule byte-equivalent to v3. The v3 archive SHA is `cfaed9a32399348d5bb1a13b817cd219c69b0f72adb03de7496735a173094a05`; worker code SHA is `1cf290c1c845e3b95ec1dcc3ce11b607484b205be20fff2f43d15306ba9dc8e2`.

The fit recipe remains fold 1, seed 3, batch 32, chunk 2,000, four epochs, AdamW, FP16, LR 0.001, weight decay 0.01, the existing teachers and distillation settings, and fixed epoch-3 checkpoint/export. In-fit validation and best-epoch selection remain disabled. After both fits complete, export and score the fixed epoch-3 checkpoints with the existing paired fold-1 screen. The screen is a one-fold paired diagnostic, not a public-validation or significance claim.

Every new run name and output path must use `t4v4`. Run both fresh production-shape one-batch probes concurrently on the two visible T4s first. Require both probe workers to pass source/input checks, complete the expected update with no AMP skip, stay below the 12 GiB operating target and 13 GiB hard stop, and stay below the 10 GiB worker RAM cap. Only after both probes pass may either full fit start. Bind each full fit to the physical T4 UUID used by its corresponding probe. Do not reuse v2 probes or receipt hashes.

## Hashing, setup, and transport controls

Use strict current full hashing for the archive, each controller phase, and each worker's portable inputs. Hash verification runs independently before the probe phase and before the fit phase; mounted inputs remain read-only and their expected path, size, and content digest checks remain required. The receipt-overlay v7 work is not in this immediate run path.

Add a watchdog as the first executable action in the fresh kernel entrypoint, before bundle lookup, archive hashing/extraction, test imports, portable verification, or CUDA initialization. The watchdog must be a separately running process with an elapsed monotonic deadline, monitor the kernel process tree (including children that create a new session), write a small timeout record, send TERM and then KILL after a bounded grace, and reap the runner where possible. Set the watchdog deadline to 25,180 seconds and the Kaggle provider kernel timeout to 25,200 seconds, leaving 20 seconds for timeout-record flush and cleanup while preserving a strict 7-hour provider bound. If the watchdog cannot start or later exits unexpectedly, fail closed before training. The worker-level timeouts below remain secondary guards.

Keep full source CUDA preflight. Increase the production-probe worker timeout to 2,400 seconds to cover the observed slow full-integrity startup plus one real update. Set each full-fit worker timeout to 12,600 seconds. Preserve current per-worker CPU, RAM, allocator, GPU operating, and stop limits. No fitting may begin unless both probes pass.

## Runtime and budget model

The W384 v3 provider log measured a 3,440.275-second gap from controller `validating_inputs` to its first worker heartbeat while verifying 84,386,156,018 declared bytes. The CPU-only v3 diagnostic measured a faster 758.186 seconds for the same declared byte total; that result is not used as the T4 planning rate. A 9-hour bound is therefore used instead of extrapolating the faster CPU diagnostic.

| Component | Conservative wall allowance |
|---|---:|
| Two controller full-hash passes, using 3,440.275 seconds per pass | 6,880.550 s |
| Paired probe workers, each capped at 2,400 seconds plus the 30-second launch stagger | 2,430 s |
| Paired full-fit workers, each capped at 12,600 seconds plus the 30-second launch stagger | 12,630 s |
| Fixed-epoch prediction, paired scoring, output closure | 1,200 s |
| Kernel setup, source preflight, and staging margin | 600 s |
| **Modeled total** | **23,740.550 s (6.595 h)** |
| **Internal watchdog deadline** | **25,180 s (6.994 h)** |
| **Hard provider timeout** | **25,200 s (7 h)** |
| **Unallocated margin inside hard timeout** | **1,459.450 s (24.324 min)** |

The 7-hour provider reservation is at most 25,200 provider-quota seconds if quota charging follows elapsed wall time. The observed dual-T4 interval charged 908.804 provider seconds over 909.405 wall seconds (ratio 0.99934); this is one observation, not a guarantee. The latest typed Kaggle quota snapshot is 80,151.692 GPU seconds remaining (22.2644 h) at 2026-10-04 20:09:56Z. Under a conservative one-provider-second-per-wall-second model, this run would leave 54,951.692 seconds (15.2644 h). Re-read typed quota immediately before admission and again after termination; do not infer a new balance from serialized duration strings.

Provider quota and allocated device-GPU-hours are separate gates. At the full 7-hour bound, a dual-T4 kernel could consume at most 14 allocated device-GPU-hours. Root's 2026-10-04 20:14Z conservative campaign upper-bound ledger is 25.9486 device-GPU-hours (25.9486h used + 14h planned + 6h recovery = 45.9486h of the 80h ceiling). Recompute from the latest root ledger at admission; if its cumulative value changes, require `cumulative + 14 + 6 <= 80`. Do not treat provider quota as the device-hour ledger.

The component sum is a planning estimate, not a guarantee that the run finishes. The 9-hour whole-job kill is the hard bound. If it fires, preserve all logs/checkpoints and report each phase actually completed; do not auto-retry.

## Admission checklist for independent review

1. Build a new immutable archive and manifest with `t4v4` identities; verify every unchanged scientific source/input hash against v3 and enumerate only intended run-identity, timeout, and transport differences.
2. Verify the private asset dataset/version inventory and every provider-expanded file digest against the frozen manifest.
3. Inspect the final kernel entrypoint to confirm the watchdog starts before any setup operation and that timeout cleanup covers the entire process tree.
4. Confirm the exact paired probe/fit output declarations and fixed epoch-3 scoring path. Any probe, memory, source, hash, quota, or transport failure blocks all fits.
5. At admission, record both typed provider quota and root's cumulative device-GPU-hour ledger; require provider remaining to cover the 9-hour reservation plus root's provider reserve, and require `cumulative_device_gpu_hours + 18 + 6 <= 80`.

No package upload or kernel push is authorized by this draft.
