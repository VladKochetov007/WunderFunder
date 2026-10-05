# E488 duration actor source receipt

This packet implements the registered 12 continuous trajectories: seeds 271839–271841 crossed with GELU, fixed mixture, learned static gate and learned recurrent-context gate. Each trajectory keeps AdamW state across three epochs, shuffles the canonical 24 fit-role positions with `default_rng(seed + zero_based_epoch)`, and resets sequence state at batch and epoch boundaries. It saves epoch 1 and 3 checkpoints plus atomic progress/telemetry receipts. The implementation reads the fit-role cohort only; scoring is absent.

The actor delegates the frozen E484 arm equations and construction to `fit_context_mixture_v2.py`, and pins E470/E474/E477, E484 actor, `wnn.nn`, `wnn.resources`, dataset, cohort, design, blind critique and controller clarification hashes before loading data. The custom update wrapper reproduces E474's global norm clip at 1.0, AdamW step, and finite-parameter check while sampling the additional registered E488 indices. At sample updates it records family preclip gradient RMS, global norm/clip factor, gate coefficient RMS/range, and parameter/head update RMS. Optimizer state finiteness is checked after every update; the trajectory also asserts optimizer step counters continue across epochs.

The E488 specification and clarification hashes are checked at runtime. Initialization uses one E474 baseline state per seed; zero-initialized gate tensors are added by E484's model constructor, while every arm construction is reseeded identically. The tests verify that the gate allocation consumes no additional RNG draw, all three mixture functions agree at initialization, shuffled orders match the zero-based rule, and the local clipping/update helper matches the frozen E474 helper over two optimizer steps.

## Bounded validation

- Synthetic CPU tests: `WNN_CPU_PCT=4 WNN_CPU_QUOTA=100% WNN_MEM_MAX=2G WNN_THREADS=1 WNN_RUNTIME_MAX=300 .venv/bin/python research/campaign_2026-10-05/context_mixture_duration/tests_duration.py` — 5 passed.
- Fit-only real-data CPU preflight: `WNN_CPU_PCT=4 WNN_CPU_QUOTA=100% WNN_MEM_MAX=2G WNN_THREADS=1 WNN_RUNTIME_MAX=2400 .venv/bin/python research/campaign_2026-10-05/context_mixture_duration/fit_duration.py --preflight-real --device cpu --output research/campaign_2026-10-05/context_mixture_duration/preflight_cpu_v1` — 12/12 actual B4×256 fit-only cells completed one update, finite changed parameters and `[2,4,64]` state; `test_rows_loaded=false`.
- Ruff, Ruff format and `py_compile` pass for actor, tests and callable CUDA probe.

No full trajectory or CUDA probe was launched. This packet is source-ready for an independent review; full fits remain contingent on the separate source gate, actual-shape CUDA probe and the preregistered 900-second timing projection/admission.

## Frozen identities at this preflight

- Design: `3764a095504af6da66a4faa908a47ff5dd3f94266249d5bea7c07cf611148824`
- Blind critique: `ccc5caba6eef86b928e31cc78ce65602ad528a6549b0017a3d91ffcf1a3aa3e1`
- Controller clarifications: `d382f867bd8c889c5c63f40228ba1133e0f52d2a9e396860fbf15009fced0a9c`
- E484 actor: `2dec963a966ba30e1dc3deed40472fd2d0129a2d24fa07585a3a7ed4e2e1dcbc`
- E488 actor: `89d38cd71da3ff8905d51da60d369bbe04b07d26789ab242085d59d384ac299e`
- E488 tests: `29a2b247d5f32e32c5242325c0bbce35f511a44fc32d01c429d29f252d33413d`
- E488 CUDA probe source: `3e1f383c9364e70b2ea0e4846a15de5281090326a1d5fe634061dbaebea5747f`
- CPU preflight manifest: `d4e25b8e8e84a7506ec5a8eecf973791b3f2aa07569bf5c64704970f6ea8fae6`
