# Independent B300 receiver review — R1 profiles / R2 fits

**Review date:** 2026-11-10  
**Decision:** R1’s group-only helper is not safe for resource accounting or stop even in profiles. ROOT is compensating for the already-running profiles with its independent native PID monitor. Do not use R1 for fits. The frozen R2 fit adapter corrects that wiring and passes the bounded local ownership/scheduling fixtures. ROOT retains native receipt review, fit admission, and process monitoring authority; this report does not approve a scientific gain.

## Scope and evidence

I read the B300 worktree `AGENTS.md`, ADR0044, the project experiment workflow, the B300 training and recipe-audit skills, the exact source capsule’s `RUNBOOK.md`, `acceptance.py`, and `sequence_config.py`, plus the receiver and fit-admission code. The pinned source is PR74 commit `8a8afec964985a6ba8b13d1c9af1ab00e14c6504`, source identity `959fc8a335f97a83de31c12dfa6a24f5db4776b78b70bf45f7890cf09b96b5d6`, manifest `1cb6bf60c668f34ea3c49e6f4c3b0c774ea6cdc7e8f8c5f29e5d9f04322f1936`, split `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`. The frozen R2 authority SHA256 is `74663454d5c8ace8f7c1cdeb6c651e3d9a93fb9b2e1c4cc756bd03fb45df3f77`.

ROOT’s native process evidence `b300-rpc/results/024-rank-ownership-progress.json` showed each torchrun rank in its own PGID/session (for example, rank 2341617 had parent 2341565 but PGID/session 2341617; rank 2341618 likewise had its own group/session). This is the concrete reason R1’s group-only ownership was insufficient. ROOT separately reported the profile completion/collection RPCs; actual native receipt qualification and the fit admission remain ROOT-owned.

## Findings

1. **Blocking for fits in R1; corrected in R2.** R1’s `owned_group_members` only included processes matching the launcher PGID and session. The observed torchrun ranks used distinct sessions, so R1 would undercount rank RSS and fail to signal them on a resource/deadline error. Do not launch fits through the R1 runner. R2 tracks the launcher’s observed PID/start-tick descendant closure across sessions, retains observed births if ranks are reparented, and signals only rechecked owned PID/birth identities.
2. **R2 fit wiring reviewed.** `b300_fit_stream_r2.py` imports R2’s runner and resource guard, injects R2 `run_owned` into the frozen R1 stage helper, and passes the R2 guard to every fit stage. Its fixture verifies the injection, cross-session descendant inclusion, unrelated-process exclusion, reparent retention, and per-cell admission checks. The R2 main also records failed terminal state if its fit phase/resource gates raise.
3. **Recipe and admission fidelity.** The frozen configs are byte-pinned to the staged archive and parsed through the exact source `load_config`; raw seed/kernel identity is checked separately, including normalized omission of K0. Production profile receipts require the exact source/config/split, two devices, BF16/BF16x9 runtime, 50 updates, both-rank model/optimizer/scheduler/RNG reload, and resource headroom. The zero-warmup profile is generated from the source helper and checked to differ only in warmup. ROOT’s six-cell admission binds profile and approved-profile bytes; the fit-only adapter revalidates it per cell and again after native profile acceptance.
4. **Fixed fit endpoint and leakage boundary.** Fits use the six exact source configs, world size 2, E4 / 3,280 updates, pinned official chronological split, 10,325 fitting members, 1,873 full-validation members / 3,528,273 scored rows, all-fit train-only normalization, and no teacher/reference ancestry. Receipts require four complete full-valid evaluation events and fixed E4 endpoint; valid scores are report-only. There is no resume or automatic retry.
5. **Nonblocking receipt limitation.** R2’s `live_owned` includes zombie entries, while `_signal_exact` skips them. If a reaped-worker zombie persists under the host init process, cleanup may record it as a survivor and wait through its bounded cleanup window. ROOT is running a native stop proof and can report the behavior; this does not justify delaying healthy admitted fits.

## Frozen receiving-code identities

All hashes below were independently recomputed from the frozen local files.

| File | SHA256 |
|---|---|
| R1 `b300_driver_r1.py` | `fe65168c8bd912886ba0d0481ec0f8626b985d74ae33a5e46ce0bae9fcdc80dc` |
| R1 `b300_driver_core_r1.py` | `f8025a90ba67624a99f705d4c02c3ed341db1215f46adcb00100bfd4184e255d` |
| R1 `b300_driver_profiles_r1.py` | `815bc9149f66d98bec7c2b2659a06ecbe661faa5879e8b6b6781b217f7d52b8c` |
| R1 `b300_driver_fits_r1.py` | `06dba6b976b8aadf2abb1b0b278bb5b27700f0f5fdeb1fb61ee55e24f091008d` |
| R1 `b300_driver_support_r1.py` | `abea048c4ab512596c35928f895291d794ac816e66be11350f902b846be29c91` |
| R1 `b300_owned_runner_r1.py` | `d8ddbfc3247b4651565c2af28c7d80fe9f26aca45f9e92865d16ece3c54b3b20` |
| R1 `test_b300_driver_r1.py` | `e3be78e848b02f2a1192c12c6503e4e98d63fbd17a64b82133165b4c0f6f31ee` |
| R2 `b300_fit_stream_r2.py` | `b8ace046e24edf4b46ef89e419b08e1db2ba530364653ff2b4d96480aa6da1c8` |
| R2 `b300_owned_runner_r2.py` | `da61faac724528de09aa5dad6ced0b402c98993507daae158e0fe700e6eb1dee` |
| R2 `test_b300_fit_stream_r2.py` | `d01ba38e55d5b6d3e51ad06c354bb307c184c147916a65059a366e07b1557a73` |

## Checks and limits

I independently ran the five bounded R2 CPU fixtures with the pinned Python and one-thread environment. Result: `Ran 5 tests in 0.010s — OK`. The actor reports Ruff, `py_compile`, CLI help, and the same five fixtures passed. No SSH, GPU, network, model training, or live process stop was performed by this reviewer; ROOT owns those native checks. No result-quality or significance claim is made.

