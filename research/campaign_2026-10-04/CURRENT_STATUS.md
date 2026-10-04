# Continuation status

As of 2026-10-04T19:24:08.971434+00:00. Best recorded public score **0.6655** (37m47s,625MB); target0.6660 remains open. The user authorized continuing until explicitly stopped. Competition uploads remain manual.

The exact public-confirmed seed-3 ZIP scores **0.700503766** over all1,873 sequences in `datasets/valid.parquet`. Previous D256 cached predictions score0.700792587. Most supplied validation was fitted. On the368 excluded sequences, seed3 scores0.699822159 versus previousD2560.697777394. Recipe four-fold complete-predictor CV is0.693658483 versus matchedcontrol0.691244104. See `VALIDATION_SCORECARD.md` for masks, metric, training overlap and precision.

**E450 seed4 is fitted and scored.** Its fixed complete predictor scores0.700771292 over suppliedVALID and0.701694557 on the excluded portion; the fixed equal seedbag scores0.701106873 /0.701420130. Excluded-panel95% paired intervals includezero. Actual new seed4 ZIP passed nativeprecision/reset/causality checks across20complete streams,400,000rows. Quiet CPU timing remains pending, so it is not declared ready for manual upload. `E450_SEED4_PAIRED_REPORT.md`.

**E426 clock comparison:3/3 full fits completed.** Constant arm completed4epochs with fixedEP3,29,840 optimizerattempts,29,828 completed updates and12AMPskips. Peak observed VRAM4.489GB/RAM20.419GB, no new OOM. Aligned and shuffled arms also completed successfully, each with the same update counters. Shuffled took2,574s, peakRAM18.489GB/VRAM4.489GB and no OOM. No scientific score yet. Exactcontroller `wnn-e426-sequence-v1.service`.

**E425 soft routing is durably queued.** Eight CPU checks and controller source/domain checks passed. The queue waits for E426's successful teardown, then a bounded quiet CPU timing window. The timing window can defer after15minutes if the host does not become quiet; an actual worker failure stops the dependency. Only one local GPU trainer is allowed. No E425 scientific fit started.

**E464 architecture completion passed8scope tests and8actual-parent preflights.** The boundedCPU worker finished14.90s,341MB peakRAM, no GPU/OOM. Eleven bounded CPU queue tests now pass; an earlier off-by-one test-root import failure is preserved. All eight fixed one-epoch head specs are now root-sealed; `wnn-e464-sequence-v1.service` has a pending start job after E425. Its cap is three actual GPU-job hours, checked with each next full-timeout reservation. Corrected pre-fit counts are94 batches and940optimizerattempts; exact16comparison and FP32 cache guards are recorded. `staging/e464/E464_SOURCE_CRITIQUE_2026-10-04.md`.

**E449 W384 Kagglev3 failed before scientific training.** Its full production probes timed out with0reported trainingsteps/0GPU allocation. Logs show repeated large integrity reads before startup:42.19GB declared inputs perprobe and a3,440s pre-worker gap. This supports an I/O/startup diagnosis; no OOM or CPU-contention measurement establishes another cause. Raw failures are preserved; no blind GPU restart. The v2CPU diagnostic failed at worker startup because of a logger argument collision, before any I/O measurement. The corrected private2CPU/4GiB diagnosticv3 is RUNNING as of19:20:41Z; no GPU allocation or scientific fit. The full AdamW width replication and separate combo-width factorial remain planned.

**E463 TPU real-data preflight remains QUEUED**, exactprivatekernel `vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1-runner`. No real-data TPU update/HBM telemetry yet. Its priorCPU FP32/BF16 one-update check passed. E407 established8v5lite devices and16,909,332,480B reportedmemory/chip; hardware probes are not predictive evidence.

**E465 two-device JAX CPU tests passed4/4 under a fresh v3 identity.** Unequal positive-mass reductions and state carry passed. The earlier all-zero fixture failure is preserved; v3 checks correctly include zero gradients and a finite zero-decay optimizer step. PeakRSS460MB under4GiB, no resourcefailure. The exact scheduler port also passed all7,461 productionTorch reference indices with zero LR/beta error; its earlier literal-float fixture failure is preserved. All-eight-TPU scientific fits are preregistered but not admitted.

**E462 native INT8 parity and paired accuracy completed.** q8r delta-0.000005052 andq8x-0.000017505 versus originali16; both95%intervals includezero. Quiet CPU timings are queued with seed4 before deployment decisions. `E462_NATIVE_PAIRED_SCORE.md`.

E424 all8parentfits/exports/commonFP32pairedscore completed; all architecture-control intervals includezero. E419 recoveredQAT scientific screen completed without demonstratedgain. E430 cadence code is staged but not launched. E427 causal volume-gate implementation passed8CPU tests, including its production-sized shared context. The input-only fit scan also completed all242,220,000rows with no OOM. Actual full recurrent GPU smoke remains required. Previous negative studies supply controls rather than an automatic veto.

Program tally E407–427: **17 completed,1partial,1running,1queued,1prepared**. These include EDA/framework programs and do not count individual model fits. E424 fits8/8 andE450 fits2/2 completed; E464 adds8planned heads to the roughly90-fit review portfolio. Hypothesis catalog now has46cards.

Resource limits persist: local36GiB/noSwap/12CPU, oneGPUtrainer,24GiB worker,4.2GBallocator/4.8GBobservedstop. PrivateT4 workers retain11GiBallocator/12GiBoperating/13GiBstop. Kaggle typed18:02:27Z quota snapshot: about22.31GPUproviderhours and19.96TPUhours remaining; serializedtimedelta dropsGPUdayportion. Providerquota is separate from per-devicejob accounting. Cumulative80deviceGPUh and local24GPU-jobh block remain bounded.

Marimo is live at `http://localhost:2718`; figures and their sourcehashes are under `eda/`. Significant small source/reports/logs/provenance/figures are saved to private main. Model weights, prediction arrays and submission ZIP binaries remain local/privateprovider artifacts identified by hashes. Unrelated historical pausemarkers and the separate `wunder/` tree are preserved.

## Timing-window incident and recovery

The actual E462 timing-v1 worker stopped at19:20:49Z because host load exceeded its predeclared quiet gate. No valid timing result, no OOM, and teardown verified. E425 did not launch; E464 remains waiting for E425. The failed attempt, log, quarantine and exact controller remain preserved. A new reviewed timing identity with stronger startup quiet admission is being prepared; worker precision, row counts, paired repeats and the load gate remain fixed.
