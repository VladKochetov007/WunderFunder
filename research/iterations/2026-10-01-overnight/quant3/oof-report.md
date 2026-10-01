# Q3 OOF completion receipt

All four frozen producers completed successfully. This is a provenance and resource report; quality has not been evaluated by this actor. Independent audit and the root's two registered comparisons remain pending.

| Fold | Engine | Sequences | Seconds | Peak process RSS KiB | Prediction SHA256 |
|---|---|---:|---:|---:|---|
| 0 | quant | 467 | 608.87 | 637936 | `34175ae04adcd6157353580ed5660f0b2c365399af6997e7d800cb1b2c4348a0` |
| 0 | fp16 | 467 | 1227.76 | 639452 | `c8ffa93199812a5848fb2020e903089019f43307afe290574a1f9d766210c9e8` |
| 1 | quant | 469 | 598.02 | 637996 | `bd94164c694bf67eb93718af4b112183955043b99adcb5a3f0fa0b8250c141ba` |
| 1 | fp16 | 469 | 1220.24 | 639908 | `0165c3c441ef4ff7dbb6e23e795acf77ff7283dfd1afb9986329d9ef54921da8` |

Allocation: `a98412b6903dd027acdcd92b4b1dbdfcb147347ed0d00a26b7c6532409d3cedf`. ZIP: `48cf940bd1c7e812fe8005a90e3f11fb3199a8940c55e0adbfef00ad1de1f10f`. Exact epoch1 children for each fold are `on_ctld256`, `on_ctl128s3`, `on_ctlc500`, in that order, with float32 equal-thirds and original-order FMA reduction. The JSON checkpoint artifacts are explicitly composite recipe manifests, not Torch checkpoints.

Inputs were rehashed before any producer and after all four jobs: raw validation parquet, raw-reader source, chain-fold metadata and serialized row-order metadata. Target/mask arrays were not decoded. All expected sizes, paths, mtimes and SHA256 hashes match. Sources, binaries, ZIP, child checkpoints, exports and four output files passed final hash checks.

Predictions remain remotely in `P/cache/pred_on_q3{quant,fp16}_f{0,1}_valid.npy`; each is float32 `(1873,20000,2)`,299,680,128bytes, with unselected sequences NaN. Local sync contains manifests, exports, metadata and execution receipts, not duplicate full prediction caches.

Resource allocation: one compute CPU per lane (36/37), BLAS/OMP threads1, GPU0. Each process had8GiB RLIMIT_AS. Sampled aggregate lane RSS after watchdog_v2 started peaked at 672,493,568/672,960,512bytes; these sampled group peaks are distinct from GNU time's per-process peaks. The scorer exposed2 OS threads with one idle library thread; affinity limited each lane to one CPU.

The original watchdog and its replacement are preserved. Version2 binds process start identities and uses a monotonic deadline; no deadline/memory kill or identity refusal occurred. Its orphan fallback intentionally refuses to signal never-observed descendants, so this is not an unconditional cgroup kill guarantee. Aggregate RSS samples can miss between-sample peaks. Actual successful durations and process peaks are preserved separately in the receipts.

These OOF execution durations do not replace the earlier complete-callback runtime gate. Only the measured v3/VNNI path has passed the50min runtime threshold. AVX2 failed its first runtime pilot, so no AVX2 OOF audit or quality query was performed. No fresh-holdout or competition-ranking claim is made.
