# CPU student architecture cost screen R1

## Result

The complete planned shape ladder ran through the production native `EnsemblePredictor.predict(DataPoint)` callback with both FP32 and dynamic Q8 VNNI. On one Intel i9-11900H host under a one-core quota, Q8's mean of the first and post-reset 20k-row callback replays was 4.7–7.7× faster than FP32 for these synthetic weights. Native weight-stream storage was 0.63–4.32 MB for Q8 versus 2.24–16.47 MB for FP32. These are runtime and storage proxies, not quality or architecture-selection evidence.

| Shape | Synthetic trainable params | FP32 two-replay mean (s) | Q8 two-replay mean (s) | FP32/Q8 | Q8 constructor quantize/pack/init (ms) | FP32/Q8 native streamed weight bytes |
|---|---:|---:|---:|---:|---:|---:|
| H256×1 | 580,866 | 1.569 | 0.205 | 7.67× | 6.7 | 2,323,520 / 659,904 |
| H192×2 | 559,682 | 1.687 | 0.225 | 7.51× | 7.4 | 2,238,784 / 631,872 |
| H256×2 | 975,618 | 2.502 | 0.360 | 6.95× | 11.3 | 3,902,528 / 1,071,552 |
| H512×1 | 2,210,306 | 5.300 | 1.119 | 4.74× | 25.5 | 8,841,280 / 2,367,424 |
| H384×2 | 2,151,554 | 5.217 | 1.093 | 4.77× | 24.8 | 8,606,272 / 2,294,976 |
| H640×1 | 3,418,242 | 9.412 | 1.682 | 5.60× | 44.0 | 13,673,024 / 3,614,400 |
| H512×2 | 3,786,242 | 12.242 | 2.046 | 5.98× | 53.5 | 15,145,024 / 3,977,152 |
| H448×3 | 4,118,466 | 13.715 | 1.815 | 7.56× | 49.2 | 16,473,920 / 4,315,200 |

Each callback replay visits all 20,000 rows, including 99 warm-up rows, and invokes the same causal raw-row feature-preprocessing and recurrent path as serving. The second replay follows an explicit native reset. All 16 complete prediction arrays matched their post-reset replays bitwise. FP32 and Q8 use the same deterministic generated tensors within each shape, but their prediction arrays were not compared as a quality endpoint. The native constructor timing includes matrix quantization/packing and model setup; generation of proxy weights and loading of the fixed feature fixture are outside that interval.

## Recipe and provenance

One synthetic model was generated for each fixed shape using seed 20261014, NumPy PCG64, independent float32 uniform matrices bounded by `±1/sqrt(fan_in)`, and zero biases. The exact 211-wide mean/std tensors from original xteam_h4 member 0 were reused. The architecture stayed fixed at a 211-wide encoder with GELU, `L` stacked H-wide GRU layers, and a `[last_state, encoder]→H` GELU→2 skip readout. Each profile used one member with coefficient 1.0, fast native activations, and no time-budget fallback.

The input was row group 0 (sequence 256678), a complete 20k raw-feature array with SHA-256 `bfec7e22df30437429f794156f75ce6f6843c882149ec6e6e7d0c57ead588f3e`. Its export manifest binds it to the original validation-file SHA `2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782`; this screen rehashed the feature fixture, manifest and source archive rather than rereading the 5.6 GB parquet. It did not load targets or scoring masks. The need-prediction mask was reconstructed from the fixed 99-row warm-up contract and matched its manifest hash.

The worker source is commit `febdb238a64f56f2d969bdf59855537527d80f2c`, with profile script SHA-256 `d87d3944ab5db7e440c7acc80aea7f4aa00deeff30a3be2ec508a002aca96614`. The [verification receipt](runs_r1/RESULTS_VERIFICATION_R1.json) checks that hash against the Git object, that the callback `utils.py` equals the integrated base-commit version, that the built native library SHA is `cd8b9692d102e5173fbe3b89a15373dcccb63382892f7621c0b2a1382f0ba362`, and that all worker result and command files match the measured run.

The workers ran sequentially with `CPUQuota=100%`, `WNN_THREADS=1`, `MemoryMax=1,500,000,000`, and swap disabled. The cgroup reported a 1,499,996,160-byte effective memory maximum and a 109,723,648-byte peak; the maximum worker process RSS was 122,732 KiB. OOM, memory-max and swap events were zero. The CPU affinity mask allowed all 16 host CPUs, so the one-core quota constrained aggregate CPU time but did not pin workers to one logical CPU.

The reported streamed-weight storage comes from the native model's total `weight_bytes`/`storage_bytes` field, which includes dynamic Q8 matrix codes and scales along with other streamed tensors. The binding's `q8_weight_bytes` getter only counts static `PackedQ8` matrices and does not expose the dynamic `DynQ8` subtotal; it is therefore recorded as unavailable rather than misreported as zero. Constructor time and total native storage still account for the Q8 packing. ZIP size is not inferred from synthetic weights.

## Evidence limits and delivery note

The [machine-readable summary](runs_r1/LADDER_RESULTS_R1.json), all 16 worker records and individual command receipts preserve callback/reset timings, selected precision, hashes, model bytes and resource observations. The full controller exited nonzero after writing the summary because its final status-print call passed `flush=True` to `json.dumps`. The exact command and traceback are preserved in [the controller failure receipt](runs_r1/CONTROLLER_FAILURE_AFTER_RESULTS_R1.json). A separate verification-only run passed all worker, provenance, bit-parity and cgroup checks; it did not rerun or retime the models.

The two earlier preflight failures are also preserved: an initial static-versus-dynamic Q8 storage-getter assumption and a manually mistyped source-commit argument. Neither wrote a full callback result; the first reached Q8 model construction, while the second stopped before native import. Neither is a native kernel, model-shape or memory failure.

This is a random-weight local CPU cost proxy. It cannot rank quality, establish training capacity, predict a public score, compare W4, certify trained ZIP size, or claim competition-host/AMD runtime. W4 was omitted because the selected sealed native binary has no straightforward W4 inference policy. No fitting or external upload occurred. Related hypotheses: H-STUDENT-LADDER-20261014 and H-CPU-POLICY-20261014.
