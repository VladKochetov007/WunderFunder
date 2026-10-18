# W18 checkpoint metadata findings

The metadata-only receipt at `TL3_W18_CHECKPOINT_METADATA_R1.json` is SHA-256 `bc99a852f2307de616002d4785e4ced2f2b898d24824fe9d2357c0e457b2992d` (58,142 bytes). Its input manifest (`583a946916a908f4fd0bb4d465bbb48a5e6efc5cfc7079eeb1a7722abec8391f`) and root admission (`b835d4418eaa505be9126e608f3af1c657743da50845bc64b9a1ce60007e0d3c`) agree with the receipt and captured launch. All eight ordered input records match the manifest and admission on logical path, byte count, generation and SHA-256; each before/after file hash agrees, and every reported resolved path equals its logical path with no symlink components. The eight files total 8,433,688 bytes.

The captured systemd command exited zero under `computer-vision-homework-tl3-w18-checkpoint-metadata-v1.service`: CPU quota 500 ms per second, memory max 1,500,000,000 bytes, swap max 0, runtime max 2 minutes, hidden CUDA devices, and the approved scratch directory. Capture reports `PASS_METADATA_ONLY` with no acceptance errors; the unit's terminal record is exited with status zero. The helper's own observation is 0.5 CPU core, 1,499,996,160-byte cgroup memory max, zero swap, 120-second runtime limit, 6.473 seconds elapsed, and CUDA not initialized. Parent memory event counters match admission exactly (`low 0`, `high 177`, `max 561`, `oom 0`, `oom_kill 0`, `oom_group_kill 0`), so the observed delta is zero. Systemd did not expose `MemoryPeak` or `MemoryCurrent`.

All eight records have the same 16-entry FP32 tensor schema, `n_in=211`, and the same top-level keys (`args`, `epoch`, `model`, `n_in`, `readout`, `readout_tilted`). Tensor values were not exported. The configuration comparisons below describe only the saved `args` metadata.

| Checkpoint group | Saved config differences |
|---|---|
| `ffull_s3/s4` ep1 | `train_source=valid`; `init=fmu128_s3/s4@ep3`; `loss=gcorr`; `optimizer=adamw`; no propensity alias; `raw_only=false`. |
| `frawW_s3/s4` ep3 | Empty `init`; `train_source=train+valid`; `raw_only=true`; `loss=wmse`; `optimizer=muon`; `propensity=prop_f0`, power 2.0, epsilon 0.03; EMA teacher 1.0 with EMA evaluation enabled. |
| `frawW_s3ft/s4ft` ep1 | `init=frawW_s3/s4@ep3`; `train_source=valid`; `loss=gcorr`; `optimizer=adamw`; no propensity alias; `raw_only=true`. |
| `fmu128_s3/s4` ep3 | Empty `init`; `train_source=train+valid`; `raw_only=false`; `loss=wmse`; `optimizer=muon`; `propensity=prop_f0`, power 2.0, epsilon 0.03; EMA teacher 1.0 with EMA evaluation enabled. |

Across the eight saved configs, the reported counts are 10,607 training sequences and 600 validation sequences; `full_valid=true`, `include_heldout_valid=true`, `valid_train_fraction=1.0`, and `eval_source=holdout`. `distill` is empty in all records; `distill_boot_h` is zero where recorded. The `full_valid` flag appears across all four groups, so it does not by itself distinguish the `ffull` run names.

These are configuration claims stored in checkpoint metadata. The receipt has no ordered row/sequence/fold-membership hashes or exact membership identity, source-data identity, or training-code revision. It records fold-related config flags, but not the corresponding member identities. The `init` strings do not prove that the named parent checkpoint supplied the loaded weights, and `prop_f0` does not identify a particular propensity checkpoint or array, sidecar, split, or clean-F0 lineage. No source/producer or clean-F0 conclusion is made from these args.
