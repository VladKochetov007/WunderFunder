The bounded I1 diagnostic found no material stream-equivalence violation in the **executed CPU float32 GRU and prediction helpers** on two frozen 256-row prefixes. This is instrument validation, not a WP result or approval of the broader proposal. The deployed callback, native implementation, GPU/FP16 path and full 20,000-row sequences remain untested.

The real checkpoint is `c20261001_d256_s3_f0@ep3`, SHA256 `f92504fbbbcf956a44e70aa3f7a08825d903b9c6a6cacce7bd9f89b6875fcad8`. It has a plain linear encoder, two GRU256 layers, 211 inputs and no alternate readout, online normalization, heuristic inputs or input propensity. The exact snapshot/provenance is in [stream_equivalence_metadata.json](stream_equivalence_metadata.json). The separate baseline provenance sidecar was loaded through the CPU wrapper and its fit membership matched checkpoint provenance; its SHA256 is recorded in [stream_equivalence_audit.json](stream_equivalence_audit.json).

I read the actual `scripts/predict_run.py` pipeline and its `train_rnn.make_model`, `build_inputs`, `forward_chunk` and `predict` helpers, plus `src/wnn/nn.py`, features, ordered feature schema and split validation. Current source hashes are recorded in [stream_equivalence_results.json](stream_equivalence_results.json) and [stream_equivalence_sample.json](stream_equivalence_sample.json). These are hashes of the current executable sources, not a claim that every historical training source version was archived by the checkpoint.

`predict_run.main` itself hardcodes CUDA and writes a full prediction cache, so I did **not** execute that entry point. I executed its real `train_rnn.predict` helper on CPU, without editing its CUDA-autocast clause. PyTorch emitted the expected warning and disabled CUDA autocast. Thus the executed path is CPU float32; GPU FP16 parity is explicitly outside the evidence. No prediction cache, trainer or model source was changed.

The sample was frozen before model execution by selecting the first two sorted fold0 evaluation row groups with distinct feature-graph components. Split validation checked manifest sources, connected-component membership and fold assignment; checkpoint provenance verified that neither selected component occurs in its actual train/valid fit. The split-manifest SHA256 is `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`.

| Valid row group | seq_ix | Graph component | Frozen steps | Required rows |
|---|---:|---:|---|---:|
| 4 | 1990332 | 2799 | 0–255 | 157 |
| 5 | 2308749 | 2800 | 0–255 | 157 |

These are **exposed existing validation** prefixes. Component exclusion is relative to this checkpoint's recorded fit, not evidence of new scientific confirmation. Only `seq_ix`, `step_in_seq`, `need_prediction` and the ordered 112 input features were streamed from parquet in batches of 256. Targets and `is_scored` were not loaded. The 512 raw rows are retained in [stream_equivalence_prefixes.npz](stream_equivalence_prefixes.npz); per-sequence raw-byte hashes are in the sample manifest. The retry verified the same frozen numeric prefix bytes instead of replacing the NPZ. The precomputed 211-feature stream SHA256 is `3766a806df94be77b25e8d9ced6fb1ef40e484034688ebdbe8973d2d56f1c88f`.

I preregistered three partitions: chunk lengths 256, 64 and 127, giving `[256]`, `[64,64,64,64]` and `[127,127,2]`. For precomputed causal streams, `train_rnn.forward_chunk` carried the real recurrent state between segments. The pause occurred after two 64-row segments; it cloned that actual hidden state and replayed the remaining two segments without resetting it. The real `predict` helper was additionally exercised under all three partitions and under reordered single-sequence calls.

| Check | Maximum absolute difference |
|---|---:|
| Repeated identical forward call | 0 |
| All model outputs, partition 64 versus 256 | 0 |
| All model outputs, partition 127 versus 256 | 6.675720e-6 |
| Final recurrent state, partition 127 versus 256 | 7.748604e-7 |
| Real `predict` primary outputs versus precomputed reference, worst partition | 4.768372e-7 |
| Batch sequence reorder, restoring original order | 0 |
| Real `predict` single-sequence call reorder | 0 |
| Actual hidden-state pause/replay | 0 |
| Replaced raw suffix: earlier causal features and predictions | 0 |
| Streaming event features versus batch torch features | 0 |
| Streaming primitive identity, pause/replay and suffix-independence | 0 |

Every returned model output and recurrent state was checked for finiteness; the helper returned finite predictions at all 512 rows, including all 314 required rows. Processing included all warm-up rows. Replacing raw rows 128–255 with the other sequence's suffix left rows 0–127 unchanged in `build_inputs`, full forward predictions and the streaming primitive.

The repeat-calibrated tolerance was frozen as `max(8*identity_error, 64*float32_epsilon*max(1,max_abs_reference_output))`. Identity error was zero, so the numerical floor set an absolute tolerance of **0.0002051933**, using the largest value across all outputs including auxiliary heads. This floor is a declared engineering tolerance, not an empirical bound on all kernels. Observed discrepancies are reported above rather than hidden behind the pass threshold. Feature tolerance was `max(8*streaming_repeat_error,8*float32_epsilon) = 9.536743e-7`; deltas, flags and held-delta columns were also required to match exactly, separately from log1p recency columns.

The separately executed stateful preprocessing primitive was `wnn.features.StreamingEventFeatures.reset/update`. One object carried state across prefix rows and pause boundaries; its output is a reusable buffer, so outputs were copied before comparison. Explicit `reset()` between sequences reproduced batch features exactly. The primitive does **not** accept `seq_ix` or enforce sequence boundaries itself. This validates its state/reset behavior when invoked correctly; it does not verify a submission's automatic reset wiring. The model helper's real internal `state=None` is initialized per batch call, matching the inspected prediction pipeline. No manufactured wrapper is claimed as the deployed submission.

Negative controls confirmed that these instruments can detect missing resets: deliberately carrying sequence A's hidden state into B changed raw model outputs by **14.72979**; calling B's first row on the unreset streaming feature object changed event features by **2.189151**. These were deliberate harness controls, not failures observed in `predict` or evidence of an actual submission bug. Raw output differences are not clipped-metric effects.

Run the diagnostic from the repository root:

```bash
bash research/campaign_2026-10-03/continuation/round2/spec_critic/stream_equivalence_run.sh
```

The exact payload is:

```bash
research/campaign_2026-10-03/continuation/run_cpu.sh round2-critic \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts /usr/bin/time -v \
  .venv/bin/python research/campaign_2026-10-03/continuation/round2/spec_critic/stream_equivalence_check.py
```

Final run [stream_equivalence_run_03.log](stream_equivalence_run_03.log): exit 0, 4.27 seconds elapsed, 3.72 user CPU seconds, 0.50 system CPU seconds, peak RSS **1,085,412 KiB**, no swap. The wrapper enforced one core, 2 GiB and a 900-second scope with CUDA hidden. No GPU process was launched. Checkpoint inspection was separately wrapped but not separately timed. Ruff/check and the metadata-only provenance/hash audit also used `run_cpu.sh round2-critic`; their logs are retained with the same filename prefix.

Two diagnostic-harness failures are preserved: run 01 failed before inference when Arrow required an explicit copy for boolean metadata (3.74 seconds, 1,053,696 KiB); run 02 completed assertions but failed to serialize a NumPy float32 tolerance (4.08 seconds, 1,087,264 KiB). Both were corrected only in this probe. The three measured diagnostic attempts totaled 12.09 seconds wall time; no resource failure occurred.

`PREMORTEM.md`, `TESTS.json`, `checks.py`, `results.json` and original `STATE.json` hashes matched before and after execution and in the final audit. No round2 actor directory/result was read, and no native file, live source, queue, evaluator or service configuration was touched. All new files use the `stream_equivalence_*` prefix in this directory. The null closes only this sampled CPU callable behavior; it establishes no absence of predictive history value, deployed equivalence or WP gain. Awaiting the controller's frozen actor assignment for independent attack; no candidate is approved here.
