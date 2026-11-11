# L4 GRU successors after the causal-stem screen

## 1. Status, lineage and owner

ROOT authorized and launched six fresh fits on 11 November. They descend from PR79. The actual immutable training source is `6b10dfe71fb0ebec9b0fc76788e2de0fd13d1dba`, source SHA `38d6dd71ec29fb8d23b9df801a448f23fb8b78c1e0832883dda1d6f25095643d`, capsule manifest `e2220d547dc92f791509e48762f164f70a3c064d01a4136156f8cc541dfb43d0`. The later loader-helper extraction in this branch keeps the repository structure check green. Running jobs use the recorded immutable source capsule. Native controller is 3756565, birth 2175927370.

## 2. Mechanisms

K64 gives the causal per-channel input FIR access to the current row and up to63 earlier rows. Width512 increases recurrent capacity. Interlayer GRU dropout0.1 tests regularization after late validation plateaus. Each arm changes one normalized model field against its same-seed K16 control. These are complete-recipe quality comparisons with different parameter counts and compute costs.

## 3. Compute advantage and cost

Six new one-L4 fits, two seeds per arm, fixed6 epochs each. Two completed K16 controls are reused. The panel adds36 model-epochs and232,560 optimizer updates. Six30-update profiles add180 engineering updates and full-state reload checks. The widest model has3,737,346 parameters. Actual profile peak reserved memory was2.381GiB/card, host RSS at most3.003GiB/rank.

## 4. Frozen recipes

GRU with two recurrent layers, raw112, causal K16 unless the K64 arm changes it. Batch32, chunk1000, accumulation1, strict FP32, AdamW3e-4, weight decay0.01, clip1, warmup5%, warmup-cosine-v2 floor0.1. Each fit has6,460 updates/epoch and38,760 total. Keep the configured final epoch6 as the primary endpoint. All epoch checkpoints and full-valid curves are retained.

## 5. Panel and placement

| Arm | Seed203301 GPU | Seed203302 GPU | Changed field |
|---|---:|---:|---|
| K64 | 0 | 3 | input_stem_kernel16 to64 |
| GRU512 K16 | 1 | 5 | hidden256 to512 |
| Dropout0.1 K16 | 2 | 6 | dropout0 to0.1 |

GPU4's existing processes retain their allocation. GPU7 remains available for bounded engineering measurements. Physical UUIDs and raw/normalized config hashes are bound by the ROOT admission.

## 6. Validation and uncertainty

Use the official earlier training file, fitting10,325 whole sequences, and complete independent valid with1,873 sequences and3,528,273 scored rows. Primary metric is global_wp_clip2_v1, pooled clipped weighted Pearson separately for t0/t1 and their mean. The validation set has repeated development exposure. Report each paired seed/target delta and its descriptive mean. Two seeds support an exploratory allocation decision.

## 7. Leakage and ancestry

Split SHA0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c is preserved. Fit-only normalization, fresh model/optimizer/RNG state, causal input history and sequence resets are checked. Source and profile receipts bind the full sparse fitting roster and independent valid roster. This follows ADR0044 and the official file boundary. Sequence identifiers supply no additional chronological ordering.

## 8. Decision rule

Mean fixed-E6 gain at least0.0005 WP with both paired seed deltas positive warrants targeted follow-up. Preserve small, mixed and negative outcomes. The current K16 control scores are0.683019498 and0.682629372. These scores compare fixed predictors on exposed development data. Public transfer is an external measurement after a supported complete submission is frozen.

## 9. Execution and timer

The shared first-GPU event is2026-11-11T01:10:28.527209UTC. Latest user instruction caps the entire wave at five hours. Fitting stops by05:10:28UTC /08:10MSK. Evaluation, collection and all owned GPU work stop by06:10:28UTC /09:10MSK. L4 profiles record their own dispatch timestamp separately. Jobs are detached and independent of the SSH transport. Owned resource guards cap RSS at128GiB, output at3.4GB and retain2.5GB free disk. Existing datasets and caches are reused.

## 10. Verification and artifacts

Seventeen focused causal/state/dropout tests passed locally and on the actual receiving runtime. K16 default model initialization, tensors, full/chunked outputs and streaming history match base336 bitwise on CPU. All six actual L4 profiles passed, including complete model/optimizer/scheduler/RNG reload and temporary-state removal. The ROOT review checks exact source/config/member rosters, normalization, runtime UUID, precision, finite losses and memory. Evidence archives and manifests are included here. Subsequent loader-helper extraction also passed17 tests, Ruff and the130-module structure check.

## 11. Results and reporting

All six fits have started. Fixed-E6 successor scores are pending. Live graphs are served locally at http://127.0.0.1:8770/. Native outputs are under /data/lc-epanov_s/tabdata-stem-successors-20261111-r1. The progress publisher records actual optimizer updates, train/full-valid loss and WP, LR, gradients/clipping and resource observations. Canonical prediction/scoring follows the completed fits and compares each same-seed output with the recorded K16 control.


## Receiving continuation

The six-fit controller was launched at01:55:57UTC. Both monitoring SSH streams disconnected at02:10UTC when the local VPN process exited. One bounded reconnect per host timed out during banner exchange. Subsequent current GPU state remains unverified pending a fresh VPN code. The detached fit timer retains its original05:10:28UTC cutoff.

The receiving helper and focused tests are under `readout/`. Transfers089/090 and launch091 are queued on the existing L4 RPC transport. Native readout launch is still pending transport restoration. The helper checks the fixed-E6 endpoint, source, sparse membership, prediction coverage and canonical scorer agreement before paired comparisons against the exact reused controls. Collection retains the original06:10:28UTC deadline.

The progress publisher is running locally as PID10068 and has posted its first report to PR80. Reports preserve observation timestamps and mirror age. Metadata inputs-r2 fixes the source identity in the manifest while preserving every R1 training/control config byte.
