# Delivery and runtime scout

The exact recorded V1 fallback is available locally at `/Users/nb/Documents/wunder/submissions/candV1_ft2.zip` and on beleriand at `/home/grinchenkoeg/WunderFunder-cpp/submissions/candV1_ft2/candV1_ft2.zip`. Both SHA256 values are `ea494921497d9f3cf53b19b1b68285133983560a249b38ef40738b3d846d319d`; compressed size is 7,382,512 bytes. Its four equal-weight members are `find256_s3h2`, `fin128_s3h2`, `fin128_s4h2`, `finc500b_s3h2`. The previous public 0.6641 and 41m42s remain journal observations, not a newly fetched leaderboard result.

## Current primary-source requirements

The organizer requires root `solution.py`, no-argument `PredictionModel`, and `predict(data_point)` returning None during warm-up or two finite values ordered t0,t1. Process every row and reset at changed seq_ix. The ZIP limit is20MB; adopt a conservative20,000,000-byte cap for new packaging. Execution is Python3.11 in Linux, CPU1, RAM16GB, GPU unavailable, no network, at most60minutes. [Official submission guide](https://wundernn.io/connectome/docs/submission_guide).

The current hidden dataset has1,970×20,000=39.4million rows; repeated inference must be deterministic. The supplied train and valid data may both be used for training/split construction; hidden scoring-mask inference is inappropriate. [Official FAQ](https://wundernn.io/connectome/docs/faq). Global scoring pools selected rows before computing the two weighted correlations; clipping follows ensemble averaging. [Official data overview](https://wundernn.io/connectome/docs/data_overview).

Final ranking is determined by the private leaderboard, not the current public rank. The organizer prohibits external market data and sharing complete solutions/strategies; the campaign already prohibits GitHub publication and any public submission. [Official rules](https://wundernn.io/connectome/docs/rules). No rule change, external communication or submission was performed by this scout. Sources were retrieved as current indexed official documentation, not by auditing the remote scoring service.

## Runtime environment and source recovery

V1 contains CPython3.11-specific `wnn_cpp` binaries for v2, avx2 and v3. The ordinary creative checkout's Rust `wnn_native` packager is not the V1 engine. Recovered `native_cpp/` plus `submission/solution_cpp.py` from existing local `cpp-engine` commit97bd27f, preserving file hashes in `recovery/source-receipt.json`. The server cpp checkout e96efe6 adds a separate class-scaled wrapper; the ordinary engine remains the recovered base. No GitHub fetch occurred and original checkouts remain untouched.

Beleriand has `~/venv311/bin/python` with pybind11 and `g++`. `~/scorer_image/run.sh` runs an already-installed image userland using its dynamic loader: Python3.11.16, glibc2.36, numpy2.2.6, pyarrow19.0.1. This works despite Docker socket permission denial. It is not complete container isolation: CPU affinity/thread limits and fresh extraction paths must be imposed separately. New builds for all baseline CPU levels succeeded and require glibc<=2.36.

The recovered packager assumes repository-root relative paths; do not invoke its copied nested location unchanged. Either integrate its source into the creative root or make an explicit isolated packager with `--repo-root`. Its export slices target head rows and must also preserve `head_extra.weight` and a separate JSON feature configuration for feature models.

## GPU/host safety

At first inspection all8×RTX A4000 GPUs carried unrelated jobs; GPU1/3 had only1.1GiB free. Later an unrelated two-GPU probe exited and GPUs6/7 became free (~15.6GiB each). Root was notified for fresh allocation checks. No job was stopped, signalled or modified. Approximately718GiB RAM and84TiB disk were available. Root's two-job/5.5GiB-per-job/24CPU-thread contract remains the campaign cap, not all apparent host capacity.

## Determinism caveat of the historical fallback

V1's runtime governor reads wall time and drops the last ensemble members when projected runtime exceeds52minutes. It approximates heterogeneous members as equally expensive. Executing the exact `_govern` method on synthetic timing trajectories at the same row boundary retains4members at60us/row but2 at120us/row. See `governor-counterexample.json`. Consequently the policy can yield different predictions on identical inputs when host speed/load differs; deterministic math alone does not satisfy full callback repeatability. The historical ZIP remains unchanged and previously scored successfully, but its timing policy is not mathematically deterministic.

Prefer an explicitly fixed ensemble in a new package if measured relative and absolute cost justify it. A static CPU-level fallback is deterministic per environment; a dynamic timer-driven selector is not. The old41m42s result leaves18m18s nominal headroom, but historical run-to-run variability and different server CPUs preclude claiming that all headroom is available. A package that fits local conditions only by dropping members must be evaluated as that actual blend, not as the ideal four-member prediction cache.

## New native feature path status

Derivative `native_transition/` adds only the 68-channel state machine, separate double-precision nuisance configuration, and a float32 extra head projection before GELU. `recovery/` remains unchanged. Source design, build provenance and independent synthetic checks are retained separately. On20,003 synthetic rows with15,579 emitted marks, native features, independent Counter-based oracle and actor batch features matched bit-for-bit for F and T. Nonzero extra weights yielded max prediction deviations of5.96e-8 float32 and1.71e-5 with existing fp16 dense weights versus the actor's PyTorch model. Reset, prefix and stream equivalence passed, including a hand-constructed single-channel nonzero head.

Three short one-core synthetic two-h128 repeats measured median49.60→54.44us/row, +4.84us or9.74% for that pair. This is a feasibility proxy; a complete trained V1-like ensemble, real rows, all relevant CPU levels and fresh ZIP loading still need final verification. No empirical target data or competition metric was used in these checks. An independent actor review of the C++ derivative is requested before deployment.

## Concrete delivery gates

1. Exact frozen epoch/checkpoint identities; intended member order/weights; legal nuisance fit provenance. Check all ZIP names for traversal/absolute paths, root solution, required per-member weights/config and conservative compressed size.
2. Fresh extracted ZIP outside repository with no repository PYTHONPATH; CPython3.11 userland; no network; no-argument construction. Warm-up None, every-row state update, shape2 finite output, reset equivalence, repeated runs identical under fixed policy.
3. Feature engine scalar/batch/native parity including threshold-adjacent values and nonzero extra projection; checkpoint-to-export tensor hashes/shapes. PyTorch/native check actual trained weights and full20k streams, not only random models.
4. One pinned CPU and numerical library threads1; compile/load overhead separate, full callback timing and peak RSS, density/warmup/reset effects. Report repetitions and ratio to unchanged V1 on the same host rather than multiplying a single historical CPU-speed factor into a guarantee.
5. If the new branch fails utility, correctness or serving gate, retain the exact V1 ZIP plus documented limitations. If a no-added-state training intervention passes, its package uses the same established native operators and still requires weight parity/runtime verification. No score claims for an unsubmitted artifact and no top-rank guarantee.


## Completed full-callback measurements

All conditions use fixed members, Python 3.11 scorer image userland, numerical library threads 1, physical CPU 18, and five predetermined complete raw validation sequences (row groups 0, 468, 936, 1404, 1872). No targets or masks were read. Each repetition processes 100,000 rows, including warm-up callbacks. The version 2 harness records process CPU time separately; the original harness is preserved. Baseline weights are the exact archived V1 weights. The feature condition adds deterministic random nonzero extra head weights solely to measure its serving path; it is not a forecast-quality candidate.

| Fixed serving composition | Repetitions | Median µs/row | Projected minutes, 39.4M rows | Observed µs/row range |
|---|---:|---:|---:|---:|
| V1: d256 + 3 h128, archived binaries | 6 | 170.81 | 112.16 | 161.93–195.65 |
| Same V1, recovered baseline binaries | 3 | 167.45 | 109.96 | 166.93–168.28 |
| V1 with extra features in two h128 heads | 6 | 177.64 | 116.65 | 168.07–188.67 |
| Plain seed3 + seed4 h128 | 3 | 53.90 | 35.40 | 53.50–54.56 |
| Plain seed3 + seed4 + chunk500 h128 | 3 | 84.09 | 55.22 | 82.35–85.05 |

Every repeated prediction hash is identical within its condition. Recovered and archived baseline binaries produce bit-identical forecasts across all 100,000 rows. Process CPU/wall ratios are approximately 0.99986, so the slow fixed4 result is computational work rather than time waiting for scheduler allocation. The host is Xeon Gold 6230 with 1 MiB L2 per core, powersave governor, and observed 3.4 GHz during inspection. Process peak RSS remained below 100 MiB in these inference probes.

The first interleaved fixed4 triplets show feature overhead of about 2.61 µs/row versus the recovered baseline; subsequent temporal variation is larger than that effect, so a single precise overhead percentage should not be treated as stable. These runs do not measure actual hidden scoring hardware. V1's recorded 41m42s could reflect different hardware, pruning, or both; no raw public governor/member-count log was found in the known local/server archive directories. The fixed4 model fails a conservative same-host 60-minute projection. Fixed3 meets the hard projection with a narrow margin; fixed2 has substantial margin. Their competition utility has not been scored or selected by this scout.

Detailed receipts and preserved input hashes are in `runtime-summary.json`, `runtime-results/`, and remote `critic/runtime-probe/`. The remote directory also retains raw fixture arrays and temporary extracted packages for reproducibility; these bulk inputs need not be copied into the source commit. `solution_fixed.py` and `package_fixed.py` provide a deterministic four-member packaging skeleton. Any smaller composition requires root's separately frozen comparison/selection rule and a corresponding packaging amendment.
