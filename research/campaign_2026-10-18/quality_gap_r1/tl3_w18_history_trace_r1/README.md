# TL3 w18 producer-history trace R1

**Status:** archived producer procedures and run-history signals are traced. The target-array bytes, checkpoint hashes, and row/preprocessing identities remain unverified.

This packet follows the archived `w18final_teacher`, `w18final_fbc1k_s5`, `w18final_family_full`, and `w18final_family_raw` producer paths. It combines the archived Git snapshot `research/creative-iteration-20260930` at commit `833aaf541155c591e930bbc6c640b8817e01a536` with one bounded, read-only probe of the exact source, run-metadata, and log paths listed in [PROBE_SCRIPT.py](PROBE_SCRIPT.py). The probe read 51,685 bytes of text under a 450,000-byte total cap and a 60,000-byte per-file cap. It did not read or hash checkpoint payloads, prediction or propensity arrays, or start or alter jobs. The raw output, capture limits, and observed metadata are preserved in [RAW_PROBE_STDOUT.json](RAW_PROBE_STDOUT.json), [RAW_PROBE_RECEIPT_R1.json](RAW_PROBE_RECEIPT_R1.json), and [REMOTE_RUN_EVIDENCE_R1.json](REMOTE_RUN_EVIDENCE_R1.json).

## Producer graph recorded by the archived source

`full_teacher_lane.sh` routes the direct aliases to these checkpoint endpoints and asks the archived `predict_run.py` to write both `train` and `valid` arrays:

| Alias | Recorded producer input | Prediction epoch |
|---|---|---:|
| `w18final_fbc1k_s5` | run `fbc1k_s5` | 3 |
| `w18final_ffull_s3` | run `ffull_s3` | 1 |
| `w18final_ffull_s4` | run `ffull_s4` | 1 |
| `w18final_frawW_s3ft` | run `frawW_s3ft` | 1 |
| `w18final_frawW_s4ft` | run `frawW_s4ft` | 1 |

The archived `prepare_full_teachers.py` then defines three positional, row-wise compositions:

- `w18final_family_full = 0.5 * w18final_ffull_s3 + 0.5 * w18final_ffull_s4`.
- `w18final_family_raw = 0.5 * w18final_frawW_s3ft + 0.5 * w18final_frawW_s4ft`.
- `w18final_teacher = 0.3 * w18final_fbc1k_s5 + 0.1 * w18final_ffull_s3 + 0.1 * w18final_ffull_s4 + 0.125 * w18final_frawW_s3ft + 0.125 * w18final_frawW_s4ft + 0.0625 * orgbase`.

The code applies those coefficients as a direct weighted sum. It has **no normalization denominator**; the total coefficient for `w18final_teacher` is `0.8125`. Each family pair sums to `1.0`. `orgbase` is included at `0.0625`, but this trace does not establish its producer or identity. The helper checks finite output rows and writes through a `.partial` file before rename. It takes shape from the first input and combines by row index; it does not validate row IDs or prove that the input arrays share the same row order.

The companion [full-teacher-manifest.json](source_snapshot/research/iterations/2026-10-05-wave18/full-teacher-manifest.json) labels these as final all-data caches and explicitly warns against reporting them as holdout metrics. The archived `final_student_lane.sh` shows intended downstream use, but this trace makes no claim about the quality or validity of those student results.

## What the source and run records establish

The remote source hashes for `full_teacher_lane.sh`, `prepare_full_teachers.py`, `full-teacher-manifest.json`, the three teacher-lane logs, the empty preparation log, `predict_run.py`, `final_rawW.sh`, and `ftpairs_wait.sh` match the copies from the pinned archived Git commit. Their full Git blob IDs, bytes, and SHA-256 values are listed in [GIT_SOURCE_BINDINGS_R1.json](GIT_SOURCE_BINDINGS_R1.json); exact content is copied under [source_snapshot](source_snapshot/).

The three remote lane logs contain only completion markers: `strong` at `2026-10-05T14:36:19+00:00`, `raw` at `2026-10-05T14:37:27+00:00`, and `full` at `2026-10-05T14:37:33+00:00`. The archived lane script uses `set -e` and writes its `DONE` marker only after the prediction loop, so the markers are consistent with successful completion if those logs came from the recorded script. No attempt receipt binds each log to a particular invocation, and the markers contain no array hashes, row-order digests, or per-file receipts. `prepare-full-teachers.log` is a zero-byte file in the probe; the script source specifies the composition step, but this trace does not independently verify the resulting composite arrays.

The probe found the seven expected run directories and `history.json` files. It reported only endpoint file sizes, not checkpoint hashes: `fbc1k_s5/model_ep3.pt` 3,934,495 bytes; `ffull_s3` and `ffull_s4` `model_ep1.pt` 1,053,479 bytes each; `frawW_s3` and `frawW_s4` `model_ep3.pt` 1,054,943 bytes each; and `frawW_s3ft` and `frawW_s4ft` `model_ep1.pt` 1,053,607 bytes each. The exact paths and history hashes are in [REMOTE_RUN_EVIDENCE_R1.json](REMOTE_RUN_EVIDENCE_R1.json). These size checks do not identify checkpoint contents.

The captured `ffull_s3` and `ffull_s4` training logs identify the parent runs `fmu128_s3@ep3` and `fmu128_s4@ep3`; their run configuration records 128 hidden units, two layers, two fine-tuning epochs, learning rate `1e-4`, `train_source=valid`, `include_heldout_valid=true`, and `full_valid=true`. The captured `frawW_s3` and `frawW_s4` logs record full-data raw-only WMSE training, with `train_source=train+valid`, `include_heldout_valid=true`, `full_valid=true`, `propensity=prop_f0`, and Muon multiplier 6. The archived `final_rawW.sh` defines the follow-on `frawW_s3ft`/`frawW_s4ft` recipe: initialize from the corresponding base model at epoch 3, train on `valid` with held-out valid included and full-valid enabled, then use epoch 1 as the teacher prediction source. These facts describe logged configuration and an archived wrapper; the probe does not bind the wrapper's source hash to a particular training execution.

## Remaining identity gaps

The exact checked `artifacts/logs/fbc1k_s5.log` path was not reported as a regular file, and the probe did not find `args.json`, `config.json`, `resolved_config.json`, `run.json`, `manifest.json`, or `provenance.json` as regular files in the `fbc1k_s5` run directory. The archived slow-fast README says that `fbc1k_s5/s6` were full-data “Vlad-recipe” jobs and describes the nearby prototype as boot-8, chunk-1000, Muon x3. That is a summary-level history note, not an exact `fbc1k_s5` command, source hash, or run-bound recipe receipt. Do not promote it into a claim that the precise prototype command was applied to `fbc1k_s5`.

The bounded probe did not inspect any target prediction file or sidecar. The archived `predict_run.py` loads checkpoint arguments, obtains group IDs from `cache/<split>_targets.npz["row_group"]`, constructs the dataset from that ordered list, and writes an FP32 `(number_of_groups, 20_000, 2)` memmap. This defines the intended row order in the archived code but does not establish the actual order, source rows, preprocessing flags, or bytes of any produced array. The predictor itself writes no sidecar, and the composition helper does not add one.

The archived `train_rnn.py` is copied and hashed from the Git snapshot, but it exceeded the probe's per-file text cap, so no remote content hash was captured for that file. No per-run source digest connects that archived trainer to the historical `ffull`, `frawW`, or `fbc1k_s5` executions. The probe also did not hash any endpoint checkpoint or inspect checkpoint metadata.

The outstanding target-array bindings are the bytes and ordered rows for each direct alias and composite, the selected checkpoint contents, the prediction-time checkpoint arguments and preprocessing flags, the exact `train`/`valid` feature and target source identities, and cross-component row alignment. The predictor source writes no target-cache sidecars, so this trace has no producer-supplied row or preprocessing receipt. Preserve `orgbase` as an unresolved component.

## Concrete next metadata and source allowlist

If a later bounded closure is authorized, these exact paths follow from the archived launch records. The current probe has already queried the fbc1k/frawW/full run metadata and the named logs; the list below identifies what to retrieve next, not a request to rerun those checks.

First bind the missing fbc1k and fine-tune records under `/home/grinchenkoeg/WunderFunder-creative`:

- `artifacts/runs/fbc1k_s5/{args.json,config.json,resolved_config.json,run.json,manifest.json,provenance.json}` and `artifacts/logs/fbc1k_s5.log`, plus a SHA-256 and non-deserializing metadata record for `artifacts/runs/fbc1k_s5/model_ep3.pt`.
- `artifacts/runs/frawW_s3ft/` and `artifacts/runs/frawW_s4ft/` equivalents for those six metadata files, their `artifacts/logs/frawW_s3ft.log` and `artifacts/logs/frawW_s4ft.log`, and a content hash for each `model_ep1.pt`. The logged full-data run starts from `frawW_s3@ep3` / `frawW_s4@ep3`; include content hashes for those two parent endpoints.
- `artifacts/runs/fmu128_s3/` and `artifacts/runs/fmu128_s4/` equivalents for those six metadata files, their run logs, and content hashes for `model_ep3.pt`, because the captured `ffull_s3/s4` launch records name these parents. The already captured `ffull_s3/s4` logs and metadata should be retained by hash, not silently replaced.

The exact historical trainer file is `/home/grinchenkoeg/WunderFunder-creative/research/iterations/2026-10-05-wave6/scripts/train_rnn.py` (83,943 bytes in the probe). The archived Git copy is pinned, but the remote probe stopped before hashing its content because it exceeded the per-file text cap. A later source-only digest comparison can resolve that identity without reading checkpoint payloads. Any job/run source-manifest or executor record should also bind the trainer hash to each named historical run; current logs do not.

For array and membership closure, the exact direct and composed output paths are `cache/pred_{w18final_fbc1k_s5,w18final_ffull_s3,w18final_ffull_s4,w18final_frawW_s3ft,w18final_frawW_s4ft,w18final_family_full,w18final_family_raw,w18final_teacher}_{train,valid}.npy`. Bind bytes, dtype, shape and SHA-256 for all 16 arrays. Derive ordered row-group digests from `cache/train_targets.npz` and `cache/valid_targets.npz`, and bind the corresponding `datasets/train.parquet` and `datasets/valid.parquet` identities. Confirm that all component inputs to each weighted sum use identical ordered row groups. Recover the prediction-time checkpoint arguments that set preprocessing in `predict_run.py`, including raw-only, normalization, sentinel, quantization and state-reset flags, from run metadata or an independently source-bound checkpoint metadata receipt. Also resolve the provenance and exact row identity of `cache/pred_orgbase_{train,valid}.npy` before claiming closure of `w18final_teacher`.

## Packet inventory

[PREPARATION_MANIFEST_R1.json](PREPARATION_MANIFEST_R1.json) seals this README, both probe artifacts, the raw-probe and remote-run evidence bindings, and all exact Git source snapshots. Its own digest is reported by the parent review handoff because a manifest cannot contain its own hash.
