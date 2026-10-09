# H232 JSON path serialization repair V3 (source-only)

The V2 Kaggle attempt failed before a timed profile because `json.dumps(vars(args))` received the parsed `PosixPath` for `--profile-trace-output`. V3 changes only `scripts/train_rnn.py` relative to V2. The explicit serializer stringifies `Path` values for stdout, provenance, resume contracts and checkpoint args while leaving all other objects subject to strict JSON serialization.

The CPU smoke invokes the actual trainer CLI and parses `--profile-trace-output` and `--split-manifest` as `Path`, then serializes them before device/data setup with CUDA hidden. It passed under the one-CPU/1.5-GB campaign wrapper; no training or provider call occurred. See [H232_JSON_PATH_SERIALIZATION_REPAIR_V3_20261008.json](H232_JSON_PATH_SERIALIZATION_REPAIR_V3_20261008.json) and [source delta](h232_profile_v3_serialization_repair/source_delta.patch). V3 is not a provider packet and was not pushed.
