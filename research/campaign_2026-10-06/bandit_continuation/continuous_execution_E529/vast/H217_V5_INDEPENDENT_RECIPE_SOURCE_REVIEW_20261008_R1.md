# H217 V5 independent recipe and source review

Status: **current packet bytes pass; worker-side exact-recipe validation has a blocking gap**. This review used the isolated V5 source tree at base commit `801b6629dc9e67becdca0514420bfe882bc9a48b`. The packet remains prepared only: no profile, fit, provider create, push, or grade is authorized by this review.

## Finding

`h217_packet_v5/worker.py` does not fail closed on all recipe drift. Its `_option()` helper reads the first occurrence of a flag, and `validate_recipe()` checks a selected subset of options without requiring the full exact argv. On a copy of the V5 recipe, the validator accepted appended `--lr 0.1`, `--seed 8`, `--class-balance`, `--valid-repeat 2`, and `--epoch-samples 1`. The actual `train_rnn.py` parser recognizes those options; duplicate `--lr` and `--seed` values are resolved by the later occurrence, `--class-balance` enables a replacement weighted sampler, `--valid-repeat` repeats fit-valid sequences, and `--epoch-samples` changes the training draw count. This can alter the declared optimizer/sampling treatment while the recipe summary still reports the intended V5 treatment.

These mutations are absent from the current V5 archive, whose bytes and manifests were independently verified below. The finding concerns the worker’s ability to reject accidental or later recipe drift. Before relying on worker-side recipe validation, compare each command to a canonical exact argv (or use a complete flag allowlist with duplicate-option rejection) and add mutation tests for the reproduced cases. Root may also bind the exact packet hash with an independent external validator before any admission.

## Current packet fidelity

The current recipe matches the root contract: two scratch TCNs, widths 128 and 256, 12 layers, mix 1, seed 7, default causal input width 211. The 211 inputs are 112 raw columns plus 99 event features; `RAW_ONLY` defaults false and is enabled only by `--raw-only`, which is absent from all four current profile/fit commands. Both use fold 0, `train+valid`, four epochs, fixed epoch 3, B8/C2000, WMSE with chunk normalization, AdamW at 0.001/0.01 with OneCycle, FP16 scale 4096, and no evaluation during fitting. Initialization, distillation, propensity, and auxiliary losses are disabled. The pinned split contains 10,535 train and 1,405 fit-valid groups (11,940 total) with 468 excluded evaluation groups. The trainer caps the shuffled fold train list at 10,607, so all 10,535 remain; `train+valid` adds the 1,405 fold-fit valid groups, and the trainer checks component separation.

Both arms independently profile first. The worker requires one config, one timing record, and one update-evidence record; 230 attempts/completions, zero skips, finite losses and gradients, and scale 4096. The measured outer-batch timing projects four epochs and rejects that arm if the 7,200-second fit cap would be exceeded. The runtime file and worker enforce per-arm profile/fit/export caps 900/7,200/1,800 seconds, a 21,600-second maximum, and reserved setup/package time. Export is fixed epoch-3 valid-only FP32 with checkpoint, row-order, shape, dtype, finiteness, and prediction-hash checks; the worker does not score predictions.

## Identity and checks

The V5 project contains 56 files and is byte-identical to both `h217_packet_v4/project` and all 56 project members of the V4 tar. The V4 packet/source-manifest hashes match the V5 predecessor fields. Every one of 61 V5 archive members matches its declared size and SHA-256 in `H217_V5_ARCHIVE_MANIFEST.json`; packet, source-manifest, and archive-manifest hashes match the build receipt. All options in the four current profile/fit command arrays are present in the actual `train_rnn.py` argparse declarations. This check parsed source syntax without importing Torch.

The focused suite passed **27 tests in 0.18s** under the 1 CPU/600 MB/120s support limit. The suite does not test the material recipe mutations above; the independent negative probe demonstrated that gap. No Torch import, fit, input-array read/NPZ decompression, parquet-footer access, GPU, provider, or credentials were used. This is an execution-fidelity review, not evidence about model quality or public transfer.

Machine-readable receipt: `H217_V5_INDEPENDENT_RECIPE_SOURCE_REVIEW_20261008_R1.json`.
