# Fresh TL3 submission candidates

[Draft PR51](https://github.com/VladKochetov007/WunderFunder/pull/51) targets `research-team`.

**Prepared, not yet exported or scored.** Fresh PR49 checkpoints are not on the
preparation Mac. This packet creates CPU submission ZIPs once the receiver has
those assets. No local tests, training, inference, remote jobs or portal upload
were run during preparation.

| Candidate | Fixed members, zero-based endpoint3 | Published individual Torch WP |
|---|---|---:|
| `tiedlead-s5` | `tl3c_fttall_tlead_s5` ×1 | 0.691415 |
| `blend` | `tl3c_fbc1k_s5` ×0.34, tiedlead s5 ×0.33, tiedlead s6 ×0.33 | Mixture unknown |

The other two individual scores are 0.691355 and0.690949. These are exposed
full-official-valid scores, not CPU or leaderboard results. The
[terminal report](https://github.com/VladKochetov007/WunderFunder/pull/49#issuecomment-6484849146)
records 16 completed fits and blocked W18 at 2026-10-29T00:56:02Z. This replacement
mixture is a new candidate, not the canonical TL3 mixture that requires W18.

## What is implemented

- `create_manifest.py`, `candidate_contract.py`, `candidate_io.py`: pin the three
  new endpoints and their four-node teacher/initialization closure, check the
  admitted split, source digests, frozen checkpoint selection and memberships.
- `build_candidate.py`, `checkpoint_export.py`: export saved evaluated EMA
  weights, retain fitted mean/std, validate exact tensor/input schemas, trim only
  the auxiliary head outputs and assemble a create-only ZIP with a receipt.
- `solution.py`: one-row native streaming callback, fresh sequence reset,
  every-row state updates and explicit float32/dynamic-INT8 modes.
- `validate_candidate.py`, `validation_io.py`, `validation_runtime.py`: offline
  Python 3.11 CPU replay, reset/mask checks, full official metric, prediction
  export, byte-bound reference comparison, time and memory evidence.
- `export_reference.py`, `reference_source.py`: reconstruct the canonical Torch
  predictor from a separate pinned PR49 checkout and export the chunk4000
  comparison reference on the receiver.
- `Dockerfile.receiver`: isolated receiver environment. The final ZIP needs
  NumPy and its included native library; Torch is only used for receiver export.
- `recipe.json`, `PR.md`: frozen candidates, decision rules and known limitations.

The old `wnn_native` packager is not used: it does not implement this tiedlead
encoder. The historical `wnn_cpp` binaries are reused as declared team assets.
No historical weights or old solution wrapper are copied into a fresh ZIP.
Matching tensor layouts **do not prove** that the opaque native binary implements
the current trainer's feature semantics.

## Receiving inputs

Use a separate checkout of this PR; do not switch a live training checkout.
All following commands are for the receiving Linux machine, **not the Mac**.
Replace `/srv/...` paths with actual locations; they are examples, not discovered
mounts or an instruction to start new training.

```bash
cd /srv/wunder-submission
PACKET="$PWD/research/experiment_drafts/tl3c-submission-20261029"
RUNS=/srv/tl3/artifacts/runs
SPLIT=/srv/tl3/chronological_primary.json
VALID=/srv/tl3/datasets/valid.parquet
NATIVE=/srv/team-assets/tl3-native-runtime-only.zip
OUT=/srv/submission-candidates/tl3c-20261029-r1
mkdir "$OUT"

python3 "$PACKET/create_manifest.py" \
  --runs-dir "$RUNS" --split "$SPLIT" --output "$OUT/inputs.json"
```

The prepared native-only handoff `tl3-native-runtime-only.zip` contains exactly
three libraries and no old weights. It is 998762 bytes, SHA256
`beb6416c19402605d68bbb467df83675d8d079914b7d2c89914a60789ba1a0c2`.
The builder checks every native member's exact name, length and SHA256.

Alternatively, the team's historical `tl3.zip` is accepted, SHA256
`d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569`.
Its three Linux CPython 3.11 binaries are copied; its old weights are ignored.
Keep it separate from the newly generated `submission.zip` files.

Required checkpoints and adjacent `provenance.json`:

| Run directories | Checkpoint |
|---|---|
| `tl3c_fbc1k_s5`, `tl3c_fttall_tlead_s5`, `tl3c_fttall_tlead_s6` | `model_ep3.pt` |
| `tl3c_fin256_s3`, `tl3c_fin256_s4` | `model_ep3.pt` |
| `tl3c_fin256ft_s3`, `tl3c_fin256ft_s4` | `model_ep1.pt` |

The split must hash to
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.
It fits 10325 train sequences, excludes overlapping train components and evaluates
all 1873 official valid sequences. Declared teacher lineage is checked, but the
legacy child sidecar does not fully bind the teacher prediction bytes actually
consumed during fitting. Obtain the finite17 export/cache receipts before claiming
complete consumed-cache ancestry; do not silently dismiss this limitation.

## Export four fixed variants

Use the receiver's existing Torch environment. Export loads saved tensors; it
does not construct a model or run training/inference. Restrict the deserializer
to `weights_only=True`; an unsupported checkpoint must fail, not fall back to
arbitrary pickle execution.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
for candidate in tiedlead-s5 blend; do
  for mode in f32 dynamic_int8; do
    python3 "$PACKET/build_candidate.py" \
      --manifest "$OUT/inputs.json" --runtime-archive "$NATIVE" \
      --candidate "$candidate" --mode "$mode" \
      --output-dir "$OUT/$candidate-$mode"
  done
done
```

Each new directory contains `submission.zip` and `receipt.json`. The primary
`f32` mode sets fast activations, half weights and dynamic INT8 all false.
`dynamic_int8` sets dynamic INT8 and fast activations true, half weights false.
Therefore the measured loss for that mode includes activation approximation.
No second EMA is applied: `model_ep3.pt["model"]` is already the scored EMA state.

The three-model float32 tensor payload is about 12.39 MB plus 2.64 MB for the native
libraries before ZIP overhead. The builder checks the actual completed archive
against 20000000 bytes; the estimate is not a size certificate. A successful export
still has pending CPU/quality/parity checks and must not be uploaded as qualified.

## Prepare the receiver image and a short profile

Build once with network access, then evaluate with no network and no GPU mounts.
The Docker image ID is the exact environment pin; keep its inspect receipt.

```bash
docker build -f "$PACKET/Dockerfile.receiver" -t tl3c-cpu-receiver "$PACKET"
IMAGE=$(docker image inspect tl3c-cpu-receiver --format '{{.Id}}')
docker image inspect "$IMAGE" > "$OUT/receiver-image.json"
VALID_SHA=$(sha256sum "$VALID" | cut -d ' ' -f 1)
mkdir "$OUT/scratch"

cpu_check() {
  timeout 3600s docker run --rm --network none --cpus 1 \
    --memory 16000000000 --memory-swap 16000000000 \
    --pids-limit 64 --cap-drop ALL --security-opt no-new-privileges \
    --read-only --user "$(id -u):$(id -g)" \
    -v "$PWD:/work:ro" -v "$VALID:/inputs/valid.parquet:ro" \
    -v "$SPLIT:/inputs/split.json:ro" -v "$OUT:/out:rw" \
    -v "$OUT/scratch:/tmp:rw" "$IMAGE" \
    python /work/research/experiment_drafts/tl3c-submission-20261029/validate_candidate.py \
    --data /inputs/valid.parquet --data-sha256 "$VALID_SHA" \
    --split /inputs/split.json "$@"
}

cpu_check --zip /out/tiedlead-s5-f32/submission.zip \
  --receipt /out/tiedlead-s5-f32/receipt.json --level v2 \
  --max-sequences 8 --output-dir /out/single-f32-profile-v2
```

Use `set -euo pipefail` in a saved Bash script so a failed export/check stops the
following commands. Run the snippets only after the asset paths are filled in.

The wrapper limits its own process to one CPU and 16 GB; the container also caps
aggregate CPU/RAM and disables swap beyond that limit. It rejects a normal online
host invocation. `v2` is the conservative included x86 build, not an assurance of
support for every possible x86 processor. Each ISA/candidate must run in a fresh
process. Profile supported `avx2`/`v3` separately before selecting a faster level;
the actual submission wrapper dispatches on CPU flags. Keep the exact tested level
and host CPU in the evidence. Do not treat a v3 benchmark as proof for an unknown
portal CPU, and verify all dispatch paths that the target may take.

`report.json` and `predictions.npy` are written to the new output directory. A
profile uses complete 20000-row sequences, including warmup, but has no primary
quality claim. Exit0 without a reference means replay completed; it does **not**
mean parity or submission approval. A timeout may interrupt receipt writing;
preserve the directory and rerun only under a new attempt path after diagnosis.

## Full score and precision comparison

Inspect the profile, the parity comparison below and CPU capability first.
In the same shell, use the proven
native level (below `v2`) for the frozen full-valid exports:

```bash
for candidate in tiedlead-s5 blend; do
  cpu_check --zip "/out/$candidate-f32/submission.zip" \
    --receipt "/out/$candidate-f32/receipt.json" --level v2 \
    --full-valid --output-dir "/out/$candidate-f32-full-v2"

  cpu_check --zip "/out/$candidate-dynamic_int8/submission.zip" \
    --receipt "/out/$candidate-dynamic_int8/receipt.json" --level v2 \
    --full-valid --output-dir "/out/$candidate-int8-full-v2" \
    --reference "/out/$candidate-f32-full-v2/predictions.npy" \
    --reference-metadata "/out/$candidate-f32-full-v2/report.json" \
    --comparison-mode quality --max-wp-drop 0.0002
done
```

This verifies 3528273 scored rows from 1873 sequences, computes the official pooled
WP, and records precision loss without requiring INT8 to be bitwise identical.
Compare mixture versus singleton metrics from the same mode, data and ISA. The
fixed useful-mixture threshold is 0.0005 WP; smaller gains favor the simpler
singleton unless another measured tradeoff changes the choice. Do not tune blend
coefficients on these reports. All results remain development-exposed.

For runtime projection, add `--test-sequences N` **only after verifying** the
official whole-test cardinality. The report uses the slowest observed sequence,
including decode, callbacks, scoring and prediction writes, plus model startup.
It cannot certify an unknown portal processor or hidden workload. The preferred
projection is below 2880 seconds against the 3600 second rule. Without a test-size
bound, the time-limit decision remains unresolved.

## The chunk-boundary parity question

PR49's Torch evaluator invokes `encode` in chunks of 4000. Its row-change feature
starts at zero at each invocation, while the historical native callback may keep
continuous row differences. Lead52 itself is assembled causally over the complete
sequence before chunking. Never reset the whole GRU every 4000 rows to hide a feature
discrepancy.

Prepare a separate clean reference checkout; this never switches a live job's
working tree. Use the receiver's Python 3.11 research environment with Torch,
NumPy, PyArrow and attrs already provisioned. The CPU command below uses no GPU.

```bash
git fetch origin experiment/tl3-chronological-20261028
REFERENCE_ROOT=/srv/wunder-reference-pr49
git worktree add --detach "$REFERENCE_ROOT" 188c0788e571c6f981cdc6827fef0d50bd38cf14

for candidate in tiedlead-s5 blend; do
  python3 "$PACKET/export_reference.py" --training-root "$REFERENCE_ROOT" \
    --manifest "$OUT/inputs.json" --receipt "$OUT/$candidate-f32/receipt.json" \
    --data "$VALID" --data-sha256 "$VALID_SHA" --device cpu \
    --max-sequences 8 --output-dir "$OUT/$candidate-torch-ref8"

  cpu_check --zip "/out/$candidate-f32/submission.zip" \
    --receipt "/out/$candidate-f32/receipt.json" --level v2 \
    --max-sequences 8 --output-dir "/out/$candidate-parity8-v2" \
    --reference "/out/$candidate-torch-ref8/predictions.npy" \
    --reference-metadata "/out/$candidate-torch-ref8/report.json" \
    --comparison-mode equivalent --equivalence-atol 1e-5
done
```

The comparison intentionally fails if native and Torch semantics differ. Preserve
the report and inspect the boundary errors. Do not loosen tolerance simply to pass.
If a distinct streaming candidate is retained, run its full valid evaluation as
described above and label that choice explicitly. A short parity run establishes
agreement only on its declared sequences, not the entire dataset. For a complete
comparison, replace `--max-sequences 8` by `--full-valid` in both commands and use
new output directories. The reference exporter also supports `--device cuda:0`
only with exactly one explicitly selected visible GPU; that is a later scheduling
choice and no current job is stopped or repurposed here.

The gate accepts an optional pinned Torch reference in float32 NPY shape
`[number_of_selected_groups,20000,2]`. Its JSON metadata must bind:

- `reference_kind: "torch_chunk4000"`, `data_sha256`, `split_sha256`;
- ordered `row_groups`, matching `sequence_ids`, `predictions.sha256`;
- the exact `member_signature` (run, coefficient, family, checkpoint SHA256)
  matching the build receipt. A different singleton/mixture is not a parity reference.

Generate that reference on the receiver from the admitted PR49 implementation
and saved endpoints; do not label a native export as Torch. Invoke the same gate
with `--reference ... --reference-metadata ... --comparison-mode equivalent
--equivalence-atol 1e-5`. It probes served rows and windows around 4000/8000/12000/
16000, returning a failing comparison when tolerance is exceeded. Without a Torch
reference, this parity remains unverified. A distinct native stream can still be
scored and considered as an explicitly separate candidate after full validation;
its score must not inherit the published Torch 0.691415.

## Before choosing the actual ZIP

Return the immutable manifest, all build/validation receipts, exact ZIP hashes,
receiver image/CPU details and any failed comparison. The checked object must be
the same ZIP that is subsequently uploaded. Select the stronger supported served
candidate, requiring the size, ABI, reset, memory, time and ancestry evidence;
retain unknowns explicitly. This packet never automatically approves or submits.
No portal score has been observed for either fresh candidate.
