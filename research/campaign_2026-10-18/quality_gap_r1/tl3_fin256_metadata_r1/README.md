# fin256 parent metadata extraction R1

Status: **prepared only; no remote launch or checkpoint load authorized here**. This isolated packet extracts safe metadata from exactly the two historical warm-start endpoints referenced by the TL3 lineage: `fin256_s3@ep3` and `fin256_s4@ep3`. It does not fit, infer, copy, export or mutate model weights. It never opens caches or large arrays.

## Exact inputs and provenance

The bounded source observation is `../tl3_ancestor_followup_r1/RAW_PROBE_RECEIPT_R1.json`, SHA-256 `93c7bda0dc3cad159d456022661ec2ef670bb4b6776257da55c8f662a0f18a18`. It reports the logical run directories under `/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/` as symlinks to the resolved `/home/grinchenkoeg/WunderFunder/artifacts/runs/` paths below. The observed `model_ep3.pt` sizes are 3,933,599 bytes each. The probe only statted checkpoint endpoints; it did not hash or read their payloads. A root owner must freshly verify each symlink target, resolved file path, regular-file identity, exact byte count and SHA-256 before creating a hash-bound manifest. The exact path, source observation receipt SHA, and resource contract are enforced by the extractor.

| ID | Resolved path | Observed bytes | SHA-256 |
|---|---|---:|---|
| `fin256_s3_ep3` | `/home/grinchenkoeg/WunderFunder/artifacts/runs/fin256_s3/model_ep3.pt` | 3,933,599 | root must freshly compute |
| `fin256_s4_ep3` | `/home/grinchenkoeg/WunderFunder/artifacts/runs/fin256_s4/model_ep3.pt` | 3,933,599 | root must freshly compute |

Combined input size is 7,867,198 bytes. The committed [input manifest template](INPUT_MANIFEST_TEMPLATE.json) intentionally has null hashes, no root admission receipt binding, and a non-runnable status. The extractor accepts only the exact two IDs, resolved paths, sizes, source-observation pin, a fresh SHA-256 for each file, and a root-admission receipt SHA.

## Extraction boundary

`extract_fin256_metadata.py` runs one file at a time with PyTorch's `torch.load(..., map_location="cpu", weights_only=True)`. It has no unrestricted-pickle fallback and does not add safe globals. An unsupported checkpoint stops the run. The helper sets `CUDA_VISIBLE_DEVICES` empty before importing Torch, checks that CUDA is not initialized, and verifies the active cgroup is capped at 0.5 CPU, 1.5 GB memory and zero swap. Root must enforce the 120-second hard limit externally with a fresh bounded unit; the internal monotonic deadline cannot interrupt a blocked loader.

For each checkpoint it verifies a regular non-symlink file, exact size, SHA-256 and stable file identity before loading, then repeats the SHA and identity checks after the restricted load. It emits bounded top-level/scalar configuration metadata and tensor names/shapes/dtypes only. Secret-shaped keys and tensor values are omitted. Checkpoints are discarded between inputs. The create-only JSON output must be a new path; existing files and symlinks are rejected.

A fresh root-generated `ROOT_HASH_BOUND` input manifest is required at launch. The intended command shape, after a separate root admission and fresh remote staging, is:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  /home/grinchenkoeg/WunderFunder/.venv/bin/python \
  /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/tl3-fin256-metadata-v1/extract_fin256_metadata.py \
  --manifest /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/tl3-fin256-metadata-v1/TL3_FIN256_METADATA_INPUTS_R1.json \
  --output /home/grinchenkoeg/computer-vision-homework-20261018/receipts/TL3_FIN256_METADATA_R1.json
```

This command is documentation only and is not a launch request. Stage only `extract_fin256_metadata.py` and the fresh root-generated `TL3_FIN256_METADATA_INPUTS_R1.json` into the clean `/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/tl3-fin256-metadata-v1/` execution directory; reject all extra files, `__pycache__` and `.pyc`. Keep launch records and captured output in separately sealed paths. Do not touch the existing eight-checkpoint packet.

## What this can establish

The receipt can reveal safe args/provenance fields and tensor schema stored in these exact checkpoint objects. It cannot establish which ordered groups were used for fitting unless the object itself contains verifiable membership information. It does not establish F0-clean ancestry, source/config identity, or a valid teacher/cache lineage by itself. Root must review any extracted metadata against the historical logs, source snapshot and canonical F0 membership before treating either endpoint as a clean initializer. This is code preparation only; no hashes of checkpoint contents are claimed here.

Local tests use tiny synthetic checkpoints only:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python -m unittest -v test_extract_fin256_metadata.py
uv run --no-sync ruff format --check extract_fin256_metadata.py test_extract_fin256_metadata.py
uv run --no-sync ruff check extract_fin256_metadata.py test_extract_fin256_metadata.py
```
