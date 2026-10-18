# W18 checkpoint metadata extractor R1

**Prepared and synthetically tested; not staged, admitted, or executed on B.** This isolated helper reads eight exact W18 teacher/parent checkpoints, restricted to CPU metadata inspection. It does not export tensor values, read prediction or propensity arrays, perform inference, train, or copy checkpoint files.

## Exact input set and evidence

The allowlist contains these logical endpoints under `/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/`:

| ID | Endpoint | Observed bytes |
|---|---|---:|
| `ffull_s3_ep1` | `ffull_s3/model_ep1.pt` | 1,053,479 |
| `ffull_s4_ep1` | `ffull_s4/model_ep1.pt` | 1,053,479 |
| `frawW_s3_ep3` | `frawW_s3/model_ep3.pt` | 1,054,943 |
| `frawW_s4_ep3` | `frawW_s4/model_ep3.pt` | 1,054,943 |
| `frawW_s3ft_ep1` | `frawW_s3ft/model_ep1.pt` | 1,053,607 |
| `frawW_s4ft_ep1` | `frawW_s4ft/model_ep1.pt` | 1,053,607 |
| `fmu128_s3_ep3` | `fmu128_s3/model_ep3.pt` | 1,054,815 |
| `fmu128_s4_ep3` | `fmu128_s4/model_ep3.pt` | 1,054,815 |

The eight files total 8,433,688 bytes. The two `fmu128` files were added only after the fresh stat-only path preflight at `2026-10-18T18:19:20.195765+00:00`; its exact receipt is `tl3_w18_checkpoint_execution_r1/ROOT_PATH_PREFLIGHT_V1.json`, SHA-256 `b555f6e282e45129be74d7614da8a77fc42b2a684da06c6c7aa6d83ae8acc5ba`. That receipt reports all eight as regular files, with each resolved path equal to its logical path and no symlink components. It read no payload bytes and provides no content hashes. Root must therefore make fresh exact-path checks and bind every file's current generation and full SHA-256 before any load.

The code also binds the bounded W18 run-evidence digest and the historical six-endpoint preparation-template digest in `SOURCE_OBSERVATION`. The latter is historical evidence for the first six endpoints; it is not the current eight-item input template or a content identity. The input template in this packet reflects all eight targets but deliberately has null SHA, generation and admission placeholders.

The exact allowlist excludes `fbc1k_s5`, `orgbase`, and the fin256 checkpoints. The old reviewed extractors remain unchanged; their fixed allowlists cannot accept this set.

## Integrity and loader contract

The runtime input manifest must contain exactly the eight IDs, in order, with the pinned logical paths and byte counts, fresh source SHA-256 and generation (`device`, `inode`, `size`, `mtime_ns`, `ctime_ns`, and `mode`), and the observed path-resolution record. For this preflight, each `source_resolution` must be exactly `{resolved_path: logical_path, symlinks: []}`. The manifest must also bind the exact root admission receipt path and SHA-256 and match the helper's fixed source-observation and resource contracts.

Each file is capped at 32 MiB; the aggregate cap is 64 MiB. The helper reads the input manifest and admission receipt through `O_NOFOLLOW` descriptors and verifies the opened-file and pathname generations against the bytes it hashes. It then verifies source path resolution and generation and hashes each checkpoint before loading. It opens the resolved regular file with `O_NOFOLLOW`, passes a duplicate of the pinned file descriptor to `torch.load(..., map_location="cpu", weights_only=True)`, then verifies the same descriptor's hash/generation and the logical path resolution again. There is no unrestricted pickle fallback. Config and tensor-schema traversal are bounded by depth, item-count and output-size caps; retained content is limited to redacted primitive config fields plus tensor names, shapes and dtypes. Each loaded object is deleted before opening the next file, and the final receipt is created exclusively and capped at 2 MiB.

Before importing Torch, the helper hides CUDA. It checks the enforced cgroup at 0.5 CPU core, no more than 1.5 GB memory and zero swap; sets Torch to one CPU thread; and requires CUDA to remain uninitialized. The 120-second in-process deadline complements the root-admitted external runtime limit. Root must use the existing B Python runtime `/home/grinchenkoeg/WunderFunder/.venv/bin/python`, a verified physical disk-backed `TMPDIR`, a fresh resource/admission check, and a new execution directory with exactly the reviewed files and no `.pyc` or `__pycache__`.

This packet has three runtime Python modules: `extract_w18_checkpoint_metadata.py`, `w18_metadata_core.py`, and `w18_metadata_summary.py`. All three must be staged with their exact manifest hashes because the runner imports both support modules. The test file is local-only and is not part of the runtime inventory.

## Root-owned invocation shape

After independent review and root's fresh source, staging, output-collision, resource and admission checks, the command shape is:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  /home/grinchenkoeg/WunderFunder/.venv/bin/python \
  /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/tl3-w18-checkpoint-metadata-v1/extract_w18_checkpoint_metadata.py \
  --manifest /home/grinchenkoeg/computer-vision-homework-20261018/receipts/TL3_W18_CHECKPOINT_METADATA_INPUTS_R1.json \
  --output /home/grinchenkoeg/computer-vision-homework-20261018/receipts/TL3_W18_CHECKPOINT_METADATA_R1.json
```

The input template is not executable. Root creates a new root-bound manifest with the live generation/hash observations and admission receipt binding; it must not edit the template or overwrite an existing receipt. The helper prints one compact JSON line binding the output receipt hash to the input-manifest and admission hashes. Preserve raw stdout/stderr and the output receipt as new attempt evidence.

## Local checks and boundaries

`test_extract_w18_checkpoint_metadata.py` uses temporary synthetic files and a tiny synthetic Torch checkpoint only. It checks the exact eight-target contract and aggregate size, exact-manifest rejection, symlink rejection for binding/source files, replacement during loading, create-only output, admission/input SHA linkage, restricted CPU loading, safe config redaction and bounded tensor-schema-only output. It never opens production checkpoint data. The exact focused-test and Ruff results are recorded in `TEST_RECEIPT_R1.json`.

No B staging, admission, metadata extraction, or training has been performed by this packet. Fresh hashes/generations, runtime versions, output collisions, resource headroom and the root admission receipt remain execution-time bindings. Artifact hashes and external evidence pins are recorded in `PREPARATION_MANIFEST.json`; its own digest must be bound by the root launcher.
