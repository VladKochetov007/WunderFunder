# TL3 checkpoint metadata extraction preparation

**Prepared, not admitted or run.** This packet describes a small CPU-only metadata pass over eight exact team checkpoint files. It does not read prediction caches, train, infer, modify checkpoints, export tensor values, or run remotely. Root owns fresh path/hash admission and any later launch.

## Why these eight files

`INPUT_MANIFEST_TEMPLATE.json` binds the paths and stat sizes observed in the source-closure probe. The selected files cover the known true TL3 weight initializer, both possible endpoints for each direct tied-lead teacher, the named `orgbase` family teacher, the propensity model, and the distilled member whose complete resolved args are absent from a separate run log.

| ID | Exact B path | Observed bytes | Known file SHA-256 |
|---|---|---:|---|
| `fbc1k_s5_ep3` | `/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/fbc1k_s5/model_ep3.pt` | 3,934,495 | not computed |
| `fin256ft_s3_model` | `/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/fin256ft_s3/model.pt` | 3,932,843 | not computed |
| `fin256ft_s3_ep1` | `/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/fin256ft_s3/model_ep1.pt` | 3,933,351 | not computed |
| `fin256ft_s4_model` | `/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/fin256ft_s4/model.pt` | 3,932,843 | not computed |
| `fin256ft_s4_ep1` | `/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/fin256ft_s4/model_ep1.pt` | 3,933,351 | not computed |
| `orgbase_model` | `/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/orgbase/model.pt` | 1,042,631 | not computed |
| `prop_f0_model` | `/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/prop_f0/model.pt` | 159,607 | not computed |
| `w18final_distilled_s5_ep1` | `/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/w18final_distilled_s5/model_ep1.pt` | 3,942,180 | not computed |

The total observed input size is 24,811,301 bytes. Each input is below 32 MiB and the inventory is below 128 MiB. The propensity cache sidecars claim model SHA-256 `9e92f26d1704f84f9170d746b372c2ba67f7fb611831865a1fc689d6b813b21c`; the model payload itself was not hashed, so this is recorded as a sidecar claim only.

No current file SHA is asserted by this packet. Before execution, root must make a **fresh** exact-path source observation, verify all eight sizes and stable generations, compute each SHA-256, and create a new `ROOT_HASH_BOUND` run manifest with those hashes and the current admission-receipt SHA. The committed template intentionally has null hashes and fails closed in the extractor. It is not a runnable input manifest.

## Extractor contract

`extract_checkpoint_metadata.py` accepts only the exact eight paths and sizes above, rejects symlinks and any other input, and requires all eight hashes plus a root admission binding. For each file it checks stat identity and a full SHA-256 before loading, uses `torch.load(..., map_location="cpu", weights_only=True)`, extracts safe scalar/config fields and tensor key/shape/dtype metadata, discards the loaded object, then repeats the source hash/stat check before advancing to the next file. It does not return or serialize tensor values, optimizer values, RNG state, or cache contents. Secret-shaped config keys are omitted. Outputs are create-only.

Execution must use a fresh root-admitted systemd unit enforcing 0.5 CPU, at most 1.5 GB memory, zero swap and a 120-second hard runtime. The helper independently checks the active cgroup CPU/memory/swap limits, hides CUDA before importing Torch, rejects an initialized CUDA runtime, caps metadata output, and checks a monotonic 120-second deadline between file operations. The external systemd limit is still required because an in-process deadline cannot interrupt a blocked deserializer.

Example interface after root has staged the sealed helper and generated a hash-bound run manifest:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  /home/grinchenkoeg/WunderFunder/.venv/bin/python \
  /home/grinchenkoeg/computer-vision-homework-20261018/research/campaign_2026-10-18/quality_gap_r1/tl3_checkpoint_metadata_r1/extract_checkpoint_metadata.py \
  --manifest /home/grinchenkoeg/computer-vision-homework-20261018/receipts/TL3_CHECKPOINT_METADATA_INPUTS_R1.json \
  --output /home/grinchenkoeg/computer-vision-homework-20261018/receipts/TL3_CHECKPOINT_METADATA_R1.json
```

Stage the reviewed helper and fresh root-bound input manifest in a new execution directory; keep launch logs and capture receipts in the separately sealed record directory. The execution directory allowlist is exactly `extract_checkpoint_metadata.py` and the root-generated `TL3_CHECKPOINT_METADATA_INPUTS_R1.json`; reject extra files, `__pycache__`, and `.pyc`. This is a command shape, not permission or evidence of a B launch. The output must remain a new path. The planned result contains no checkpoint tensors; reviewers should inspect the manifest, full source identities, and extracted ancestry/config fields before deciding whether any new fit is scientifically justified.

## Local tests and limits

`test_extract_checkpoint_metadata.py` exercises the exact eight-item allowlist, root-hash requirement, fail-closed path and hash checks, metadata redaction, tensor shape/type summaries without numerical values, bounded hashing, and create-only output rejection. It also uses the local uv Python 3.13.11 / PyTorch 2.14.1 runtime to save and extract a tiny synthetic CPU checkpoint; no production checkpoint was loaded. A later admitted B preflight is the first test of PyTorch's restricted loader against the exact team checkpoint objects; it must stop on any unsupported safe-unpickling type and must never fall back to unrestricted pickle loading.
