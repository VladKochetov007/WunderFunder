#!/usr/bin/env python3
"""Bounded one-sequence CUDA shape profile on the real clean-F0 producer data.

This is a resource-safety probe, not an architecture-quality experiment. It
reads one source-qualified sequence and its homogeneous target, copies only the
fit-derived 211-feature normalization buffers from the frozen H256 control,
then optionally performs exactly one FP32 AdamW update on its first 2,000 rows.
It does not validate the full distillation-cache payload or export predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.data import N_FEATURES, SEQUENCE_LENGTH, WARMUP, SequenceFile
from wnn.features import event_features_torch
from wnn.nn import RNNModel
from wnn.splits import load_campaign_split
from wnn.training_sampling import EpochOrderSampler

SHAPES = ((256, 1), (192, 2), (256, 2), (512, 1), (384, 2), (640, 1), (512, 2), (448, 3))
EXPECTED_CACHE_META_SHA256 = "1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8"
EXPECTED_SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
EXPECTED_CONTROL_SHA256 = "e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda"
EXPECTED_CONTROL_PROVENANCE_SHA256 = "bdf643043884b1344564b636a2972ad52b8202ee30cf22598b1afd90405dd961"
EXPECTED_TEACHER_SOURCES = {
    "clean_t3_f0_muonall_h384_e3_s3_r5": (
        "201221c6b4140dda70b078db2a9d7a7246ec40b813f054570d195cecefc61582",
        "740b0d96ee311f215153db62272262e0153ce11f8f91560b03efae8e739f9ed0",
    ),
    "clean_t3_f0_muonall_h384_e3_s4_r5": (
        "714ac9b04cd1be0f63fb5f9f60cce2059e5e9284848470cae9d7198794767df7",
        "f738ef25f1b7d4244d6c900ebe4fba861efc24ddd106ff425fc60d7e3caccf11",
    ),
    "clean_t3_f0_muonall_h384_e3_s5_r5": (
        "763fac7f0aa24507167425e9905f1514a0a28ace48036b1c1c5819c7eb02c8d1",
        "44728c0c60981166f63038f2a6696b6714ce57c52e039e304883ffe7449def18",
    ),
}
EXPECTED_TARGET_SHA256 = "6a19f7ac45c6e8a02d3b89c0ac4986ffab4885ab7a4ebc5fb4599b3047f4e7b3"
EXPECTED_TARGET_VALID_SHA256 = "d1cd78537f3b6aaa830861ac771cf5c4621f1f2c1f19a73f6affb813c33ca3a2"
CUDA_ALLOCATOR_CAP_BYTES = 3_500_000_000
OBSERVED_VRAM_LIMIT_BYTES = 5_000_000_000


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Return a lowercase SHA256 digest."""
    return hashlib.sha256(value).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash an exact little-endian float32 normalization buffer."""
    array = value.detach().cpu().contiguous().to(torch.float32).numpy().astype("<f4", copy=False)
    return sha256_bytes(array.tobytes())


def cgroup_memory_current() -> int | None:
    """Read this process's cgroup-v2 memory charge when available."""
    try:
        entries = [line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                   if line.startswith("0::")]
        if len(entries) != 1:
            return None
        path = (Path("/sys/fs/cgroup") / entries[0].lstrip("/")).resolve()
        root = Path("/sys/fs/cgroup").resolve()
        if not path.is_relative_to(root):
            return None
        value = (path / "memory.current").read_text().strip()
        return int(value) if value.isascii() and value.isdigit() else None
    except (OSError, ValueError):
        return None


def nvidia_smi_used_bytes() -> int:
    """Return a conservative sum of current device usage reported by NVML."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not rows or any(not row.isascii() or not row.isdigit() for row in rows):
        raise RuntimeError(f"unparseable nvidia-smi memory output: {result.stdout!r}")
    return sum((int(row) + 1) * 1024**2 for row in rows)


def source_manifest(script: Path, repo: Path) -> dict[str, str]:
    """Bind the profile and the production model/data primitives it calls."""
    paths = [script, repo / "scripts/train_rnn.py", repo / "src/wnn/data.py",
             repo / "src/wnn/distillation_cache.py", repo / "src/wnn/features.py",
             repo / "src/wnn/nn.py", repo / "src/wnn/resources.py",
             repo / "src/wnn/splits.py", repo / "src/wnn/training_controls.py",
             repo / "src/wnn/training_sampling.py"]
    return {str(path.relative_to(repo)): sha256_file(path) for path in paths}


def load_profile_inputs(data_root: Path, contract: dict[str, Any], seed: int) -> dict[str, Any]:
    """Validate real producer identities and load one deterministic first-batch row."""
    split_path = data_root / contract["split_manifest"]
    split = load_campaign_split(split_path, data_root)
    if split.sha256 != EXPECTED_SPLIT_SHA256:
        raise ValueError("source-qualified F0 split manifest SHA256 changed")

    seed_order = np.random.default_rng(seed).permutation(np.asarray(split.train_groups, dtype=np.int64))
    train_indices = seed_order[: int(contract["sample_order"]["train_seqs_argument"])]
    if not np.array_equal(np.sort(train_indices), np.sort(np.asarray(split.train_groups))):
        raise ValueError("profile expects the clean-F0 train membership without truncation")
    combined_groups = np.concatenate((train_indices, np.asarray(split.valid_groups, dtype=np.int64)))
    order_batch_size = int(contract["sample_order"]["batch_size_for_order"])
    sampler = EpochOrderSampler(len(combined_groups), seed, batch_size=order_batch_size, drop_last=True)
    sampler.set_epoch(0)
    first_batch_indices = np.fromiter(iter(sampler), dtype=np.int64, count=order_batch_size)
    first_batch_groups = combined_groups[first_batch_indices]
    first_split = "train" if int(first_batch_groups[0]) in set(split.train_groups) else "valid"
    group = int(first_batch_groups[0])
    if first_split == "train" and group not in set(split.train_groups):
        raise ValueError("profile sampler selected a non-fit train row group")
    if first_split == "valid" and group not in set(split.valid_groups):
        raise ValueError("profile sampler selected a non-fit valid row group")
    split.assert_fit_groups(
        [group] if first_split == "train" else [],
        [group] if first_split == "valid" else [],
    )
    if first_split == "valid" and group in set(split.evaluation_groups):
        raise ValueError("profile sampler selected a held-out evaluation row group")

    metadata_path = data_root / contract["cache_root"] / "metadata.json"
    metadata_size = metadata_path.stat().st_size
    if metadata_size <= 0 or metadata_size > int(contract["cache_metadata_max_bytes"]):
        raise ValueError("clean-T3 metadata exceeds the explicit profile read bound")
    metadata_raw = metadata_path.read_bytes()
    if len(metadata_raw) != metadata_size:
        raise ValueError("clean-T3 metadata changed while being read")
    metadata_sha = sha256_bytes(metadata_raw)
    if metadata_sha != EXPECTED_CACHE_META_SHA256 or metadata_sha != contract["cache_metadata_sha256"]:
        raise ValueError("clean-T3 cache metadata identity changed")
    metadata = json.loads(metadata_raw)
    expected_student = {
        "epochs": 6, "checkpoint_policy": "fixed", "checkpoint_epoch": 5,
        "hidden": 256, "layers": 2, "batch": 16, "chunk": 2000,
        "distill_alpha": 0.9, "since_weight": 0.0,
        "inherit_init_preprocessing": False,
    }
    if metadata.get("student_contract") != expected_student:
        raise ValueError("historical producer student_contract changed; profile adapter cannot relax it")
    if (metadata.get("cache_profile") != "cleanfold_homogeneous_strong_t3_v1"
            or metadata.get("split_manifest_sha256") != split.sha256
            or metadata.get("fold") != 0
            or metadata.get("fit_groups") != {"train": list(split.train_groups), "valid": list(split.valid_groups)}
            or metadata.get("evaluation_valid_groups") != list(split.evaluation_groups)
            or metadata.get("target_contract", {}).get("clipping") != "none during cache construction"
            or metadata.get("target_contract", {}).get("formula")
            != "float32(0.8125 * float32_mean(three_epoch2_predictions))"):
        raise ValueError("clean-T3 producer target or source-qualified membership differs")
    target_contract = metadata["files"][f"main_homogeneous_{first_split}"]
    expected_target_sha = EXPECTED_TARGET_SHA256 if first_split == "train" else EXPECTED_TARGET_VALID_SHA256
    if target_contract.get("sha256") != expected_target_sha:
        raise ValueError("target file SHA does not match the frozen producer metadata")

    source_records = {item["run"]: item for item in metadata["sources"]}
    for run, (checkpoint_sha, provenance_sha) in EXPECTED_TEACHER_SOURCES.items():
        record = source_records.get(run)
        if record is None or record.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError(f"source teacher checkpoint binding differs for {run}")
        checkpoint_path = data_root / "artifacts/runs" / run / "model_ep2.pt"
        if sha256_file(checkpoint_path) != checkpoint_sha:
            raise ValueError(f"source teacher checkpoint bytes differ for {run}")
        provenance_path = data_root / "artifacts/runs" / run / "provenance.json"
        if sha256_file(provenance_path) != provenance_sha:
            raise ValueError(f"source teacher provenance binding differs for {run}")

    target_path = data_root / contract["cache_root"] / target_contract["path"]
    target_array = np.load(target_path, mmap_mode="r", allow_pickle=False)
    if (list(target_array.shape) != target_contract["shape"]
            or target_array.dtype != np.float32
            or target_path.stat().st_size != target_contract["bytes"]):
        raise ValueError("profile target mmap disagrees with producer shape or dtype")
    row_groups = np.asarray(target_contract["row_groups"], dtype=np.int64)
    row_groups_sha = sha256_bytes(np.asarray(row_groups, dtype="<i8").tobytes())
    if row_groups_sha != target_contract["row_groups_sha256"]:
        raise ValueError("ordered target row-group index differs from the producer metadata")
    expected_fit_groups = split.train_groups if first_split == "train" else split.valid_groups
    if len(row_groups) != len(expected_fit_groups) or set(map(int, row_groups)) != set(expected_fit_groups):
        raise ValueError("ordered target rows do not have exact source-qualified F0 fit membership")
    target_pos = np.flatnonzero(row_groups == group)
    if target_pos.size != 1:
        raise ValueError(f"target index does not uniquely contain sampled group {group}")
    target_row = np.array(target_array[int(target_pos[0])], dtype=np.float32, copy=True)
    if target_row.shape != (SEQUENCE_LENGTH, 2) or not np.isfinite(target_row).all():
        raise ValueError("sampled teacher target row has wrong shape or non-finite values")

    source_path = data_root / "datasets" / f"{first_split}.parquet"
    split_manifest = json.loads(split_path.read_bytes())
    seq_ix_by_group = {
        (source_name, int(item["row_group"])): int(item["seq_ix"])
        for source_name, nodes in split_manifest["sequences"].items()
        for item in nodes
    }
    sequence = SequenceFile(source_path).read(group)
    expected_seq_ix = seq_ix_by_group.get((first_split, group))
    if expected_seq_ix is None or sequence.seq_ix != expected_seq_ix:
        raise ValueError("sampled parquet sequence ID disagrees with the frozen split manifest")
    if sequence.targets is None:
        raise ValueError("selected producer sequence lacks fitting labels")
    scored_mask_present = sequence.scored is not None
    scored = sequence.scored if scored_mask_present else np.zeros(SEQUENCE_LENGTH, dtype=np.bool_)
    if sequence.features.shape != (SEQUENCE_LENGTH, N_FEATURES):
        raise ValueError("selected producer sequence feature schema changed")
    control_path = data_root / contract["normalization_source"]["checkpoint"]
    control_provenance_path = data_root / contract["normalization_source"]["provenance"]
    if sha256_file(control_path) != EXPECTED_CONTROL_SHA256:
        raise ValueError("frozen H256 control checkpoint identity changed")
    if sha256_file(control_provenance_path) != EXPECTED_CONTROL_PROVENANCE_SHA256:
        raise ValueError("frozen H256 control provenance identity changed")
    control = torch.load(control_path, map_location="cpu", weights_only=False)
    mean = control["model"]["mean"].detach().to(torch.float32).clone()
    std = control["model"]["std"].detach().to(torch.float32).clone()
    del control
    if mean.shape != (211,) or std.shape != (211,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("H256 control normalization buffers must be finite vectors of width 211")
    if torch.any(std <= 0):
        raise ValueError("H256 control normalization standard deviations must be positive")

    return {
        "split": split,
        "group": group,
        "seq_ix": sequence.seq_ix,
        "source_split": first_split,
        "first_batch_groups": [int(item) for item in first_batch_groups],
        "train_order_sha256": sha256_bytes(np.asarray(train_indices, dtype="<i8").tobytes()),
        "first_batch_order_sha256": sha256_bytes(np.asarray(first_batch_groups, dtype="<i8").tobytes()),
        "features": sequence.features,
        "labels": sequence.targets,
        "scored": scored,
        "teacher": target_row,
        "feature_row_sha256": sha256_bytes(np.asarray(sequence.features, dtype="<f4").tobytes()),
        "label_row_sha256": sha256_bytes(np.asarray(sequence.targets, dtype="<f4").tobytes()),
        "scored_row_sha256": sha256_bytes(np.asarray(scored, dtype=np.uint8).tobytes()),
        "scored_mask_present_in_source": scored_mask_present,
        "teacher_row_sha256": sha256_bytes(np.asarray(target_row, dtype="<f4").tobytes()),
        "target_metadata_sha256": metadata_sha,
        "target_expected_file_sha256": target_contract["sha256"],
        "target_row_groups_sha256": target_contract["row_groups_sha256"],
        "target_whole_file_sha256_recomputed": False,
        "norm_mean": mean,
        "norm_std": std,
        "norm_mean_sha256": tensor_sha256(mean),
        "norm_std_sha256": tensor_sha256(std),
    }


def run_update(inputs: dict[str, Any], hidden: int, layers: int, seed: int,
               chunk: int, device: torch.device) -> dict[str, Any]:
    """Perform one small FP32 diagnostic update and report finite/resource evidence."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = RNNModel(n_in=211, hidden=hidden, layers=layers, dropout=0.0, n_out=2,
                     cell="gru", enc="linear", readout_activation="gelu").to(device=device, dtype=torch.float32)
    model.mean.copy_(inputs["norm_mean"].to(device))
    model.std.copy_(inputs["norm_std"].to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    raw = torch.as_tensor(inputs["features"][:chunk], dtype=torch.float32, device=device).unsqueeze(0)
    labels = torch.as_tensor(inputs["labels"][:chunk], dtype=torch.float32, device=device).unsqueeze(0)
    teacher = torch.as_tensor(inputs["teacher"][:chunk], dtype=torch.float32, device=device).unsqueeze(0)
    event = event_features_torch(raw)
    model_inputs = torch.cat((raw, event), dim=-1)
    if model_inputs.shape != (1, chunk, 211):
        raise ValueError(f"profile input shape changed: {tuple(model_inputs.shape)}")
    target = 0.1 * labels.clamp(-2, 2) + 0.9 * teacher.clamp(-2, 2)
    valid = torch.arange(chunk, device=device).ge(WARMUP).unsqueeze(0)
    # This shape probe keeps the clean trainer's label-magnitude WMSE and warmup
    # support; it intentionally omits propensity remapping and auxiliary losses.
    weight = labels.clamp(-2, 2).abs() * valid.unsqueeze(-1)
    if not weight.sum().isfinite() or weight.sum() <= 0:
        raise ValueError("profile sequence has no positive label weight after warmup")

    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    prediction, _state = model(model_inputs, None)
    prediction = prediction[..., :2]
    loss = (weight * (prediction.float() - target) ** 2).sum() / weight.sum().clamp_min(1e-8)
    if not torch.isfinite(loss):
        raise FloatingPointError("profile loss is non-finite")
    loss.backward()
    grad_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                      for parameter in model.parameters())
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if not grad_finite or not torch.isfinite(grad_norm):
        raise FloatingPointError("profile gradients are non-finite before optimizer update")
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.step()
    updated = any(not torch.equal(old, parameter.detach())
                  for old, parameter in zip(before, model.parameters(), strict=True))
    param_finite = all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
    torch.cuda.synchronize(device)
    if not updated or not param_finite:
        raise FloatingPointError("profile optimizer did not produce a finite parameter update")
    return {
        "loss_finite": True,
        "loss_value_diagnostic_only": float(loss.detach().cpu()),
        "gradient_finite_before_clip": grad_finite,
        "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
        "parameters_finite_after_update": param_finite,
        "parameter_update_observed": updated,
        "optimizer": "torch.optim.AdamW",
        "optimizer_attempts": 1,
        "optimizer_completed_updates": 1,
        "precision": "float32",
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "cuda_memory_allocated_bytes_after_update": int(torch.cuda.memory_allocated(device)),
        "cuda_free_bytes_after_update": int(torch.cuda.mem_get_info(device)[0]),
    }


def parse_args() -> argparse.Namespace:
    """Parse the deliberately narrow profile interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--hidden", type=int, required=True, choices=sorted({h for h, _ in SHAPES}))
    parser.add_argument("--layers", type=int, required=True,
                        choices=sorted({layer_count for _, layer_count in SHAPES}))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--execute", action="store_true", help="perform one finite FP32 update on CUDA")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Run contract-checked one-sequence setup and optionally one CUDA update."""
    args = parse_args()
    if (args.hidden, args.layers) not in SHAPES:
        raise SystemExit("shape is outside the explicit eight-shape architecture ladder")
    if args.seed != 3:
        raise SystemExit("this initial F0 profile is frozen to seed 3")
    contract_raw = args.contract.read_bytes()
    if len(contract_raw) > 64_000:
        raise ValueError("profile contract exceeds its 64 KB bound")
    contract = json.loads(contract_raw)
    if contract.get("format") != "wnn.student-frontier-shape-profile-contract" or contract.get("version") != 1:
        raise ValueError("unsupported student-frontier shape profile contract")
    if (contract.get("shape") != [args.hidden, args.layers]
            or contract.get("seed") != args.seed
            or contract.get("batch_size") != 1
            or contract.get("chunk_rows") != 2000
            or contract.get("precision") != "float32"):
        raise ValueError("CLI shape or update settings differ from the frozen profile contract")
    repo = Path(__file__).resolve().parents[1]
    source_hashes = source_manifest(Path(__file__).resolve(), repo)
    expected_source_hashes = contract.get("source_sha256")
    if expected_source_hashes != source_hashes:
        raise ValueError("profile source hashes differ from the frozen contract")
    t0 = time.perf_counter()
    loaded = load_profile_inputs(args.data_root.resolve(), contract, args.seed)
    setup_seconds = time.perf_counter() - t0
    norm_hashes = {"mean": loaded["norm_mean_sha256"], "std": loaded["norm_std_sha256"]}
    if norm_hashes != contract.get("normalization_sha256"):
        raise ValueError("shared H256 normalization-buffer hashes differ from contract")

    torch.set_num_threads(1)
    result: dict[str, Any] = {
        "format": "wnn.student-frontier-one-sequence-profile-result",
        "version": 1,
        "profile_contract_sha256": sha256_bytes(contract_raw),
        "profile_script_sha256": source_hashes["scripts/profile_student_frontier.py"],
        "source_sha256": source_hashes,
        "data_root": str(args.data_root.resolve()),
        "cache_metadata_sha256": loaded["target_metadata_sha256"],
        "target_file_expected_sha256": loaded["target_expected_file_sha256"],
        "target_whole_file_sha256_recomputed": loaded["target_whole_file_sha256_recomputed"],
        "split_manifest_sha256": loaded["split"].sha256,
        "source_split": loaded["source_split"],
        "sampled_row_group": loaded["group"],
        "sampled_seq_ix": loaded["seq_ix"],
        "first_seeded_batch_row_groups": loaded["first_batch_groups"],
        "train_order_sha256": loaded["train_order_sha256"],
        "first_batch_order_sha256": loaded["first_batch_order_sha256"],
        "sample_payload_sha256": {
            key: loaded[key] for key in
            ("feature_row_sha256", "label_row_sha256", "scored_row_sha256", "teacher_row_sha256")
        },
        "scored_mask_present_in_source": loaded["scored_mask_present_in_source"],
        "shape": [args.hidden, args.layers],
        "features": 211,
        "seed": args.seed,
        "precision": "float32",
        "batch_size": 1,
        "chunk_rows": 2000,
        "rows_after_warmup": 2000 - WARMUP,
        "normalization_source": contract["normalization_source"],
        "normalization_sha256": norm_hashes,
        "setup_seconds": setup_seconds,
        "host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "host_cgroup_memory_current_bytes": cgroup_memory_current(),
        "torch_version": torch.__version__,
        "cuda_profile_requested": args.execute,
        "quality_claim": None,
        "interpretation": "one-sequence shape/resource diagnostic only; not a fit, score, checkpoint, or architecture selection result",
    }
    if args.execute:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profile requested but no CUDA device is available")
        from wnn.resources import configure_torch

        if os.environ.get("WNN_GPU_MEMORY_BYTES") != str(CUDA_ALLOCATOR_CAP_BYTES):
            raise RuntimeError("the admitted profile must set WNN_GPU_MEMORY_BYTES=3500000000")
        configure_torch()
        device = torch.device("cuda")
        nvidia_start = nvidia_smi_used_bytes()
        if nvidia_start > OBSERVED_VRAM_LIMIT_BYTES:
            raise RuntimeError(f"initial observed VRAM exceeds the {OBSERVED_VRAM_LIMIT_BYTES} byte ceiling")
        result["cuda_device"] = torch.cuda.get_device_name(device)
        result["cuda_total_memory_bytes"] = int(torch.cuda.get_device_properties(device).total_memory)
        result["nvidia_smi_used_before_update_bytes"] = nvidia_start
        step_start = time.perf_counter()
        result["update"] = run_update(loaded, args.hidden, args.layers, args.seed,
                                      int(contract["chunk_rows"]), device)
        result["update_seconds"] = time.perf_counter() - step_start
        nvidia_after = nvidia_smi_used_bytes()
        result["nvidia_smi_used_after_update_bytes"] = nvidia_after
        result["nvidia_smi_max_of_start_and_after_bytes"] = max(nvidia_start, nvidia_after)
        if result["update"]["cuda_max_reserved_bytes"] > CUDA_ALLOCATOR_CAP_BYTES:
            raise RuntimeError("PyTorch CUDA allocator exceeded the explicitly configured byte cap")
        if nvidia_after > OBSERVED_VRAM_LIMIT_BYTES:
            raise RuntimeError(f"post-update observed VRAM exceeds the {OBSERVED_VRAM_LIMIT_BYTES} byte ceiling")
        result["host_max_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        result["host_cgroup_memory_current_bytes"] = cgroup_memory_current()
    else:
        result["update"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"profile_result": str(args.output), "shape": result["shape"],
                      "sampled_row_group": result["sampled_row_group"],
                      "update_completed": result["update"] is not None}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
