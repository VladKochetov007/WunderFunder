"""Stream two selected GRU teachers directly into one transient W18 target cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gru_checkpoint_io import load_gru_models

from wnn.data import SEQUENCE_LENGTH, SequenceFile
from wnn.distillation_cache import (
    SEQUENCE_ROWS,
    file_sha256,
    row_groups_sha256,
    validate_distillation_cache,
)
from wnn.resources import configure_strict_fp32, configure_torch
from wnn.splits import load_campaign_split
from wnn.training.environment import TrainingEnvironment
from wnn.training.forward import forward_chunk
from wnn.training.inputs import build_inputs
from wnn.training.prediction_inputs import configure_prediction_inputs

GPU_LIMIT = 12 * 1024**3
CHUNK = 1000


def _array_record(path: Path, row_groups: list[int], array: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": "float32",
        "row_groups": row_groups,
        "row_groups_sha256": row_groups_sha256(row_groups),
    }


def _normalizer_sha(model: torch.nn.Module) -> str:
    if not hasattr(model, "mean") or not hasattr(model, "std"):
        raise ValueError("selected GRU checkpoint lacks fitted normalization buffers")
    mean, std = model.mean.detach().cpu().contiguous(), model.std.detach().cpu().contiguous()
    if (
        mean.shape != std.shape
        or mean.ndim != 1
        or mean.shape != (211,)
        or not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
    ):
        raise ValueError("checkpoint normalization buffers are invalid")
    if bool((std <= 0).any()):
        raise ValueError("checkpoint normalization scales must be positive")
    digest = hashlib.sha256()
    for name, value in (("mean", mean), ("std", std)):
        digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes() + value.numpy().tobytes())
    return digest.hexdigest()


def _predict_one_teacher(
    record: dict[str, Any],
    groups: list[int],
    data: SequenceFile,
    project: Path,
    source: Path,
    target: np.ndarray,
    teacher_index: int,
    batch_size: int,
    device: torch.device,
) -> None:
    model = record["model"]
    model.eval()
    environment = TrainingEnvironment(root=project, source_root=source)
    configure_prediction_inputs(environment, record["args"])
    for first in range(0, len(groups), batch_size):
        batch_groups = groups[first : first + batch_size]
        sequences = [data.read(group, targets=False, mask=False) for group in batch_groups]
        raw = np.stack([sequence.features for sequence in sequences])
        x = torch.from_numpy(raw).to(device=device, dtype=torch.float32)
        inputs = build_inputs(x, environment=environment)
        del x, raw, sequences
        if inputs.shape != (len(batch_groups), SEQUENCE_LENGTH, 211):
            raise ValueError(f"teacher inputs have unexpected shape {tuple(inputs.shape)}")
        state = None
        for start in range(0, SEQUENCE_LENGTH, CHUNK):
            stop = min(start + CHUNK, SEQUENCE_LENGTH)
            with torch.inference_mode():
                output, state = forward_chunk(model, inputs, slice(start, stop), state)
            prediction = output[..., :2].float().clamp(-2.0, 2.0).cpu().numpy()
            view = target[first : first + len(batch_groups), start:stop]
            if teacher_index == 0:
                np.copyto(view, prediction)
            else:
                np.add(view, prediction, out=view)
        del inputs
    target.flush()


def _cache_metadata(
    old: dict[str, Any],
    old_cache: Path,
    partial: Path,
    target_path: Path,
    target_array: np.ndarray,
    groups: list[int],
    teachers: list[dict[str, Any]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for key, entry in old["files"].items():
        if key == "main_source_train":
            files[key] = _array_record(target_path, groups, target_array)
            continue
        source_path = old_cache / entry["path"]
        linked = partial / entry["path"]
        if source_path.stat().st_dev != partial.stat().st_dev:
            raise OSError("old cache and transient target must share a filesystem for hardlinks")
        os.link(source_path, linked)
        files[key] = dict(entry)
    metadata = dict(old)
    metadata.update(
        {
            "cache_profile": "l4_objective_two_gru_direct_target_v1",
            "estimated_new_cache_bytes": target_path.stat().st_size,
            "prediction_banks_persisted": False,
            "source_coefficients_scope": "unchanged qualified PR60 base sources; new family is applied to the target separately",
            "base_cache": {
                "path": str(old_cache),
                "metadata_sha256": validation["metadata_sha256"],
                "main_source_train_sha256": old["files"]["main_source_train"]["sha256"],
                "families_train_sha256": old["files"]["families_train"]["sha256"],
            },
            "target_contract": {
                "formula": "float32(0.8*T_old + 0.2*0.8125*mean(float32_clip(each_new_teacher,-2,2)))",
                "family": "selected_gru_pair",
                "teacher_runs_equal_weight": [teacher["run"] for teacher in teachers],
                "teacher_epoch_zero_based": 3,
                "teacher_mean": "ordered float32 clipped sum divided by two",
                "clipping": {"lower": -2.0, "upper": 2.0, "applied_per_teacher_before_mean": True},
                "base_weight": 0.8,
                "new_family_weight": 0.2,
                "new_family_source_scale": 0.8125,
                "fit_groups_sha256": row_groups_sha256(groups),
                "teachers": teachers,
                "prediction_banks_persisted": False,
            },
            "retained_auxiliary_targets": {
                "source_cache": str(old_cache),
                "family_order": old["family_order"],
                "train_sha256": old["files"]["families_train"]["sha256"],
                "train_hardlink_reused_without_writes": True,
                "valid_sha256": old["files"]["families_valid"]["sha256"],
            },
            "transient_target_lifecycle": {
                "status": "transient_for_two_student_comparison",
                "cleanup_after": "both_student_grades_and_required_weights_and_metadata_are_retained",
            },
            "files": files,
        }
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--cell-id", action="append", required=True)
    parser.add_argument("--old-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if len(args.cell_id) != 2 or args.batch_size < 1 or args.batch_size > 32:
        parser.error("supply exactly two --cell-id values and --batch-size in [1,32]")
    output = args.output_cache.absolute()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("output or partial exists; preserve it and choose a fresh path")
    panel_path, old_cache = args.panel.resolve(strict=True), args.old_cache.resolve(strict=True)
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    source_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(source_root))
    from scripts.l4_teacher_capacity_20261104.stage2_stream_common import (
        PR60_CACHE_METADATA_SHA256,
        PR60_FAMILIES_TRAIN_SHA256,
        PR60_MAIN_TRAIN_SHA256,
        TARGET_BYTES,
        TRAIN_SEQUENCES,
        combine_target_chunk,
    )

    project = Path(panel["paths"]["project_root"])
    source = Path(panel["paths"]["source_root"])
    split = load_campaign_split(Path(panel["paths"]["split_manifest"]), project)
    groups = list(map(int, split.train_groups))
    if len(groups) != TRAIN_SEQUENCES or len(split.evaluation_groups) != 1873:
        raise ValueError("official chronological membership counts differ")
    validation = validate_distillation_cache(
        old_cache, project, split, "source", require_families=True
    )
    old = validation["metadata"]
    if (
        validation["metadata_sha256"] != PR60_CACHE_METADATA_SHA256
        or old["files"]["main_source_train"]["sha256"] != PR60_MAIN_TRAIN_SHA256
        or old["files"]["families_train"]["sha256"] != PR60_FAMILIES_TRAIN_SHA256
        or old["files"]["main_source_train"]["row_groups"] != groups
    ):
        raise ValueError("input is not the qualified PR60 target cache in official fit order")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free - TARGET_BYTES < 3_000_000_000:
        raise OSError("direct target output would breach the 3 GB free-space floor")
    if old_cache.stat().st_dev != output.parent.stat().st_dev:
        raise OSError("target output must share the old-cache filesystem for family hardlinks")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("selected GRU target export requires one CUDA device")
    torch.cuda.set_device(device)
    os.environ["WNN_THREADS"] = "8"
    cap = min(GPU_LIMIT, int(os.environ.get("WNN_GPU_MEMORY_BYTES", str(GPU_LIMIT))))
    os.environ["WNN_GPU_MEMORY_BYTES"] = str(cap)
    configure_torch()
    configure_strict_fp32()
    torch.set_num_threads(8)
    teachers, _ = load_gru_models(panel, split, device, args.cell_id, (3,))
    if [teacher["seed"] for teacher in teachers] != [1001, 1002]:
        raise ValueError("the two teachers must be the matched seed1001/seed1002 pair")
    if any(
        teacher["args"].get("hidden") != 256 or teacher["args"].get("layers") != 2
        for teacher in teachers
    ):
        raise ValueError("selected teachers must use the frozen GRU256x2 architecture")
    teacher_records = [
        {
            "cell_id": t["cell_id"],
            "run": t["run_name"],
            "seed": t["seed"],
            "epoch": t["epoch"],
            "checkpoint": t["checkpoint"],
            "checkpoint_sha256": t["checkpoint_sha256"],
            "normalization_buffers_sha256": _normalizer_sha(t["model"]),
            "teacher_panel_sha256": file_sha256(panel_path),
            "source_binding_sha256": panel["identity"]["source_binding_sha256"],
            "data_binding_sha256": panel["identity"]["data_binding_sha256"],
        }
        for t in teachers
    ]
    partial.mkdir(parents=True)
    target_path = partial / "main_source_train.npy"
    target = np.lib.format.open_memmap(
        target_path, mode="w+", dtype=np.float32, shape=(TRAIN_SEQUENCES, SEQUENCE_ROWS, 2)
    )
    target.fill(0.0)
    data = SequenceFile(project / "datasets" / "train.parquet")
    for index, teacher in enumerate(teachers):
        _predict_one_teacher(
            teacher, groups, data, project, source, target, index, args.batch_size, device
        )
    old_main_path = old_cache / old["files"]["main_source_train"]["path"]
    old_main = np.load(old_main_path, mmap_mode="r", allow_pickle=False)
    for first in range(0, TRAIN_SEQUENCES, args.batch_size):
        stop = min(first + args.batch_size, TRAIN_SEQUENCES)
        for start in range(0, SEQUENCE_ROWS, CHUNK):
            end = min(start + CHUNK, SEQUENCE_ROWS)
            summed = np.array(target[first:stop, start:end], dtype=np.float32, copy=True)
            target[first:stop, start:end] = combine_target_chunk(
                np.asarray(old_main[first:stop, start:end], dtype=np.float32), summed, len(teachers)
            )
    target.flush()
    if target_path.stat().st_size != TARGET_BYTES:
        raise ValueError("direct target array allocation differs from the frozen byte budget")
    metadata = _cache_metadata(
        old, old_cache, partial, target_path, target, groups, teacher_records, validation
    )
    metadata_path = partial / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8"
    )
    if metadata_path.stat().st_size > 512_000:
        raise ValueError("compact target metadata exceeds the canonical 512 KB reader bound")
    del target, old_main, teachers
    validate_distillation_cache(partial, project, split, "source", require_families=True)
    partial.replace(output)
    print(
        json.dumps(
            {
                "status": "complete",
                "cache": str(output),
                "target_bytes": TARGET_BYTES,
                "metadata_sha256": file_sha256(output / "metadata.json"),
                "teacher_runs": [teacher["run"] for teacher in teacher_records],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
