"""Validate source-bound fit, checkpoint, and full-fold prediction artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grade_prediction_bank import aligned, sha256  # noqa: E402

from wnn.kalman_group_features import FEATURE_GROUPS  # noqa: E402
from wnn.splits import file_sha256  # noqa: E402

SPLITS = {
    0: ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json",
    2: ROOT / "research/campaign_2026-09-29/artifacts/splits/fold2.json",
}
SUBSETS = {
    0: ROOT / "research/campaign_2026-10-20/plateau_review_r1/fit_f0_10pct_s1001.json",
    2: ROOT / "research/campaign_2026-10-21/next_cycle_r1/i1/fit_f2_10pct_s1002.json",
}
SEEDS = {0: 1001, 2: 1002}
ROWS = 20_000


def _npy_header(path: Path) -> dict[str, int]:
    """Parse an NPY header and require an exact, uncompressed payload length."""
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        readers = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
            (3, 0): np.lib.format.read_array_header_2_0,
        }
        if version not in readers:
            raise ValueError(f"unsupported NPY version {version}")
        shape, fortran_order, dtype = readers[version](stream)
        header_bytes = stream.tell()
    payload_bytes = math.prod(shape) * np.dtype(dtype).itemsize
    file_bytes = path.stat().st_size
    if fortran_order or file_bytes != header_bytes + payload_bytes:
        raise ValueError("prediction NPY layout or payload length differs from its header")
    return {"header_bytes": header_bytes, "payload_bytes": payload_bytes, "file_bytes": file_bytes}


def _target_normalization_identity(metadata: dict[str, Any]) -> str:
    """Hash the exact stored target means and population standard deviations."""
    payload = json.dumps(
        {"mean": metadata["target_mean"], "std": metadata["target_std"]},
        allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _verify_target_metadata(fold: int, metadata: Any, subset: dict[str, Any]) -> dict[str, Any]:
    """Bind target statistics to the exact fit-only groups and source implementation."""
    if not isinstance(metadata, dict):
        raise ValueError(f"F{fold}: dynamics target-normalization provenance is absent")
    selected = subset["selected_groups"]
    members = metadata.get("fit_members", {})
    if members.get("train") != selected["train"] or members.get("valid") != selected["valid"]:
        raise ValueError(f"F{fold}: target statistics use different fit groups")
    ordered = sorted(
        [("train", int(group)) for group in selected["train"]]
        + [("valid", int(group)) for group in selected["valid"]]
    )
    expected_member_sha = hashlib.sha256(
        json.dumps(ordered, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if metadata.get("fit_members_sha256") != expected_member_sha:
        raise ValueError(f"F{fold}: target-normalization member hash differs")
    groups = metadata.get("groups", [])
    canonical_groups = [
        {"name": name, "start": start, "stop": stop} for name, start, stop in FEATURE_GROUPS
    ]
    cursor = 0
    for group in groups:
        if group.get("start") != cursor or group.get("stop", cursor) <= cursor:
            raise ValueError(f"F{fold}: target group registry is not an ordered partition")
        cursor = group["stop"]
    if (
        groups != canonical_groups
        or len(groups) != 13
        or cursor != 112
        or metadata.get("horizons") != [16, 256]
        or metadata.get("groups_source") != "wnn.kalman_group_features.FEATURE_GROUPS"
        or metadata.get("target_order")
        != "per horizon, then registry group, mean standardized change followed by log mean squared standardized change"
        or metadata.get("change_definition")
        != "z[t+k]-z[t+k-1] for k=1..h; z=(raw-model.mean[:112])/model.std[:112]"
        or metadata.get("boundary")
        != "target state t is valid when t >= 99 and t+h < sequence length; raw fit reads are complete, unthinned sequences"
        or metadata.get("target_normalization")
        != "per-output fitting-membership mean and population standard deviation, floored at 1e-3"
        or metadata.get("input_normalization")
        != "existing fitting-only model mean/std, raw channels 0:112"
        or metadata.get("fit_reader")
        != "SequenceFile.read(targets=False, mask=False); training targets and scoring masks are not loaded"
        or metadata.get("module_source_sha256")
        != file_sha256(ROOT / "src/wnn/training/dynamics_auxiliary.py")
    ):
        raise ValueError(
            f"F{fold}: target-normalization metadata differs from the frozen mechanism"
        )
    means, deviations = metadata.get("target_mean"), metadata.get("target_std")
    counts = metadata.get("target_counts")
    if not (
        isinstance(means, list)
        and isinstance(deviations, list)
        and isinstance(counts, list)
        and len(means) == len(deviations) == len(counts) == 52
        and all(math.isfinite(value) for value in means + deviations)
        and all(value >= 1e-3 for value in deviations)
    ):
        raise ValueError(f"F{fold}: target statistics are missing, nonfinite, or malformed")
    sequence_count = len(selected["train"]) + len(selected["valid"])
    expected_counts = [19_885 * sequence_count] * 26 + [19_645 * sequence_count] * 26
    if counts != expected_counts:
        raise ValueError(f"F{fold}: fit-only target counts do not cover complete sequences")
    return {
        "fit_members_sha256": expected_member_sha,
        "target_statistics_sha256": _target_normalization_identity(metadata),
        "target_counts": counts,
        "module_source_sha256": metadata["module_source_sha256"],
    }


def _checkpoint_head_identity(path: Path, expected_normalization: dict[str, Any]) -> dict[str, str]:
    """Hash the saved training head and verify its buffers match provenance."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model")
    metadata = checkpoint.get("provenance", {}).get("dynamics_auxiliary")
    if not isinstance(state, Mapping) or not isinstance(metadata, dict):
        raise ValueError(f"{path}: checkpoint lacks the auxiliary head or target statistics")
    head = {name: value for name, value in state.items() if name.startswith("dynamics_aux_head.")}
    if set(head) != {"dynamics_aux_head.weight", "dynamics_aux_head.bias"}:
        raise ValueError(f"{path}: checkpoint auxiliary head parameter names differ")
    if tuple(head["dynamics_aux_head.weight"].shape) != (52, 512) or tuple(
        head["dynamics_aux_head.bias"].shape
    ) != (52,):
        raise ValueError(f"{path}: checkpoint auxiliary head shape differs")
    head_hash = hashlib.sha256()
    for name, tensor in sorted(head.items()):
        value = tensor.detach().cpu().contiguous()
        head_hash.update(name.encode("ascii"))
        head_hash.update(str(tuple(value.shape)).encode("ascii"))
        head_hash.update(value.numpy().tobytes())
    target_hash = _target_normalization_identity(metadata)
    if target_hash != expected_normalization["target_statistics_sha256"]:
        raise ValueError(f"{path}: checkpoint target buffers and provenance differ")
    for key, field in (
        ("dynamics_target_mean", "target_mean"),
        ("dynamics_target_std", "target_std"),
    ):
        value = state.get(key)
        if value is None or not np.array_equal(
            value.detach().cpu().numpy(), np.asarray(metadata[field], dtype=np.float32)
        ):
            raise ValueError(f"{path}: checkpoint {key} differs from target provenance")
    return {"trained_head_sha256": head_hash.hexdigest(), "target_statistics_sha256": target_hash}


def _verify_run(
    fold: int, arm: str, run: str, groups: list[int], split_sha: str, subset: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate fit metadata, matched initialization, checkpoint head, and full-fold export."""
    run_dir = ROOT / "artifacts/runs" / run
    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    args = provenance.get("args", {})
    weight = 0.0 if arm == "control" else 0.05
    updates_per_epoch = {0: 1_540, 2: 1_610}[fold]
    diagnostics = provenance.get("update_diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != 3:
        raise ValueError(f"{run}: expected exactly three epoch update records")
    for index, record in enumerate(diagnostics):
        epoch = record.get("epoch", {})
        gradient_stats = record.get("grad_stats", {})
        if (
            record.get("epoch_index") != index
            or epoch.get("optimizer_attempts") != updates_per_epoch
            or epoch.get("optimizer_completed_updates") != updates_per_epoch
            or epoch.get("amp_skipped_updates") != 0
            or gradient_stats.get("steps") != updates_per_epoch
            or gradient_stats.get("nonfinite_steps") != 0
        ):
            raise ValueError(
                f"{run}: epoch {index} update evidence differs from the fixed schedule"
            )
    cumulative = diagnostics[-1].get("cumulative", {})
    if (
        cumulative.get("optimizer_attempts") != 3 * updates_per_epoch
        or cumulative.get("optimizer_completed_updates") != 3 * updates_per_epoch
        or cumulative.get("amp_skipped_updates") != 0
    ):
        raise ValueError(f"{run}: full-fit update evidence differs from the fixed schedule")
    expected_provenance = {
        "seed": SEEDS[fold],
        "split_manifest_sha256": split_sha,
        "fit_subset_manifest_sha256": file_sha256(SUBSETS[fold]),
        "fit_subset_members_sha256": subset["selected_members_sha256"],
        "fit_row_groups": subset["selected_groups"],
        "selected_epoch": 2,
    }
    expected_args = {
        "name": run,
        "seed": SEEDS[fold],
        "arch": "gru",
        "hidden": 512,
        "layers": 1,
        "head_layers": 1,
        "dropout": 0.0,
        "batch": 8,
        "chunk": 2000,
        "epochs": 3,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 2,
        "no_eval_during_fit": True,
        "loss": "wmse",
        "loss_norm": "chunk",
        "optim": "adamw",
        "sched": "onecycle",
        "warmup_frac": 0.05,
        "lr": 0.001,
        "wd": 0.01,
        "beta2": 0.999,
        "clip": 1.0,
        "precision": "fp32",
        "research_features": "i1_history",
        "research_feature_control": "all",
        "drop_i1": False,
        "dynamics_aux_head": True,
        "dynamics_aux_weight": weight,
        "fit_subset_experiment_id": subset["experiment_id"],
        "split_manifest": SPLITS[fold].as_posix(),
        "fit_subset_manifest": SUBSETS[fold].as_posix(),
        "save_training_state": True,
        "grad_stats": True,
        "deterministic": True,
        "batch_order": "seeded",
        "train_source": "train+valid",
        "train_seqs": 10_607,
        "valid_repeat": 1,
    }
    if (
        any(provenance.get(key) != value for key, value in expected_provenance.items())
        or not provenance.get("initial_parameter_sha256")
        or not provenance.get("normalization_identity")
        or any(args.get(key) != value for key, value in expected_args.items())
        or any(args.get(key) for key in ("init", "distill", "distill_cache_root", "propensity"))
    ):
        raise ValueError(f"{run}: fit provenance differs from the frozen screen")

    target_identity = _verify_target_metadata(fold, provenance.get("dynamics_auxiliary"), subset)
    checkpoint = run_dir / "model_ep2.pt"
    head_identity = _checkpoint_head_identity(checkpoint, target_identity)
    prediction_path = ROOT / "cache" / f"pred_{run}_valid.npy"
    sidecar_path = prediction_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected_groups_sha = hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()
    expected_sidecar = {
        "run": run,
        "split": "valid",
        "epoch": 2,
        "checkpoint_sha256": file_sha256(checkpoint),
        "split_manifest_sha256": split_sha,
        "campaign_split_manifest_sha256": split_sha,
        "campaign_export_scope": "evaluation",
        "feature_only": True,
        "inference_precision": "fp32",
        "row_groups": groups,
        "evaluation_membership_sha256": expected_groups_sha,
        "prediction_sha256": file_sha256(prediction_path),
    }
    if any(sidecar.get(key) != value for key, value in expected_sidecar.items()):
        raise ValueError(f"{run}: prediction sidecar does not bind the full fixed evaluation fold")
    header = _npy_header(prediction_path)
    array = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if array.shape != (len(groups), ROWS, 2) or array.dtype != np.float32:
        raise ValueError(f"{run}: prediction NPY shape or dtype differs")
    aligned_result = aligned(
        {"prediction": prediction_path.relative_to(ROOT).as_posix()}, groups, split_sha
    )
    if isinstance(aligned_result, str):
        raise ValueError(f"{run}: {aligned_result}")
    _, order = aligned_result
    if order != list(range(len(groups))):
        raise ValueError(f"{run}: evaluation row order is not exact")
    identity = {
        "run": run,
        "checkpoint_sha256": sidecar["checkpoint_sha256"],
        "training_provenance_sha256": sha256(provenance_path),
        "update_diagnostics": diagnostics,
        "prediction_sha256": sidecar["prediction_sha256"],
        "sidecar_sha256": sha256(sidecar_path),
        "fit_subset_sha256": file_sha256(SUBSETS[fold]),
        "fit_subset_members_sha256": subset["selected_members_sha256"],
        "initial_parameter_sha256": provenance["initial_parameter_sha256"],
        "normalization_identity": provenance["normalization_identity"],
        **target_identity,
        **head_identity,
        "evaluation_membership_sha256": expected_groups_sha,
        "prediction_shape": list(array.shape),
        "prediction_dtype": str(array.dtype),
        **header,
    }
    return array, identity
