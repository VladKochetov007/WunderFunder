"""Bind B03's independently scored GRU predictions to the eight declared E6 endpoints."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from runtime_source import PACKET, read, source_contract
from runtime_source import sha256 as digest

# Native JSON combines nested source, hardware and scoring records. Check before using each field.
Record = dict[str, Any]
BASE = "33542502a1404a2843359b3999fa34f3a157bf3f"
SPLIT = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
TEACHER = "759165147d99b33034b74f4af63a30e8a202dafc78fd5e5699d05b6cfe9bef90"
CACHE = "138ef43c6ef03755a3b6195ca83f2c346f7dca4e3fe365f9e8e151ad2c71792d"
SEEDS = (203301, 203302)
ARMS = {"labels": (0, 0.0), "since32": (32, 0.0), "kd": (0, 0.5), "both": (32, 0.5)}
METRICS = ("wp", "t0", "t1")
EPOCHS = 6
STEPS = 38_760
ROWS = 3_528_273


def canonical(value: object) -> str:
    """Match the pinned runtime's sorted JSON digest."""
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def qualification(root: Path, split_path: Path) -> tuple[Record, list, list]:
    """Verify the reproducible source tree, teacher, frozen cells and official membership."""
    q = read(root / "qualification.json")
    runtime = root / "runtime"
    manifest = read(PACKET / "runtime_manifest.json")
    source_contract(runtime, manifest)
    source = manifest["source_sha256"]
    if (
        q["schema"] != "wnn.b03.qualification.v1"
        or q["base_commit"] != BASE
        or q["source_sha256"] != source
        or digest(split_path) != SPLIT
        or q["split_sha256"] != SPLIT
        or q["teacher_sha256"] != TEACHER
        or q["cache_manifest_sha256"] != CACHE
        or set(q["configs"]) != {f"{arm}_s{seed}" for seed in SEEDS for arm in ARMS}
    ):
        raise ValueError("B03 source, split, teacher, or cell qualification differs")
    split = read(split_path)
    fitting = [["train", group] for group in split["fit"]["train"]]
    valid = [["valid", group] for group in split["evaluation"]["valid"]]
    if len(fitting) != 10_325 or split["fit"]["valid"] or len(valid) != 1_873:
        raise ValueError("unexpected source-qualified membership")
    return q, fitting, valid


def monitor(path: Path, valid: list) -> Record:
    """Read all six full-valid epoch observations, keeping E6 separate from earlier diagnostics."""
    if path.is_symlink() or not 0 < path.stat().st_size <= 4 << 20:
        raise ValueError("invalid full-validation monitor journal")
    events = [json.loads(line) for line in path.read_bytes().splitlines()]
    if len(events) != EPOCHS:
        raise ValueError("incomplete epoch monitoring")
    for epoch, event in enumerate(events, start=1):
        metrics, context = event["metrics"], event["context"]
        if (
            event["kind"] != "evaluation"
            or event["epoch"] != epoch
            or event["step"] != epoch * 6460
            or context["split_sha256"] != SPLIT
            or context["metric_id"] != "global_wp_clip2_v1"
            or context["evaluation_member_sha256"] != canonical(valid)
            or metrics["valid/sequence_count"] != len(valid)
            or metrics["valid/selected_rows"] != ROWS
            or not all(
                math.isfinite(metrics[key]) for key in ("valid/wp", "valid/wp/t0", "valid/wp/t1")
            )
        ):
            raise ValueError(f"monitor coverage or epoch disagrees at E{epoch}")
    metrics = events[-1]["metrics"]
    return dict(
        zip(
            METRICS,
            (metrics["valid/wp"], metrics["valid/wp/t0"], metrics["valid/wp/t1"]),
            strict=True,
        )
    )


def endpoint(
    root: Path, arm: str, seed: int, q: Record, fitting: list, valid: list
) -> tuple[Record, Record, str]:
    """Admit a single fixed E6 endpoint with the native profile, terminal, export and grade."""
    cell = f"{arm}_s{seed}"
    run = root / cell
    fit = run / "fit"
    manifest_path = run / "predictions/manifest.json"
    metadata = read(fit / "run_metadata.json")
    terminal = read(fit / "terminal_status.json")
    index = read(fit / "epoch_index.json")
    approved = read(run / "approved_profile.json")
    profile = read(run / "profile/profile_receipt.json")
    manifest = read(manifest_path)
    grade = read(run / "score.json")
    z = grade["producer"]
    provenance = z["provenance"]
    identity = terminal["identity"]
    cfg, data = z["config"], identity["data"]
    tau, alpha = ARMS[arm]
    hashes = {
        "model_sha256": digest(fit / "model.pt"),
        "config_sha256": canonical(cfg),
        "profile_sha256": digest(run / "profile/profile_receipt.json"),
        "approved_profile_sha256": digest(run / "approved_profile.json"),
        "predictions_manifest_sha256": digest(manifest_path),
        "score_sha256": digest(run / "score.json"),
        "monitor_sha256": digest(fit / "monitor_evaluations.jsonl"),
        "terminal_sha256": digest(fit / "terminal_status.json"),
    }
    model_hash = hashes["model_sha256"]
    source = metadata["source"]
    if (
        cfg != metadata["config"]
        or q["configs"][cell] != hashes["config_sha256"]
        or identity["config_sha256"] != hashes["config_sha256"]
        or cfg["seed"] != seed
        or (cfg["since_weight"], cfg["kd_weight"], cfg["since_floor"]) != (tau, alpha, 0.25)
        or cfg["objective_mode"] != "standalone_b03_convex_kd_wmse_v1"
        or cfg["precision_mode"] != "fp32"
        or cfg["epochs"] != EPOCHS
        or (cfg["model"]["architecture"], cfg["model"]["input_stem_kernel"]) != ("gru", 16)
        or cfg["required_world_size"] != 1
        or cfg["required_device"] != "L4"
        or source["sha256"] != q["source_sha256"]
        or identity["source_sha256"] != source["sha256"]
        or canonical(source["files"]) != source["sha256"]
        or source["files"] != identity["source_files"]
        or identity["world_size"] != 1
        or data["split_sha256"] != SPLIT
        or identity["data_sha256"] != canonical(data)
        or metadata["provenance"]["data"] != data
        or data["objective_contract"]["kind"] != "convex_teacher_target_wmse"
        or data["objective_contract"]["target_alpha"] != alpha
        or data["objective_contract"]["since_weight"] != tau
        or data["objective_contract"]["since_floor"] != 0.25
        or data["objective_contract"]["shared_denominator"] != "sum_abs_clipped_label_targets"
        or data["fit_members"] != fitting
        or data["evaluation_members"] != valid
        or data["teacher"] != z["teacher"]
        or z["teacher"]["manifest_sha256"] != CACHE
        or z["teacher"]["checkpoint_sha256"] != TEACHER
        or z["teacher"]["producer"]["checkpoint_sha256"] != TEACHER
        or z["teacher"]["producer"]["fit_members"] != fitting
        or z["teacher"]["producer"]["split_sha256"] != SPLIT
        or z["fit_members"] != fitting
        or "reference" in z
        or "kind" in z
        or z["split_sha256"] != SPLIT
    ):
        raise ValueError(f"{cell}: source, scientific factors, or teacher/data lineage differs")

    if (
        profile["identity"] != identity
        or approved["identity"] != identity
        or profile["status"] != "profile_complete"
        or approved["approved"] is not True
        or approved["profile_sha256"] != hashes["profile_sha256"]
        or provenance["profile_receipt"]["sha256"] != hashes["approved_profile_sha256"]
        or terminal["status"] != "completed"
        or terminal["completed_epochs"] != EPOCHS
        or terminal["global_updates"] != STEPS
        or terminal["model_sha256"] != model_hash
        or provenance["identity"] != identity
        or z["checkpoint_sha256"] != model_hash
        or provenance["checkpoint_purpose"] != "fixed_configured_final_endpoint"
        or provenance["completed_epochs"] != EPOCHS
        or provenance["global_updates"] != STEPS
        or index["identity"] != identity
        or index["completed_epochs"] != EPOCHS
        or index["global_step"] != STEPS
        or terminal["epoch_index_sha256"] != index["index_sha256"]
        or provenance["epoch_index_sha256"] != index["index_sha256"]
        or index["index_sha256"]
        != canonical({k: v for k, v in index.items() if k != "index_sha256"})
    ):
        raise ValueError(f"{cell}: fixed E6, profile or checkpoint binding differs")
    metrics = grade["score"]
    if (
        grade["split_sha256"] != SPLIT
        or grade["sequences"] != len(valid)
        or metrics["selected_rows"] != ROWS
        or metrics["metric_id"] != "global_wp_clip2_v1"
        or grade["predictions_manifest_sha256"] != hashes["predictions_manifest_sha256"]
        or manifest["schema"] != "sequence-predictions-v1"
        or manifest["membership"] != "evaluation"
        or manifest["complete"] is not True
        or manifest["split_sha256"] != SPLIT
        or manifest["input_schema"] != "raw112-v1"
        or manifest["sequence_length"] != 20000
        or manifest["hint_dim"] != 0
        or manifest["world_size"] != 1
        or manifest["producer"] != z
        or [[m["split"], m["row_group"]] for m in manifest["sequences"]] != valid
        or not all(math.isfinite(metrics[k]) for k in METRICS)
        or not math.isclose(
            metrics["wp"], (metrics["t0"] + metrics["t1"]) / 2, abs_tol=1e-10, rel_tol=0
        )
    ):
        raise ValueError(f"{cell}: canonical exported grade or evaluation coverage differs")
    monitoring = monitor(fit / "monitor_evaluations.jsonl", valid)
    drift = {k: metrics[k] - monitoring[k] for k in METRICS}
    # Full environment identity is bound separately to each fitted cell's approved profile.
    # Across cells compare precision, model, training protocol and data on the same GPU class.
    runtime = identity["runtime"]
    devices = runtime["devices"]
    if len(devices) != 1 or devices[0]["rank"] != 0 or "L4" not in devices[0]["name"]:
        raise ValueError(f"{cell}: missing single L4 rank")
    common = {
        "recipe": {k: v for k, v in cfg.items() if k not in ("since_weight", "kd_weight", "seed")},
        "data": {k: v for k, v in data.items() if k != "objective_contract"},
        "normalization": provenance["normalization"],
        "runtime": {
            k: runtime[k]
            for k in (
                "precision",
                "compiler",
                "attention_backend",
                "autocast",
                "amp_scaler",
                "matmul_precision",
                "matmul_tf32",
                "cudnn_tf32",
                "cudnn_rnn_fp32_precision",
                "nvidia_tf32_override",
                "torch_allow_tf32_cublas_override",
                "cublas_workspace_config",
            )
        },
        "device": {k: devices[0][k] for k in ("name", "capability", "total_memory_bytes")},
        "model_parameters": identity["model_parameters"],
        "optimizer": identity["optimizer"],
    }
    return (
        {
            "grade": {k: metrics[k] for k in METRICS},
            "monitor_e6": monitoring,
            "grade_minus_monitor": drift,
            "review_monitor_export_drift": any(abs(delta) > 1e-5 for delta in drift.values()),
            "environment": {
                k: runtime[k]
                for k in ("python", "platform", "torch", "cuda_runtime", "cudnn", "nvidia_driver")
            },
            **hashes,
        },
        grade["component_centered_moments"],
        canonical(common),
    )
