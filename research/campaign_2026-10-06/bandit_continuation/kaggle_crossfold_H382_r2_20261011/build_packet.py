"""Build the immutable H382 Kaggle bundle and its runnable metadata."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SNAPSHOT = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/"
    "h230_source_v8_arch"
)
SPLIT_REL = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
REGISTRY_REL = "research/campaign_2026-10-04/portable_inputs.json"
ASSET_MANIFEST_REL = "research/campaign_2026-10-04/kaggle/fold0_assets/assets-manifest.json"
PACKET_ID = "vladkochetov07/wnn-kaggle-crossfold-h382-20261011-r2"
BUNDLE_ID = "vladkochetov07/wnn-kaggle-crossfold-h382-bundle-20261011-r2"
KERNEL_TITLE = "WNN Kaggle Crossfold H382 20261011 R2"
KERNEL_SLUG = "wnn-kaggle-crossfold-h382-20261011-r2"
ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"
RAW_DATASET = "vladkochetov07/wunder-funder"
ASSET_SELECTION = {
    "artifacts/runs/c20260929_prop_f0/model.pt",
    "cache/chain_folds.npz",
    "cache/chains.npz",
    "cache/propcache_c20260929_prop_f0_train.json",
    "cache/propcache_c20260929_prop_f0_train.npy",
    "cache/propcache_c20260929_prop_f0_valid.json",
    "cache/propcache_c20260929_prop_f0_valid.npy",
    "cache/train_targets.npz",
    "cache/valid_targets.npz",
    SPLIT_REL,
    REGISTRY_REL,
}
JOBS = (
    ("kg_h382r2_f0_muonall_prop2_e3_s3", 2.0, 3, 2, "fold0 all-scope control"),
    ("kg_h382r2_f0_muonall_prop0_e3_s3", 0.0, 3, 2, "propensity-power contrast"),
    ("kg_h382r2_f0_muonall_prop0_e4_s3", 0.0, 4, 3, "longer fixed schedule contrast"),
)
EXPECTED_FIT_MEMBERSHIP = {
    "train": {
        "count": 10535,
        "ordered_int64_le_sha256": "98a3e8da248f52e307af61efaeab845a76cf8e89e005ea5630c0af9a133508a9",
    },
    "valid": {
        "count": 1405,
        "ordered_int64_le_sha256": "0ef790f8c27723b8f0ca1ebe045c2dcbebb4c4531a17952985802382c9568d75",
    },
    "valid_repeat": 1,
    "dataset_length": 11940,
}


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    """Encode stable JSON bytes for packet files."""
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def source_entry(path: Path, archive_path: str) -> tuple[str, bytes, dict[str, object]]:
    """Return a source payload and its source-manifest entry."""
    payload = path.read_bytes()
    return archive_path, payload, {
        "path": archive_path,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def validate_split_and_audit(split: dict[str, Any], audit: dict[str, Any]) -> None:
    """Check fold-0 membership against the historical recipe audit before packaging."""
    if split.get("fold") != 0 or split.get("n_folds") != 4:
        raise ValueError("H382 requires the frozen four-fold fold-0 manifest")
    manifest_fit = split["fit"]
    train_order = np.asarray(
        np.random.default_rng(3).permutation(np.asarray(manifest_fit["train"], dtype=np.int64)),
        dtype="<i8",
    )
    valid_order = np.asarray(manifest_fit["valid"], dtype="<i8")
    observed = {
        "train": train_order,
        "valid": valid_order,
        "valid_repeat": 1,
        "dataset_length": len(manifest_fit["train"]) + len(manifest_fit["valid"]),
    }
    for split_name in ("train", "valid"):
        groups = observed[split_name]
        observed[split_name] = {
            "count": len(groups),
            "ordered_int64_le_sha256": hashlib.sha256(groups.tobytes()).hexdigest(),
        }
    if observed != EXPECTED_FIT_MEMBERSHIP:
        raise ValueError(f"fold-0 fit membership changed: {observed}")
    old_control = audit["vt_f0_prop2_e3_f0"]["fitting_membership"]
    old_prop0 = audit["vt_f0_prop0_e3_f0"]["fitting_membership"]
    if old_control != EXPECTED_FIT_MEMBERSHIP or old_prop0 != EXPECTED_FIT_MEMBERSHIP:
        raise ValueError("fold-0 existing recipe audit does not bind expected fitting membership")
    for run, prop_power in (("vt_f0_prop2_e3_f0", 2.0), ("vt_f0_prop0_e3_f0", 0.0)):
        args = audit[run]["args"]
        if (
            args["muon_scope"] != "rnn"
            or args["seed"] != 3
            or args["valid_fold"] != 0
            or args["prop_power"] != prop_power
            or args["train_source"] != "train+valid"
            or args["epochs"] != 3
            or args["checkpoint_epoch"] != 2
        ):
            raise ValueError(f"historical fold-0 recipe changed: {run}")
    components = {
        (split_name, int(record["row_group"])): int(record["component"])
        for split_name, records in split["sequences"].items()
        for record in records
    }
    fit_components = {
        components[(split_name, int(group))]
        for split_name in ("train", "valid")
        for group in manifest_fit[split_name]
    }
    evaluation_components = {
        components[("valid", int(group))] for group in split["evaluation"]["valid"]
    }
    if fit_components & evaluation_components:
        raise ValueError("fold-0 component isolation is violated")
    if len(split["evaluation"]["valid"]) != 468:
        raise ValueError("fold-0 evaluation membership changed")


def source_payloads() -> tuple[list[tuple[str, bytes]], list[dict[str, object]]]:
    """Freeze the reviewed V8 trainer source plus the current V6 predictor helper."""
    payloads: list[tuple[str, bytes]] = []
    entries: list[dict[str, object]] = []
    helper_name = "scripts/predict_run.py"
    for folder in ("src", "scripts"):
        for path in sorted((SNAPSHOT / folder).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(SNAPSHOT).as_posix()
            if relative == helper_name:
                continue
            archive_path, payload, entry = source_entry(path, relative)
            payloads.append((archive_path, payload))
            entries.append(entry)
    predictor = ROOT / helper_name
    archive_path, payload, entry = source_entry(predictor, helper_name)
    payloads.append((archive_path, payload))
    entries.append(entry)
    entries.sort(key=lambda item: str(item["path"]))
    if not any(item["path"] == "scripts/train_rnn.py" for item in entries):
        raise ValueError("frozen V8 source is missing its trainer entry point")
    if not any(item["path"] == "src/wnn/optim.py" for item in entries):
        raise ValueError("frozen V8 source is missing the optimizer implementation")
    return payloads, entries


def fit_argv(name: str, epochs: int, checkpoint_epoch: int, prop_power: float) -> list[str]:
    """Make one exact fresh-fit command from the fold-0 control recipe."""
    args = [
        "--device", "cuda",
        "--arch", "gru",
        "--cell", "gru",
        "--enc", "linear",
        "--hidden", "256",
        "--layers", "2",
        "--dropout", "0",
        "--seed", "3",
        "--split-manifest", f"/root/wnn/{SPLIT_REL}",
        "--valid-fold", "0",
        "--eval-source", "vfold",
        "--train-source", "train+valid",
        "--train-seqs", "10607",
        "--epochs", str(epochs),
        "--checkpoint-policy", "fixed",
        "--checkpoint-epoch", str(checkpoint_epoch),
        "--no-eval-during-fit",
        "--batch", "32",
        "--chunk", "1000",
        "--loss", "wmse",
        "--loss-norm", "chunk",
        "--lr", "0.001",
        "--wd", "0.01",
        "--optim", "muon",
        "--muon-scope", "all",
        "--sched", "onecycle",
        "--precision", "fp16",
        "--propensity", "c20260929_prop_f0",
        "--prop-power", str(int(prop_power)),
        "--prop-eps", "0.03",
        "--valid-true-mask", "0.03",
        "--aux-struct", "0.3",
        "--aux-next", "0.3",
        "--since-weight", "32",
        "--fast",
        "--save-training-state",
    ]
    return ["python", "/root/wnn/scripts/train_rnn.py", "--name", name, *args]


def make_job_list() -> list[dict[str, object]]:
    """Bind the three preregistered comparisons to fit and export commands."""
    result: list[dict[str, object]] = []
    for name, prop_power, epochs, checkpoint_epoch, contrast in JOBS:
        result.append(
            {
                "name": name,
                "prop_power": prop_power,
                "epochs": epochs,
                "checkpoint_epoch": checkpoint_epoch,
                "contrast": contrast,
                "steps": [
                    {
                        "label": "fit",
                        "timeout": 5400,
                        "argv": fit_argv(name, epochs, checkpoint_epoch, prop_power),
                    },
                    {
                        "label": "export",
                        "timeout": 1800,
                        "argv": [
                            "python",
                            "/root/wnn/scripts/predict_run.py",
                            name,
                            "valid",
                            "--epoch",
                            str(checkpoint_epoch),
                        ],
                    },
                ],
            }
        )
    return result


def tar_add(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    """Add bytes with stable tar metadata."""
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(payload))


def build(force: bool = False) -> dict[str, object]:
    """Create all upload files and a hash-bound review manifest."""
    output = HERE / "dataset"
    kernel_dir = HERE / "kernel"
    output.mkdir(exist_ok=True)
    kernel_dir.mkdir(exist_ok=True)
    bundle_path = output / "bundle.wnn"
    paths = {
        "split": ROOT / SPLIT_REL,
        "registry": ROOT / REGISTRY_REL,
        "asset_manifest": ROOT / ASSET_MANIFEST_REL,
        "audit": HERE / "FOLD0_EXISTING_RECIPE_AUDIT_R1.json",
        "preregistration": HERE / "H382_PREREG_R1.md",
        "retry_note": HERE / "H382_R2_STARTUP_REPAIR.md",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required H382 input is missing ({name}): {path}")
    split = json.loads(paths["split"].read_text())
    registry = json.loads(paths["registry"].read_text())
    asset_manifest = json.loads(paths["asset_manifest"].read_text())
    audit = json.loads(paths["audit"].read_text())
    validate_split_and_audit(split, audit)
    if registry.get("split_manifest_sha256") != sha256_file(paths["split"]):
        raise ValueError("portable registry does not bind the packaged fold-0 split")
    if asset_manifest.get("dataset") != ASSET_DATASET:
        raise ValueError("fold-0 asset manifest names an unexpected private dataset")
    registry_sources = registry["sources"]
    for logical in ("datasets/train.parquet", "datasets/valid.parquet"):
        if logical not in registry_sources:
            raise ValueError(f"portable input registry lacks {logical}")
        if split["sources"][logical] != registry_sources[logical]["original_identity"]:
            raise ValueError(f"fold-0 split and portable registry disagree for {logical}")
    all_asset_items = {item["path"]: item for item in asset_manifest["files"]}
    if not ASSET_SELECTION.issubset(all_asset_items):
        raise ValueError(f"fold-0 assets omit required files: {sorted(ASSET_SELECTION - set(all_asset_items))}")
    asset_files = [all_asset_items[path] for path in sorted(ASSET_SELECTION)]
    for item in asset_files:
        if item["sha256"] not in item["filename"]:
            raise ValueError(f"asset is not named by its content digest: {item['path']}")
        source_record = split["sources"].get(item["path"])
        if source_record is not None and (
            source_record.get("bytes") != item["size"]
            or (
                source_record.get("sha256") is not None
                and source_record.get("sha256") != item["sha256"]
            )
        ):
            raise ValueError(f"fold-0 split and asset manifest disagree for {item['path']}")
        registry_record = registry_sources.get(item["path"])
        if registry_record is not None:
            identity = registry_record["original_identity"]
            if identity.get("bytes") != item["size"]:
                raise ValueError(f"portable registry and asset manifest disagree for {item['path']}")

    payloads, source_entries = source_payloads()
    payloads.extend(
        [
            (SPLIT_REL, paths["split"].read_bytes()),
            (REGISTRY_REL, paths["registry"].read_bytes()),
            ("h382_fold0_asset_manifest.json", paths["asset_manifest"].read_bytes()),
            ("h382_fold0_existing_recipe_audit.json", paths["audit"].read_bytes()),
            ("h382_preregistration.md", paths["preregistration"].read_bytes()),
            ("h382_r2_startup_repair.md", paths["retry_note"].read_bytes()),
        ]
    )
    source_entries.extend(
        [
            {
                "path": SPLIT_REL,
                "bytes": paths["split"].stat().st_size,
                "sha256": sha256_file(paths["split"]),
            },
            {
                "path": REGISTRY_REL,
                "bytes": paths["registry"].stat().st_size,
                "sha256": sha256_file(paths["registry"]),
            },
            {
                "path": "h382_fold0_asset_manifest.json",
                "bytes": paths["asset_manifest"].stat().st_size,
                "sha256": sha256_file(paths["asset_manifest"]),
            },
            {
                "path": "h382_fold0_existing_recipe_audit.json",
                "bytes": paths["audit"].stat().st_size,
                "sha256": sha256_file(paths["audit"]),
            },
            {
                "path": "h382_preregistration.md",
                "bytes": paths["preregistration"].stat().st_size,
                "sha256": sha256_file(paths["preregistration"]),
            },
            {
                "path": "h382_r2_startup_repair.md",
                "bytes": paths["retry_note"].stat().st_size,
                "sha256": sha256_file(paths["retry_note"]),
            },
        ]
    )
    source_entries.sort(key=lambda item: str(item["path"]))
    if len({item["path"] for item in source_entries}) != len(source_entries):
        raise ValueError("duplicate source path in H382 source manifest")
    source_manifest = {"schema": "wnn.h382-source-manifest.v1", "files": source_entries}
    source_manifest_bytes = canonical_json(source_manifest)
    source_manifest_sha = sha256_bytes(source_manifest_bytes)
    fit_membership = audit["vt_f0_prop2_e3_f0"]["fitting_membership"]
    split_sha = sha256_file(paths["split"])
    registry_sha = sha256_file(paths["registry"])
    asset_manifest_sha = sha256_file(paths["asset_manifest"])
    jobs = {
        "schema": "wnn.kaggle-crossfold-cohort.v1",
        "packet_id": PACKET_ID,
        "bundle_id": BUNDLE_ID,
        "source_manifest_sha256": source_manifest_sha,
        "source_bindings": [
            item for item in source_entries if item["path"] in ("scripts/train_rnn.py", "scripts/predict_run.py", "src/wnn/optim.py")
        ],
        "asset_manifest_sha256": asset_manifest_sha,
        "asset_manifest_path": "h382_fold0_asset_manifest.json",
        "asset_files": asset_files,
        "split_manifest_path": SPLIT_REL,
        "split_manifest_sha256": split_sha,
        "portable_registry_path": REGISTRY_REL,
        "portable_registry_sha256": registry_sha,
        "bundled_materialized_paths": [SPLIT_REL, REGISTRY_REL],
        "prop_model_sha256": all_asset_items["artifacts/runs/c20260929_prop_f0/model.pt"]["sha256"],
        "fit_membership": fit_membership,
        "env": {
            "WNN_PORTABLE_DATA_MANIFEST": f"/root/wnn/{REGISTRY_REL}",
            "WNN_LOADER_WORKERS": "1",
        },
        "slots": [{"gpu": 0, "threads": 4}, {"gpu": 1, "threads": 4}],
        "stagger_seconds": 30,
        "cohort_runtime_seconds": 14_400,
        "jobs": make_job_list(),
    }
    bundle_entries = payloads + [
        ("h382_source_manifest.json", source_manifest_bytes),
        ("kaggle_jobs.json", canonical_json(jobs)),
    ]
    if len({name for name, _ in bundle_entries}) != len(bundle_entries):
        raise ValueError("duplicate H382 bundle path")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    compressed = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=compressed, compresslevel=9, mtime=0) as stream,
        tarfile.open(fileobj=stream, mode="w") as archive,
    ):
        for name, payload in sorted(bundle_entries):
            tar_add(archive, name, payload)
    bundle_bytes = compressed.getvalue()
    if bundle_path.exists() and not force and bundle_path.read_bytes() != bundle_bytes:
        raise FileExistsError(f"existing H382 bundle differs; use --force after reviewing: {bundle_path}")
    bundle_path.write_bytes(bundle_bytes)
    bundle_sha = sha256_bytes(bundle_bytes)
    kernel_template = (kernel_dir / "template.py").read_text()
    kernel = kernel_template.replace("__BUNDLE_SHA256__", bundle_sha).replace(
        "__BUNDLE_BYTES__", str(len(bundle_bytes))
    )
    kernel_path = kernel_dir / "kernel.py"
    if kernel_path.exists() and not force and kernel_path.read_text() != kernel:
        raise FileExistsError(f"existing H382 kernel differs; use --force after reviewing: {kernel_path}")
    kernel_path.write_text(kernel)
    metadata = {
        "id": PACKET_ID,
        "title": KERNEL_TITLE,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [RAW_DATASET, ASSET_DATASET, BUNDLE_ID],
        "competition_sources": [],
        "kernel_sources": [],
    }
    dataset_metadata = {
        "title": "wnn kaggle crossfold h382 bundle 20261011 r2",
        "id": BUNDLE_ID,
        "licenses": [{"name": "CC0-1.0"}],
    }
    metadata_path = kernel_dir / "kernel-metadata.json"
    dataset_metadata_path = output / "dataset-metadata.json"
    metadata_path.write_bytes(canonical_json(metadata))
    dataset_metadata_path.write_bytes(canonical_json(dataset_metadata))
    packet = {
        "schema": "wnn.kaggle-h382-packet.v1",
        "attempt": "R2",
        "supersedes_attempt": "R1 startup failure only; no model fit or export began",
        "packet_id": PACKET_ID,
        "kernel_title": KERNEL_TITLE,
        "kernel_slug": KERNEL_SLUG,
        "bundle_id": BUNDLE_ID,
        "assets_dataset": ASSET_DATASET,
        "raw_dataset": RAW_DATASET,
        "source_snapshot": str(SNAPSHOT.relative_to(ROOT)),
        "source_snapshot_note_sha256": sha256_file(SNAPSHOT / "SNAPSHOT_NOTE.md"),
        "source_snapshot_manifest_sha256": sha256_file(
            SNAPSHOT / "INHERITED_V2_SNAPSHOT_MANIFEST_V1.json"
        ),
        "source_manifest_sha256": source_manifest_sha,
        "source_file_count": len(source_entries),
        "source_entrypoints": {
            "trainer": next(item for item in source_entries if item["path"] == "scripts/train_rnn.py"),
            "predictor": next(item for item in source_entries if item["path"] == "scripts/predict_run.py"),
            "optimizer": next(item for item in source_entries if item["path"] == "src/wnn/optim.py"),
        },
        "source_binding_note": (
            "V8 frozen trainer/tree used for the completed V6 same-flag GRU baseline; "
            "the current project predictor helper is packaged once to match V6's effective extraction."
        ),
        "bundle_sha256": bundle_sha,
        "bundle_bytes": len(bundle_bytes),
        "kernel_sha256": sha256_file(kernel_path),
        "kernel_bytes": kernel_path.stat().st_size,
        "kernel_metadata_sha256": sha256_file(metadata_path),
        "dataset_metadata_sha256": sha256_file(dataset_metadata_path),
        "split_manifest_path": SPLIT_REL,
        "split_manifest_sha256": split_sha,
        "portable_registry_path": REGISTRY_REL,
        "portable_registry_sha256": registry_sha,
        "asset_manifest_sha256": asset_manifest_sha,
        "existing_recipe_audit_path": "h382_fold0_existing_recipe_audit.json",
        "existing_recipe_audit_sha256": sha256_file(paths["audit"]),
        "preregistration_path": "h382_preregistration.md",
        "preregistration_sha256": sha256_file(paths["preregistration"]),
        "startup_repair_path": "h382_r2_startup_repair.md",
        "startup_repair_sha256": sha256_file(paths["retry_note"]),
        "fit_membership": fit_membership,
        "fit_sequence_count": fit_membership["dataset_length"],
        "full_valid_prediction_shape": [1873, 20_000, 2],
        "full_valid_prediction_dtype": "float32",
        "cohort": [
            {
                "name": name,
                "prop_power": prop_power,
                "epochs": epochs,
                "checkpoint_epoch": checkpoint_epoch,
                "contrast": contrast,
            }
            for name, prop_power, epochs, checkpoint_epoch, contrast in JOBS
        ],
        "recipe_note": (
            "Fresh fold-0 seed-3 GRU-256x2; Muon scope all; batch32/chunk1000 WMSE chunk; "
            "LR .001, WD .01, OneCycle, fp16; all other flags fixed to the V6 control. "
            "The e4 arm is a fresh four-epoch OneCycle run, not an epoch continuation."
        ),
        "optimizer_scope_caveat": (
            "Muon scope all routes eligible parameters through Muon; fallback parameters use AdamW "
            "with the shared LR. This is not a pure-Muon optimizer claim."
        ),
        "comparison_note": (
            "Historical fold-0 prop0/prop2 runs used Muon scope rnn; they are not direct duplicates. "
            "The H382 prop2/e3 arm is a fresh all-scope anchor; prop0/e3 changes only propensity power "
            "relative to that anchor, and prop0/e4 changes the fixed schedule length/checkpoint."
        ),
        "device_timing_reference": {
            "device": "2x NVIDIA Tesla T4 (Kaggle)",
            "v6_completed_same_shape_fit_seconds": [2899, 2902, 2932, 3545],
            "v6_completed_same_shape_export_seconds": [97, 140, 140, 142],
            "scope": "Actual fold-1 V6 GRU256x2 runs; representative timing only, not a fold-0 profile or speedup claim.",
        },
        "limits": {
            "max_fit_seconds_per_job": 5400,
            "max_export_seconds_per_job": 1800,
            "cohort_hardwall_seconds": 14_400,
            "maximum_gpu_seconds_from_two_slots": 28_800,
            "source_bootstrap_bytes_max": 1_000_000,
        },
        "hardware_request": {
            "accelerator": "NvidiaTeslaT4",
            "expected_visible_device_count": 2,
            "pre_fit_device_gate": "require exactly two nvidia-smi devices and T4 model token before extracting inputs or starting jobs",
        },
        "admission": "Prepared for root review only; no provider upload or dispatch is performed by this builder.",
    }
    packet_path = HERE / "H382_PACKET_R2.json"
    packet_path.write_bytes(canonical_json(packet))
    return packet


def main() -> None:
    """Build packet; `--force` is for intentional regeneration in this new H382 ID."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    packet = build(force=args.force)
    print(
        json.dumps(
            {
                "packet_id": packet["packet_id"],
                "bundle_sha256": packet["bundle_sha256"],
                "bundle_bytes": packet["bundle_bytes"],
                "kernel_sha256": packet["kernel_sha256"],
                "kernel_bytes": packet["kernel_bytes"],
                "source_file_count": packet["source_file_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
