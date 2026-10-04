"""Build fixed q8 variants of the byte-verified E412 public archive."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
E416 = CAMPAIGN / "E416_native_q8_build"
SOURCE_ZIP = ROOT / "submissions/c20261004_combo_s3_i16/submission.zip"
SOURCE_SHA256 = "21f9b7368eddd64ad8ae19ae0702990a57a1806531a85148b657e5aed719dafb"
PARENT_RUN = "c20261004_combo_s3_final_r1"
HEAD_RUN = "c20261004_combo_head_s3_final"
PARENT_SHA256 = "64c2b28651b414778d6af3c33bb9d54f653a46fb0be087de2e3c44ca88c31166"
HEAD_SHA256 = "f1cb88f38b6d5b5b67859346b439e69d610928bc836baac8867e84e7b1c95d85"
E416_LIB_SHA256 = {
    "v2": "fafb05afc04a7aa2ad96997c2ad20233349092d0fe5dc06cbcbcecddd2582aef",
    "v3": "8bba2b94b474790287fb9d69a9ea7b7d2c4aff23d91671b561087617c5ab101d",
}
VARIANTS = {"q8r": "recurrent", "q8x": "recurrent_head"}

if TYPE_CHECKING:
    import torch


def sha256_file(path: Path) -> str:
    """Hash a file without loading its full contents into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_state(checkpoint_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load one fixed checkpoint into contiguous float32 arrays and inspect its metadata."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"checkpoint has no model state: {checkpoint_path}")
    state = {
        name: np.ascontiguousarray(value.detach().cpu().float().numpy())
        for name, value in checkpoint["model"].items()
    }
    return state, checkpoint


def _fit_provenance(run: str, epoch: int, expected_init: str | None) -> dict:
    """Verify a source run's fixed epoch, checkpoint, and holdout-excluded fit membership."""
    run_dir = ROOT / "artifacts/runs" / run
    checkpoint_path = run_dir / f"model_ep{epoch}.pt"
    provenance_path = run_dir / "provenance.json"
    history_path = run_dir / "history.json"
    checkpoint_hash = sha256_file(checkpoint_path)
    expected_hash = PARENT_SHA256 if run == PARENT_RUN else HEAD_SHA256
    if checkpoint_hash != expected_hash:
        raise ValueError(f"{run}: checkpoint hash differs from E412 source identity")
    provenance = json.loads(provenance_path.read_text())
    args = provenance.get("args", {})
    expected_source = "train+valid" if run == PARENT_RUN else "valid"
    expected_seed = 3
    if (
        args.get("name") != run
        or args.get("train_source") != expected_source
        or args.get("seed") != expected_seed
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != epoch
        or provenance.get("selected_epoch") != epoch
        or bool(args.get("include_heldout_valid"))
    ):
        raise ValueError(f"{run}: not the preregistered fixed, holdout-excluded checkpoint")
    actual_init = args.get("init")
    if actual_init != expected_init:
        raise ValueError(f"{run}: initializer changed ({actual_init!r})")
    if provenance.get("checkpoint_sha256", {}).get(checkpoint_path.name) != checkpoint_hash:
        raise ValueError(f"{run}: checkpoint hash differs from its original provenance")
    fit_valid = set(map(int, provenance["fit_row_groups"].get("valid", [])))
    return {
        "run": run,
        "epoch": epoch,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "provenance_sha256": sha256_file(provenance_path),
        "history_sha256": sha256_file(history_path),
        "fit_valid_row_groups": sorted(fit_valid),
        "provenance": provenance,
    }


def _holdout_groups() -> set[int]:
    """Recover the historical holdout group IDs from frozen target identities."""
    target_path = ROOT / "cache/valid_targets.npz"
    with np.load(target_path, allow_pickle=False) as target:
        groups = np.asarray(target["row_group"], dtype=np.int64)
        sequence_ids = np.asarray(target["seq_ix"], dtype=np.int64)
    if groups.shape != sequence_ids.shape or len(np.unique(groups)) != len(groups):
        raise ValueError("malformed validation target index")
    return {
        int(group)
        for group, sequence_id in zip(groups, sequence_ids, strict=True)
        if int.from_bytes(hashlib.sha256(str(int(sequence_id)).encode()).digest(), "big") % 5 == 0
    }


def _merge_heads(parent: dict[str, np.ndarray], head: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reproduce package_submission's exact equal-weight shared-trunk fusion."""
    if parent.keys() != head.keys():
        raise ValueError("parent and head tensor names differ")
    for name in parent:
        if not name.startswith("head.") and not np.array_equal(parent[name], head[name]):
            raise ValueError(f"head changed frozen trunk tensor {name}")
    merged = {name: value for name, value in parent.items() if not name.startswith("head.")}
    merged["head.0.weight"] = np.concatenate(
        [parent["head.0.weight"], head["head.0.weight"]], axis=0
    )
    merged["head.0.bias"] = np.concatenate(
        [parent["head.0.bias"], head["head.0.bias"]]
    )
    merged["head.2.weight"] = np.concatenate(
        [parent["head.2.weight"][:2], head["head.2.weight"][:2]], axis=1
    ) / np.float32(2.0)
    merged["head.2.bias"] = (
        parent["head.2.bias"][:2] + head["head.2.bias"][:2]
    ) / np.float32(2.0)
    return {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in merged.items()}


def _source_member() -> tuple[dict[str, np.ndarray], dict, dict[str, str]]:
    """Verify ZIP identity, fixed member manifest, and exact checkpoint-to-member fusion."""
    if sha256_file(SOURCE_ZIP) != SOURCE_SHA256:
        raise ValueError("public E412 source ZIP hash changed")
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        manifest_bytes = archive.read("members.json")
        source_manifest = json.loads(manifest_bytes)
        if source_manifest != [
            {
                "weights": "weights_0.npz",
                "coef": 1.0,
                "source": f"{PARENT_RUN}@ep3|{HEAD_RUN}@ep0",
            }
        ]:
            raise ValueError("public E412 member manifest differs from fixed source")
        member_bytes = archive.read("weights_0.npz")
        source_package_hashes = {
            "solution.py": hashlib.sha256(archive.read("solution.py")).hexdigest(),
            **{
                f"{level}/wnn_native.abi3.so": hashlib.sha256(
                    archive.read(f"{level}/wnn_native.abi3.so")
                ).hexdigest()
                for level in ("v2", "avx2", "v3")
            },
        }
        with np.load(io.BytesIO(member_bytes), allow_pickle=False) as stored:
            weights = {name: np.ascontiguousarray(stored[name]) for name in stored.files}
    parent_info = _fit_provenance(PARENT_RUN, 3, None)
    head_info = _fit_provenance(HEAD_RUN, 0, f"{PARENT_RUN}@ep3")
    parent_state, parent_checkpoint = _read_state(ROOT / parent_info["checkpoint"])
    head_state, head_checkpoint = _read_state(ROOT / head_info["checkpoint"])
    if parent_checkpoint.get("epoch") != 3 or head_checkpoint.get("epoch") != 0:
        raise ValueError("source checkpoint epoch metadata mismatch")
    if head_info["provenance"].get("initial_checkpoint_sha256") != PARENT_SHA256:
        raise ValueError("head provenance does not bind its exact parent checkpoint")
    heldout = _holdout_groups()
    if set(parent_info["fit_valid_row_groups"]) & heldout:
        raise ValueError("parent fit includes historical holdout groups")
    if set(head_info["fit_valid_row_groups"]) & heldout:
        raise ValueError("head fit includes historical holdout groups")
    fused = _merge_heads(parent_state, head_state)
    if fused.keys() != weights.keys() or any(
        not np.array_equal(fused[name], weights[name]) for name in fused
    ):
        raise ValueError("public ZIP member is not the exact fused source checkpoints")
    proof = {
        "source_zip_sha256": SOURCE_SHA256,
        "source_members_json_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_package_hashes": source_package_hashes,
        "source_member_sha256": hashlib.sha256(member_bytes).hexdigest(),
        "source_member": source_manifest[0],
        "parent": {key: value for key, value in parent_info.items() if key != "provenance"},
        "head": {key: value for key, value in head_info.items() if key != "provenance"},
        "parent_job_sha256": sha256_file(
            CAMPAIGN / "runner/attempts/e412-combo-final-s3-v2.json"
        ),
        "head_job_sha256": sha256_file(CAMPAIGN / "runner/attempts/e412-final-head-v1.json"),
        "holdout_groups_sha256": hashlib.sha256(
            np.asarray(sorted(heldout), dtype="<i8").tobytes()
        ).hexdigest(),
        "fusion_matches_archived_member_bitwise": True,
        "non_head_trunk_bitwise_equal": True,
    }
    for key in ("parent_job_sha256", "head_job_sha256"):
        name = "e412-combo-final-s3-v2.json" if key.startswith("parent") else "e412-final-head-v1.json"
        job = json.loads((CAMPAIGN / "runner/attempts" / name).read_text())
        if job.get("status") != "finished" or job.get("exit_code") != 0:
            raise ValueError(f"source training job is not recorded successful: {name}")
        checkpoint_path = parent_info["checkpoint"] if key.startswith("parent") else head_info["checkpoint"]
        expected_hash = PARENT_SHA256 if key.startswith("parent") else HEAD_SHA256
        artifacts = {
            artifact["path"]: artifact["sha256"] for artifact in job.get("artifacts", [])
        }
        if artifacts.get(checkpoint_path) != expected_hash:
            raise ValueError(f"source job does not bind its exact checkpoint: {name}")
    return weights, proof, {
        "source_member_sha256": proof["source_member_sha256"],
        "parent_checkpoint_sha256": PARENT_SHA256,
        "head_checkpoint_sha256": HEAD_SHA256,
    }


def _import_package(package_dir: Path, preset: str) -> tuple[object, object]:
    """Import an isolated package with its q8 precision and invoke one warmup row."""
    import importlib

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(package_dir))
    os.environ["WNN_NATIVE_LEVEL"] = "v3"
    os.environ.pop("WNN_PRECISION", None)
    sys.modules.pop("wnn_native", None)
    module_name = f"e462_solution_{preset}"
    spec = importlib.util.spec_from_file_location(module_name, package_dir / "solution.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import package solution at {package_dir}")
    solution = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = solution
    spec.loader.exec_module(solution)
    model = solution.PredictionModel()
    from utils import DataPoint

    if model.predict(DataPoint(987654321, 0, False, np.zeros(112, dtype=np.float32))) is not None:
        raise ValueError(f"{preset}: callback predicted during warmup")
    expected = "q8r" if preset == "recurrent" else "q8x"
    if model.model.precision != expected:
        raise ValueError(f"{preset}: wrong native precision {model.model.precision}")
    return solution, model


def build(output: Path) -> None:
    """Export two immutable q8 packages and perform bounded native constructor checks."""
    from wnn.quantization import quantize_model
    from wnn.quantization_export import read_q8_export, write_q8_export

    if output.exists():
        raise FileExistsError(output)
    e416_manifest = json.loads((CAMPAIGN / "E416_native_q8_confirmation/manifest.json").read_text())
    for level, expected_hash in E416_LIB_SHA256.items():
        library = E416 / level / "wnn_native.abi3.so"
        if sha256_file(library) != expected_hash:
            raise ValueError(f"E416 {level} library hash changed")
    output.mkdir(parents=True, exist_ok=False)
    source_weights, lineage, _identities = _source_member()
    package_solution = Path(__file__).with_name("solution_q8.py")
    variant_records = {}
    for label, preset in VARIANTS.items():
        represented, codes, quantization = quantize_model(source_weights, preset)
        case_dir = output / label
        package_dir = case_dir / "pkg"
        package_dir.mkdir(parents=True)
        write_q8_export(package_dir / "weights_0.npz", source_weights, preset, codes)
        restored, restored_codes, restored_preset = read_q8_export(package_dir / "weights_0.npz")
        if restored_preset != preset:
            raise ValueError(f"{label}: q8 preset changed on export/reload")
        if restored.keys() != represented.keys() or any(
            not np.array_equal(restored[name], represented[name]) for name in represented
        ):
            raise ValueError(f"{label}: represented weights changed on export/reload")
        if restored_codes.keys() != codes.keys() or any(
            restored_codes[name].dtype != codes[name].dtype
            or not np.array_equal(restored_codes[name].view(np.uint8), codes[name].view(np.uint8))
            for name in codes
        ):
            raise ValueError(f"{label}: q8 code/scale bytes changed on export/reload")
        shutil.copyfile(package_solution, package_dir / "solution.py")
        with zipfile.ZipFile(SOURCE_ZIP) as source_archive:
            package_dir.joinpath("members.json").write_bytes(source_archive.read("members.json"))
        (package_dir / "precision.txt").write_text(f"{label}\n")
        for level in ("v2", "v3"):
            shutil.copytree(E416 / level, package_dir / level)
        package_proof = {
            "schema": "e462-public-combo-q8-v1",
            "variant": label,
            "preset": preset,
            "quantization": quantization,
            "lineage": lineage,
            "native_library_sha256": {
                level: sha256_file(package_dir / level / "wnn_native.abi3.so")
                for level in ("v2", "v3")
            },
            "q8_export_sha256": sha256_file(package_dir / "weights_0.npz"),
        }
        (package_dir / "E462_PROVENANCE.json").write_text(
            json.dumps(package_proof, indent=2, sort_keys=True) + "\n"
        )
        zip_path = case_dir / "submission.zip"
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package_dir).as_posix())
        with zipfile.ZipFile(zip_path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"{label}: ZIP integrity check failed")
        _, callback = _import_package(package_dir, preset)
        variant_records[label] = {
            "zip": str(zip_path.relative_to(ROOT)),
            "zip_sha256": sha256_file(zip_path),
            "zip_bytes": zip_path.stat().st_size,
            "weights_sha256": sha256_file(package_dir / "weights_0.npz"),
            "precision": callback.model.precision,
            "weight_bytes": callback.model.weight_bytes[0],
            "q8_weight_bytes": callback.model.q8_weight_bytes[0],
            "quantization": quantization,
        }
    report = {
        "schema": "e462-public-combo-q8-build-v1",
        "source": lineage,
        "e416_confirmation_manifest_sha256": sha256_file(
            CAMPAIGN / "E416_native_q8_confirmation/manifest.json"
        ),
        "e416_build_manifest": e416_manifest,
        "variants": variant_records,
        "scope": "export, package integrity, E416-library identity and one-row native constructor/warmup checks only",
        "scientific_ready": False,
        "parity_and_scoring_pending": True,
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    """Create the admitted output directory from immutable source identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
