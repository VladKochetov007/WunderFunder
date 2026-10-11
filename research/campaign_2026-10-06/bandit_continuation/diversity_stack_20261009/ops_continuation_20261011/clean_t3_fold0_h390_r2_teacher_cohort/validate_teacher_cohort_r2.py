"""Static validation of the frozen R2 bundle; no model, device, provider, or data-array loads."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[6]
EXPECTED_SOURCE_SHA = "fb32325fa1010615c12a45e67e81edfd18141e288c620a5c73664d45beb397ea"
EXPECTED_SPLIT_SHA = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
EXPECTED_PROP_SHA = "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7"
EXPECTED_R1_ARCHIVE_SHA = "0ba1b1f7b99dc5dabd892c7d99aa030733465aebc73bb652455c9bc527e0d65f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(r1: Path) -> dict[str, Any]:
    bundle = PACKET / "bundle_teacher_cohort_r2"
    manifest_path = bundle / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads((bundle / "kernel/kernel-metadata.json").read_text())
    archive_path = bundle / "bundle.wnn"
    if sha256(archive_path) != manifest["archive_sha256"]:
        raise ValueError("bundle archive differs from its manifest")
    if manifest["dataset"] != "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r2":
        raise ValueError("dataset slug differs from identified R2")
    if metadata["id"] != "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r2-runner":
        raise ValueError("kernel ID is not title-derived from the R2 slug")
    if len(manifest["external_mounts"]) != 13 or len(manifest["spec_paths"]) != 12:
        raise ValueError("unexpected external-input or job inventory")

    sys.path.insert(0, str(REPO / "scripts"))
    from kaggle_common import canonical_sha256, read_spec, verify_files  # noqa: PLC0415

    fit_specs = []
    export_specs = []
    declared_outputs: list[str] = []
    for relative in sorted((PACKET / "jobs").glob("*.json")):
        spec = read_spec(relative)
        if (
            canonical_sha256(sorted(spec["code"], key=lambda row: row["path"]))
            != spec["code_sha256"]
        ):
            raise ValueError(f"code dependency hash mismatch: {relative.name}")
        verify_files(REPO, spec["code"])
        if spec["environment"].get("WNN_LOADER_WORKERS") != "0":
            raise ValueError(f"loader worker count is not zero: {spec['name']}")
        if spec["name"].startswith("fit_"):
            fit_specs.append(spec)
            args = spec["argv"]
            if (
                args[args.index("--epochs") + 1] != "3"
                or args[args.index("--checkpoint-epoch") + 1] != "2"
                or "--save-training-state" not in args
                or spec["limits"]["timeout_seconds"] > 5400
                or len(spec["artifacts"]) != 5
            ):
                raise ValueError(f"fit recipe/output contract differs: {spec['name']}")
        elif spec["name"].startswith("export_"):
            export_specs.append(spec)
            args = spec["argv"]
            if args[args.index("--epoch") + 1] != "2" or "--campaign-split-manifest" not in args:
                raise ValueError(f"prediction checkpoint/split binding differs: {spec['name']}")
            is_full = spec["name"].endswith("_full_valid")
            if is_full != ("--campaign-full-valid-feature-only" in args):
                raise ValueError(f"full-valid export mode differs: {spec['name']}")
            if (
                len(spec["artifacts"]) != 2
                or not spec["artifacts"][0].endswith(".npy")
                or not spec["artifacts"][1].endswith(".json")
            ):
                raise ValueError(f"prediction array/sidecar are not both declared: {spec['name']}")
            declared_outputs.extend(spec["artifacts"])
        else:
            raise ValueError(f"unexpected job type: {spec['name']}")
    if len(fit_specs) != 3 or len(export_specs) != 9 or len(set(declared_outputs)) != 18:
        raise ValueError("fit/export stages or unique output set differ from 3 + 9")
    if (
        sum(
            spec["argv"][spec["argv"].index("scripts/predict_run.py") + 2] == "train"
            for spec in export_specs
        )
        != 3
    ):
        raise ValueError("expected exactly three train-fit exports")
    if (
        sum(
            spec["argv"][spec["argv"].index("scripts/predict_run.py") + 2] == "valid"
            for spec in export_specs
        )
        != 6
    ):
        raise ValueError("expected six valid-source exports")
    if sum("--campaign-full-valid-feature-only" in spec["argv"] for spec in export_specs) != 3:
        raise ValueError("expected exactly three separate full-valid feature-only exports")

    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name for member in archive.getmembers()}
    declared_files = {
        row["path"] for row in manifest["files"] if row["path"] not in manifest["external_mounts"]
    }
    if members != declared_files | set(manifest["spec_paths"]):
        raise ValueError("archive contents differ from the sealed manifest/spec closure")

    source_r1 = r1 / "source_snapshot"
    source_r2 = PACKET / "source_snapshot"
    source_files = sorted(path for path in source_r1.rglob("*") if path.is_file())
    for source in source_files:
        copy = source_r2 / source.relative_to(source_r1)
        if not copy.is_file() or sha256(source) != sha256(copy):
            raise ValueError(
                f"R2 source snapshot differs from R1 at {source.relative_to(source_r1)}"
            )
    if sha256(source_r2 / "SOURCE_MANIFEST_H390.json") != EXPECTED_SOURCE_SHA:
        raise ValueError("R2 source manifest hash differs from frozen R1")
    if sha256(r1 / "reference_inputs/fold0.json") != EXPECTED_SPLIT_SHA:
        raise ValueError("fold-0 reference split differs from expected SHA")
    preflight = json.loads((PACKET / "ACTUAL_PRODUCER_PREFLIGHT_R1_RECEIPT.json").read_text())
    if (
        preflight.get("status") != "pass_static_actual_producer_fixtures_only"
        or preflight.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA
    ):
        raise ValueError("copied R1 actual-producer fixture receipt is not valid for fold 0")
    if preflight.get("source_snapshot_manifest_sha256") != EXPECTED_SOURCE_SHA:
        raise ValueError("copied R1 actual-producer receipt does not bind R1 source")
    source_binding = json.loads((PACKET / "SOURCE_BINDING_R2.json").read_text())
    if source_binding["copied_source_manifest_sha256"] != EXPECTED_SOURCE_SHA:
        raise ValueError("R2 source binding changed")
    input_binding = json.loads((PACKET / "INPUT_BINDING_R2.json").read_text())
    if (
        input_binding["own_propensity_checkpoint_sha256"] != EXPECTED_PROP_SHA
        or input_binding["split_manifest_sha256"] != EXPECTED_SPLIT_SHA
    ):
        raise ValueError("R2 split/propensity input binding changed")
    propensity = next(
        row
        for row in input_binding["external_inputs"]
        if row["path"] == "artifacts/runs/c20260929_prop_f0/model.pt"
    )
    if propensity["sha256"] != EXPECTED_PROP_SHA:
        raise ValueError("R2 mounted own-fold propensity model hash differs")

    gate_path = PACKET / "CONDITIONAL_FIT_ADMISSION_R2.json"
    gate = json.loads(gate_path.read_text())
    kernel_path = bundle / "kernel/kernel.py"
    kernel = kernel_path.read_text()
    compile(kernel, str(kernel_path), "exec")
    gate_sha = sha256(gate_path)
    if (
        gate.get("release_authorized") is not False
        or gate.get("status") != "pending_actual_t4_profile_and_root_release"
    ):
        raise ValueError("R2 gate is not pending/closed")
    if gate_sha not in kernel or kernel.index(
        'PermissionError("R2 fit admission is pending'
    ) > kernel.index('for logical, mapping in manifest["external_mounts"]'):
        raise ValueError(
            "bootstrap does not hash-check and refuse pending gate before external mounts"
        )
    if kernel.index('for logical, mapping in manifest["external_mounts"]') > kernel.index(
        "subprocess.Popen(command"
    ):
        raise ValueError("bootstrap worker launch precedes input mount validation")
    if "KERNEL_PROCESS_WALL_SECONDS_MAX = 14_400" not in kernel:
        raise ValueError("R2 bounded kernel wall ceiling is missing")

    return {
        "schema": "wnn.kaggle.teacher_cohort_static_validation.r2",
        "status": "pass_static_actual_bundle_specs_and_guard",
        "scope": "local static validation only; no provider, GPU, external mount, Torch model, or prediction-array access",
        "r1_archive_sha256": EXPECTED_R1_ARCHIVE_SHA,
        "r1_source_manifest_sha256": EXPECTED_SOURCE_SHA,
        "r2_archive_sha256": manifest["archive_sha256"],
        "r2_manifest_sha256": sha256(manifest_path),
        "r2_bootstrap_sha256": sha256(kernel_path),
        "r2_kernel_id": metadata["id"],
        "archive_members": len(members),
        "external_mount_count": len(manifest["external_mounts"]),
        "source_files_copied_and_compared": len(source_files),
        "fit_specs": len(fit_specs),
        "prediction_specs": len(export_specs),
        "unique_prediction_payloads": len(declared_outputs),
        "fit_admission": "pending_profile_and_root_release",
        "remote_asset_version_readiness": "not_checked; root-owned gate",
        "full_cohort_eta_seconds": None,
        "wall_ceiling_seconds": 14400,
        "two_device_gpu_seconds_upper_bound": 28800,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-r1-packet", type=Path, required=True)
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    receipt = validate(args.profile_r1_packet.resolve())
    output = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.write_receipt:
        (PACKET / "STATIC_VALIDATION_R2.json").write_text(output)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
