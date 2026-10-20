"""Pinned source binding for the fixed H512-in-TL3 endpoint screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

# Versioned nested receipt fields stay schema-checked at their trust boundary.
H512_PACKET = Path("research/campaign_2026-10-20/submission_delivery_r1/h512_f0_followup_r1")
SCRATCH_ROOT = Path("/home/vlad/development/wunder/research_tmp")
SOURCE_ZIP_SHA256 = "d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569"
SOURCE_ARCHIVE_RECEIPT_SHA256 = "73cd201b165424f4b7ec72fe4bf4d9d573f56d93c0501519ed3bc6a3ea11a2bd"
FIT_RUN = "r01_h512_f0_s1003_local_cp5_r1"
SWA_RUN = "r01_h512_f0_s1003_local_swa345_r1"
TRAINING_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
FIT_PLAN_SHA256 = "51d797d1d568d98b0d9553c934721aab2826ba5d4d61b96a24e496ce514c7ec9"
FIT_SCREEN_SHA256 = "f6b3b954cd2253a9eac10032c9a053f7abcd55e359c41c2062ed3bdeff2920d8"
FIT_ADMISSION_SHA256 = "a17a468f5d0d247db92cbaac0e3d4f0800536c46d6682f2665abb4b781e776f1"
FIT_RECEIPT_PRODUCER_SHA256 = "64bb60c2661a8d402fe8561c149b2eaf944da7b6df50d3f58b9af6e9faffc502"
DERIVER_SHA256 = "c43042f15ca245c5e8dfc0cc856c0b42f03e55f2db3f3d5c74c87b09344b031a"
PACKAGER_FILES = {
    "packager": "14dac5e42f57595043cd773ff6cbe8ba987d50b79a6265757e69d51e627f869a",
    "checkpoint_exporter": "51468297d2da604d8f3c8236fb7d6722b8b0d57a637eace94ccb1810b03cf4c4",
    "source_archive_helper": "4960d4b77419a717503fd5f1b0dc2f214881bf6010e576ce59511068bf36ee46",
    "architecture_helper": "d7d27a1debe2ec5354c60d0adf7a119b0d6321c334b55a74f7222de4b3252775",
    "native_input_contract": "93d323e5e7abfb077ceff56a24f3056c430cdf923264e30895c9f77794c6ff80",
}
ENDPOINTS = {
    "cp5": {"run": FIT_RUN},
    "swa345": {"run": SWA_RUN},
}
FIXED_MEMBERS = (
    ("weights_tl5.npz", "fttall_tlead_s5@ep3", 0.33),
    ("weights_tl6.npz", "fttall_tlead_s6@ep3", 0.33),
)


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one bounded evidence object."""
    if path.stat().st_size > 16_000_000:
        raise ValueError(f"evidence JSON exceeds 16 MB: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def physical_scratch(path: Path) -> Path:
    """Require outputs under the configured scratch mount, outside RAM filesystems."""
    scratch = SCRATCH_ROOT.resolve(strict=True)
    output = path.resolve()
    if not output.is_relative_to(scratch):
        raise ValueError(f"output is outside approved scratch: {output}")
    mounts: list[tuple[int, str]] = []
    for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        mountpoint = Path(fields[1].replace("\\040", " "))
        if scratch.is_relative_to(mountpoint):
            mounts.append((len(str(mountpoint)), fields[2]))
    if not mounts or max(mounts)[1] in {"tmpfs", "ramfs"}:
        raise ValueError("approved research scratch is not a verified physical filesystem")
    return output


def _fit_terminal(root: Path) -> tuple[Path, str, dict[str, Any]]:
    path = (root / H512_PACKET / "FIT_TERMINAL_ACCEPTANCE_R1.json").resolve(strict=True)
    receipt = read_json(path)
    digest = sha256_file(path)
    producer = root / H512_PACKET / "record_h512_fit_terminal.py"
    if (
        receipt.get("schema") != "wnn.h512-fit-terminal.v1"
        or receipt.get("status") != "FIT_COMPLETE_ACCEPTED"
        or receipt.get("run_name") != FIT_RUN
        or receipt.get("source_execution_commit") != TRAINING_COMMIT
        or receipt.get("plan_sha256") != FIT_PLAN_SHA256
        or receipt.get("screen_sha256") != FIT_SCREEN_SHA256
        or receipt.get("admission_sha256") != FIT_ADMISSION_SHA256
        or receipt.get("receipt_producer_sha256") != FIT_RECEIPT_PRODUCER_SHA256
        or sha256_file(producer) != FIT_RECEIPT_PRODUCER_SHA256
        or receipt.get("selected_fixed_epoch") != 5
        or receipt.get("fit_time_evaluation") is not False
    ):
        raise ValueError("H512 source lacks the accepted frozen CP5 fit receipt")
    return path, digest, receipt


def _check_member_map(actual: object, run: str) -> list[dict[str, Any]]:
    """Require the one H512 replacement and the two unchanged TL3 members."""
    expected = [
        {"weights": "weights_s5.npz", "name": run, "coef": 0.34},
        {"weights": FIXED_MEMBERS[0][0], "name": FIXED_MEMBERS[0][1], "coef": 0.33},
        {"weights": FIXED_MEMBERS[1][0], "name": FIXED_MEMBERS[1][1], "coef": 0.33},
    ]
    if actual != expected:
        raise ValueError("candidate ZIP changed TL3 member identity, order, or fixed coefficients")
    return expected


def _accepted_checkpoint(
    root: Path, endpoint: str, fit_path: Path, fit_sha: str, fit: dict[str, Any]
) -> tuple[Path, str, Path, str, Path | None, str | None]:
    """Bind the package member to the accepted CP5 fit or its fixed SWA345 child."""
    spec = ENDPOINTS[endpoint]
    run = str(spec["run"])
    checkpoint = root / "artifacts/runs" / run / "model_ep5.pt"
    provenance = root / "artifacts/runs" / run / "provenance.json"
    if endpoint == "cp5":
        checkpoint_sha = fit.get("checkpoint_sha256", {}).get("model_ep5.pt")
        provenance_sha = str(fit.get("provenance_sha256", ""))
        if fit_path != (root / H512_PACKET / "FIT_TERMINAL_ACCEPTANCE_R1.json").resolve():
            raise ValueError("accepted CP5 receipt path differs from the frozen source")
        derivation_path = None
        derivation_sha = None
    else:
        derivation_path = root / "artifacts/runs" / SWA_RUN / "weight_derivation_receipt.json"
        derivation_path = derivation_path.resolve(strict=True)
        derivation = read_json(derivation_path)
        derivation_sha = sha256_file(derivation_path)
        fit_ref = derivation.get("fit_terminal_receipt", {})
        output = derivation.get("outputs", {})
        source_provenance = derivation.get("source_provenance", {})
        source_checkpoints = derivation.get("source_checkpoints", [])
        input_manifest = derivation.get("input_manifest", {})
        checkpoint_sha = output.get("model_ep5.pt", {}).get("sha256")
        provenance_sha = str(output.get("provenance.json", {}).get("sha256", ""))
        source_provenance_path = root / "artifacts/runs" / FIT_RUN / "provenance.json"
        expected_source_checkpoints = [
            {
                "epoch": epoch,
                "path": f"artifacts/runs/{FIT_RUN}/model_ep{epoch}.pt",
                "sha256": fit.get("checkpoint_sha256", {}).get(f"model_ep{epoch}.pt"),
            }
            for epoch in (3, 4, 5)
        ]
        source_manifest_path = root / H512_PACKET / "SWA345_SOURCE_MANIFEST_R1.json"
        if (
            derivation.get("schema") != "wnn.h512-swa345-derivation.v1"
            or derivation.get("method") != "equal_parameter_mean_epochs_3_4_5"
            or derivation.get("source_run_name") != FIT_RUN
            or derivation.get("destination_run_name") != SWA_RUN
            or derivation.get("source_training_commit") != TRAINING_COMMIT
            or derivation.get("training_run") is not False
            or fit_ref.get("sha256") != fit_sha
            or (root / str(fit_ref.get("path", ""))).resolve() != fit_path
            or source_provenance.get("path") != str(source_provenance_path.relative_to(root))
            or source_provenance.get("sha256") != fit.get("provenance_sha256")
            or source_checkpoints != expected_source_checkpoints
            or input_manifest.get("path") != str(source_manifest_path.relative_to(root))
            or input_manifest.get("sha256") != sha256_file(source_manifest_path)
            or read_json(source_manifest_path).get("status") != "READY_FOR_DERIVATION"
            or derivation.get("derivation_code_sha256") != DERIVER_SHA256
            or sha256_file(root / H512_PACKET / "derive_h512_swa345.py") != DERIVER_SHA256
            or sha256_file(checkpoint) != checkpoint_sha
            or sha256_file(provenance) != provenance_sha
        ):
            raise ValueError("SWA345 source differs from the accepted, fixed derivation")
    checkpoint = checkpoint.resolve(strict=True)
    provenance = provenance.resolve(strict=True)
    if (
        not checkpoint_sha
        or not provenance_sha
        or sha256_file(checkpoint) != checkpoint_sha
        or sha256_file(provenance) != provenance_sha
    ):
        raise ValueError("H512 checkpoint/provenance bytes lack an accepted source binding")
    if endpoint == "cp5" and fit.get("checkpoint_sha256", {}).get("model_ep5.pt") != checkpoint_sha:
        raise ValueError("CP5 checkpoint hash differs from accepted fit receipt")
    return (
        checkpoint,
        str(checkpoint_sha),
        provenance,
        provenance_sha,
        derivation_path,
        derivation_sha,
    )


def validate_candidate(
    root: Path, endpoint: str, archive_path: Path, build_receipt_path: Path
) -> dict[str, Any]:
    """Validate a fixed package against accepted H512 source and protected TL3 assets."""
    if endpoint not in ENDPOINTS:
        raise ValueError(f"unknown endpoint: {endpoint}")
    root = root.resolve(strict=True)
    archive_path = archive_path.resolve(strict=True)
    build_receipt_path = build_receipt_path.resolve(strict=True)
    build_sha = sha256_file(build_receipt_path)
    build = read_json(build_receipt_path)
    fit_path, fit_sha, fit = _fit_terminal(root)
    checkpoint, checkpoint_sha, provenance, provenance_sha, derivation_path, derivation_sha = (
        _accepted_checkpoint(root, endpoint, fit_path, fit_sha, fit)
    )
    run = str(ENDPOINTS[endpoint]["run"])
    new_member = build.get("new_member", {})
    artifact = build.get("artifact", {})
    source = build.get("source_archive", {})
    source_zip = root / "submissions/team_tl3_original_20261018.zip"
    if (
        build.get("schema") != "wnn.tl3-h512-member-package.v1"
        or build.get("status") != "archive_built; independent_runtime_and_quality_gates_pending"
        or build.get("artifact_role") != "production_candidate"
        or source.get("path") != str(source_zip)
        or source.get("sha256") != SOURCE_ZIP_SHA256
        or source.get("receipt_sha256") != SOURCE_ARCHIVE_RECEIPT_SHA256
        or sha256_file(source_zip) != SOURCE_ZIP_SHA256
        or new_member.get("run_reference") != run
        or new_member.get("checkpoint_path") != str(checkpoint)
        or new_member.get("checkpoint_sha256") != checkpoint_sha
        or new_member.get("provenance_path") != str(provenance)
        or new_member.get("provenance_sha256") != provenance_sha
        or artifact.get("path") != str(archive_path)
        or artifact.get("sha256") != sha256_file(archive_path)
        or artifact.get("bytes") != archive_path.stat().st_size
    ):
        raise ValueError("candidate package does not bind to the accepted H512 checkpoint")
    code = build.get("code_sha256", {})
    for name, expected in PACKAGER_FILES.items():
        path = {
            "packager": root / "scripts/package_tl3_h512_member.py",
            "checkpoint_exporter": root / "scripts/package_submission.py",
            "source_archive_helper": root / "scripts/tl3_source_archive.py",
            "architecture_helper": root / "scripts/team_member_export_lib.py",
            "native_input_contract": root / "src/wnn/native_representation.py",
        }[name]
        if code.get(name) != expected or sha256_file(path) != expected:
            raise ValueError(f"candidate package source changed: {name}")
    source_entries = source.get("entries")
    output_entries = artifact.get("zip_entries")
    if not isinstance(source_entries, dict) or not isinstance(output_entries, dict):
        raise ValueError("package receipt lacks source/output entry hashes")
    if set(source_entries) != set(output_entries):
        raise ValueError("candidate ZIP entry set differs from the pinned TL3 source")
    for name, original in source_entries.items():
        if name not in {"weights_s5.npz", "members.json"} and output_entries[name] != original:
            raise ValueError(f"protected TL3 entry changed: {name}")
    export = new_member.get("export", {})
    if export.get("path") != "h512_weights_s5.npz" or output_entries["weights_s5.npz"].get(
        "sha256"
    ) != export.get("sha256"):
        raise ValueError("exported H512 weights differ from the candidate ZIP entry")
    with ZipFile(archive_path) as candidate:
        members = json.loads(candidate.read("members.json"))
    expected_members = _check_member_map(members, run)
    if build.get("member_map", {}).get("members") != expected_members:
        raise ValueError("build receipt member map differs from fixed endpoint weights")
    return {
        "endpoint": endpoint,
        "zip_path": str(archive_path),
        "zip_sha256": str(artifact["sha256"]),
        "build_receipt_path": str(build_receipt_path),
        "build_receipt_sha256": build_sha,
        "fit_terminal_path": str(fit_path),
        "fit_terminal_sha256": fit_sha,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "provenance_path": str(provenance),
        "provenance_sha256": provenance_sha,
        "derivation_receipt_path": str(derivation_path) if derivation_path else None,
        "derivation_receipt_sha256": derivation_sha,
        "members": expected_members,
        "whole_endpoint_ancestry": "H512 is the admitted F0-trained member; the two retained TL3 members have unknown ancestry.",
        "evidence_classification": "full-endpoint fitted diagnostic; not clean CV",
    }
