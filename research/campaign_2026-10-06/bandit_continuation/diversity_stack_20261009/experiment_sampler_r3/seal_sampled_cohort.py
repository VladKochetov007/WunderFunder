from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DIV = HERE.parent
R5 = DIV / "d02_muon_loss_successor_r5_profile_export_r2"
SOURCE_ASSET = R5 / "source_asset"
SOURCE_DATASET = "vladkochetov07/wnn-d02-muon-loss-code-20261009-r5"
SOURCE_COMMIT = "845405d16dc6b3768080aa99c3663a440d343ba5"
OVERLAY_DATASET = "vladkochetov07/wnn-sampled-recipe-runner-code-20261009-r1"
OVERLAY_TITLE = "WNN Sampled Recipe Runner Code 20261009 R1"
KERNEL_ID = "vladkochetov07/wnn-sampled-recipe-cohort-20261009-r1"
KERNEL_TITLE = "WNN Sampled Recipe Cohort 20261009 R1"
KERNEL_FILENAME = "wnn-sampled-recipe-cohort-20261009-r1.py"
SOURCE_CODE_MANIFEST_SHA256 = "dbd98f4b73f5cdc7dd8df137ae9d0bd23495fd8ee2067da02cc94d45699bd6e9"
SOURCE_ARCHIVE_SHA256 = "7c85e1175f3dfff01fc8ab5b2fee81f449cfb873c8f1303fece8b330ad67dcdc"
SOURCE_ASSET_MANIFEST_SHA256 = "9ab15bc1efae1b56e045413d3999c350ffdf75edbc541b49f053b41c3bb264c6"
FOLD0_ASSET_MANIFEST_SHA256 = "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
CANDIDATE_POOL_SHA256 = "e5d37d2e5b427fee3648bb6cae1aebcd189bf9b4db642a1c07b47c64743935ee"
CONTROL_CONTRACT_SHA256 = "5b63ed39aa6046e9f02878e7dcb4b48982240bffa45ec09b7466dfb9eeac63f7"
CACHE_MANIFEST_SHA256 = "7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b"
CACHE_KEY = "949587f77794935c08589bd5d5e81641062de547526b92fc1b01bc4109cd82a6"
CPU_SOURCE_KERNEL = "vladkochetov07/wnn-parsed-cache-source-stage-20261009-v1"
CONTROL_KERNEL = "vladkochetov07/wnn-loss-pair-export-recovery-20261009-r1"
R5_KERNEL = "vladkochetov07/wnn-d02-muon-loss-gpu-20261009-v5"
FOLD0_ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"
MEMBERSHIP_SHA256 = "dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
REGISTRY_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
R5_PROFILE_GATE_SHA256 = "a5445694b64988811b3dbf9b1dd5ea50a94ef2b7ee3f02c337c2647dd23d88d9"
OVERLAY_PATHS = (
    "scripts/cohort_spec_r1.json",
    "scripts/sampled_recipe_worker.py",
)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def title_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> bytes:
    raw = canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _admission_payload(
    spec_sha: str,
    overlay_manifest_sha: str,
    runner_sha: str,
    source_manifest_sha: str,
    *,
    push_allowed: bool,
) -> dict[str, Any]:
    return {
        "schema": "wnn.sampled-cohort.root-admission.v1",
        "kernel_id": KERNEL_ID,
        "push_allowed": push_allowed,
        "conditional_fit_allowed": push_allowed,
        "source_commit": SOURCE_COMMIT,
        "source_dataset": SOURCE_DATASET,
        "candidate_pool_sha256": CANDIDATE_POOL_SHA256,
        "cohort_spec_sha256": spec_sha,
        "source_code_manifest_sha256": source_manifest_sha,
        "cache_manifest_sha256": CACHE_MANIFEST_SHA256,
        "cache_key": CACHE_KEY,
        "control_contract_sha256": CONTROL_CONTRACT_SHA256,
        "runner_overlay_manifest_sha256": overlay_manifest_sha,
        "runner_worker_sha256": runner_sha,
        "runner_overlay_dataset": OVERLAY_DATASET,
        "runtime_seconds": 3300,
        "provider_seconds": 3600,
        "gpu_model": "NvidiaTeslaT4",
        "gpu_count": 2,
        "maximum_output_bytes": 12_000_000_000,
    }


def seal_candidate(output: Path, admission_path: Path | None = None) -> dict[str, Any]:
    """Create an immutable local packet; this function never contacts Kaggle."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"preserve existing sealed candidate: {output}")
    if title_slug(OVERLAY_TITLE) != OVERLAY_DATASET.rsplit("/", 1)[1]:
        raise ValueError("overlay dataset title-derived slug differs from its metadata ID")
    if title_slug(KERNEL_TITLE) != KERNEL_ID.rsplit("/", 1)[1]:
        raise ValueError("GPU kernel title-derived slug differs from its metadata ID")
    if sha_file(SOURCE_ASSET / "assets-manifest.json") != SOURCE_ASSET_MANIFEST_SHA256:
        raise ValueError("the reused R5 source asset manifest changed")
    source_manifest = _read_json(SOURCE_ASSET / "assets-manifest.json")
    if source_manifest.get("dataset") != SOURCE_DATASET:
        raise ValueError("the reused R5 source dataset identity differs")

    spec_path = HERE / "cohort_spec_r1.json"
    worker_path = HERE / "sampled_recipe_worker.py"
    template_path = HERE / "kernel_template.py"
    spec_sha = sha_file(spec_path)
    worker_sha = sha_file(worker_path)
    code_manifest_sha = SOURCE_CODE_MANIFEST_SHA256

    overlay_payloads = {relative: (HERE / Path(relative).name).read_bytes() for relative in OVERLAY_PATHS}
    overlay_rows = [
        {
            "filename": f"{sha_bytes(payload)}.wnn",
            "path": relative,
            "sha256": sha_bytes(payload),
            "size": len(payload),
        }
        for relative, payload in sorted(overlay_payloads.items())
    ]
    overlay_manifest = {
        "dataset": OVERLAY_DATASET,
        "private": True,
        "version": 1,
        "files": overlay_rows,
    }
    overlay_manifest_bytes = canonical_json(overlay_manifest)
    overlay_manifest_sha = sha_bytes(overlay_manifest_bytes)

    if admission_path is None:
        admission = _admission_payload(
            spec_sha, overlay_manifest_sha, worker_sha, code_manifest_sha, push_allowed=False
        )
    else:
        admission = _read_json(admission_path)
        expected_admission = _admission_payload(
            spec_sha, overlay_manifest_sha, worker_sha, code_manifest_sha, push_allowed=True
        )
        for key, expected in expected_admission.items():
            if admission.get(key) != expected:
                raise ValueError(f"root admission does not bind the final sealed packet at {key}")

    output.mkdir(parents=True)
    overlay_root = output / "runner_overlay_asset"
    overlay_root.mkdir()
    for relative, payload in overlay_payloads.items():
        row = next(item for item in overlay_rows if item["path"] == relative)
        (overlay_root / row["filename"]).write_bytes(payload)
    (overlay_root / "assets-manifest.json").write_bytes(overlay_manifest_bytes)
    _write_json(
        overlay_root / "dataset-metadata.json",
        {
            "id": OVERLAY_DATASET,
            "title": OVERLAY_TITLE,
            "isPrivate": True,
            "licenses": [{"name": "other"}],
        },
    )

    kernel_root = output / "kernel"
    kernel_root.mkdir()
    bootstrap = template_path.read_text(encoding="utf-8")
    source_entries = {row["path"]: row for row in source_manifest["files"]}
    replacements = {
        "__ASSET_MANIFEST_SHA256__": SOURCE_ASSET_MANIFEST_SHA256,
        "__ARCHIVE_SHA256__": SOURCE_ARCHIVE_SHA256,
        "__CODE_MANIFEST_SHA256__": SOURCE_CODE_MANIFEST_SHA256,
        "__SOURCE_COMMIT__": SOURCE_COMMIT,
        "__SAMPLED_ADMISSION_B64__": base64.b64encode(canonical_json(admission)).decode(),
        "__OVERLAY_MANIFEST_SHA256__": overlay_manifest_sha,
        "__COHORT_SPEC_SHA256__": spec_sha,
        "__RUNNER_WORKER_SHA256__": worker_sha,
    }
    for placeholder, value in replacements.items():
        bootstrap = bootstrap.replace(placeholder, value)
    if any(placeholder in bootstrap for placeholder in replacements):
        raise ValueError("kernel bootstrap retained an unresolved content pin")
    kernel_path = kernel_root / KERNEL_FILENAME
    kernel_path.write_text(bootstrap, encoding="utf-8")
    if kernel_path.stat().st_size >= 1_000_000:
        raise ValueError("Kaggle kernel bootstrap exceeds the 1 MB envelope")
    metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": KERNEL_FILENAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [SOURCE_DATASET, FOLD0_ASSET_DATASET, OVERLAY_DATASET],
        "kernel_sources": [CPU_SOURCE_KERNEL, CONTROL_KERNEL, R5_KERNEL],
        "competition_sources": [],
        "model_sources": [],
    }
    metadata_bytes = _write_json(kernel_root / "kernel-metadata.json", metadata)
    _write_json(output / "root_admission.json", admission)

    packet = {
        "schema": "wnn.diversity.sampled-cohort-packet.v1",
        "format_version": 1,
        "packet_id": KERNEL_ID,
        "authorized_for_execution": admission.get("push_allowed") is True,
        "root_admission_required_before_provider_mutation": True,
        "kernel": {
            "id": KERNEL_ID,
            "title": KERNEL_TITLE,
            "title_slug_matches_id": title_slug(KERNEL_TITLE) == KERNEL_ID.rsplit("/", 1)[1],
            "accelerator": "NvidiaTeslaT4",
            "gpu_count": 2,
            "worker_wall_seconds": 3300,
            "provider_timeout_seconds": 3600,
            "bootstrap_sha256": sha_file(kernel_path),
            "bootstrap_size_bytes": kernel_path.stat().st_size,
            "metadata_sha256": sha_bytes(metadata_bytes),
        },
        "source": {
            "dataset": SOURCE_DATASET,
            "source_commit": SOURCE_COMMIT,
            "code_manifest_sha256": SOURCE_CODE_MANIFEST_SHA256,
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "asset_manifest_sha256": SOURCE_ASSET_MANIFEST_SHA256,
            "source_asset_files": source_entries,
            "runner_overlay_dataset": OVERLAY_DATASET,
            "runner_overlay_title": OVERLAY_TITLE,
            "runner_overlay_manifest_sha256": overlay_manifest_sha,
            "runner_overlay_files": overlay_rows,
            "runner_worker_sha256": worker_sha,
            "cohort_spec_sha256": spec_sha,
        },
        "producer": {
            "cache_kernel": CPU_SOURCE_KERNEL,
            "cache_manifest_sha256": CACHE_MANIFEST_SHA256,
            "cache_key": CACHE_KEY,
            "cache_source_stage_receipt_sha256": "557cd0a61f1129f811dc104398c3c9c7295c3343a1c9a302018fcb588c0c1de7",
            "cache_validation_sha256": "84686dff926036b37362a3ea28edcf2166d0bb44bee11b2fcbd64082e4b5cf35",
            "split_sha256": SPLIT_SHA256,
            "registry_sha256": REGISTRY_SHA256,
            "membership_sha256": MEMBERSHIP_SHA256,
            "fit_train_count": 350,
            "fit_valid_count_unused": 28,
            "evaluation_count": 468,
            "cache_payload_bytes": 7_732_440_000,
            "runtime_mapping": "each fit process reopens the pinned cache as read-only memmaps; it is not one persistent parsed-array process",
        },
        "controls": {
            "kernel": CONTROL_KERNEL,
            "training_contract_sha256": CONTROL_CONTRACT_SHA256,
            "profile_gate_kernel": R5_KERNEL,
            "profile_gate_sha256": R5_PROFILE_GATE_SHA256,
            "r5_fit_transport_review": {
                "path": "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ROOT_GPU_R5_TERMINAL_REVIEW_R1.json",
                "sha256": "60e9770de432a6b39d3ae41c818d60b2d47b5298f630b33ee3483a0e7fecc7f9",
            },
            "retrain_controls": False,
        },
        "cohort_spec": _read_json(spec_path),
        "expected_artifacts": [
            "boot_marker.json",
            "phase.jsonl",
            "gpu_inventory.json",
            "fold0_assets_validation.json",
            "out/bootstrap_receipt.json",
            "out/final_status.json",
            "out/failure.json",
            "out/cohort_progress.json",
            "out/training_contract.json",
            "out/sampled_cohort_run_receipt.json",
            "out/shared/*",
            "out/arms/<queue_id>/model/model.pt",
            "out/arms/<queue_id>/model/model_ep0.pt",
            "out/arms/<queue_id>/model/model_ep1.pt",
            "out/arms/<queue_id>/model/training_state.pt",
            "out/arms/<queue_id>/model/provenance.json",
            "out/arms/<queue_id>/predictions_valid.npy",
            "out/arms/<queue_id>/prediction_membership.json",
            "out/arms/<queue_id>/update_trace.jsonl",
            "out/arms/<queue_id>/schedule_trace.json",
            "out/arms/<queue_id>/schedule_trace_interpreted.json",
            "out/logs/*",
        ],
        "limits": {
            "maximum_output_bytes": 12_000_000_000,
            "minimum_disk_free_bytes": 4_000_000_000,
            "maximum_cgroup_current_bytes": 24_000_000_000,
            "minimum_mem_available_bytes": 2_000_000_000,
            "provider_calls": "one admitted private overlay dataset create and one admitted kernel push; status/collection are read-only",
        },
        "root_admission": admission,
    }
    packet_bytes = _write_json(output / "sampled_cohort.packet.json", packet)
    receipt = {
        "schema": "wnn.diversity.sampled-cohort-seal-receipt.v1",
        "state": "sealed_pending_admission" if admission.get("push_allowed") is not True else "sealed_admitted",
        "packet_sha256": sha_bytes(packet_bytes),
        "bootstrap_sha256": sha_file(kernel_path),
        "metadata_sha256": sha_bytes(metadata_bytes),
        "cohort_spec_sha256": spec_sha,
        "runner_worker_sha256": worker_sha,
        "runner_overlay_manifest_sha256": overlay_manifest_sha,
        "root_admission_sha256": sha_file(output / "root_admission.json"),
    }
    _write_json(output / "seal_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-admission", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(seal_candidate(arguments.output, arguments.root_admission), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
