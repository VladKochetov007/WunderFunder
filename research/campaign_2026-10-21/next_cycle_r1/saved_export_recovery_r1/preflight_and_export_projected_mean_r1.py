from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRAINING_ROOT = Path("/home/vlad/development/wunder/research_tmp/overnight-nll-adaptation-20261020")
SCRATCH_ROOT = Path("/home/vlad/development/wunder/research_tmp/overnight-nll-projected-mean-f0-r1")
RUN_NAME = "overnight_nll_f0_s3_wmse_projected_mean_r1"
PLAN_PATH = (
    PROJECT_ROOT / "research/campaign_2026-10-21/next_cycle_r1/weight_mean_projected_r1/plan.json"
)
DERIVATION_PATH = SCRATCH_ROOT / "receipts/PROJECTED_MEAN_DERIVATION_R1.json"
PREDICTOR_PATH = TRAINING_ROOT / "scripts/predict_run.py"
EXPECTED_PLAN_SHA256 = "3b0e0e7b072be59d6125e518b3f84443f011d6bb2713e7f6463940ce4d2115f1"
EXPECTED_TRAINING_HEAD = "d3055423a185d2ecfe7baf1c20206abeb743ad8b"
EXPECTED_SERVING_SOURCE_COMMIT = "c2e2fb57ebb9fb513701d5cd39b9be19505dd35b"
EXPECTED_DERIVE_SCRIPT_SHA256 = "a3cac2ee7040837f0e91ee7e7303d6d3a46d114e8e332cdfb765bd664376f923"
EXPECTED_CHECKPOINT_SHA256 = "fc5e432eff521ad4829a6d1e0aefa6325a3d706b4dc02e17f602d3b33cf26a55"
EXPECTED_DERIVATION_RECEIPT_SHA256 = (
    "a0049df0aeea57e5dc267d9ee498162dc489560eb9cad277b057799fe6b19ac3"
)
EXPECTED_DERIVED_PROVENANCE_SHA256 = (
    "faf01caebd4cb29b2d69c8e571af43ab22000796a380db7499edb630563834d3"
)
FOLD_PATH = PROJECT_ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
SCORE_PATH = SCRATCH_ROOT / "receipts/SCORE_PROJECTED_MEAN_F0_R1.json"


def sha256(path: Path) -> str:
    """Return a streamed SHA256 digest for one pinned file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a small pinned JSON manifest at the execution boundary."""
    # These frozen manifests have heterogeneous schemas, so Any stays at this boundary.
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def require(condition: bool, message: str) -> None:
    """Stop before a write or export when a recovery invariant fails."""
    if not condition:
        raise RuntimeError(message)


def verify_clean_training_source() -> None:
    """Require the exact committed NLL predictor worktree used by derivation."""
    head = subprocess.run(
        ["git", "-C", str(TRAINING_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(head == EXPECTED_TRAINING_HEAD, f"unexpected NLL predictor HEAD: {head}")
    status = subprocess.run(
        ["git", "-C", str(TRAINING_ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(not status, "NLL predictor worktree is dirty")


def verify_projected_mean_pins() -> tuple[dict[str, str], dict[str, dict[str, str | bool]]]:
    """Verify frozen source, checkpoint, derivation and split identities without loading tensors."""
    verify_clean_training_source()
    require(sha256(PLAN_PATH) == EXPECTED_PLAN_SHA256, "projected-mean plan bytes changed")
    plan = read_json(PLAN_PATH)
    source_checked: dict[str, str] = {}
    current_serving_source: dict[str, dict[str, str | bool]] = {}
    for role, entries in plan["source_pins"].items():
        for relative, expected in entries.items():
            if role == "training":
                actual = sha256(TRAINING_ROOT / relative)
            elif role == "serving":
                completed = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(PROJECT_ROOT),
                        "show",
                        f"{EXPECTED_SERVING_SOURCE_COMMIT}:{relative}",
                    ],
                    check=True,
                    capture_output=True,
                )
                actual = hashlib.sha256(completed.stdout).hexdigest()
                current_actual = sha256(PROJECT_ROOT / relative)
                current_serving_source[relative] = {
                    "historical_git_blob_sha256": actual,
                    "current_project_file_sha256": current_actual,
                    "current_file_differs": current_actual != actual,
                }
            else:
                raise ValueError(f"unexpected source pin role: {role}")
            require(actual == expected, f"source pin mismatch for {role}:{relative}")
            source_checked[f"{role}:{relative}"] = actual

    for item in plan["checkpoints"]:
        checkpoint_path = (PROJECT_ROOT / item["path"]).resolve(strict=True)
        provenance_path = (PROJECT_ROOT / item["provenance_path"]).resolve(strict=True)
        require(checkpoint_path.is_relative_to(PROJECT_ROOT), "checkpoint escapes project root")
        require(
            provenance_path.is_relative_to(PROJECT_ROOT), "checkpoint sidecar escapes project root"
        )
        actual_checkpoint = sha256(checkpoint_path)
        actual_provenance = sha256(provenance_path)
        require(actual_checkpoint == item["sha256"], f"checkpoint pin mismatch: {item['role']}")
        require(
            actual_provenance == item["provenance_sha256"],
            f"checkpoint sidecar pin mismatch: {item['role']}",
        )
        source_checked[f"sidecar:{item['role']}"] = actual_provenance

    require(
        sha256(DERIVATION_PATH) == EXPECTED_DERIVATION_RECEIPT_SHA256,
        "projected-mean derivation receipt bytes changed",
    )
    receipt = read_json(DERIVATION_PATH)
    expected_sources = [
        {"role": item["role"], "sha256": item["sha256"]} for item in plan["checkpoints"]
    ]
    require(
        receipt.get("status") == "derived_cpu_projection_parity_passed_quality_not_evaluated",
        "projected-mean derivation receipt status changed",
    )
    require(receipt.get("plan_sha256") == EXPECTED_PLAN_SHA256, "derivation plan binding changed")
    require(
        receipt.get("output_checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256,
        "derived checkpoint identity in receipt changed",
    )
    require(
        receipt.get("source_checkpoints") == expected_sources, "derivation checkpoint pins changed"
    )
    require(
        receipt.get("derive_script_sha256") == EXPECTED_DERIVE_SCRIPT_SHA256,
        "projected-mean derivation source hash changed",
    )
    require(
        receipt.get("source_code_sha256") == source_checked,
        "derivation source pins differ",
    )
    checkpoint_path = SCRATCH_ROOT / "artifacts/runs" / RUN_NAME / "model_ep1.pt"
    require(
        sha256(checkpoint_path) == EXPECTED_CHECKPOINT_SHA256, "derived checkpoint hash changed"
    )
    provenance_path = checkpoint_path.parent / "provenance.json"
    require(
        sha256(provenance_path) == EXPECTED_DERIVED_PROVENANCE_SHA256,
        "derived checkpoint provenance bytes changed",
    )
    output_provenance = read_json(provenance_path)
    require(
        output_provenance.get("checkpoint_sha256", {}).get(checkpoint_path.name)
        == EXPECTED_CHECKPOINT_SHA256,
        "derived checkpoint provenance hash changed",
    )
    require(
        sha256(FOLD_PATH) == plan["common_fitting_identity"]["split_manifest_sha256"],
        "F0 split manifest hash changed",
    )
    require(
        sha256(PREDICTOR_PATH) == plan["source_pins"]["training"]["scripts/predict_run.py"],
        "NLL predictor bytes differ from the frozen source pin",
    )
    return source_checked, current_serving_source


def reserve_alias_and_check_outputs() -> Path:
    """Create the unique artifact alias only after all output paths pass the no-overwrite gate."""
    alias = PROJECT_ROOT / "artifacts/runs" / RUN_NAME
    target = SCRATCH_ROOT / "artifacts/runs" / RUN_NAME
    outputs = [
        alias,
        PROJECT_ROOT / "cache" / f"pred_{RUN_NAME}_valid.npy",
        PROJECT_ROOT / "cache" / f"pred_{RUN_NAME}_valid.npy.partial",
        PROJECT_ROOT / "cache" / f"pred_{RUN_NAME}_valid.json",
        PROJECT_ROOT / "cache" / f"seqsums_{RUN_NAME}.npz",
        SCORE_PATH,
    ]
    collisions = [str(path) for path in outputs if os.path.lexists(path)]
    require(not collisions, f"refusing to overwrite projected-mean outputs: {collisions}")
    require((PROJECT_ROOT / "artifacts/runs").is_dir(), "project run directory is absent")
    require(target.is_dir(), "derived run directory is absent")
    require((target / "model_ep1.pt").is_file(), "derived epoch-1 checkpoint is absent")
    alias.symlink_to(target, target_is_directory=True)
    require(
        alias.is_symlink() and alias.resolve(strict=True) == target.resolve(strict=True),
        "alias check failed",
    )
    return alias


def main() -> None:
    """Pin, safely alias, and export the full-valid feature-only projected-mean predictions."""
    checked, current_serving_source = verify_projected_mean_pins()
    sys.stdout.write(
        json.dumps(
            {
                "status": "source_pins_verified",
                "training_head": EXPECTED_TRAINING_HEAD,
                "source_code_sha256": checked,
                "historical_serving_source_commit": EXPECTED_SERVING_SOURCE_COMMIT,
                "historical_serving_source_current_file_comparison": current_serving_source,
            },
            indent=2,
        )
        + "\n"
    )
    sys.stdout.flush()
    alias = reserve_alias_and_check_outputs()
    environment = os.environ.copy()
    environment.update(
        {
            "WNN_PROJECT_ROOT": str(PROJECT_ROOT),
            "PYTHONPATH": os.pathsep.join((str(TRAINING_ROOT), str(TRAINING_ROOT / "src"))),
            "NVIDIA_TF32_OVERRIDE": "0",
            "WNN_GPU_MEMORY_BYTES": "4000000000",
            "WNN_LOADER_WORKERS": "0",
            "WNN_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "ARROW_NUM_THREADS": "1",
            "TMPDIR": "/home/vlad/development/wunder/research_tmp",
        }
    )
    argv = [
        sys.executable,
        str(PREDICTOR_PATH),
        RUN_NAME,
        "valid",
        "--campaign-split-manifest",
        str(FOLD_PATH),
        "--campaign-full-valid-feature-only",
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
        "--loader-workers",
        "0",
        "--epoch",
        "1",
    ]
    alias_target = SCRATCH_ROOT / "artifacts/runs" / RUN_NAME
    try:
        subprocess.run(argv, cwd=TRAINING_ROOT, env=environment, check=True)
    finally:
        if alias.is_symlink() and os.readlink(alias) == str(alias_target):
            alias.unlink()
    sys.stdout.write(
        json.dumps(
            {
                "status": "predictor_returned_successfully",
                "argv": argv,
                "training_head": EXPECTED_TRAINING_HEAD,
                "verified_source_and_sidecar_pin_count": len(checked),
                "alias_removed": not os.path.lexists(alias),
                "prediction_path": str(PROJECT_ROOT / "cache" / f"pred_{RUN_NAME}_valid.npy"),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
