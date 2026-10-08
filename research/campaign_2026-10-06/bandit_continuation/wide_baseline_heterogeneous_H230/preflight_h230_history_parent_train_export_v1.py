from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_AUTH_SHA256 = "ff19ae75d35cc60c73326060170bc5663b337091d6943724c06a4dde05c4deaf"
PROFILE_RESULT_SHA256 = "37936f99a16648a86b3574f045a30933059c546c205df28b074b845aa4392a4c"
PROFILE_TERMINAL_SHA256 = "f0d8001f6dc16d997e8d0cb60d0c0e7f72cfcfb36c230381d437e606d8afa63e"
PROFILE_CONTRACT_SHA256 = "d3b6641b01984e8f6bdfdc54fda9222ec8a5a27fc171545d2c5d73dc094a2d70"
SOURCE_MANIFEST_SHA256 = "01aa939ddb95f91b24859ddf324a66188a3254c11691cd39b40ce77865b3b2f2"
SOURCE_REVIEW_SHA256 = "3c4ea40cc53f8c05a47998e33932b6284716bc36e8c6659902a35c0b73615b6c"
RECIPE_SHA256 = "6d9d11f3116ebf61a2bc615e498d669ef4d6c080b588a2691752dc3f617cd8d1"
FIT_CONTRACT_SHA256 = "2d98c38c18c143175e286fcff56911aecf6516871776b9945f102e6169bb915b"
FIT_TERMINAL_SHA256 = "29baa03564f124712bd596f4132288492d549073a7acc4c9adea655c7384a216"
PROVENANCE_SHA256 = "3becd92c4775098cf16b9005cf6c45361bbe776eb5601a6ebe89faa8605e22dd"
CHECKPOINT_SHA256 = "478773bd3d0754d261f8f3c9d46a135d56c3596c269d2a856b760caef05b93c7"
H218_GRADE = "research/campaign_2026-10-06/bandit_continuation/attention_training_H218/H218_FOLD0_REFERENCE_BLEND_20261008_R1.json"
ATTEMPT_ID = "h230-history-parent-train-fp32-export-v1-20261008"
RUN_NAME = "bd26_h230_all_data_parent_b16_ep3_retry3_v1"
OUTPUT_NAME = "h230_history_parent_ep3_fp32_v1"


def sha256_file(path: Path) -> str:
    """Hash a file with bounded memory use."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stat_identity(path: Path) -> dict[str, int]:
    """Return size and nanosecond mtime without reading file contents."""
    info = path.stat()
    return {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def require_hash(root: Path, relative: str, expected: str) -> dict[str, str]:
    """Verify one small or stream-hashed artifact against its frozen digest."""
    path = root / relative
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {relative}: {actual}")
    return {"path": relative, "sha256": actual}


def main() -> int:
    """Validate the frozen train-export inputs without loading data arrays."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    campaign = Path("research/campaign_2026-10-06/bandit_continuation")
    h230 = campaign / "wide_baseline_heterogeneous_H230"
    runtime = campaign / "runtime/attempts"
    source_manifest_rel = Path(
        "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/"
        "H230_HISTORY_SWAP_SOURCE_MANIFEST_V1_20261008.json"
    )
    contract_rel = (
        h230 / "H230_HISTORY_PARENT_INFERENCE_PROFILE_AND_PHASE_SEQUENCE_V3_20261008.json"
    )
    profile_rel = h230 / "H230_HISTORY_PARENT_INFERENCE_PROFILE_ATTEMPT3_RESULT_V3_20261008.json"
    profile_terminal_rel = (
        runtime / "h230-history-parent-inference-profile-attempt3-v3-20261008.json"
    )
    recipe_rel = h230 / "H230_HISTORY_SWAP_SEED3_SOURCE_CACHE_RECIPE_V1_20261008.json"
    fit_contract_rel = h230 / "H230_ALLDATA_PARENT_FIT_RETRY3_CONTRACT_V1_20261008.json"
    fit_terminal_rel = runtime / "h230-all-data-parent-fit-b16-ep3-retry3-v1-20261008.json"
    provenance_rel = Path("artifacts/runs") / RUN_NAME / "provenance.json"
    checkpoint_rel = Path("artifacts/runs") / RUN_NAME / "model_ep3.pt"
    source_review_rel = Path(
        "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/"
        "H230_HISTORY_SWAP_ROOT_SOURCE_REVIEW_20261008_R1.json"
    )
    auth_rel = h230 / "H230_HISTORY_PARENT_TRAIN_EXPORT_ROOT_AUTHORIZATION_20261008_R1.json"
    script_rel = Path(__file__).resolve().relative_to(root)

    verified = [
        require_hash(root, str(auth_rel), ROOT_AUTH_SHA256),
        require_hash(root, str(profile_rel), PROFILE_RESULT_SHA256),
        require_hash(root, str(profile_terminal_rel), PROFILE_TERMINAL_SHA256),
        require_hash(root, str(contract_rel), PROFILE_CONTRACT_SHA256),
        require_hash(root, str(source_manifest_rel), SOURCE_MANIFEST_SHA256),
        require_hash(root, str(source_review_rel), SOURCE_REVIEW_SHA256),
        require_hash(root, str(recipe_rel), RECIPE_SHA256),
        require_hash(root, str(fit_contract_rel), FIT_CONTRACT_SHA256),
        require_hash(root, str(fit_terminal_rel), FIT_TERMINAL_SHA256),
        require_hash(root, str(provenance_rel), PROVENANCE_SHA256),
        require_hash(root, str(checkpoint_rel), CHECKPOINT_SHA256),
        {"path": H218_GRADE, "sha256": sha256_file(root / H218_GRADE)},
    ]
    profile = json.loads((root / profile_rel).read_text(encoding="utf-8"))
    manifest = json.loads((root / source_manifest_rel).read_text(encoding="utf-8"))
    auth = json.loads((root / auth_rel).read_text(encoding="utf-8"))

    source_root = root / manifest["source_root"]
    checked_sources = 0
    for member in manifest["files"]:
        path = source_root / member["path"]
        if path.stat().st_size != member["bytes"] or sha256_file(path) != member["sha256"]:
            raise ValueError(f"sealed source member mismatch: {member['path']}")
        checked_sources += 1
    if (
        checked_sources != 59
        or auth.get("status") != "approved_conditional_fresh_resource_admission"
    ):
        raise ValueError("source closure or root authorization status is unexpected")

    train_profile = profile["profile"]["splits"][0]
    input_identities: dict[str, Any] = {}
    for key, relative in (
        ("dataset_identity_stat_only", "datasets/train.parquet"),
        ("target_index_identity", "cache/train_targets.npz"),
    ):
        expected = train_profile[key]
        path = root / relative
        actual = stat_identity(path)
        if actual != {"bytes": expected["bytes"], "mtime_ns": expected["mtime_ns"]}:
            raise ValueError(f"profile-bound stat identity changed: {relative}")
        input_identities[relative] = {
            **actual,
            "profile_sha256": expected.get("sha256"),
            "content_was_read_or_rehashed": False,
        }

    cache_output = root / "cache" / f"pred_{OUTPUT_NAME}_train.npy"
    sidecar_output = root / "cache" / f"pred_{OUTPUT_NAME}_train.json"
    output_paths = [
        cache_output,
        cache_output.with_name(cache_output.name + ".partial"),
        sidecar_output,
        sidecar_output.with_name(sidecar_output.name + ".partial"),
    ]
    present = [str(path.relative_to(root)) for path in output_paths if path.exists()]
    if present:
        raise FileExistsError(f"train export destination already exists: {present}")
    attempt_path = root / runtime / f"{ATTEMPT_ID}.json"
    if attempt_path.exists():
        raise FileExistsError(f"durable attempt ID already exists: {ATTEMPT_ID}")
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < 25 * 1024**3:
        raise OSError(f"free disk below 25 GiB guard: {free_bytes}")

    receipt = {
        "schema": "h230-history-parent-train-export-preflight-v1",
        "record_id": "H230_HISTORY_PARENT_TRAIN_EXPORT_PREFLIGHT_V1_20261008",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "metadata_source_and_destination_checks_passed_pending_fresh_resource_admission",
        "owner": "h230_actual_delivery",
        "authorization": {"path": str(auth_rel), "sha256": ROOT_AUTH_SHA256},
        "attempt_id": ATTEMPT_ID,
        "run": RUN_NAME,
        "output_name": OUTPUT_NAME,
        "verified_frozen_artifacts": verified,
        "source_members_rehashed": checked_sources,
        "train_inputs": input_identities,
        "output_paths_confirmed_absent_at": datetime.now(UTC).isoformat(),
        "output_paths_confirmed_absent": [str(path.relative_to(root)) for path in output_paths],
        "free_disk_bytes": free_bytes,
        "large_input_policy": "Target-index NPZ validated by the frozen V3 profile SHA256 plus exact size/mtime stat; its 1.91GB contents were not loaded or rehashed.",
        "script_sha256": sha256_file(root / script_rel),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
