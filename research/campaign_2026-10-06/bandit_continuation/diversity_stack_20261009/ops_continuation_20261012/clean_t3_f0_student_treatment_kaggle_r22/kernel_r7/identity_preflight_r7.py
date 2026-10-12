from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

EXPECTED_ID = "vladkochetov07/wnn-clean-t3-f0-student-20261012-r24"
EXPECTED_TITLE = "wnn-clean-t3-f0-student-20261012-r24"
OLD_ID = "vladkochetov07/wnn-clean-t3-f0-muonall-student-20261012-r22-runner"
ROOT = Path(__file__).resolve().parent.parent
R6_DIR = ROOT / "kernel_r6"
R7_DIR = ROOT / "kernel_r7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def import_kernel(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("r7_kernel_identity_fixture", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import kernel source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def release_fixture(kernel: ModuleType) -> tuple[dict[str, object], dict[str, object]]:
    source_manifest_sha = "a" * 64
    plan: dict[str, object] = {
        "source_manifest_sha256": source_manifest_sha,
        "cache": {
            "metadata": {"sha256": "b" * 64},
            "train": {"sha256": "c" * 64},
            "fit_valid": {"sha256": "d" * 64},
        },
        "parents": {
            "control": {"checkpoint": {"sha256": "e" * 64}, "provenance": {"sha256": "f" * 64}},
            "treatment": {"checkpoint": {"sha256": "1" * 64}, "provenance": {"sha256": "2" * 64}},
        },
    }
    bindings: dict[str, object] = {
        "pair_plan_sha256": kernel.PAIR_PLAN_SHA256,
        "asset_manifest_sha256": kernel.ASSET_MANIFEST_SHA256,
        "bundle_sha256": kernel.BUNDLE_SHA256,
        "bundle_manifest_sha256": kernel.SOURCE_BUNDLE_MANIFEST_SHA256,
        "source_manifest_sha256": source_manifest_sha,
        "cache_metadata_sha256": plan["cache"]["metadata"]["sha256"],
        "cache_train_sha256": plan["cache"]["train"]["sha256"],
        "cache_valid_sha256": plan["cache"]["fit_valid"]["sha256"],
        "parent_checkpoint_sha256": {
            arm: item["checkpoint"]["sha256"] for arm, item in plan["parents"].items()
        },
        "parent_provenance_sha256": {
            arm: item["provenance"]["sha256"] for arm, item in plan["parents"].items()
        },
    }
    receipt: dict[str, object] = {
        "schema": "wnn.clean_f0_student_treatment_root_release.r2",
        "status": "admitted",
        "authorized": True,
        "target": {
            "kernel_id": EXPECTED_ID,
            "source_dataset": kernel.SOURCE_DATASET,
            "asset_dataset": kernel.ASSET_DATASET,
        },
        "binding": bindings,
        "quota_budget": {
            "wall_ceiling_seconds": kernel.WALL_SECONDS,
            "single_device_gpu_seconds_upper_bound": kernel.WALL_SECONDS,
            "provisioned_gpu_count": kernel.PROVISIONED_GPU_COUNT,
            "cohort_gpu_seconds_upper_bound": kernel.COHORT_GPU_SECONDS_UPPER_BOUND,
            "paid_spending_authorized": False,
        },
        "profile_authorized": True,
        "fit_authorized": False,
    }
    return receipt, plan


def validate() -> dict[str, object]:
    old_kernel = (R6_DIR / "kernel.py").read_bytes()
    new_kernel = (R7_DIR / "kernel.py").read_bytes()
    old_literal = f'KERNEL_ID = "{OLD_ID}"'.encode()
    new_literal = f'KERNEL_ID = "{EXPECTED_ID}"'.encode()
    if old_kernel.count(old_literal) != 1:
        raise AssertionError("R6 identity literal is not unique in the frozen parent")
    if new_kernel != old_kernel.replace(old_literal, new_literal, 1):
        raise AssertionError("R7 kernel differs from R6 outside the KERNEL_ID literal")

    old_metadata = load_json(R6_DIR / "kernel-metadata.json")
    metadata = load_json(R7_DIR / "kernel-metadata.json")
    expected_metadata = dict(old_metadata)
    expected_metadata["id"] = EXPECTED_ID
    expected_metadata["title"] = EXPECTED_TITLE
    if metadata != expected_metadata:
        raise AssertionError("R7 metadata differs beyond its id/title fields")
    title = metadata.get("title")
    kernel_id = metadata.get("id")
    if not isinstance(title, str) or not 5 <= len(title) <= 50:
        raise AssertionError("Kaggle title must contain between 5 and 50 characters")
    if re.fullmatch(r"[a-z0-9-]+", title) is None:
        raise AssertionError("title is not a valid title-derived slug")
    if kernel_id != f"vladkochetov07/{title}" or kernel_id != EXPECTED_ID:
        raise AssertionError("metadata id does not match the title-derived target slug")

    kernel = import_kernel(R7_DIR / "kernel.py")
    if metadata["id"] != kernel.KERNEL_ID:
        raise AssertionError("runtime KERNEL_ID differs from metadata id")
    receipt, plan = release_fixture(kernel)
    kernel.validate_root_release(receipt, plan, plan["source_manifest_sha256"])
    old_target = dict(receipt)
    old_target["target"] = dict(receipt["target"], kernel_id=OLD_ID)
    try:
        kernel.validate_root_release(old_target, plan, plan["source_manifest_sha256"])
    except PermissionError:
        pass
    else:
        raise AssertionError("root release for the previous R6 kernel ID was accepted")

    return {
        "schema": "wnn.clean_t3_f0_student_kernel_identity_preflight.r7",
        "result": "pass",
        "fixture_only": True,
        "kernel_id": kernel_id,
        "title": title,
        "title_characters": len(title),
        "title_limit": 50,
        "title_derived_slug_matches": True,
        "runtime_release_target_matches": True,
        "old_r6_target_rejected": True,
        "r6_kernel_sha256": sha256(R6_DIR / "kernel.py"),
        "r7_kernel_sha256": sha256(R7_DIR / "kernel.py"),
        "r6_metadata_sha256": sha256(R6_DIR / "kernel-metadata.json"),
        "r7_metadata_sha256": sha256(R7_DIR / "kernel-metadata.json"),
        "preflight_source_sha256": sha256(Path(__file__).resolve()),
        "provider_or_data_action": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the R7 Kaggle identity-only successor.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite pre-existing receipt: {args.output}")
    result = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
