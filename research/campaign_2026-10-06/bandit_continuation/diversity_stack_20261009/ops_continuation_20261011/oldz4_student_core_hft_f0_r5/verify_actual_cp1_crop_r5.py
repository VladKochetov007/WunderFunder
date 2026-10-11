from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

from prepare_core_crop_r5 import PACKET_ROOT, read_object, sha256_file, verify_crop


def _load_crop_module(source_path: Path) -> ModuleType:
    """Import the pinned crop primitive from its staged snapshot."""
    spec = importlib.util.spec_from_file_location("r5_actual_crop_primitive", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import crop source {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_actual_cp1(project_root: Path) -> dict[str, Any]:
    """Crop a temporary copy of the actual CP1 and compare every model tensor."""
    project_root = project_root.resolve()
    contract = read_object(PACKET_ROOT / "CORE_CROP_CONTRACT_R5.json")
    bindings = read_object(PACKET_ROOT / "SOURCE_BINDINGS_R5.json")
    crop_binding = bindings["snapshots"]["core_crop"]
    source_script = (
        PACKET_ROOT / crop_binding["root_relative_path"] / "scripts/make_core_checkpoint.py"
    )
    if sha256_file(source_script) != crop_binding["files"]["scripts/make_core_checkpoint.py"]:
        raise ValueError("crop primitive differs from its frozen source binding")
    actual_checkpoint = project_root / contract["source_checkpoint_relative_path"]
    actual_provenance = project_root / contract["source_provenance_relative_path"]
    if sha256_file(actual_checkpoint) != contract["source_checkpoint_sha256"]:
        raise ValueError("actual CP1 checkpoint hash differs from the R5 binding")
    if sha256_file(actual_provenance) != contract["source_provenance_sha256"]:
        raise ValueError("actual CP1 provenance hash differs from the R5 binding")

    with tempfile.TemporaryDirectory(prefix="oldz4-corecrop-r5-") as temporary:
        temporary_root = Path(temporary)
        source_run_dir = temporary_root / "artifacts/runs" / contract["source_run"]
        source_run_dir.mkdir(parents=True)
        copied_checkpoint = source_run_dir / actual_checkpoint.name
        copied_provenance = source_run_dir / "provenance.json"
        shutil.copyfile(actual_checkpoint, copied_checkpoint)
        shutil.copyfile(actual_provenance, copied_provenance)
        before_checkpoint_sha = sha256_file(copied_checkpoint)
        before_provenance_sha = sha256_file(copied_provenance)

        module = _load_crop_module(source_script)
        module.ROOT = temporary_root
        result = module.derive_core(
            temporary_root,
            contract["source_run"],
            contract["source_checkpoint_epoch"],
            contract["destination_run"],
        )
        output_dir = temporary_root / "artifacts/runs" / contract["destination_run"]
        output_checkpoint = output_dir / copied_checkpoint.name
        output_provenance = output_dir / "provenance.json"
        verification = verify_crop(
            copied_checkpoint,
            copied_provenance,
            output_checkpoint,
            output_provenance,
            source_run=contract["source_run"],
            source_epoch=contract["source_checkpoint_epoch"],
        )
        if sha256_file(copied_checkpoint) != before_checkpoint_sha:
            raise ValueError("crop primitive mutated its source checkpoint")
        if sha256_file(copied_provenance) != before_provenance_sha:
            raise ValueError("crop primitive mutated source provenance")
        if result.get("checkpoint_sha256") != verification["output_checkpoint_sha256"]:
            raise ValueError("crop primitive result differs from actual output bytes")
        return {
            "schema": "wnn.oldz4-core-crop-producer-fixture.v1",
            "status": "PASS_ACTUAL_CP1_TEMPORARY_CROP",
            "actual_project_checkpoint_sha256": sha256_file(actual_checkpoint),
            "actual_project_provenance_sha256": sha256_file(actual_provenance),
            "temporary_source_checkpoint_sha256": before_checkpoint_sha,
            "temporary_source_provenance_sha256": before_provenance_sha,
            "output": verification,
            "source_tree_sha256": crop_binding["tree_sha256"],
        }


def main() -> None:
    """Run the actual-checkpoint crop fixture without writing project outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_actual_cp1(args.project_root), sort_keys=True))


if __name__ == "__main__":
    main()
