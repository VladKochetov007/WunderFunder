from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

S1_SHA = "7d3324230effb050ff083b939e26ad638e5df5c5b72ea70b0f8aed169997db4a"
S3_SHA = "a28f86a74c6d8617c6b4df37fbb9c972dd9148e2ea1e520a27888109a744e8f6"
RESUME = "src/wnn/training/sequence_fit_resume.py"
S3_CHANGES = {
    "src/wnn/training/sequence_monitor_gates.py",
    "src/wnn/training/sequence_monitor_diagnostics.py",
    "src/wnn/training/sequence_config_validation.py",
}
OLD_RESTORE = """        coordinated_call(
            lambda: (
                monitor.restore_offsets(monitor_offsets, global_step)
                if context.is_primary and monitor is not None
                else True
            ),
            "safe-boundary monitor log restore",
            context,
        )"""
NEW_RESTORE = '''        def restore_monitor() -> bool:
            """Restore monitor prefixes and acknowledge the coordinated side effect."""
            if context.is_primary and monitor is not None:
                monitor.restore_offsets(monitor_offsets, global_step)
            return True

        coordinated_call(restore_monitor, "safe-boundary monitor log restore", context)'''


def resume_source(original: str) -> str:
    """Replace only the None-returning monitor callback, retaining exception semantics."""
    if original.count(OLD_RESTORE) != 1:
        raise ValueError("frozen canonical resume anchor differs; refuse implicit adaptation")
    result = original.replace(OLD_RESTORE, NEW_RESTORE, 1)
    ast.parse(result)
    return result


def checked_inventory(root: Path, manifest: Path, expected_sha: str) -> dict[str, str]:
    """Verify every file in one externally pinned620-file source capsule."""
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("source inventory differs from its native frozen pin")
    files = json.loads(manifest.read_bytes())["files"]
    if not isinstance(files, dict) or len(files) != 620:
        raise ValueError("source inventory must contain exactly620 files")
    for name, sha in files.items():
        relative = PurePosixPath(name)
        path = root.joinpath(*relative.parts)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != sha
        ):
            raise ValueError(f"frozen source bytes differ: {name}")
    return files


def prepare(
    source: Path, inventory: Path, s3: Path, s3_inventory: Path, destination: Path
) -> dict[str, object]:
    """Build S4 directly against S1 ancestry, preserving exact S3 diagnostic bytes."""
    old = checked_inventory(source, inventory, S1_SHA)
    diagnostic = checked_inventory(s3, s3_inventory, S3_SHA)
    if (
        old.keys() != diagnostic.keys()
        or {name for name in old if old[name] != diagnostic[name]} != S3_CHANGES
    ):
        raise ValueError("S3 differs from S1 outside the qualified three-file amendment")
    if (
        destination.exists()
        or destination.resolve().is_relative_to(source.resolve())
        or destination.resolve().is_relative_to(s3.resolve())
    ):
        raise ValueError("S4 requires a fresh destination outside both frozen sources")
    patched = resume_source((source / RESUME).read_text())
    if len(patched.splitlines()) > 300:
        raise ValueError("resume module exceeds the300-line bound")
    destination.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for name in diagnostic:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == RESUME:
            target.write_text(patched, encoding="utf-8", newline="\n")
        else:
            shutil.copyfile(s3 / name, target)
        outputs[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    changes = {
        name: {"parent_sha256": old[name], "new_sha256": sha}
        for name, sha in outputs.items()
        if old[name] != sha
    }
    if set(changes) != S3_CHANGES | {RESUME} or {
        name for name in outputs if outputs[name] != diagnostic[name]
    } != {RESUME}:
        raise ValueError("S4 escaped the exact fourth-file resume amendment")
    manifest_path = destination.parent / f"{destination.name}-manifest.json"
    manifest = {
        "schema": "wnn.diagnostic-source-capsule.v1",
        "parent_inventory_sha256": S1_SHA,
        "diagnostic_parent_inventory_sha256": S3_SHA,
        "files": outputs,
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    receipt = {
        "schema": "wnn.gru-monitor-resume-source.v1",
        "parent_inventory_sha256": S1_SHA,
        "new_inventory_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "diagnostic_parent_inventory_sha256": S3_SHA,
        "source": str(destination),
        "manifest": str(manifest_path),
        "changed_files": changes,
        "changed_files_relative_to_S3": [RESUME],
        "scope": "S3 unchanged except coordinated monitor restoration returnsTrue; genuine restore exceptions propagate",
        "numerical_training_changes": [],
        "historical_S1_and_S3_changed": False,
        "external_safe_state_archival_required": True,
        "models_executed": 0,
    }
    with (destination.parent / f"{destination.name}-patch-receipt.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main() -> None:
    """Prepare only new source bytes; native profiles and admission remain ROOT-owned."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--s3-source", type=Path, required=True)
    parser.add_argument("--s3-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.source, args.inventory, args.s3_source, args.s3_inventory, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
