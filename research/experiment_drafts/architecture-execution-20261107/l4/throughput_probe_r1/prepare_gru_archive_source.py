from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

from prepare_monitor_source import DIAGNOSTICS, EXPECTED_PARENT, GATES, patched_sources

VALIDATOR = "src/wnn/training/sequence_config_validation.py"
OLD_CONDITION = (
    '    if not extended_runtime.issubset(value) or checkpoint_retention != "all_epochs":'
)
NEW_CONDITION = """    if not extended_runtime.issubset(value) or (
        checkpoint_retention != "all_epochs"
        and (checkpoint_retention != "final_only" or "early_stopping" in value)
    ):"""


def validator_source(original: str) -> str:
    """Allow report-only extended monitoring with externally archived safe states."""
    if original.count(OLD_CONDITION) != 1:
        raise ValueError("frozen validator condition differs; refuse an implicit adaptation")
    result = original.replace(OLD_CONDITION, NEW_CONDITION, 1)
    ast.parse(result)
    return result


def prepare(source: Path, inventory: Path, destination: Path) -> dict[str, object]:
    """Create a new three-file source capsule from frozen S1; preserve S1 and S2."""
    if hashlib.sha256(inventory.read_bytes()).hexdigest() != EXPECTED_PARENT:
        raise ValueError("parent inventory differs from the frozen620-file source-r1")
    parent = json.loads(inventory.read_bytes())["files"]
    if (
        len(parent) != 620
        or destination.exists()
        or destination.resolve().is_relative_to(source.resolve())
    ):
        raise ValueError("requires620 parent files and a fresh outside destination")
    gates, diagnostics = patched_sources(
        (source / GATES).read_text(), (source / DIAGNOSTICS).read_text()
    )
    changes = {
        GATES: gates,
        DIAGNOSTICS: diagnostics,
        VALIDATOR: validator_source((source / VALIDATOR).read_text()),
    }
    if any(len(text.splitlines()) > 300 for text in changes.values()):
        raise ValueError("changed source module exceeds the300-line bound")
    destination.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for name, sha in parent.items():
        relative = PurePosixPath(name)
        original = source.joinpath(*relative.parts)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or original.is_symlink()
            or not original.resolve().is_relative_to(source.resolve())
            or hashlib.sha256(original.read_bytes()).hexdigest() != sha
        ):
            raise ValueError(f"parent source bytes differ: {name}")
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in changes:
            target.write_text(changes[name], encoding="utf-8", newline="\n")
        else:
            shutil.copyfile(original, target)
        outputs[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    differences = {
        name: {"parent_sha256": parent[name], "new_sha256": sha}
        for name, sha in outputs.items()
        if parent[name] != sha
    }
    if differences.keys() != changes.keys():
        raise ValueError("source changes differ from the exact three-file whitelist")
    manifest = {
        "schema": "wnn.diagnostic-source-capsule.v1",
        "parent_inventory_sha256": EXPECTED_PARENT,
        "files": outputs,
    }
    manifest_path = destination.parent / f"{destination.name}-manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    receipt = {
        "schema": "wnn.gru-monitor-archive-source.v1",
        "parent_inventory_sha256": EXPECTED_PARENT,
        "new_inventory_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source": str(destination),
        "manifest": str(manifest_path),
        "changed_files": differences,
        "scope": "bounded GRU prefix gate telemetry; extended final_only monitoring allowed only without early_stopping",
        "external_archival_required_for_these_two_continuations": True,
        "historical_S1_and_S2_changed": False,
        "models_executed": 0,
        "checkpoint_migration_authorized_by_this_receipt": False,
    }
    receipt_path = destination.parent / f"{destination.name}-patch-receipt.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main() -> None:
    """Prepare a source artifact only; native qualification remains ROOT-owned."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.inventory, args.output), indent=2))


if __name__ == "__main__":
    main()
