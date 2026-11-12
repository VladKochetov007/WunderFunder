from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from admission import digest, read


def main() -> None:
    """Create one ROOT-reviewable concrete receiver plan from the frozen profile/config packet."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-plan", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path)
    args = parser.parse_args()
    plan = read(args.profile_plan)
    if digest(args.source_manifest) != args.source_manifest_sha256:
        raise ValueError("trusted source manifest SHA differs")
    manifest = read(args.source_manifest)
    os.environ.update(plan["environment"])
    sys.path.insert(0, str(Path(plan["source_root"]) / "src"))
    from wnn.training.sequence_config import load_config
    from wnn.training.sequence_provenance import canonical_sha256, source_identity

    plan["expected_training_source_sha256"] = source_identity(
        Path(plan["source_root"]), Path(plan["source_root"]) / "scripts/train_sequence.py"
    )["sha256"]
    plan.update(
        runtime=str(args.runtime),
        python=str(args.python),
        source_manifest=str(args.source_manifest),
        source_manifest_sha256=args.source_manifest_sha256,
        source_commit=manifest["commit"],
        authority=str(args.authority),
        authority_sha256=digest(args.authority),
    )
    for cell in plan["cells"]:
        cell["config_path"] = str(args.runtime / "configs" / cell["config"])
        if digest(Path(cell["config_path"])) != cell["config_sha256"]:
            raise ValueError("staged production config differs")
        cell["normalized_config_sha256"] = canonical_sha256(load_config(Path(cell["config_path"])))
        cell["profile_receipt_path"] = str(
            (args.profile_root or args.runtime / "profiles-r2")
            / cell["cell"]
            / "profile_receipt.json"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({"plan": str(args.output), "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
