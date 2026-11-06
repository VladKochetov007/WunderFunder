from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from reference_assets import MEMBER_RUNS, audit_ancestors, bank_path, check_source, write_new_json

from wnn.splits import file_sha256, load_campaign_split
from wnn.training.sequence_reference_contract import canonical_sha256
from wnn.training.sequence_reference_pins import PUBLISHED_RUNS


def main() -> None:
    """Export one fixed PR49 member/source with the actual pinned legacy predictor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-source-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--member", required=True, choices=tuple(MEMBER_RUNS))
    parser.add_argument("--source", required=True, choices=("train", "valid"))
    args = parser.parse_args()
    source_root, data_root = args.legacy_source_root.resolve(), args.data_root.resolve()
    split_path = args.split_manifest.resolve()
    source_commit = check_source(source_root)
    split = load_campaign_split(split_path, data_root)
    producer = audit_ancestors(data_root, split)
    run = MEMBER_RUNS[args.member]
    bank = bank_path(data_root, run, args.source)
    receipt = bank.with_suffix(".export.json")
    for path in (bank, bank.with_suffix(".json"), bank.with_name(bank.name + ".partial"), receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"preserve the existing reference export: {path}")
    predictor = source_root / "scripts/predict_run.py"
    command = [
        sys.executable,
        str(predictor),
        run,
        args.source,
        "--epoch",
        str(PUBLISHED_RUNS[run]["epoch"]),
        "--output-name",
        f"tl3ref_{run}",
        "--campaign-split-manifest",
        str(split_path),
        "--inference-precision",
        "fp32",
        "--device",
        "cuda",
    ]
    command.append(
        "--campaign-full-train-index-feature-only"
        if args.source == "train"
        else "--campaign-evaluation-only"
    )
    env = dict(os.environ)
    env.update(
        WNN_PROJECT_ROOT=str(data_root),
        PYTHONPATH=str(source_root / "src"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    subprocess.run(command, env=env, cwd=source_root, check=True)
    check_source(source_root)
    if audit_ancestors(data_root, split) != producer:
        raise ValueError("reference ancestors changed during prediction export")
    write_new_json(
        receipt,
        {
            "schema": "pr49-reference-export-v1",
            "complete": True,
            "run": run,
            "source": args.source,
            "source_commit": source_commit,
            "predictor_sha256": file_sha256(predictor),
            "argv": command,
            "reference_sha256": canonical_sha256(producer),
            "prediction_sha256": file_sha256(bank),
            "metadata_sha256": file_sha256(bank.with_suffix(".json")),
            "producer": producer,
        },
    )


if __name__ == "__main__":
    main()
