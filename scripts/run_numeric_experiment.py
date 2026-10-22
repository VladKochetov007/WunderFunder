"""Run one explicit stage of one fixed PR35 cell; no queue or automatic admission."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Dispatch a frozen cell to the shared prepare, fit, predict and score entrypoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "profile", "fit", "evaluate"))
    parser.add_argument("--fold", required=True, type=int, choices=(0, 2))
    parser.add_argument("--arm", required=True, choices=("raw_events", "uniform16", "quantile16"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--profile-receipt", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    split = args.split_manifest.resolve()
    if json.loads(split.read_text())["fold"] != args.fold:
        parser.error("split manifest fold does not match the selected cell")
    if args.stage == "fit" and args.profile_receipt is None:
        parser.error("fit requires the approved real-device --profile-receipt")
    if args.stage != "fit" and (args.profile_receipt or args.resume):
        parser.error("profile receipt/resume apply only to fit")
    repository = Path(__file__).resolve().parents[1]
    output = args.output_root.resolve()
    data = ["--data-root", str(args.data_root.resolve()), "--split-manifest", str(split)]
    cell = f"f{args.fold}_{args.arm}"
    config = (
        repository
        / "research/experiment_drafts/typed-bin-feature-encoding/configs"
        / f"{cell}.json"
    )
    artifact = output / "preprocessing" / f"f{args.fold}.json"
    run = output / "runs" / cell
    if args.stage == "prepare":
        commands = [["scripts/prepare_numeric_inputs.py", *data, "--output", str(artifact)]]
    elif args.stage in ("profile", "fit"):
        destination = output / "profiles" / cell if args.stage == "profile" else run
        command = [
            "scripts/train_sequence.py",
            *data,
            "--config",
            str(config),
            "--input-artifact",
            str(artifact),
            "--output",
            str(destination),
        ]
        if args.stage == "profile":
            command += ["--profile-updates", "4"]
        else:
            command += ["--profile-receipt", str(args.profile_receipt.resolve())]
            if args.resume:
                command += ["--resume", str(args.resume.resolve())]
        commands = [command]
    else:
        predictions = run / "evaluation"
        commands = [
            [
                "scripts/predict_sequence.py",
                *data,
                "--checkpoint",
                str(run / "model.pt"),
                "--membership",
                "evaluation",
                "--output",
                str(predictions),
            ],
            [
                "scripts/score_sequence.py",
                *data,
                "--predictions",
                str(predictions),
                "--output",
                str(run / "score.json"),
            ],
        ]
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=repository, check=True)


if __name__ == "__main__":
    main()
