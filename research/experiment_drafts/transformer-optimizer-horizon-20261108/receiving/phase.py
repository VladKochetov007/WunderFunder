from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from admission import read


def main() -> None:
    """Execute one cache-free fixed4E phase from the independently frozen source capsule."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--cell", default="")
    args = parser.parse_args()
    plan = read(args.plan)
    root, python, runtime = Path(plan["source_root"]), plan["python"], Path(plan["runtime"])
    launch = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
    ]
    if args.action == "source-check":
        subprocess.run(
            [
                python,
                "-m",
                "wnn.training.sequence_source_capsule",
                "--root",
                str(root),
                "--manifest",
                plan["source_manifest"],
                "--manifest-sha256",
                plan["source_manifest_sha256"],
                "--commit",
                plan["source_commit"],
            ],
            check=True,
        )
        return
    if args.action == "compare":
        subprocess.run(
            [python, str(Path(__file__).with_name("compare.py")), "--plan", str(args.plan)],
            check=True,
        )
        return
    if args.action == "plots":
        subprocess.run(
            [python, str(Path(__file__).with_name("plot_runs.py")), "--plan", str(args.plan)],
            check=True,
        )
        return
    cell = next(item for item in plan["cells"] if item["cell"] == args.cell)
    folder = runtime / "runs" / args.cell
    common = ["--data-root", plan["data_root"], "--split-manifest", plan["split_manifest"]]
    if args.action == "fit":
        trainer = [
            *launch,
            str(root / "scripts/train_sequence.py"),
            "--config",
            cell["config_path"],
            *common,
            "--output",
            str(folder / "fit"),
            "--profile-receipt",
            cell["approved_profile_receipt_path"],
        ]
        subprocess.run(
            [
                python,
                str(root / "scripts/observe_sequence.py"),
                "--output",
                str(folder / "fit"),
                "--interval",
                "10",
                "--max-seconds",
                "25200",
                "--",
                *trainer,
            ],
            check=True,
        )
    elif args.action == "evaluate":
        subprocess.run(
            [
                *launch,
                str(root / "scripts/predict_sequence.py"),
                "--checkpoint",
                str(folder / "fit/model.pt"),
                *common,
                "--membership",
                "evaluation",
                "--output",
                str(folder / "evaluation"),
            ],
            check=True,
        )
        subprocess.run(
            [
                python,
                str(root / "scripts/score_sequence.py"),
                *common,
                "--predictions",
                str(folder / "evaluation"),
                "--output",
                str(folder / "fit/score.json"),
            ],
            check=True,
        )
    elif args.action == "bundle":
        command = [
            python,
            str(root / "scripts/research_result_bundle.py"),
            "--root",
            str(folder),
            "--output-dir",
            str(folder / "bundle"),
        ]
        for name in (
            "run_metadata.json",
            "metrics.jsonl",
            "terminal_status.json",
            "score.json",
            "rank0.jsonl",
            "rank1.jsonl",
            "monitor_events.jsonl",
            "monitor_evaluations.jsonl",
            "monitor_progress.json",
            "monitor_observer.jsonl",
        ):
            command += ["--evidence", f"fit/{name}"]
        command += [
            "--evidence",
            "evaluation/manifest.json",
            "--weight",
            "fit/model.pt",
            "--weight",
            "fit/training_state.pt",
        ]
        for epoch in range(1, 5):
            for extension in ("json", "png", "svg"):
                command += ["--evidence", f"fit/plots/epoch-{epoch:02d}.{extension}"]
        subprocess.run(command, check=True)
    else:
        raise ValueError("unknown phase")


if __name__ == "__main__":
    main()
