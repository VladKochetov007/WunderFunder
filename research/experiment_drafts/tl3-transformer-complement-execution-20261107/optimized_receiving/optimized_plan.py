from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

RUNTIME = PurePosixPath("/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1")
CELLS = (
    ("labels_s101001", "0,1"),
    ("complement_s101001", "2,3"),
    ("labels_s101002", "4,5"),
    ("complement_s101002", "6,7"),
)
COMMIT = "6ab2f679e70928f913b97d5a29f0d1e09ceacaa3"


def sha(path: Path) -> str:
    """Hash immutable, bounded configuration and control files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_updates(batch: int) -> int:
    """Use the canonical padded-rank batch and A4 temporal window count."""
    return math.ceil(math.ceil(10325 / 2) / batch) * 20 * 30


def read_panel(path: Path, expected_sha: str) -> dict:
    """Bind every exact production recipe before profile or fit admission."""
    if sha(path) != expected_sha:
        raise ValueError("panel manifest digest changed")
    value = json.loads(path.read_text())
    if value.get("source_commit") != COMMIT or value.get("batch_per_rank") not in (128, 256):
        raise ValueError("unsupported optimized source or batch")
    if [(item["cell"], item["pair"]) for item in value["cells"]] != list(CELLS):
        raise ValueError("optimized panel must preserve exactly four cells and physical pairs")
    for item in value["cells"]:
        config_path = Path(item["config"])
        if not config_path.is_absolute() or sha(config_path) != item["config_sha256"]:
            raise ValueError("production config path/digest differs")
        config = json.loads(config_path.read_text())
        if (
            config["batch_size"] != value["batch_per_rank"]
            or config["required_world_size"] != 2
            or config["epochs"] != 30
            or config["accumulate_chunks"] != 4
            or config["seed"] != int(item["cell"].split("_s")[1])
            or item["expected_updates"] != expected_updates(config["batch_size"])
        ):
            raise ValueError("production config budget differs")
    return value


def approval_ready(folder: Path) -> bool:
    """Accept only ROOT's exact copy of the completed native production profile."""
    path = folder / "approved_profile.json"
    if not path.exists():
        return False
    approved = json.loads(path.read_text())
    native = json.loads((folder / "profile/profile_receipt.json").read_text())
    if (
        approved.get("approved") is not True
        or approved.get("reviewer") != "ROOT"
        or native.get("status") != "profile_complete"
        or native.get("updates") != 50
        or native.get("errors") != []
        or native.get("scientific_checkpoint_written") is not False
        or native["identity"]["world_size"] != 2
        or any(approved.get(key) != item for key, item in native.items() if key != "approved")
    ):
        raise ValueError("ROOT approval changed or lacks successful exact native50-update evidence")
    return True


def commands(
    action: str, item: dict, runtime: Path, python: Path, run_root: Path
) -> list[list[str]]:
    """Build direct canonical S3 commands, preserving its original inference/score API."""
    source = runtime / "source-r3"
    cell, config = item["cell"], item["config"]
    folder = run_root / cell
    launch = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
    ]
    train = [
        str(source / "scripts/train_sequence.py"),
        "--config",
        config,
        "--data-root",
        str(runtime / "project"),
        "--split-manifest",
        "/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1/data/chronological_primary.json",
        "--monitor-reference-root",
        str(runtime / "reference"),
    ]
    if cell.startswith("complement_"):
        train += ["--reference-cache", str(runtime / "reference/fit")]
    if action in ("profile", "fit"):
        output = folder / action
        extra = (
            ["--profile-updates", "50"]
            if action == "profile"
            else ["--profile-receipt", str(folder / "approved_profile.json")]
        )
        return [
            [
                str(python),
                str(source / "scripts/observe_sequence.py"),
                "--output",
                str(output),
                "--interval",
                "10",
                "--max-seconds",
                "86400",
                "--",
                *launch,
                *train,
                "--output",
                str(output),
                *extra,
            ]
        ]
    if action == "evaluate":
        shared = [
            "--data-root",
            str(runtime / "project"),
            "--split-manifest",
            train[train.index("--split-manifest") + 1],
        ]
        return [
            [
                *launch,
                str(source / "scripts/predict_sequence.py"),
                "--checkpoint",
                str(folder / "fit/model.pt"),
                *shared,
                "--membership",
                "evaluation",
                "--output",
                str(folder / "evaluation"),
                "--reference-cache",
                str(runtime / "reference/evaluation"),
                "--blend-alpha",
                "0.1",
            ],
            [
                str(python),
                str(source / "scripts/score_sequence.py"),
                *shared,
                "--predictions",
                str(folder / "evaluation"),
                "--output",
                str(folder / "fit/score.json"),
            ],
        ]
    if action == "bundle":
        evidence = [
            "fit/" + name
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
            )
        ]
        evidence += [
            "evaluation/manifest.json",
            "approved_profile.json",
            "profile/profile_receipt.json",
        ]
        command = [
            str(python),
            str(source / "scripts/research_result_bundle.py"),
            "--root",
            str(folder),
            "--output-dir",
            str(folder / "bundle"),
        ]
        for name in evidence:
            command += ["--evidence", name]
        return [[*command, "--weight", "fit/model.pt", "--weight", "fit/training_state.pt"]]
    raise ValueError("unknown optimized cell action")


def main() -> None:
    """Freeze a separately identified matched batch128 or256 panel; launch nothing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(128, 256), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config-root", type=Path)
    args = parser.parse_args()
    original = args.base_config_root or Path(__file__).resolve().parents[1] / "configs"
    remote = RUNTIME / "throughput-engineering-r1/production"
    (args.output / "configs").mkdir(parents=True, exist_ok=True)
    cells = []
    for cell, pair in CELLS:
        config = json.loads((original / f"{cell}.json").read_text())
        config["batch_size"] = args.batch
        raw = (json.dumps(config, indent=2) + "\n").encode()
        path = args.output / "configs" / f"{cell}.json"
        if path.exists() and path.read_bytes() != raw:
            raise FileExistsError("existing production config differs")
        path.write_bytes(raw)
        cells.append(
            {
                "cell": cell,
                "pair": pair,
                "config": str(remote / "configs" / path.name),
                "config_sha256": hashlib.sha256(raw).hexdigest(),
                "expected_updates": expected_updates(args.batch),
            }
        )
    manifest = {
        "schema": "wnn.pr67-optimized-panel.v1",
        "source_commit": COMMIT,
        "batch_per_rank": args.batch,
        "cells": cells,
        "new_scientific_fit_cap": 4,
        "production_profile_cap": 8,
        "max_seconds": 86400,
        "collection_reserve_seconds": 3600,
    }
    path = args.output / "panel.json"
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("existing production panel differs")
    path.write_bytes(raw)


if __name__ == "__main__":
    main()
