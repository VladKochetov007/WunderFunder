"""Prepare two bounded one-factor screens against immutable PR43 controls."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PREVIOUS = Path("/data/epanov_s/tabdata/runs/compact-reg-20261026")
OPERATORS = Path("/data/epanov_s/tabdata/runtime/compact-reg-20261026")
WAVES = ("regularization_factors", "learning_rate")


def read(path: Path) -> dict:
    """Read one bounded experiment document."""
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    """Hash a small contract or receipt."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    """Write a new artifact without replacing historical files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def prepare(root: Path) -> None:
    """Freeze eight configs with identical controls except their declared factor."""
    parent = read(PREVIOUS / "plan-selected.json")
    if len(list((PREVIOUS / "cohort-b128").glob("*/EXIT.json"))) != 4:
        raise ValueError("parent panel must be terminal before reusing its GPU allocation")
    for wave in WAVES:
        folder = root / wave
        plan = copy.deepcopy(parent)
        plan.update(
            experiment_wave=wave,
            run_root=str(folder / "cohort-b128"),
            tmp_root=str(root / "scratch" / wave),
            max_seconds=10800,
        )
        for job in plan["jobs"]:
            cell = job["name"]
            fold = int(cell[-1])
            control = next(j for j in parent["jobs"] if j["name"] == f"ctl_f{fold}")
            cfg = read(Path(control["config"]))
            if wave == "regularization_factors":
                if cell.startswith("ctl"):
                    cfg["model"]["dropout"] = 0.1
                    label = "dropout_only"
                else:
                    cfg["weight_decay"] = 0.1
                    label = "decay_only"
            else:
                cfg["lr"] = 5e-5 if cell.startswith("ctl") else 2e-4
                label = "lr_low" if cell.startswith("ctl") else "lr_high"
            path = folder / "configs" / f"{cell}.json"
            write(path, cfg)
            job.update(
                config=str(path),
                profile_receipt=str(folder / "profiles" / cell / "approved.json"),
                compile_cache=str(root / "scratch" / wave / "profiles" / cell / "inductor"),
                scientific_arm=label,
                config_sha256=sha(path),
                split_sha256=sha(Path(job["split_manifest"])),
            )
            job.pop("profile_sha256", None)
        plan.pop("frozen_unix", None)
        plan.pop("validation_monitor_sha256", None)
        write(folder / "plan-profile.json", plan)
    controls = []
    for job in parent["jobs"]:
        for path in sorted(
            (Path(parent["run_root"]) / job["name"] / "validation").glob("*/RESULT.json")
        ):
            x = read(path)
            if x["status"] != "complete":
                raise ValueError("only complete parent evaluations can be reused")
            controls.append(
                {
                    "path": str(path),
                    "sha256": sha(path),
                    "checkpoint_sha256": x["checkpoint_sha256"],
                }
            )
    write(root / "REUSED_RESULTS.json", controls)
    write(
        root / "CONTRACT.json",
        {
            "objective_mode": "competition",
            "verification": "development_holdout_not_fresh_confirmation",
            "waves": list(WAVES),
            "new_fits": 8,
            "epochs_per_fit": 5,
            "source_commit": parent["source_commit"],
            "source_root": parent["source_root"],
            "prior_campaign": str(PREVIOUS),
            "operators": str(OPERATORS),
            "epoch_selection": "per-arm common epoch maximizes mean F0/F2 inner WP; earliest tie",
            "scientific_limits": "One seed per fold; LR includes its coupled decay effect; no asymptotic claim.",
            "max_campaign_seconds": 21600,
        },
    )


def profile(root: Path) -> None:
    """Run both finite four-job resource probes before scientific admission."""
    for wave in WAVES:
        folder = root / wave
        with (folder / "PROFILE.log").open("xb") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(OPERATORS / "profile_panel.py"),
                    "--plan",
                    str(folder / "plan-profile.json"),
                    "--output",
                    str(folder / "profiles"),
                    "--updates",
                    "12",
                    "--max-seconds",
                    "1200",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    write(root / "PROFILES_COMPLETE.json", {"waves": list(WAVES), "scientific_fits_started": 0})


def main() -> None:
    """Prepare or profile only; approval is a separate inspected decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["prepare", "profile"], required=True)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else profile)(args.root)


if __name__ == "__main__":
    main()
