from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    """Hash an artifact without retaining its contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pins(root: Path, pins: dict[str, str]) -> None:
    """Reject any drift from the admitted source/checkpoint identities."""
    for relative, expected in pins.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"frozen artifact changed: {relative}")


def output_paths(root: Path, run: str, split: str) -> list[Path]:
    """Include every path the predictor can replace or remove."""
    final = root / "cache" / f"pred_{run}_{split}.npy"
    return [
        final,
        final.with_name(final.name + ".partial"),
        final.with_suffix(".json"),
        root / "cache" / f"seqsums_{run}.npz",
    ]


def verify_absent(paths: list[Path]) -> None:
    """Treat dangling symlinks as existing outputs too."""
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to replace output: {path}")


def main() -> None:
    """Export a finite reviewed teacher cohort within the campaign GPU guardian."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if sha256(args.packet) != args.packet_sha256:
        raise ValueError("export packet hash changed")
    packet = json.loads(args.packet.read_text())
    root = Path(packet["root"]).resolve(strict=True)
    if packet["status"] != "admitted_exact_eight_exports":
        raise ValueError("packet is not admitted")
    if len(packet["exports"]) != 8:
        raise ValueError("expected exactly eight reviewed exports")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    from scripts.bandit_resources import atomic_json, utc_now
    from scripts.build_distill_cache import _audit_ancestor_closure, _inspect_teacher
    from wnn.splits import load_campaign_split

    verify_pins(root, packet["pins_sha256"])
    split = load_campaign_split(root / packet["split_manifest"], root)
    orders = {}
    for name, expected in packet["target_orders"].items():
        with np.load(root / "cache" / f"{name}_targets.npz", allow_pickle=False) as data:
            groups = np.asarray(data["row_group"], dtype="<i8")
        if (
            len(groups) != expected["rows"]
            or hashlib.sha256(groups.tobytes()).hexdigest() != expected["sha256"]
        ):
            raise ValueError(f"target order changed: {name}")
        orders[name] = groups.tolist()
    checked = set()
    for item in packet["exports"]:
        run, epoch = item["run"], item["epoch"]
        verify_absent(output_paths(root, run, item["split"]))
        if (run, epoch) not in checked:
            _audit_ancestor_closure(run, split, root / "artifacts/runs", f"model_ep{epoch}.pt")
            checked.add((run, epoch))
    required = packet["disk_reserve_bytes"] + sum(item["bytes"] for item in packet["exports"])
    if shutil.disk_usage(root).free < required:
        raise RuntimeError(f"insufficient free disk for cohort plus reserve: {required}")
    if args.preflight:
        print(json.dumps({"status": "preflight_pass", "exports": 8, "required_bytes": required}))
        return

    receipt_path = root / packet["receipt_path"]
    log_dir = root / packet["log_directory"]
    verify_absent([receipt_path, log_dir])
    log_dir.mkdir(parents=True)
    receipt = {
        "packet_sha256": args.packet_sha256,
        "started_utc": utc_now(),
        "status": "running",
        "exports": [],
    }
    atomic_json(receipt_path, receipt)
    try:
        for index, item in enumerate(packet["exports"]):
            if (root / packet["runtime"] / "STOP_ADMISSION.json").exists():
                receipt.update(status="drained", finished_utc=utc_now())
                break
            verify_pins(root, packet["pins_sha256"])
            verify_absent(output_paths(root, item["run"], item["split"]))
            required = packet["disk_reserve_bytes"] + item["bytes"]
            if shutil.disk_usage(root).free < required:
                raise RuntimeError("insufficient disk headroom before export")
            command = [
                sys.executable,
                "scripts/predict_run.py",
                item["run"],
                item["split"],
                "--epoch",
                str(item["epoch"]),
                "--inference-precision",
                "checkpoint",
            ]
            row = {
                "run": item["run"],
                "split": item["split"],
                "command": command,
                "started_utc": utc_now(),
                "phase": "predicting",
            }
            receipt["exports"].append(row)
            atomic_json(receipt_path, receipt)
            started = time.monotonic()
            with (log_dir / f"{index:02d}_{item['run']}_{item['split']}.log").open("xb") as output:
                subprocess.run(
                    command,
                    cwd=root,
                    check=True,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=packet["timeout_seconds_per_export"],
                )
            row.update(predict_seconds=time.monotonic() - started, phase="validating")
            atomic_json(receipt_path, receipt)
            final = output_paths(root, item["run"], item["split"])[0]
            if (
                final.stat().st_size != item["bytes"]
                or final.with_name(final.name + ".partial").exists()
            ):
                raise ValueError("prediction is incomplete or has wrong NPY file size")
            _teacher, arrays, records = _inspect_teacher(
                item["run"],
                split,
                root / "cache",
                root / "artifacts/runs",
                {item["split"]: orders[item["split"]]},
                item["epoch"],
            )
            del arrays
            row.update(
                phase="complete",
                finished_utc=utc_now(),
                total_seconds=time.monotonic() - started,
                validation=records[item["split"]],
            )
            atomic_json(receipt_path, receipt)
        else:
            receipt.update(status="complete", finished_utc=utc_now())
        atomic_json(receipt_path, receipt)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        receipt.update(
            status="needs_review",
            failure=f"{type(error).__name__}: {error}",
            finished_utc=utc_now(),
        )
        atomic_json(receipt_path, receipt)
        raise


if __name__ == "__main__":
    main()
