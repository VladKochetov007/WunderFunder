"""Plot recovery training/resource telemetry; never reads labels or scores."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import time
from pathlib import Path

F = Path("/data/epanov_s/tabdata/runtime/final-submit-all-labeled-20261031-r1")
R = Path("/data/epanov_s/tabdata/runtime/w18-submit-recovery-20261103-r1")
PROGRESS = re.compile(
    r"^ep (\d+) batch (\d+)/(\d+) loss ([-+0-9.eE]+) "
    r"lr ([-+0-9.eE]+) seq/s ([-+0-9.eE]+)$"
)
os.environ.update(CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", MPLCONFIGDIR=str(R / "reports/.mpl"))
os.environ.update(
    {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
mpl = importlib.import_module("matplotlib")
mpl.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")


def _train(path: Path) -> list[tuple[float, float, float, float]]:
    """Parse only legacy progress rows, skipping the concurrent partial tail."""
    if not path.is_file():
        return []
    raw = path.read_bytes()
    raw = raw[: raw.rfind(b"\n") + 1] if raw and not raw.endswith(b"\n") else raw
    rows = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        match = PROGRESS.fullmatch(line)
        if match:
            epoch, batch, total = map(int, match.group(1, 2, 3))
            loss, lr, speed = map(float, match.group(4, 5, 6))
            if total and all(map(math.isfinite, (loss, lr, speed))):
                rows.append((epoch + (batch + 1) / total, loss, lr, speed))
    return rows


def _gib(value: object) -> float | None:
    match = (
        re.match(r"\s*([-+0-9.eE]+)\s*(kB|KB|bytes|B)?", value) if isinstance(value, str) else None
    )
    try:
        amount = float(match.group(1)) if match else float(value)
    except (TypeError, ValueError):
        return None
    scale = (
        1024**2
        if match and match.group(2) in ("kB", "KB")
        else 1024**3
        if match and match.group(2)
        else 1
    )
    return amount / scale if math.isfinite(amount) else None


def _resources(path: Path) -> list[tuple[float, float, float, float]]:
    if not path.is_file():
        return []
    raw = path.read_bytes()
    raw = raw[: raw.rfind(b"\n") + 1] if raw and not raw.endswith(b"\n") else raw
    records = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
            gpu = [float(value) for value in row["shared_gpu_util_used_free"].split(",")]
            host = row["host_memory"]
            total, available = _gib(host["MemTotal"]), _gib(host["MemAvailable"])
            if len(gpu) >= 2 and total is not None and available is not None:
                records.append(
                    (float(row["epoch_time"]), gpu[0], gpu[1] / 1024, max(0, total - available))
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if records:
        start = records[0][0]
        return [(t - start, util, mem, host) for t, util, mem, host in records]
    return []


def _save(path: Path, title: str, xlabel: str, rows: list, columns: tuple) -> None:
    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title)
    for axis, (column, label) in zip(axes, columns, strict=True):
        axis.plot([row[0] for row in rows], [row[column] for row in rows], lw=0.8)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel(xlabel)
    temporary = path.with_name(path.stem + ".tmp.png")
    fig.savefig(temporary, format="png", dpi=120)
    plt.close(fig)
    os.replace(temporary, path)


def main() -> None:
    """Poll at most once per minute; stop on W18 receipt or hard deadline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be in 1..60")
    out = R / "reports"
    out.mkdir(parents=True, exist_ok=True)
    deadline = float(json.loads((R / "ROOT_AUTHORITY_R1.json").read_text())["hard_deadline_epoch"])
    log = R / "logs/fit-r1.log"
    receipt = F / "control/final-submit-chain/receipts/fits/w18final_distilled_s5.json"
    while True:
        _save(
            out / "training_loss_lr_throughput.png",
            "W18 fit: 12,480 train+valid sequences; no independent evaluation",
            "Fractional epoch",
            _train(log),
            ((1, "Online batch loss"), (2, "Learning rate"), (3, "Sequences / second")),
        )
        _save(
            out / "resources_shared.png",
            "Shared resources; no quality or validation score",
            "Wall time from first sample (seconds)",
            _resources(out / "resources.jsonl"),
            ((1, "GPU utilization (%)"), (2, "GPU memory (GiB)"), (3, "Host RAM used (GiB)")),
        )
        try:
            done = json.loads(receipt.read_text(encoding="utf-8")).get("status") in {
                "completed",
                "failed",
            }
        except (OSError, json.JSONDecodeError):
            done = False
        if done or time.time() >= deadline:
            return
        time.sleep(min(args.poll_seconds, max(0, deadline - time.time())))


if __name__ == "__main__":
    main()
