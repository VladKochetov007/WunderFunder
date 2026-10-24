"""Select a measured PR32 batch after the user's larger-memory continuation."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path("/data/epanov_s/tabdata")
CAMPAIGN = ROOT / "runs/pr32-blackwell-20261024"


def sha(path: Path) -> str:
    """Hash a small engineering receipt."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate(directory: Path) -> dict:
    """Read completed real-data work with rank-aligned warmed timing."""
    operator_path = directory / "operator.json"
    operator = json.loads(operator_path.read_text())
    config = json.loads((directory / "config.json").read_text())
    profile_path = directory / "profile/profile_receipt.json"
    profile = json.loads(profile_path.read_text())
    if operator["exit_code"] != 0 or profile["errors"] or profile["updates"] != 12:
        raise ValueError(f"Unsuccessful engineering profile: {directory}")
    ranks = [
        {
            row["step"]: row
            for row in map(
                json.loads,
                (directory / "profile" / f"rank{rank}.jsonl").read_text().splitlines(),
            )
        }
        for rank in range(2)
    ]
    seconds = [max(ranks[rank][s]["window_seconds"] for rank in range(2)) for s in range(6, 13)]
    rates = [ranks[0][s]["forwarded_rows"] / seconds[s - 6] for s in range(6, 13)]
    peak = profile["peak_reserved_bytes"]
    total = min(profile["device_total_memory_bytes"])
    return {
        "model": "t" + str(config["model"]["hidden"]),
        "batch_size": config["batch_size"],
        "profile_receipt": str(profile_path),
        "profile_sha256": sha(profile_path),
        "operator_receipt": str(operator_path),
        "operator_sha256": sha(operator_path),
        "status": "complete",
        "median_rows_per_second": statistics.median(rates),
        "median_rankmax_seconds": statistics.median(seconds),
        "peak_reserved_bytes": peak,
        "device_total_memory_bytes": total,
        "eligible": peak <= 0.9 * total,
    }


def main() -> None:
    """Retain every bounded candidate or failed extension; select without holdout scores."""
    extension = CAMPAIGN / "extended-batch-r2-w2"
    if not (extension / "COMPLETE.json").is_file():
        raise RuntimeError("The extended profile round is still running")
    candidates = []
    for name in ("batch-round8-16-r1-w2", "batch-round32-64-r1-w2"):
        directory = CAMPAIGN / name
        for cell in ("t1280_f0", "t1792_f0", "t1280_f2", "t1792_f2"):
            candidates.append(candidate(directory / cell))
    failed = []
    for cell in ("t1280_f0", "t1280_f2"):
        directory = extension / cell
        operator_path = directory / "operator.json"
        operator = json.loads(operator_path.read_text())
        if operator["exit_code"] == 0:
            candidates.append(candidate(directory))
        else:
            log = directory / "console.log"
            config = json.loads((directory / "config.json").read_text())
            failed.append(
                {
                    "model": "t1280",
                    "batch_size": config["batch_size"],
                    "status": "oom" if "out of memory" in log.read_text().lower() else "failed",
                    "operator_receipt": str(operator_path),
                    "operator_sha256": sha(operator_path),
                    "log": str(log),
                    "log_sha256": sha(log),
                }
            )
    # The newly admitted large B64 must also work on both actual production folds.
    for cell in ("t1792_f0", "t1792_f2"):
        proof = candidate(extension / cell)
        if not proof["eligible"]:
            raise RuntimeError(f"{cell}: actual B64 production fold exceeds the new memory gate")
    selected = {}
    for model in ("t1280", "t1792"):
        eligible = [row for row in candidates if row["model"] == model and row["eligible"]]
        best = max(row["median_rows_per_second"] for row in eligible)
        near = [row for row in eligible if row["median_rows_per_second"] >= 0.97 * best]
        selected[model] = min(near, key=lambda row: row["batch_size"])["batch_size"]
    cpu = CAMPAIGN / "batch-loader-b64-128-r1.json"
    result = {
        "schema": "wnn.pr32.batch-selection.v2",
        "source_commit": "f1beeb4c210a9da01735e03b1ff6e27be878b873",
        "engineering_only": True,
        "quality_equivalence_claim": False,
        "minimum_headroom_fraction": 0.10,
        "tie_tolerance_fraction": 0.03,
        "rule": "Maximize warmed rank-max global rows/sec among clean profiles with at least10% VRAM headroom; within3% choose the smaller batch.",
        "authority": "User explicitly asked to admit the roughly13% headroom B64 large model and test B128 on630M.",
        "candidates": candidates,
        "failed_profiles": failed,
        "selected_per_rank_batch": selected,
        "cpu_loading_receipt": str(cpu),
        "cpu_loading_receipt_sha256": sha(cpu),
        "heldout_used_for_selection": False,
    }
    destination = ROOT / "receipts/pr32-batch-r4-selection.json"
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "selected": selected,
                "failed_profiles": failed,
                "candidates": [
                    {
                        k: row[k]
                        for k in (
                            "model",
                            "batch_size",
                            "median_rows_per_second",
                            "median_rankmax_seconds",
                            "peak_reserved_bytes",
                            "device_total_memory_bytes",
                            "eligible",
                        )
                    }
                    for row in candidates
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
