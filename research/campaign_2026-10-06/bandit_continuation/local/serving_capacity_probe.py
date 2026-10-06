"""Measure fixed native single-trunk and two-trunk workloads under current load."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

import engine_fp16_check as engine  # noqa: E402
from package_submission import load_state  # noqa: E402


def sha256(path: Path) -> str:
    """Hash a small, specified probe input without retaining its bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Replay one complete stream with fixed weights and counterbalanced order."""
    base = Path(__file__).resolve().parent
    output = base / "E509_SERVING_CAPACITY_RESULT.json"
    if output.exists():
        raise FileExistsError(output)
    cpu = max(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    package = ROOT / "submissions/c20261004_combo_s4_i16/pkg"
    library = package / "v3"
    native = engine.load_native(library)
    states = []
    checkpoint_paths = []
    for seed in (3, 4):
        run = f"c20261003_combo_s{seed}_f0"
        checkpoint_paths.append(ROOT / "artifacts/runs" / run / "model_ep3.pt")
        state = load_state(f"{run}@ep3")
        for key in ("head.2.weight", "head.2.bias"):
            state[key] = np.ascontiguousarray(state[key][:2])
        states.append(state)
    cases = {
        "public_champion_single_trunk": [engine.load_npz(package / "weights_0.npz")],
        "two_warm_parent_trunks": states,
    }
    data = ROOT / "cache/valid_subset20.parquet"
    columns = ["seq_ix", "need_prediction", *engine.FEATURE_COLUMNS]
    table = pq.ParquetFile(data).read_row_group(0, columns=columns, use_threads=False)
    sequence = int(table["seq_ix"][0].as_py())
    features = np.column_stack([table[c].to_numpy() for c in engine.FEATURE_COLUMNS])
    features = np.ascontiguousarray(features, dtype=np.float32)
    need = table["need_prediction"].to_numpy().astype(bool)
    del table

    def measure(name: str) -> dict[str, object]:
        model = engine.ensemble(native, cases[name], "i16", budget_s=None)
        if model.precision != "i16":
            raise ValueError(f"unexpected effective precision: {model.precision}")
        cycles = [0]
        seconds, predictions = engine.harness(model, sequence, features, need, cycles)
        return {
            "case": name,
            "seconds": seconds,
            "microseconds_per_row": seconds * 1e6 / len(features),
            "cycles_per_row": cycles[0] / len(features) if engine.CYCLES.fd >= 0 else None,
            "prediction_sha256": hashlib.sha256(predictions[need].tobytes()).hexdigest(),
        }

    names = list(cases)
    warmups = [measure(name) for name in names]
    trials = []
    for repeat in range(4):
        for name in names if repeat % 2 == 0 else list(reversed(names)):
            trials.append({"repeat": repeat, **measure(name)})
    for name in names:
        hashes = {row["prediction_sha256"] for row in warmups + trials if row["case"] == name}
        if len(hashes) != 1:
            raise ValueError(f"non-repeatable callback: {name}")
    medians = {
        name: statistics.median(row["seconds"] for row in trials if row["case"] == name)
        for name in names
    }
    ratio = medians[names[1]] / medians[names[0]]
    result = {
        "id": "E-509",
        "finished_utc": datetime.now(UTC).isoformat(),
        "status": "completed_capacity_screen_only",
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "loadavg": list(os.getloadavg()),
        "cgroup": Path("/proc/self/cgroup").read_text(),
        "inputs": {
            str(path): sha256(path)
            for path in [
                *checkpoint_paths,
                package / "weights_0.npz",
                library / "wnn_native.abi3.so",
                data,
                Path(__file__),
                ROOT / "scripts/engine_fp16_check.py",
            ]
        },
        "row_group": 0,
        "rows": len(features),
        "precision": "i16",
        "warmups": warmups,
        "trials": trials,
        "median_seconds": medians,
        "pair_to_champion_median_ratio": ratio,
        "requires_serving_cost_followup": ratio > 1.4,
        "limitation": "Concurrent GPU training and support work; one process and laptop CPU. Warm parents size the student architecture but are not candidate checkpoints. No public-time guarantee, quality score, or submission approval.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "median_ratio": ratio}))


if __name__ == "__main__":
    main()
