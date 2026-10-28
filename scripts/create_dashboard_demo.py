from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEMO_RUNS = ("synthetic-lstm-seed1", "synthetic-gru-seed1")


def _event(step: int, architecture: str, start: datetime) -> dict[str, object]:
    """Build one clearly synthetic event with a complete demo metric set."""
    phase = step / 8
    baseline = 0.5 if architecture == "LSTM" else 0.54
    metrics = {
        "train/loss": 1.1 * math.exp(-phase / 3) + 0.03,
        "train_eval/loss": 0.92 * math.exp(-phase / 2.5) + baseline / 10,
        "valid/loss": 0.98 * math.exp(-phase / 2.2) + baseline / 9,
        "train_eval/score": 0.2 + 0.68 * (1 - math.exp(-phase / 2.8)),
        "valid/score": 0.18 + 0.61 * (1 - math.exp(-phase / 2.5)),
        "train_eval/score/t0": 0.22 + 0.65 * (1 - math.exp(-phase / 2.7)),
        "valid/score/t0": 0.2 + 0.58 * (1 - math.exp(-phase / 2.3)),
        "train_eval/score/t1": 0.19 + 0.7 * (1 - math.exp(-phase / 2.9)),
        "valid/score/t1": 0.17 + 0.63 * (1 - math.exp(-phase / 2.4)),
        "optimizer/lr": 0.001 * (0.95**step),
        "optimizer/grad_norm_preclip": 1.3 + 0.2 * math.sin(phase),
        "optimizer/clip_fraction": 0.15 + 0.08 * abs(math.sin(phase)),
        "local/parameter_norm": 8.2 + phase * 0.03,
        "global/parameter_norm": 11.4 + phase * 0.02,
        "local/grad_norm": 0.7 + 0.12 * abs(math.sin(phase)),
        "global/grad_norm": 1.0 + 0.15 * abs(math.cos(phase)),
        "fusion/gate_saturation": 0.11 + 0.05 * abs(math.sin(phase / 2)),
        "attention/entropy": 2.3 - 0.12 * math.sin(phase),
        "attention/lag": 6.0 + 2.5 * abs(math.cos(phase)),
        "gpu/utilization_pct": 72 + 14 * math.sin(phase / 2),
        "gpu/memory_used_mb": 6_800 + step * 3,
        "gpu/power_w": 205 + 12 * math.sin(phase),
        "cpu/utilization_pct": 48 + 10 * math.cos(phase),
        "cpu/rss_mb": 3_100 + step * 1.7,
        "throughput/rows_per_second": 1_650 + 90 * math.sin(phase),
        "timing/train_step_seconds": 0.43 + 0.04 * abs(math.sin(phase)),
        "timing/evaluation_seconds": 4.8 + 0.35 * abs(math.cos(phase)),
    }
    return {
        "schema": "wnn.monitor.v1",
        "kind": "status" if step == 0 else ("evaluation" if step % 4 == 0 else "train_step"),
        "step": step,
        "epoch": step / 10,
        "wall_time": (start + timedelta(seconds=step * 3)).isoformat(),
        "metrics": metrics,
        "context": {"synthetic": True, "label": "demo fixture; no scientific result"},
    }


def create_demo(output: Path) -> list[Path]:
    """Create two new synthetic run folders without overwriting prior files."""
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / run_id for run_id in DEMO_RUNS]
    if any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError("a synthetic dashboard demo folder already exists")
    start = datetime.now(UTC)
    for index, path in enumerate(targets):
        architecture = "LSTM" if index == 0 else "GRU"
        path.mkdir()
        metadata = {
            "source": {"kind": "synthetic fixture", "name": "local demo generator"},
            "config": {
                "architecture": architecture,
                "seed": 1,
                "hidden_size": 128 if architecture == "LSTM" else 96,
                "validation_policy": "illustrative only; not an evaluation result",
            },
            "provenance": {"label": "Synthetic fixture only; no scientific result."},
            "synthetic": True,
        }
        (path / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        with (path / "monitor_events.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for step in range(40):
                handle.write(
                    json.dumps(_event(step, architecture, start), separators=(",", ":")) + "\n"
                )
    return targets


def main() -> None:
    """Parse an explicit destination and create a non-scientific demo packet."""
    parser = argparse.ArgumentParser(description="Create labeled synthetic dashboard demo runs.")
    parser.add_argument(
        "--output", required=True, type=Path, help="new or existing output directory"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        created = create_demo(args.output)
    except (OSError, FileExistsError) as exc:
        raise SystemExit(f"Cannot create dashboard demo: {exc}") from exc
    logging.getLogger(__name__).info(
        "Created synthetic demo runs: %s", ", ".join(path.name for path in created)
    )


if __name__ == "__main__":
    main()
