from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def changes(x: np.ndarray) -> np.ndarray:
    """Identify adjacent changes in an input block without future access."""
    return np.any(x[1:] != x[:-1], axis=1)


def conditional_rates(targets: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    """Count diagnostic label changes conditional on unchanged inputs."""
    delta = np.diff(targets, axis=0)
    sum_changed = np.abs(delta.sum(axis=1)) > 1e-5
    difference_changed = np.abs(delta[:, 0] - delta[:, 1]) > 1e-5
    any_changed = np.any(np.abs(delta) > 1e-5, axis=1)
    return {
        name: {
            "rows": int(mask.sum()),
            "sum_changed": int((sum_changed & mask).sum()),
            "difference_changed": int((difference_changed & mask).sum()),
            "either_target_changed": int((any_changed & mask).sum()),
            "cancelling_target_changes": int((any_changed & ~sum_changed & mask).sum()),
            "sum_change_rate": float(np.mean(sum_changed[mask])) if mask.any() else None,
        }
        for name, mask in masks.items()
    }


def recurrence_witnesses(prices: np.ndarray, targets: np.ndarray) -> list[dict]:
    """Extract explicit witnesses for the already frozen exact-state test."""
    first_visit: dict[bytes, int] = {}
    witnesses = []
    event_count = np.r_[0, np.cumsum(changes(prices))]
    for row in range(99, len(prices)):
        key = prices[row].tobytes()
        previous = first_visit.setdefault(key, row)
        if abs(targets[row].sum() - targets[previous].sum()) > 1e-5:
            witnesses.append(
                {
                    "steps": [previous, row],
                    "targets": [targets[previous].tolist(), targets[row].tolist()],
                    "exact_price_equal": bool(np.array_equal(prices[previous], prices[row])),
                    "intervening_price_events": int(event_count[row] - event_count[previous]),
                    "price_row_sha256": hashlib.sha256(key).hexdigest(),
                }
            )
            if len(witnesses) == 3:
                break
    return witnesses


def main() -> None:
    """Run the frozen event falsifier on six previously exposed sequences."""
    frozen_bytes = (HERE / "HYPOTHESES_followup.json").read_bytes()
    hypothesis = json.loads(frozen_bytes)
    state = json.loads((HERE / "STATE.json").read_text())
    state["phase"] = "running_event_followup"
    state["commands"].append(
        "run_cpu.sh eda-puzzle-a .venv/bin/python research/campaign_2026-10-03/eda/puzzle_a/continuation_events.py"
    )
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2))
    records = []
    for split in ["train", "valid"]:
        parquet = pq.ParquetFile(ROOT / "datasets" / f"{split}.parquet")
        inputs = [c for c in parquet.schema_arrow.names if c.startswith(("i0_", "i1_", "a"))]
        columns = inputs + ["seq_ix", "t0", "t1"]
        for group in hypothesis["cohorts"][f"{split}_row_groups"]:
            table = parquet.read_row_group(group, columns=columns)
            x = np.column_stack([table[c].to_numpy() for c in inputs])
            y = np.column_stack([table[c].to_numpy() for c in ["t0", "t1"]]).astype(np.float64)
            p_indices = [inputs.index(f"i0_p{i}") for i in range(22)]
            i0_indices = [i for i, c in enumerate(inputs) if c.startswith("i0_")]
            valid_rows = np.arange(1, len(y)) >= 99
            masks = {
                "all_rows": valid_rows,
                "unchanged_i0_prices": valid_rows & ~changes(x[:, p_indices]),
                "unchanged_i0_all": valid_rows & ~changes(x[:, i0_indices]),
                "unchanged_all_inputs": valid_rows & ~changes(x),
            }
            delta_sum = np.diff(y.sum(axis=1))
            witness_rows = (
                np.flatnonzero(masks["unchanged_all_inputs"] & (np.abs(delta_sum) > 1e-5))[:5] + 1
            )
            record = {
                "split": split,
                "row_group": group,
                "seq_ix": int(table["seq_ix"][0].as_py()),
                "input_columns": inputs,
                "selected_input_target_sha256": hashlib.sha256(
                    x.tobytes() + y.tobytes()
                ).hexdigest(),
                "aligned": conditional_rates(y, masks),
                "shifted_control": conditional_rates(np.roll(y, 997, axis=0), masks),
                "recurrent_price_sum_conflict_witnesses": recurrence_witnesses(x[:, p_indices], y),
                "unchanged_input_sum_conflict_witnesses": [
                    {
                        "step": int(row),
                        "previous_targets": y[row - 1].tolist(),
                        "current_targets": y[row].tolist(),
                        "input_equal": bool(np.array_equal(x[row - 1], x[row])),
                    }
                    for row in witness_rows
                ],
            }
            records.append(record)
    result = {"hypotheses_sha256": hashlib.sha256(frozen_bytes).hexdigest(), "records": records}
    (HERE / "continuation_events_results.json").write_text(json.dumps(result, indent=2))
    fig, ax = plt.subplots(figsize=(9, 4))
    indices = np.arange(len(records))
    for condition in ["aligned", "shifted_control"]:
        for name in ["unchanged_i0_prices", "unchanged_i0_all"]:
            rates = [r[condition][name]["sum_change_rate"] for r in records]
            ax.plot(indices, rates, ".-", label=f"{condition}: {name}")
    ax.set_title("No unchanged-all-input transitions; that condition is untested")
    ax.set_xticks(indices, [f"{r['split']}:{r['row_group']}" for r in records])
    ax.set_ylabel("Adjacent sum-change rate, unchanged input block")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "continuation_events.png", dpi=140)
    plt.close(fig)
    state["phase"] = "event_followup_complete"
    state["artifacts"] += [
        "HYPOTHESES_initial_frozen.json",
        "HYPOTHESES_followup.json",
        "continuation_events.py",
        "continuation_events_results.json",
        "continuation_events.png",
    ]
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
