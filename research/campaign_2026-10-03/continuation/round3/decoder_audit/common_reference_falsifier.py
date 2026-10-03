from __future__ import annotations

import json
import logging
import resource
import time
import traceback
from collections import Counter, defaultdict

import numpy as np

from audit_common import OWN, digest, phase, price_code, save_json

LOGGER = logging.getLogger(__name__)


def generate_paths() -> list[dict[str, np.ndarray]]:
    """Generate causal current-book codes and separately defined future labels."""
    rng = np.random.default_rng(20261003)
    mixing = rng.integers(-2, 3, size=(22, 22)).astype(np.float32)
    paths = []
    for _ in range(8):
        offset = int(rng.integers(-1000, 1001))
        direction = int(rng.choice([-1, 1]))
        moves = direction * np.where(np.arange(64) % 2 == 0, 1, -1)
        midpoint = np.r_[offset, offset + np.cumsum(moves)].astype(np.float64)
        bid, ask = midpoint[:-1] - 2, midpoint[:-1] + 2
        gap = np.where(moves > 0, 1, 2)
        levels = np.c_[
            bid[:, None] - gap[:, None] * np.arange(11),
            ask[:, None] + gap[:, None] * np.arange(11),
        ]
        centered = levels - ((ask + bid) / 2)[:, None]
        prices = (centered.astype(np.float32) @ mixing).astype(np.float32)
        references = {
            "changing": midpoint[1:],
            "fixed": np.full(64, midpoint[0] + moves[0] / 2),
        }
        paths.append(
            {
                "bid": bid,
                "ask": ask,
                "prices": prices,
                "reference_changing": references["changing"],
                "reference_fixed": references["fixed"],
            }
        )
    return paths


def supervised_table(paths: list[dict[str, np.ndarray]], mode: str) -> dict:
    """Fit a literal code-pair table from interior labels only, not true moves."""
    votes = defaultdict(Counter)
    for path in paths:
        reference = path[f"reference_{mode}"]
        labels = np.c_[reference - path["ask"], path["bid"] - reference]
        codes = [price_code(row) for row in path["prices"]]
        for index in range(1, len(labels)):
            if (labels[index - 1 : index + 1] <= 0).all():
                difference = labels[index] - labels[index - 1]
                votes[(codes[index - 1], codes[index])][
                    (-float(difference[0]), float(difference[1]))
                ] += 1
    accepted = {}
    for key, outcomes in votes.items():
        candidate, support = outcomes.most_common(1)[0]
        if support >= 3 and support == outcomes.total():
            accepted[key] = candidate
    return accepted


def grade_mode(paths: list[dict[str, np.ndarray]], mode: str) -> dict[str, object]:
    """Measure held-out label exactness and physical move exactness separately."""
    table = supervised_table(paths[:4], mode)
    counts = Counter(
        eligible=0,
        mapped=0,
        label_difference_exact=0,
        true_quote_move_exact=0,
        gauge_adjusted_exact=0,
    )
    witnesses = []
    max_error = 0.0
    spread_errors = []
    for path_index, path in enumerate(paths[4:], start=4):
        reference = path[f"reference_{mode}"]
        labels = np.c_[reference - path["ask"], path["bid"] - reference]
        spread_errors.extend(
            (-labels.sum(axis=1) - (path["ask"] - path["bid"])).tolist()
        )
        codes = [price_code(row) for row in path["prices"]]
        for index in range(1, len(labels)):
            if not (labels[index - 1 : index + 1] <= 0).all():
                continue
            counts["eligible"] += 1
            inferred = table.get((codes[index - 1], codes[index]))
            if inferred is None:
                continue
            counts["mapped"] += 1
            delta_y = labels[index] - labels[index - 1]
            from_labels = np.array([-delta_y[0], delta_y[1]])
            actual = np.array(
                [
                    path["ask"][index] - path["ask"][index - 1],
                    path["bid"][index] - path["bid"][index - 1],
                ]
            )
            delta_reference = reference[index] - reference[index - 1]
            counts["label_difference_exact"] += int(
                np.array_equal(inferred, from_labels)
            )
            counts["true_quote_move_exact"] += int(np.array_equal(inferred, actual))
            gauge_error = np.array(inferred) - actual + delta_reference
            counts["gauge_adjusted_exact"] += int(np.array_equal(gauge_error, [0, 0]))
            max_error = max(
                max_error, float(np.max(np.abs(np.array(inferred) - actual)))
            )
            if len(witnesses) < 4:
                witnesses.append(
                    {
                        "heldout_path": path_index,
                        "step": index,
                        "true_quote_move_ask_bid": actual.tolist(),
                        "inferred_quote_move_ask_bid": list(inferred),
                        "reference_move": float(delta_reference),
                        "labels_before_after": labels[index - 1 : index + 1].tolist(),
                        "current_code": codes[index],
                    }
                )
    return {
        "counts": dict(counts),
        "accepted_directed_codes": len(table),
        "max_true_move_error_ticks": max_error,
        "maximum_spread_coordinate_error_ticks": float(np.max(np.abs(spread_errors))),
        "witnesses": witnesses,
        "appears_exact_but_physical_moves_differ": bool(
            counts["mapped"] > 0
            and counts["label_difference_exact"] == counts["mapped"]
            and counts["true_quote_move_exact"] < counts["mapped"]
        ),
    }


def algebra_controls() -> dict[str, object]:
    """Check entry inclusion, execution and clipping qualifications analytically."""
    entry_bid, entry_ask = 100.0, 101.0
    future_bid, future_ask = 97.0, 98.0
    spread = entry_ask - entry_bid
    inclusive = [
        max(entry_bid, future_bid) - entry_ask,
        entry_bid - min(entry_ask, future_ask),
    ]
    exclusive = [future_bid - entry_ask, entry_bid - future_ask]
    raw_label, wide_spread = -4.0, 3.0
    return {
        "entry_spread": spread,
        "entry_inclusive_extrema_returns": inclusive,
        "entry_inclusive_bound_holds": all(value >= -spread for value in inclusive),
        "entry_excluding_or_execution_returns": exclusive,
        "entry_excluding_bound_violations": sum(value < -spread for value in exclusive),
        "clipping_fixture": {
            "raw_label": raw_label,
            "bound": -wide_spread,
            "raw_violates": raw_label < -wide_spread,
            "clipped_violates_unchanged_bound": float(np.clip(raw_label, -2, 2))
            < -wide_spread,
        },
    }


def main() -> None:
    """Run the separately frozen falsifier and its fixed-reference control."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    state = json.loads((OWN / "STATE.json").read_text())
    commands = state["commands"] + [
        "run_cpu.sh round3-hft /usr/bin/time -v -o decoder_audit/synthetic_resource.log .venv/bin/python -B -u decoder_audit/common_reference_falsifier.py"
    ]
    phase("synthetic_falsifier_running", commands=commands)
    start = time.monotonic()
    paths = generate_paths()
    arrays = {
        f"path{index}_{key}": value
        for index, path in enumerate(paths)
        for key, value in path.items()
    }
    np.savez_compressed(OWN / "synthetic_quote_arrays.npz", **arrays)
    results = {
        "changing_reference": grade_mode(paths, "changing"),
        "fixed_reference_control": grade_mode(paths, "fixed"),
        "algebra_controls": algebra_controls(),
        "code_information_boundary": "Centered current book with current deeper gap phase; no future reference or targets enter price codes.",
        "prereg_sha256": digest(OWN / "PREREG.json"),
        "synthetic_array_sha256": digest(OWN / "synthetic_quote_arrays.npz"),
        "cost": {
            "wall_seconds": time.monotonic() - start,
            "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
            + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    save_json("synthetic_results.json", results)
    phase("synthetic_falsifier_complete", cost_synthetic=results["cost"])
    LOGGER.info("SYNTHETIC %s", json.dumps(results))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure = traceback.format_exc()
        (OWN / "synthetic_failure.log").write_text(failure)
        phase("synthetic_failed", last_failure=failure)
        raise
