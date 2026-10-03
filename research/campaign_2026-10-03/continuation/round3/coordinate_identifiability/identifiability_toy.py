from __future__ import annotations

import hashlib
import json
import logging
import resource
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
FROZEN = OWN.parents[1] / "round2" / "first_passage"
LOGGER = logging.getLogger(__name__)
QUANTA = (0.12543, 0.24133, 0.46854, 0.65687)


def sha256(path: Path) -> str:
    """Hash bounded documentation/table/code artifacts only."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(name: str, data: object) -> None:
    """Preserve strict JSON within the new write scope."""
    (OWN / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def key(values: np.ndarray) -> str:
    """Replay the frozen numerical price-code grammar without fitting."""
    return np.asarray(np.around(values.astype(np.float32), 4), dtype="<f4").tobytes().hex()


def choose_pair(model: dict) -> tuple[dict[str, object] | None, int]:
    """Select one frozen reversible, overlapping coordinate pair deterministically."""
    spreads = {row[0]: tuple(row[1:]) for row in model["spreads"]}
    moves = {(row[0], row[1]): tuple(row[2:]) for row in model["moves"]}
    eligible = []
    for first, second in sorted(moves):
        if first >= second or first not in spreads or second not in spreads:
            continue
        cls, spread0 = spreads[first]
        cls1, spread1 = spreads[second]
        forward = moves[first, second]
        backward = moves.get((second, first))
        if cls != cls1 or backward is None or tuple(-v for v in forward) != backward:
            continue
        ask1, bid1 = spread0 + forward[0], forward[1]
        if ask1 - bid1 != spread1:
            continue
        if (ask1 + bid1) / 2 == spread0 / 2:
            continue
        lower, upper = max(0, bid1), min(spread0, ask1)
        if lower > upper:
            continue
        eligible.append({
            "codes": [first, second], "class": int(cls), "q": QUANTA[cls],
            "spreads": [int(spread0), int(spread1)], "forward_move_ask_bid": list(forward),
            "Uask": [float(spread0), float(ask1)], "Ubid": [0.0, float(bid1)],
            "K": (lower + upper) / 2,
        })
    return (eligible[0] if eligible else None), len(eligible)


def observations(
    ask_bid: np.ndarray, frozen_codes: list[str], phases: np.ndarray
) -> np.ndarray:
    """Encode112 causal fields; four most recent trades occur at each current row."""
    rows = []
    lookup = {1: np.frombuffer(bytes.fromhex(frozen_codes[0]), dtype="<f4"),
              2: np.frombuffer(bytes.fromhex(frozen_codes[1]), dtype="<f4")}
    for (ask0, bid0), phase in zip(ask_bid, phases, strict=True):
        spread0 = ask0 - bid0
        midpoint0 = (ask0 + bid0) / 2
        gap = int(phase) + 1
        depth = np.arange(11)
        book0 = np.r_[bid0 - gap * depth, ask0 + gap * depth]
        # Phase is inferred from the CURRENT book's first deeper gap, not a label.
        observed_gap = int(book0[0] - book0[1])
        p0 = lookup[observed_gap].copy()
        midpoint1 = midpoint0 + 40 + phase
        spread1 = 2 + phase
        bid1, ask1 = midpoint1 - spread1 / 2, midpoint1 + spread1 / 2
        book1 = np.r_[bid1 - gap * depth, ask1 + gap * depth]
        p1 = book1 - (ask1 + bid1) / 2
        v0 = 20 + 3 * phase + np.arange(22)
        v1 = 30 + 2 * phase + np.arange(22)
        current_trades0 = np.array([bid0, bid0, ask0, ask0])
        current_trades1 = np.array([bid1, bid1, ask1, ask1])
        dp0 = current_trades0 - midpoint0
        dp1 = current_trades1 - midpoint1
        dv0 = 1 + phase + np.arange(4)
        dv1 = 2 + phase + np.arange(4)
        extra = [spread0, spread1, gap, 40 + phase, phase,
                 dv0.sum(), dv1.sum(), 1]
        rows.append(np.r_[p0, v0, dp0, dv0, p1, v1, dp1, dv1, extra])
    return np.asarray(rows, dtype=np.float32)


def replay_prefix(inputs: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    """Replay input-only frozen table and first-spread anchor on the toy cycle."""
    spreads = {row[0]: tuple(row[1:]) for row in model["spreads"]}
    moves = {(row[0], row[1]): tuple(row[2:]) for row in model["moves"]}
    codes = [key(row[:22]) for row in inputs]
    cls, initial_spread = spreads[codes[0]]
    values = np.zeros((len(inputs), 2))
    values[0] = [initial_spread, 0]
    labels = np.empty(len(inputs), dtype=int)
    labels[0] = cls
    for row in range(1, len(inputs)):
        values[row] = values[row - 1] + moves[codes[row - 1], codes[row]]
        labels[row], expected_spread = spreads[codes[row]]
        if values[row, 0] - values[row, 1] != expected_spread:
            raise ValueError("Frozen cycle loses spread consistency")
    return values, labels


def experiment(pair: dict[str, object], model: dict) -> dict[str, object]:
    """Evaluate identical base inputs and explicitly stronger anchor controls."""
    count = 32
    phases = np.arange(count + 1) % 2
    u = np.c_[np.asarray(pair["Uask"])[phases], np.asarray(pair["Ubid"])[phases]]
    midpoint_u = u.mean(axis=1)
    reference_level = float(pair["K"])
    gauge = np.empty(count + 1)
    gauge[0] = 100
    for row in range(count):
        gauge[row + 1] = gauge[row] + reference_level - midpoint_u[row + 1]
    world0 = u + 100
    world1 = u + gauge[:, None]
    f0 = np.full(count, reference_level + 100)
    f1 = world1[1:].mean(axis=1)
    inputs0 = observations(world0[:count], pair["codes"], phases[:count])
    inputs1 = observations(world1[:count], pair["codes"], phases[:count])
    decoded0, class0 = replay_prefix(inputs0, model)
    decoded1, class1 = replay_prefix(inputs1, model)
    q = float(pair["q"])
    y0 = q * np.c_[f0 - world0[:count, 0], world0[:count, 1] - f0]
    y1 = q * np.c_[f1 - world1[:count, 0], world1[:count, 1] - f1]
    physical_moves0, physical_moves1 = np.diff(world0[:count], axis=0), np.diff(world1[:count], axis=0)
    held_trade = float(world0[0].mean())
    z0 = held_trade - world0[:count].mean(axis=1)
    z1 = held_trade - world1[:count].mean(axis=1)
    anchored_midpoint0 = world0[0].mean() - (z0 - z0[0])
    anchored_midpoint1 = world1[0].mean() - (z1 - z1[0])
    # These anchor channels are controls, NOT claims about supplied fields.
    absolute_bid0, absolute_bid1 = world0[:count, 1], world1[:count, 1]
    known_reference_recovery = decoded1 + (f1 - reference_level)[:, None]
    known_bid_recovery = np.c_[absolute_bid1 + (decoded1[:, 0] - decoded1[:, 1]), absolute_bid1]
    result = {
        "rows": count, "feature_count": inputs0.shape[1],
        "base_observations_bit_identical": bool(np.array_equal(inputs0, inputs1)),
        "all112_max_difference": float(np.max(np.abs(inputs0 - inputs1))),
        "frozen_code_roundtrip_exact": all(key(inputs0[row, :22]) == pair["codes"][int(phases[row])] for row in range(count)),
        "frozen_decoder_paths_identical": bool(np.array_equal(decoded0, decoded1)),
        "frozen_decoder_classes_identical": bool(np.array_equal(class0, class1)),
        "frozen_decoder_matches_coordinate_U": bool(np.array_equal(decoded1, u[:count])),
        "labels_identical_diagnostic_only": bool(np.array_equal(y0, y1)),
        "all_labels_interior": bool((y0 <= 0).all() and (y1 <= 0).all()),
        "physical_quotes_differ_rows": int(np.any(world0[:count] != world1[:count], axis=1).sum()),
        "physical_moves_differ_transitions": int(np.any(physical_moves0 != physical_moves1, axis=1).sum()),
        "true_spreads_identical": bool(np.array_equal(np.diff(world0, axis=1), np.diff(world1, axis=1))),
        "first_physical_anchor_identical": bool(np.array_equal(world0[0], world1[0])),
        "next_step_future_reference_identity_error": float(np.max(np.abs(f1 - (reference_level + gauge[:count])))),
        "negative_controls": {
            "initial_absolute_anchor_breaks_equivalence": not bool(np.array_equal(world0[0], world1[0])),
            "refreshing_trade_slots_break_equivalence": not bool(np.array_equal(inputs0[:, 44:52], inputs1[:, 44:52])),
            "second_instrument_and_extras_break_equivalence": not bool(np.array_equal(inputs0[:, 52:], inputs1[:, 52:])),
            "first_known_spread_anchor_break_equivalence": not bool(np.array_equal(decoded0, decoded1)),
        },
        "positive_controls_extra_assumptions": {
            "known_absolute_bid_each_row_breaks_equivalence": not bool(np.array_equal(absolute_bid0, absolute_bid1)),
            "known_absolute_bid_reconstruction_max_error": float(np.max(np.abs(known_bid_recovery - world1[:count]))),
            "known_common_reference_each_row_breaks_equivalence": not bool(np.array_equal(f0, f1)),
            "known_common_reference_reconstruction_max_error": float(np.max(np.abs(known_reference_recovery - world1[:count]))),
            "immutable_trade_same_current_mid_normalization_breaks_equivalence": not bool(np.array_equal(z0, z1)),
            "immutable_trade_anchor_midpoint_max_error": float(max(np.max(np.abs(anchored_midpoint0 - world0[:count].mean(axis=1))), np.max(np.abs(anchored_midpoint1 - world1[:count].mean(axis=1))))),
        },
        "world0_reference_future_realization": "Append a terminal midpoint K+100 after the observed window; it defines the constant future reference without entering inputs.",
    }
    result["unidentifiability_witness_succeeded"] = bool(
        result["base_observations_bit_identical"] and result["labels_identical_diagnostic_only"]
        and result["frozen_code_roundtrip_exact"] and result["frozen_decoder_paths_identical"]
        and result["physical_moves_differ_transitions"] > 0
    )
    np.savez_compressed(OWN / "two_world_arrays.npz", inputs0=inputs0, inputs1=inputs1,
                        physical0=world0, physical1=world1, U=u, gauge=gauge,
                        future_reference0=f0, future_reference1=f1, labels0=y0, labels1=y1,
                        held_trade_centered0=z0, held_trade_centered1=z1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(world0[:count].mean(axis=1), "o-", label="Physical world0 midpoint")
    axes[0].plot(world1[:count].mean(axis=1), "x--", label="Physical world1 midpoint")
    axes[0].set_title("Different physical paths, same112 inputs and labels")
    axes[0].legend(fontsize=8)
    axes[1].plot(decoded0.mean(axis=1), "o-", label="Frozen input-decoded coordinate")
    axes[1].plot(gauge[:count] - 100, "x--", label="Unobserved gauge relative to row0")
    axes[1].set_title("A prefix reset cannot reveal time-varying origin")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel("Toy row")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OWN / "two_world_identifiability.png", dpi=120)
    plt.close(fig)
    return result


def main() -> None:
    """Run one frozen table-conditioned toy, with no raw-data or target fitting."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start = time.monotonic()
    state = json.loads((OWN / "STATE.json").read_text())
    state["phase"] = "toy_running"
    save_json("STATE.json", state)
    model = json.loads((FROZEN / "decoder_tables.json").read_text())
    pair, eligible = choose_pair(model)
    save_json("selected_frozen_fixture.json", {"eligible_pairs": eligible, "selected": pair})
    results = {"eligible_pairs": eligible, "fixture_available": pair is not None,
               "experiment": experiment(pair, model) if pair is not None else None,
               "prereg_sha256": sha256(OWN / "PREREG.json"),
               "decoder_table_sha256": sha256(FROZEN / "decoder_tables.json"),
               "field_documentation_sha256": sha256(ROOT / "research/campaign_2026-10-03/eda/HOST_DATA_DESCRIPTION.txt"),
               "cost": {"wall_seconds": time.monotonic() - start,
                        "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    save_json("results.json", results)
    state["phase"] = "toy_complete"
    state["cost"] = results["cost"]
    save_json("STATE.json", state)
    LOGGER.info("RESULTS %s", json.dumps(results))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure = traceback.format_exc()
        (OWN / "failure.log").write_text(failure)
        state = json.loads((OWN / "STATE.json").read_text())
        state.update(phase="toy_failed", last_failure=failure)
        save_json("STATE.json", state)
        raise
