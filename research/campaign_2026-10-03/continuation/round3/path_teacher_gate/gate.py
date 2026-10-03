from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np

OWN = Path(__file__).resolve().parent
ROUND = OWN.parent
AUDIT = ROUND / "decoder_audit"


def digest(path: Path) -> str:
    """Hash only small frozen inputs and own outputs."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path) -> ModuleType:
    """Load frozen specifications without invoking their mains or writing caches."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(name: str, value: object) -> None:
    (OWN / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def state(ask: np.ndarray, bid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """One candidate: spread and past reversal distances, never future labels.

    The prefix must begin at a verified anchor. An unknown transition poisons
    subsequent states; a later known island is not an implicit reanchor.
    Each row uses no more than 32 current/past events, including current event.
    """
    output = np.full((len(ask), 4), np.nan)
    output[:, 3] = 0
    for t in range(len(ask)):
        if not valid[: t + 1].all():
            continue
        start = max(0, t - 31)
        output[t] = [
            ask[t] - bid[t],
            np.max(bid[start : t + 1]) - bid[t],
            ask[t] - np.min(ask[start : t + 1]),
            1,
        ]
    return output


def reconstruct(path: dict, table: dict, code_fn: object) -> tuple:
    """Use only current/past codes and an initial spread anchor after fit."""
    codes = [code_fn(row) for row in path["prices"]]
    ask, bid = np.full(len(codes), np.nan), np.full(len(codes), np.nan)
    valid = np.zeros(len(codes), dtype=bool)
    # Toy has known spread=4; deployment would require frozen input-code map.
    ask[0], bid[0], valid[0] = 4, 0, True
    for t in range(1, len(codes)):
        move = table.get((codes[t - 1], codes[t]))
        if move is None or not valid[t - 1]:
            continue
        ask[t], bid[t] = ask[t - 1] + move[0], bid[t - 1] + move[1]
        valid[t] = True
    return ask, bid, valid


def main() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    wall = time.monotonic()
    cpu = time.process_time()
    inputs = [
        AUDIT / "REPORT.txt",
        AUDIT / "common_reference_falsifier.py",
        AUDIT / "audit_common.py",
        ROUND / "hybrid_pair" / "REPORT.md",
        ROUND / "hybrid_pair" / "laws.py",
        OWN / "PREREG.json",
    ]
    before = {str(p): digest(p) for p in inputs}
    sys.path.insert(0, str(AUDIT))
    audit = load_module("audit_falsifier_readonly", inputs[1])
    laws = load_module("hybrid_laws_readonly", inputs[4])
    paths = audit.generate_paths()
    mode_results = {}
    prefix_checks = translation_checks = 0
    witnesses = []
    for mode in ("fixed", "changing"):
        table = audit.supervised_table(paths[:4], mode)
        rows = known = exact = identity = spread_exact = 0
        for p, path in enumerate(paths[4:], start=4):
            ask, bid, valid = reconstruct(path, table, audit.price_code)
            feature = state(ask, bid, valid)
            truth = state(path["ask"], path["bid"], np.ones(64, bool))
            translated = state(ask + 91, bid + 91, valid)
            for t in range(64):
                rows += 1
                known += int(valid[t])
                exact += int(np.array_equal(feature[t], truth[t]))
                spread_exact += int(feature[t, 0] == truth[t, 0])
                truncated = {k: v[: t + 1] for k, v in path.items()}
                a, b, v = reconstruct(truncated, table, audit.price_code)
                assert np.array_equal(state(a, b, v)[-1], feature[t], equal_nan=True)
                prefix_checks += 1
                assert np.array_equal(translated[t], feature[t], equal_nan=True)
                translation_checks += 1
                reference = path[f"reference_{mode}"]
                if t > 0:
                    labels = np.c_[reference - path["ask"], path["bid"] - reference]
                    delta = labels[t] - labels[t - 1]
                    identity += int(
                        np.array_equal(
                            [ask[t] - ask[t - 1], bid[t] - bid[t - 1]], [-delta[0], delta[1]]
                        )
                    )
                if (
                    mode == "changing"
                    and not np.array_equal(feature[t], truth[t])
                    and len(witnesses) < 4
                ):
                    witnesses.append(
                        {
                            "toy_path": p,
                            "step": t,
                            "decoded_state": feature[t].tolist(),
                            "physical_state": truth[t].tolist(),
                        }
                    )
        mode_results[mode] = {
            "all_origins": rows,
            "known": known,
            "unknown": rows - known,
            "physical_state_exact": exact,
            "spread_exact": spread_exact,
            "label_transition_exact": identity,
            "eligible_transitions": 252,
            "frozen_table_entries": len(table),
        }

    rng = np.random.default_rng(632031)
    positive_exact = complete = censored = 0
    arrays = {}
    for p in range(64):
        bid = np.r_[0, np.cumsum(rng.choice([-3, -1, 0, 1, 3], 32))].astype(float)
        ask = bid + rng.integers(1, 4, 33)
        expected = laws.scalar_reference(ask, bid, "shared_profitable_exit")
        observed = laws.run_law(ask, bid, np.ones(33, bool), "shared_profitable_exit")["pair"]
        assert np.array_equal(expected, observed, equal_nan=True)
        positive_exact += 1
        complete += int(np.isfinite(expected).all())
        censored += int(not np.isfinite(expected).all())
        arrays[f"positive_{p}_bid"] = bid
        arrays[f"positive_{p}_ask"] = ask
    # Same current/past input; future order differs, both spans have equal range.
    bids = [np.r_[x, np.zeros(28)] for x in ([0, 2, 4, 2, 0], [0, 2, 1, 4, 0])]
    futures = [
        laws.run_law(b + 1, b, np.ones(33, bool), "shared_profitable_exit")["pair"] for b in bids
    ]
    prefixes = [state((b + 1)[:2], b[:2], np.ones(2, bool))[-1] for b in bids]
    assert np.array_equal(prefixes[0], prefixes[1])
    assert not np.array_equal(futures[0], futures[1])
    gap_valid = np.ones(33, bool)
    gap_valid[1] = False
    gap_state = state(bids[0] + 1, bids[0], gap_valid)
    assert np.isnan(gap_state[1:, :3]).all()
    for i, path in enumerate(paths):
        for key, array in path.items():
            arrays[f"audit_{i}_{key}"] = array
    np.savez_compressed(OWN / "toy_arrays.npz", **arrays)
    after = {str(p): digest(p) for p in inputs}
    assert before == after
    physical_pass = mode_results["changing"]["physical_state_exact"] == 256
    assert mode_results["fixed"]["physical_state_exact"] == 256
    assert mode_results["changing"]["label_transition_exact"] == 252
    assert not physical_pass
    result = {
        "decision": "REJECT_PHYSICAL_EVENT_STATE_TEACHER",
        "causal_prefix_implementation_pass": True,
        "physical_identifiability_pass": physical_pass,
        "target_law_gate": "FAIL: own frozen TRAIN report rejects both exact hybrid laws; not rescored",
        "modes": mode_results,
        "known_generator": {
            "cases": 64,
            "engine_reference_exact": positive_exact,
            "complete": complete,
            "censored": censored,
        },
        "controls": {
            "prefix_identity_checks": prefix_checks,
            "constant_translation_checks": translation_checks,
            "unknown_gap_rejected_states": 32,
            "same_prefix": prefixes[0].tolist(),
            "different_future_pairs": [x.tolist() for x in futures],
        },
        "witnesses": witnesses,
        "IDs": {
            "fit_audit_paths": [0, 1, 2, 3],
            "test_audit_paths": [4, 5, 6, 7],
            "test_steps_inclusive": [0, 63],
            "positive_paths": list(range(64)),
            "positive_seed": 632031,
            "audit_seed": 20261003,
            "supplied_origins_newly_scored": 0,
        },
        "provenance": {
            "input_hashes": before,
            "script_sha256": digest(Path(__file__)),
            "synthetic_arrays_sha256": digest(OWN / "toy_arrays.npz"),
            "audit_fixture_reuse": True,
        },
        "cost": {
            "wall_seconds": time.monotonic() - wall,
            "cpu_seconds": time.process_time() - cpu,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "raw_rows_read": 0,
            "GPU": False,
            "monetary_cost": "not exposed",
        },
    }
    save("results.json", result)
    save(
        "STATE.json",
        {
            "phase": "gate_complete",
            "candidate_count": 1,
            "decision": result["decision"],
            "raw_data_reads": 0,
            "prior_reports_preserved": True,
            "resource_role": "round3-hybrid-pair",
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
