from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np
from laws import LAWS, run_law, scalar_reference

HERE = Path(__file__).resolve().parent


def json_safe(value: object) -> object:
    """Serialize unresolved synthetic outputs as null rather than NaN."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    """Run known-generator positives and adversarial failures before raw TRAIN reads."""
    started = time.monotonic()
    rng = np.random.default_rng(20261003)
    bid_paths = np.column_stack(
        [np.zeros(256), np.cumsum(rng.choice([-3, -1, 0, 1, 3], (256, 64)), axis=1)]
    )
    spreads = rng.integers(1, 4, 256)
    ask_paths = bid_paths + spreads[:, None]
    labels = np.full((256, 3, 2), np.nan)
    checks = {
        "scalar_reference": 0,
        "identity": 0,
        "translation": 0,
        "reflection": 0,
        "unknown_gap": 0,
    }
    for i, (ask, bid) in enumerate(zip(ask_paths, bid_paths, strict=True)):
        valid = np.ones(len(ask), bool)
        for k, law in enumerate(LAWS):
            known_label = scalar_reference(ask, bid, law)
            labels[i, k] = known_label
            actual = run_law(ask, bid, valid, law)["pair"]
            assertions = {
                "scalar_reference": known_label,
                "identity": run_law(ask.copy(), bid.copy(), valid.copy(), law)["pair"],
                "translation": run_law(ask + 53, bid + 53, valid, law)["pair"],
                "reflection": run_law(-bid, -ask, valid, law)["pair"][::-1],
            }
            for name, expected in assertions.items():
                if not np.array_equal(actual, expected, equal_nan=True):
                    raise AssertionError(f"{name} sample{i} law{law}")
                checks[name] += 1
            # A later valid island cannot repair a gap at the first future event.
            gapped = valid.copy()
            gapped[1] = False
            output = run_law(ask, bid, gapped, law)
            assert not np.isfinite(output["pair"]).any()
            assert output["reasons"] == ["unknown_transition"] * 2
            checks["unknown_gap"] += 1
    confusion = {}
    for generated, name in enumerate(LAWS):
        confusion[name] = {}
        for candidate, other in enumerate(LAWS):
            known = np.isfinite(labels[:, generated]).all(axis=1) & np.isfinite(
                labels[:, candidate]
            ).all(axis=1)
            exact = known & np.all(labels[:, generated] == labels[:, candidate], axis=1)
            confusion[name][other] = {
                "all_paths": 256,
                "joint_known": int(known.sum()),
                "exact_pairs": int(exact.sum()),
            }
    adversaries = {}
    p0 = np.r_[[0.0, 2, 4, 2, 0], np.zeros(28)]
    p1 = np.r_[[0.0, 2, 1, 4, 0], np.zeros(28)]
    assert p0.max() == p1.max() and p0.min() == p1.min()
    for law in LAWS:
        a = run_law(p0 + 1, p0, np.ones(33, bool), law)
        b = run_law(p1 + 1, p1, np.ones(33, bool), law)
        adversaries[law] = {
            "pair_a": a["pair"].tolist(),
            "pair_b": b["pair"].tolist(),
            "different": bool(not np.array_equal(a["pair"], b["pair"], equal_nan=True)),
        }
    assert all(v["different"] for v in adversaries.values())
    # Two laws genuinely differ even with the same winning extreme.
    assert not np.array_equal(
        np.array(adversaries[LAWS[0]]["pair_b"]), np.array(adversaries[LAWS[1]]["pair_b"])
    )
    ramp = np.arange(33, dtype=float)
    ramp_results = {
        law: json_safe(run_law(ramp + 1, ramp, np.ones(33, bool), law)["pair"].tolist())
        for law in LAWS
    }
    assert all(v[0] is None for v in ramp_results.values())
    result = {
        "stage": "SYNTHETIC PASSED BEFORE RAW TRAIN SCORING",
        "checks": checks,
        "cross_generator_confusion": confusion,
        "same_range_different_order": adversaries,
        "monotone_cap_pairs": ramp_results,
        "unknown_gap_late_recovery_rejected": True,
        "cost": {
            "wall_seconds": time.monotonic() - started,
            "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
            + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    np.savez(
        HERE / "known_generator_paths.npz",
        asks=ask_paths,
        bids=bid_paths,
        family_labels=labels,
        laws=LAWS,
    )
    (HERE / "synthetic_results.json").write_text(
        json.dumps(json_safe(result), indent=2, allow_nan=False)
    )
    print(json.dumps(json_safe(result)))


if __name__ == "__main__":
    main()
