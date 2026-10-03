from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np
from path_laws import FAMILIES, functional, independent_reference

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Generate known-family labels and adversarial near-equivalent stopping controls."""
    start = time.monotonic()
    rng = np.random.default_rng(20261003)
    bids = np.column_stack(
        [np.zeros(256), np.cumsum(rng.choice([-3, -1, 0, 1, 3], (256, 64)), axis=1)]
    )
    spreads = rng.integers(1, 4, 256)
    asks = bids + spreads[:, None]
    labels = np.full((256, 3, 2), np.nan)
    checks = {"independent_reference": 0, "identity": 0, "translation": 0, "reflection": 0}
    for i, (ask, bid) in enumerate(zip(asks, bids, strict=True)):
        valid = np.ones(len(ask), dtype=bool)
        for k, family in enumerate(FAMILIES):
            output = functional(ask, bid, valid, family)["pair"]
            labels[i, k] = output
            controls = {
                "independent_reference": independent_reference(ask, bid, family),
                "identity": functional(ask.copy(), bid.copy(), valid.copy(), family)["pair"],
                "translation": functional(ask + 53, bid + 53, valid, family)["pair"],
                "reflection": functional(-bid, -ask, valid, family)["pair"][::-1],
            }
            for name, control in controls.items():
                if not np.array_equal(output, control, equal_nan=True):
                    raise AssertionError(f"{name} failed sample{i} family{family}")
                checks[name] += 1
    confusion = {}
    for generated, name in enumerate(FAMILIES):
        confusion[name] = {}
        for candidate, other in enumerate(FAMILIES):
            known = np.isfinite(labels[:, generated]).all(axis=1) & np.isfinite(
                labels[:, candidate]
            ).all(axis=1)
            exact = known & np.all(labels[:, generated] == labels[:, candidate], axis=1)
            confusion[name][other] = {
                "all_paths": 256,
                "joint_known": int(known.sum()),
                "exact_pairs": int(exact.sum()),
                "exact_all_fraction": float(exact.mean()),
            }
    # Same future extrema, different order: fixed-horizon extrema cannot distinguish these.
    prefix_a = np.array([0, 2, 4, 2, 0] + [0] * 12, dtype=float)
    prefix_b = np.array([0, 2, 1, 4, 0] + [0] * 12, dtype=float)
    order_control = {}
    for family in FAMILIES:
        pair_a = functional(prefix_a + 1, prefix_a, np.ones(17, bool), family)["pair"]
        pair_b = functional(prefix_b + 1, prefix_b, np.ones(17, bool), family)["pair"]
        order_control[family] = {
            "a": pair_a.tolist(),
            "b": pair_b.tolist(),
            "same": bool(np.array_equal(pair_a, pair_b, equal_nan=True)),
        }
    assert order_control["fixed_extrema"]["same"]
    assert not order_control["trailing_extrema"]["same"]
    # Same first32 future events. Longer trailing stops return different extremes.
    common = np.arange(33, dtype=float)
    suffix_a = np.r_[common, 31, 30]
    suffix_b = np.r_[common, 33, 34, 35, 34, 33]
    suffix_control = {
        "same_prefix_33": bool(np.array_equal(suffix_a[:33], suffix_b[:33])),
        "trailing_long_extreme_a": 32 - 1,
        "trailing_long_extreme_b": 35 - 1,
        "bounded_output_a": functional(
            suffix_a + 1, suffix_a, np.ones(len(suffix_a), bool), "trailing_extrema"
        )["pair"].tolist(),
        "bounded_output_b": functional(
            suffix_b + 1, suffix_b, np.ones(len(suffix_b), bool), "trailing_extrema"
        )["pair"].tolist(),
    }
    assert suffix_control["trailing_long_extreme_a"] != suffix_control["trailing_long_extreme_b"]
    unknown = np.ones(65, bool)
    unknown[2:] = False
    gap = {family: functional(asks[0], bids[0], unknown, family)["reasons"] for family in FAMILIES}
    result = {
        "checks": checks,
        "cross_family_confusion": confusion,
        "adversarial_order": order_control,
        "adversarial_identical_prefix_different_suffix": suffix_control,
        "unknown_gap_reasons": gap,
        "cost": {
            "wall_seconds": time.monotonic() - start,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    np.savez(
        HERE / "synthetic_paths_labels.npz",
        asks=asks,
        bids=bids,
        family_labels=labels,
        families=FAMILIES,
    )
    (HERE / "synthetic_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
