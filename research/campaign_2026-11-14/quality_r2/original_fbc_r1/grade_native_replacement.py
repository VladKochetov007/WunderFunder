"""Grade only the completed actual replacement ZIP; keep proxy evidence separate."""

from __future__ import annotations

import json

import numpy as np
from grade_replacement import (
    CACHE,
    OUT,
    ROOT,
    component_layout,
    digest,
    native,
    score,
    score_from_sums,
    wp_sums,
)


def main() -> None:
    """Verify native scores, paired component sensitivity and measured serving evidence."""
    with np.load(CACHE / "valid_targets.npz") as data:
        groups, mask, y = data["row_group"], data["mask"], data["y"]
    selected_y = y[mask]
    baseline, base_source = native(
        ROOT / "submissions/tl3c_pr49_20261029/blend-int8-full-v3", groups, mask
    )
    actual, source = native(OUT / "replacement_full_valid", groups, mask)
    receipt = json.loads((OUT / "replacement_receipt.json").read_text())
    if source["artifact"]["sha256"] != receipt["artifact"]["sha256"]:
        raise ValueError("actual candidate ZIP identity differs")
    values = [score(selected_y, p)["wp"] for p in (baseline, actual)]
    if (
        max(abs(values[0] - base_source["recorded_wp"]), abs(values[1] - source["recorded_wp"]))
        > 1e-8
    ):
        raise ValueError("independent native score reconstruction differs")
    components, mapping, _ = component_layout(groups)
    positions = {value: i for i, value in enumerate(components)}
    sums = np.zeros((2, len(components), 2, 6), np.float64)
    offset = 0
    for group, m in zip(groups, mask, strict=True):
        count = int(m.sum())
        selected = slice(offset, offset + count)
        for k, pred in enumerate((baseline, actual)):
            sums[k, positions[mapping[int(group)]]] += wp_sums(
                selected_y[selected].T, pred[selected].T
            )
        offset += count
    assert offset == 3528273
    assert max(abs(score_from_sums(sums[k].sum(axis=0)) - values[k]) for k in (0, 1)) < 1e-8
    rng = np.random.default_rng(20261114)
    differences = []
    for _ in range(2000):
        draw = rng.integers(0, len(components), len(components))
        differences.append(
            score_from_sums(sums[1, draw].sum(axis=0)) - score_from_sums(sums[0, draw].sum(axis=0))
        )
    report = json.loads((OUT / "replacement_full_valid/report.json").read_text())
    proxy = json.loads((OUT / "REPLACEMENT_PROXY_R1.json").read_text())
    delta = values[1] - values[0]
    result = {
        "status": "actual_native_complete",
        "baseline_wp": values[0],
        "candidate_wp": values[1],
        "delta": delta,
        "paired_component_interval95": np.quantile(differences, [0.025, 0.975]).tolist(),
        "native_minus_proxy": values[1] - proxy["scores"]["replacement_proxy"],
        "point_estimate_improved": delta > 0,
        "artifact": source["artifact"],
        "report_sha256": digest(OUT / "replacement_full_valid/report.json"),
        "prediction_sha256": source["prediction_sha256"],
        "state_checks": report["state_checks"],
        "peak_rss_bytes": report["peak_rss_bytes"],
        "timing": source["timing"],
        "runtime_projection": report["runtime_projection"],
        "thermal_runtime_amendment": json.loads(
            (OUT.parent / "THERMAL_RESPONSE_R4.json").read_text()
        ),
        "fit_count": 0,
        "limits": [
            "Exposed full-valid, one KD90 seed; no unseen-CV or public gain claim.",
            "Component intervals are conditional on these fitted predictions.",
            "Host full-valid runtime does not establish portal whole-test runtime.",
        ],
    }
    (OUT / "ACTUAL_REPLACEMENT_R1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "candidate_wp",
                    "delta",
                    "paired_component_interval95",
                    "native_minus_proxy",
                    "point_estimate_improved",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
