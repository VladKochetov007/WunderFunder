"""Freeze descriptive numerical envelopes, not a contraction or scoring fit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    result = json.loads((HERE / "results.json").read_text())
    cases = result["cases"]
    floor_values = [f["output"] for case in cases for f in case["floors"]]
    h_floors = [v for case in cases for f in case["floors"] for v in f["hidden_per_layer"]]
    initial_rms = [v for case in cases for v in case["scale"]["hidden_rms"]]
    hidden_ratios = [
        r["hidden_l2"] / r["initial_effective_l2"]
        for case in cases
        for r in case["responses"]
        if r["lag"] == 288 and r["initial_effective_l2"]
    ]
    summary = {
        "cases": len(cases),
        "components": len({c["case"]["component"] for c in cases}),
        "output_floor_range": [min(floor_values), max(floor_values)],
        "hidden_per_layer_floor_range": [min(h_floors), max(h_floors)],
        "incoming_hidden_layer_rms_range": [min(initial_rms), max(initial_rms)],
        "end_hidden_l2_over_initial_delta_l2_range": [min(hidden_ratios), max(hidden_ratios)],
        "max_identity_reference_discrepancy": max(c["identity_reference_maxabs"] for c in cases),
        "max_repeat_output_discrepancy": max(c["repeat_output_maxabs"] for c in cases),
        "max_repeat_hidden_discrepancy": max(c["repeat_hidden_maxabs"] for c in cases),
        "max_future_canary_discrepancy": max(c["future_canary_maxabs"] for c in cases),
        "zero_input_direction_layers": sum(sum(c["scale"]["zero_direction_layers"]) for c in cases),
        "effective_zero_direction_layers": sum(
            sum(r["effective_zero_layers"]) for c in cases for r in c["responses"] if r["lag"] == 0
        ),
        "perturbed_initial_outside_gru_unit_interval_counts": [
            r["initial_outside_gru_unit_interval"]
            for c in cases
            for r in c["responses"]
            if r["lag"] == 0
        ],
        "normalized_input_maxabs_range": [
            min(c["numerical_envelope"]["normalized_maxabs"] for c in cases),
            max(c["numerical_envelope"]["normalized_maxabs"] for c in cases),
        ],
        "encoded_gelu_max_range": [
            min(c["numerical_envelope"]["encoded_gelu_max"] for c in cases),
            max(c["numerical_envelope"]["encoded_gelu_max"] for c in cases),
        ],
        "hidden_abs_over_099_fraction_range": [
            min(c["numerical_envelope"]["hidden_abs_over_099_fraction"] for c in cases),
            max(c["numerical_envelope"]["hidden_abs_over_099_fraction"] for c in cases),
        ],
        "envelope": result["envelope"],
        "cost": result["cost"],
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    # Capture manifest for completed scientific runs, before human report freeze.
    files = [
        p
        for p in sorted(HERE.iterdir())
        if p.is_file() and p.name not in {"HASHES.json", "measurement_hashes.json", "summary.log"}
    ]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (HERE / "measurement_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
