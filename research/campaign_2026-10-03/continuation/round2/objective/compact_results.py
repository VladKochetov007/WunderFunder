from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Extract frozen measurement tables without any new fit or selection."""
    results = json.loads((HERE / "results.json").read_text())
    compact = []
    for i, result in enumerate(results["actual"]):
        compact.append(
            {
                "chunk": result["chunk"],
                "permutation": i % 3,
                "main_cv": result["main_W"]["cv"],
                "main_cv_no_warmup": result["main_W_excluding_warmup_chunk"]["cv"],
                "row_cv": result["row_R"]["cv"],
                "label_only_cv": result["label_only_W"]["cv"],
                "inverse_mass_quantiles": result["relative_inverse_mass"]["q01_q10_q50_q90_q99"],
                "inverse_mass_minmax": [
                    result["relative_inverse_mass"]["min"],
                    result["relative_inverse_mass"]["max"],
                ],
                "gradient_proxy": result["gradient_proxies"],
                "binned_mean_shifts": {
                    name: {
                        key: value
                        for key, value in entry.items()
                        if key in {"weighted_RMS_shift", "max_abs_nonempty_shift"}
                    }
                    for name, entry in result["conditional_input_bins"].items()
                },
                "weighted_label_to_row_mass": result["weighted_abs_label_sum_per_row"],
                "within_position_cv_range": [
                    min(result["within_position_cv"]),
                    max(result["within_position_cv"]),
                ],
            }
        )
    (HERE / "compact_results.json").write_text(json.dumps(compact, indent=2) + "\n")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
