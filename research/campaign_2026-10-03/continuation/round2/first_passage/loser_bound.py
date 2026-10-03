from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from decoder import HERE, ROOT, digest, keys_for, load_tables, read_group


def main() -> None:
    """Test the law's horizon-independent loss bound using frozen input-derived spreads."""
    samples = json.loads((HERE / "sampled_origins.json").read_text())
    spreads, _ = load_tables()
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    counts = Counter()
    weights = Counter()
    witnesses = []
    for group in sorted({r["group"] for r in samples}):
        x, _, _, _ = read_group(source, group)
        rows = [r for r in samples if r["group"] == group]
        codes = keys_for(x[[r["step"] for r in rows]])
        for r, code in zip(rows, codes, strict=True):
            y = np.array(r["target_raw"])
            if not (y > 0).any():
                counts["all_interior_origins"] += 1
                lookup = spreads.get(code)
                if lookup is not None:
                    counts["mapped_interior_spread_checks"] += 1
                    expected = -lookup[1] * r["quantum_from_fit_code"]
                    counts["exact_interior_spread_checks"] += int(abs(y.sum() - expected) <= 0.002)
                continue
            counts["all_outside_origins"] += 1
            weights["all_outside_origins"] += r["diagnostic_weight"]
            lookup = spreads.get(code)
            if lookup is None:
                counts["missing_current_spread"] += 1
                weights["missing_current_spread"] += r["diagnostic_weight"]
                continue
            cls, spread = lookup
            assert cls == r["fit_code_class"]
            bound = -spread * r["quantum_from_fit_code"]
            counts["mapped_outside_origins"] += 1
            weights["mapped_outside_origins"] += r["diagnostic_weight"]
            violated = bool(y.min() < bound - 0.002)
            counts["trailing_extrema_bound_violated"] += int(violated)
            weights["trailing_extrema_bound_violated"] += r["diagnostic_weight"] * violated
            if violated and len(witnesses) < 5:
                witnesses.append(
                    {
                        "group": group,
                        "seq_ix": r["seq_ix"],
                        "step": r["step"],
                        "class_from_frozen_fit_code": cls,
                        "current_spread_ticks": spread,
                        "minimum_extreme_return_raw": bound,
                        "supplied_loser_raw": float(y.min()),
                    }
                )
    with np.load(HERE / "synthetic_paths_labels.npz") as synthetic:
        p = synthetic["family_labels"][:, 2]
        spread = synthetic["asks"][:, 0] - synthetic["bids"][:, 0]
        synthetic_violations = int(np.sum(np.isfinite(p) & (p < -spread[:, None])))
    assert synthetic_violations == 0
    result = {
        "prereg_sha256": digest(HERE / "BOUND_PREREG.json"),
        "counts": dict(counts),
        "weights": dict(weights),
        "synthetic_bound_violations": synthetic_violations,
        "witnesses": witnesses,
        "claim": "Rules out extrema-return law including entry for mapped outside labels at any stopping horizon, conditional on supervised spread coordinate. Does not identify the actual loser execution law.",
    }
    (HERE / "loser_bound_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
