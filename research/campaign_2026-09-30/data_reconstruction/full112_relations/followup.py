from __future__ import annotations

import hashlib
import json
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from probe import HERE, ROOT, context, read
from scipy.special import ndtr
from scipy.stats import norm, spearmanr
from sklearn.isotonic import IsotonicRegression

TRADE_STARTS = (44, 48, 96, 100)
CANDIDATES = (
    [f"age_block{i}" for i in range(6)]
    + ["age_any_price", "age_any_trade"]
    + [f"count{window}_block{i}" for window in (32, 128) for i in range(6)]
)


def causal(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build publication-age/count controls using only current and previous rows."""
    flags, log_age = context(x)
    counts = []
    prefix = np.vstack((np.zeros((1, 6)), flags.cumsum(axis=0)))
    row = np.arange(len(x))
    for window in (32, 128):
        counts.append(prefix[row + 1] - prefix[np.maximum(0, row + 1 - window)])
    return flags, np.column_stack((np.expm1(log_age), *counts))


def main() -> None:
    """Test transformed trade baselines and causal inverse-CDF associations."""
    start = time.monotonic()
    panel = json.loads((HERE.parent / "selection.json").read_text())
    classes = {
        r["group"]: r["diagnostic_target_class"]
        for r in json.loads((HERE.parent / "results.json").read_text())["by_group"]
    }
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    mode_counts = [Counter() for _ in TRADE_STARTS]
    fit_x, fit_y = [], []
    for group in panel["fit_train_groups"]:
        x = read(source, group)
        _, feature = causal(x)
        loc = np.linspace(0, len(x) - 1, 512, dtype=int)
        fit_x.append(feature[loc])
        fit_y.append(ndtr(x[loc, 104:]))
        for i, first in enumerate(TRADE_STARTS):
            mode_counts[i].update(
                v.tobytes() for v in x[loc[::2], first : first + 4].astype(np.float32)
            )
    fx, fy = np.concatenate(fit_x), np.concatenate(fit_y)
    models = []
    train_error = []
    for a in range(8):
        model_row, errors = [], []
        for j in range(len(CANDIDATES)):
            increasing = bool(spearmanr(fx[:, j], fy[:, a]).statistic >= 0)
            model = IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit(
                fx[:, j], fy[:, a]
            )
            model_row.append(model)
            errors.append(float(np.square(model.predict(fx[:, j]) - fy[:, a]).mean()))
        models.append(model_row)
        train_error.append(errors)
    baseline = [np.frombuffer(v.most_common(1)[0][0], dtype=np.float32) for v in mode_counts]
    test_x, test_y, test_cls = [], [], []
    totals = [Counter() for _ in TRADE_STARTS]
    marker = [Counter() for _ in range(8)]
    by_group = []
    discrete = [Counter() for _ in range(8)]
    endpoint = np.float32(norm.ppf(1e-7 - np.finfo(float).eps))
    for group in panel["test_train_groups"]:
        x = read(source, group)
        flags, feature = causal(x)
        loc = np.linspace(0, len(x) - 1, 512, dtype=int)
        test_x.append(feature[loc])
        test_y.append(ndtr(x[loc, 104:]))
        test_cls.extend([classes[group]] * len(loc))
        local = []
        for i, first in enumerate(TRADE_STARTS):
            block = x[:, first : first + 4].astype(np.float32)
            active = np.all(block == baseline[i], 1)
            changed = np.any(block[1:] != block[:-1], 1)
            v = Counter(
                {
                    "rows": len(x),
                    "baseline_rows": int(active.sum()),
                    "changed_rows": int(changed.sum()),
                    "into_baseline_events": int((changed & active[1:]).sum()),
                    "outof_baseline_events": int((changed & active[:-1]).sum()),
                }
            )
            for j in range(6):
                v[f"baseline_with_block{j}_refresh_rows"] += int((active & flags[:, j]).sum())
                v[f"into_baseline_with_block{j}_refresh_events"] += int(
                    (changed & active[1:] & flags[1:, j]).sum()
                )
            totals[i].update(v)
            local.append(dict(v))
        for a in range(8):
            values = x[:, 104 + a].astype(np.float32)
            discrete[a].update(values.tolist())
            low = values == endpoint
            marker[a].update({"rows": len(x), "lowerendpoint_rows": int(low.sum())})
            for j in range(6):
                marker[a][f"lowerendpoint_with_block{j}_refresh"] += int((low & flags[:, j]).sum())
        by_group.append(
            {"group": group, "diagnostic_class": classes[group], "trade_baseline_counts": local}
        )
    tx, ty, cls = np.concatenate(test_x), np.concatenate(test_y), np.array(test_cls)
    reconstruction = []
    for a in range(8):
        methods = []
        for j, name in enumerate(CANDIDATES):
            pred = models[a][j].predict(tx[:, j])
            var = np.square(ty[:, a] - ty[:, a].mean()).sum()
            unknown = ~np.isin(tx[:, j], np.unique(fx[:, j]))
            methods.append(
                {
                    "name": name,
                    "train_mse": train_error[a][j],
                    "test_r2": float(1 - np.square(pred - ty[:, a]).sum() / var),
                    "test_mae": float(np.abs(pred - ty[:, a]).mean()),
                    "test_candidate_level_unknown_fraction": float(unknown.mean()),
                    "by_class_mae": {
                        str(c): float(np.abs(pred[cls == c] - ty[cls == c, a]).mean())
                        for c in range(4)
                    },
                }
            )
        best = int(np.argmin(train_error[a]))
        reconstruction.append(
            {
                "a": a,
                "train_selected_method": CANDIDATES[best],
                "heldout_selected": methods[best],
                "all_controls": methods,
            }
        )
    cdf = []
    for a, values in enumerate(discrete):
        unique = np.array(sorted(values))
        probability = ndtr(unique)
        cdf.append(
            {
                "a": a,
                "unique_codes": len(unique),
                "markers": dict(marker[a]),
                "fixed_grid_checks": {
                    str(n): {
                        "unique_fraction_near_grid1e6": float(
                            (np.abs(probability * n - np.rint(probability * n)) / n <= 1e-6).mean()
                        ),
                        "max_probability_grid_residual": float(
                            (np.abs(probability * n - np.rint(probability * n)) / n).max()
                        ),
                    }
                    for n in (198, 1998, 19998)
                },
                "a7_code_levels": unique.tolist() if a == 7 else None,
                "a7_cdf_levels": probability.tolist() if a == 7 else None,
            }
        )
    result = {
        "contract_sha256": hashlib.sha256(
            (HERE / "followup_contract.json").read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "endpoint": {
            "value_float32": float(endpoint),
            "cdf_of_endpoint": float(ndtr(float(endpoint))),
            "source": "sklearn.preprocessing._data.BOUNDS_THRESHOLD=1e-7; scipy.stats.norm.ppf(1e-7-float64eps)",
            "source_url": "https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/preprocessing/_data.py",
            "match_is_not_identification": True,
        },
        "train_trade_baselines": [
            {
                "first_field": first,
                "baseline_float32": b.tolist(),
                "fit_support": counts.most_common(1)[0][1],
                "fit_sample_rows": counts.total(),
            }
            for first, b, counts in zip(TRADE_STARTS, baseline, mode_counts, strict=True)
        ],
        "test_trade_baselines": [dict(v) for v in totals],
        "by_group": by_group,
        "cdf_diagnostics": cdf,
        "causal_univariate_inverseCDF_reconstruction": reconstruction,
        "resources": {
            "wall_seconds": time.monotonic() - start,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    (HERE / "followup_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(HERE / "followup_results.json"), "resources": result["resources"]}
        )
    )


if __name__ == "__main__":
    main()
