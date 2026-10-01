from __future__ import annotations

import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
NAMES = [
    f"i{i}_{kind}{j}"
    for i in range(2)
    for kind, width in (("p", 22), ("v", 22), ("dp", 4), ("dv", 4))
    for j in range(width)
] + [f"a{i}" for i in range(8)]
BLOCKS = [(0, 22), (22, 44), (44, 52), (52, 74), (74, 96), (96, 104)]


def read(source: pq.ParquetFile, group: int) -> np.ndarray:
    """Read feature-only TRAIN sequence."""
    tab = source.read_row_group(group, columns=NAMES, use_threads=False)
    return np.column_stack([tab.column(k).to_numpy() for k in NAMES]).astype(np.float64)


def context(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute causal equality-clock ages without assuming physical dimensions."""
    change = x[1:] != x[:-1]
    flags = np.vstack(
        (np.zeros((1, 6), bool), np.column_stack([change[:, a:b].any(1) for a, b in BLOCKS]))
    )
    age_flags = np.column_stack((flags, flags[:, 0] | flags[:, 3], flags[:, 2] | flags[:, 5]))
    index = np.arange(len(x))
    age = index[:, None] - np.maximum.accumulate(np.where(age_flags, index[:, None], 0), axis=0)
    return flags, np.log1p(age)


def fit_beta(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a fixed ridge feature reconstruction control without target labels."""
    mean, scale = x.mean(0), x.std(0)
    scale = np.maximum(scale, 1e-6)
    z = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    beta = np.linalg.solve(z.T @ z + np.diag([0] + [0.001] * x.shape[1]), z.T @ y)
    return mean, scale, beta


def stats(y: np.ndarray, p: np.ndarray) -> dict[str, object]:
    """Report per-a-field reconstruction diagnostics."""
    var = np.square(y - y.mean(0)).sum(0)
    err = np.square(y - p).sum(0)
    r2 = np.divide(err, var, out=np.full(8, np.nan), where=var > 0)
    return {"rows": len(y), "r2": (1 - r2).tolist(), "mae": np.abs(y - p).mean(0).tolist()}


def main() -> None:
    """Audit all112 current/past feature relations on fixed TRAIN chain groups."""
    start = time.monotonic()
    panel = json.loads((HERE.parent / "selection.json").read_text())
    primary = json.loads((HERE.parent / "results.json").read_text())
    classes = {v["group"]: v["diagnostic_target_class"] for v in primary["by_group"]}
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    field_votes = [defaultdict(Counter) for _ in range(4)]
    fit_x, fit_age, fit_y = [], [], []
    field_starts = [0, 11, 52, 63]
    for group in panel["fit_train_groups"]:
        x = read(source, group)
        _, age = context(x)
        loc = np.linspace(0, len(x) - 1, 256, dtype=int)
        fit_x.append(x[loc, :104])
        fit_age.append(age[loc])
        fit_y.append(x[loc, 104:])
        for i, first in enumerate(field_starts):
            values = x[:, first : first + 11].astype(np.float32)
            ev = np.r_[0, np.flatnonzero(np.any(values[1:] != values[:-1], 1)) + 1]
            for row in ev:
                field_votes[i][float(values[row, 0])][values[row, 1:].tobytes()] += 1
    xx, ages, yy = np.concatenate(fit_x), np.concatenate(fit_age), np.concatenate(fit_y)
    controls = {
        "current104": fit_beta(xx, yy),
        "current104_plus_causal_age8": fit_beta(np.column_stack((xx, ages)), yy),
    }
    tables = [
        {
            value: votes.most_common(1)[0][0]
            for value, votes in v.items()
            if votes.most_common(1)[0][1] >= 3
            and votes.most_common(1)[0][1] / votes.total() >= 0.95
        }
        for v in field_votes
    ]
    counts = Counter()
    by_group = []
    eval_y = []
    eval_p = {k: [] for k in controls}
    eval_class, eval_event = [], []
    extra = [[] for _ in range(8)]
    for group in panel["test_train_groups"]:
        x = read(source, group)
        flags, age = context(x)
        local = Counter({"rows": len(x) - 1})
        for i in range(6):
            local[f"block{i}_refresh"] += int(flags[1:, i].sum())
        for i in range(6):
            for j in range(i + 1, 6):
                local[f"blocks{i}_{j}_joint_refresh"] += int((flags[1:, i] & flags[1:, j]).sum())
        for first in (44, 48, 96, 100):
            block = x[:, first : first + 4]
            changed = np.any(block[1:] != block[:-1], 1)
            a, b = block[1:][changed], block[:-1][changed]
            local[f"field{first}_changed"] += len(a)
            local[f"field{first}_shift_left_exact"] += int(np.all(a[:, :-1] == b[:, 1:], 1).sum())
            local[f"field{first}_shift_right_exact"] += int(np.all(a[:, 1:] == b[:, :-1], 1).sum())
            local[f"field{first}_clear_exact"] += int(np.all(a == 0, 1).sum())
            local[f"field{first}_allzero_rows"] += int(np.all(block == 0, 1).sum())
        for i, first in enumerate(field_starts):
            block = x[:, first : first + 11].astype(np.float32)
            ev = np.r_[0, np.flatnonzero(np.any(block[1:] != block[:-1], 1)) + 1]
            local[f"priceblock{i}_event_states"] += len(ev)
            for row in ev:
                expected = tables[i].get(float(block[row, 0]))
                if expected is not None:
                    local[f"priceblock{i}_mapped_states"] += 1
                    local[f"priceblock{i}_exact_states"] += expected == block[row, 1:].tobytes()
        loc = np.linspace(0, len(x) - 1, 256, dtype=int)
        eval_y.append(x[loc, 104:])
        eval_class.extend([classes[group]] * len(loc))
        eval_event.extend(flags[loc, 0].tolist())
        for name, (mean, scale, beta) in controls.items():
            features = (
                x[loc, :104] if name == "current104" else np.column_stack((x[loc, :104], age[loc]))
            )
            eval_p[name].append(
                np.column_stack((np.ones(len(loc)), (features - mean) / scale)) @ beta
            )
        for i in range(8):
            extra[i].append(x[:, 104 + i])
        by_group.append(
            {"group": group, "class_diagnostic_only": classes[group], "counts": dict(local)}
        )
        counts.update(local)
    y = np.concatenate(eval_y)
    cls, event = np.array(eval_class), np.array(eval_event)
    regression = {}
    for name, pieces in eval_p.items():
        pred = np.concatenate(pieces)
        regression[name] = {
            "all": stats(y, pred),
            "by_class": {str(c): stats(y[cls == c], pred[cls == c]) for c in range(4)},
            "i0_price_quiet": stats(y[~event], pred[~event]),
            "i0_price_event": stats(y[event], pred[event]),
        }
    extra_stats = []
    for pieces in extra:
        values = np.concatenate(pieces)
        delta = np.diff(values)
        extra_stats.append(
            {
                "min_max": [float(values.min()), float(values.max())],
                "unique_values": len(np.unique(values)),
                "positive_fraction": float((values > 0).mean()),
                "quantiles": np.quantile(values, [0.01, 0.5, 0.99]).tolist(),
                "warning": "Concatenated ranges only; derivatives across group boundaries are excluded from claims",
                "row_change_fraction_including_boundaries_descriptive": float((delta != 0).mean()),
            }
        )
    result = {
        "preregistration_sha256": hashlib.sha256(
            (HERE / "preregistration.json").read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "columns": NAMES,
        "fit_field_maps": [
            {
                "first_column": NAMES[first],
                "fit_values": len(v),
                "ambiguous_firstfield_values": sum(len(c) > 1 for c in v.values()),
                "accepted_values": len(t),
            }
            for first, v, t in zip(field_starts, field_votes, tables, strict=True)
        ],
        "test_counts": dict(counts),
        "by_group": by_group,
        "a0_to_a7": extra_stats,
        "a_feature_only_reconstruction": regression,
        "resources": {
            "wall_seconds": time.monotonic() - start,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps({"output": str(HERE / "results.json"), "resources": result["resources"]}))


if __name__ == "__main__":
    main()
