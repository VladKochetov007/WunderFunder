"""Does any predeclared causal feature help on the quiet rows that hold the loss?

The cell is deployable: no i0_p0/i0_p11 change and age since the last OR event
at least 32. Fit and eval are the same non-holdout split as wave1_anatomy.
Features are standardized on the fit cell. Alpha is predeclared at 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.gmetric import GlobalAccumulator
from wnn.splits import is_holdout

from wave1_anatomy import (
    COLS,
    FEATURE_NAMES,
    PRIMARY_ALPHA,
    ROOT,
    _clock_and_features,
    _mean_or_gap,
    _solve_ridge,
    _split_bit,
)

OUT = Path(__file__).resolve().parent


def _wp(y: np.ndarray, p: np.ndarray) -> float:
    acc = GlobalAccumulator()
    acc.add(y[:, None, :], p[:, None, :], np.ones(y.shape[0], dtype=bool)[:, None])
    return float(acc.result()["wp"])


def _standardize_fit(
    features: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sw = weight
    mean = (features * sw[:, None]).sum(axis=0) / sw.sum()
    var = ((features - mean) ** 2 * sw[:, None]).sum(axis=0) / sw.sum()
    scale = np.sqrt(np.maximum(var, 1e-8))
    return (features - mean) / scale, mean, scale


def main() -> None:
    cache = ROOT / "cache"
    with np.load(cache / "valid_targets.npz") as valid:
        y_all = valid["y"]
        mask = valid["mask"]
        seq_ix = valid["seq_ix"]
        row_group = valid["row_group"]
        market_class = valid["market_class"]
    parent = np.load(cache / "pred_c20261004_combo_s4_final_fp32_valid.npy", mmap_mode="r")
    head = np.load(cache / "pred_c20261004_combo_head_s4_final_fp32_valid.npy", mmap_mode="r")
    train = pq.ParquetFile(ROOT / "datasets" / "train.parquet")
    decay = float(np.exp(-1.0 / _mean_or_gap(train)))
    valid_pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")

    fit_f: list[np.ndarray] = []
    fit_y: list[np.ndarray] = []
    fit_p: list[np.ndarray] = []
    fit_c: list[np.ndarray] = []
    ev_f: list[np.ndarray] = []
    ev_y: list[np.ndarray] = []
    ev_p: list[np.ndarray] = []
    ev_c: list[np.ndarray] = []
    class_weight = np.zeros(4, dtype=np.float64)
    cell_rows = 0
    scored_rows = 0

    for npz_index in range(y_all.shape[0]):
        if is_holdout(int(seq_ix[npz_index])):
            continue
        table = valid_pf.read_row_group(
            int(row_group[npz_index]), columns=list(COLS), use_threads=False
        )
        cols = {name: table.column(name).to_numpy() for name in COLS}
        state = _clock_and_features(cols, decay)
        scored = mask[npz_index]
        quiet = (~state["or_ev"]) & (state["age_or"] >= 32) & scored
        scored_rows += int(scored.sum())
        cell_rows += int(quiet.sum())
        if not quiet.any():
            continue
        feat = state["features"][quiet]
        yy = y_all[npz_index][quiet]
        pred = (
            0.5
            * (
                parent[npz_index][quiet].astype(np.float64)
                + head[npz_index][quiet].astype(np.float64)
            )
        ).astype(np.float32)
        w = np.abs(np.clip(yy, -2, 2)).sum()
        class_weight[int(market_class[npz_index])] += w
        bucket_f, bucket_y, bucket_p, bucket_c = (
            (fit_f, fit_y, fit_p, fit_c)
            if _split_bit(int(seq_ix[npz_index]), "wave1b") == 0
            else (ev_f, ev_y, ev_p, ev_c)
        )
        bucket_f.append(feat.astype(np.float32))
        bucket_y.append(yy.astype(np.float32))
        bucket_p.append(pred)
        bucket_c.append(np.full(feat.shape[0], int(market_class[npz_index]), dtype=np.int8))

    def pack(
        fs: list[np.ndarray], ys: list[np.ndarray], ps: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.concatenate(fs).astype(np.float64),
            np.concatenate(ys).astype(np.float64),
            np.concatenate(ps).astype(np.float64),
        )

    x_fit, y_fit, p_fit = pack(fit_f, fit_y, fit_p)
    x_ev, y_ev, p_ev = pack(ev_f, ev_y, ev_p)
    w_fit = np.abs(np.clip(y_fit, -2, 2)).mean(axis=1)
    x_std, mean, scale = _standardize_fit(x_fit, w_fit)
    x_ev_std = (x_ev - mean) / scale

    def fit_predict(x_train: np.ndarray, extra_train: np.ndarray | None, x_test: np.ndarray, extra_test: np.ndarray | None) -> np.ndarray:
        if extra_train is None:
            design_train = x_train
            design_test = x_test
        else:
            design_train = np.column_stack([x_train, extra_train])
            design_test = np.column_stack([x_test, extra_test])
        sw = np.sqrt(np.maximum(w_fit, 0.0))
        weighted = np.column_stack([np.ones(design_train.shape[0]), design_train]) * sw[:, None]
        xtx = weighted.T @ weighted
        xty = weighted.T @ (y_fit * sw[:, None])
        beta = _solve_ridge(xtx, xty, PRIMARY_ALPHA)
        pred = np.column_stack([np.ones(design_test.shape[0]), design_test]) @ beta
        return np.clip(pred, -2, 2).astype(np.float32)

    incumbent = _wp(y_ev.astype(np.float32), p_ev.astype(np.float32))
    feat_only = _wp(y_ev.astype(np.float32), fit_predict(x_std, None, x_ev_std, None))
    plus = _wp(
        y_ev.astype(np.float32),
        fit_predict(x_std, p_fit, x_ev_std, p_ev),
    )
    # Weighted correlation of each standardized feature with the incumbent residual.
    residual = y_ev - p_ev
    w_ev = np.abs(np.clip(y_ev, -2, 2)).mean(axis=1)
    sw = w_ev / w_ev.sum()
    corrs = []
    for j, name in enumerate(FEATURE_NAMES):
        col = x_ev_std[:, j]
        pair = []
        for t in range(2):
            a = col - np.sum(sw * col)
            b = residual[:, t] - np.sum(sw * residual[:, t])
            denom = np.sqrt(np.sum(sw * a * a) * np.sum(sw * b * b))
            pair.append(float(np.sum(sw * a * b) / denom) if denom > 1e-12 else 0.0)
        corrs.append({"feature": name, "t0": pair[0], "t1": pair[1]})

    # One small MLP on the same standardized features, residual added to the incumbent.
    import torch

    torch.manual_seed(0)
    device = torch.device("cpu")
    # Keep the MLP inside the memory budget: CPU, width 32, four passes over a 200k subsample.
    rng = np.random.default_rng(0)
    take = rng.choice(x_std.shape[0], size=min(200_000, x_std.shape[0]), replace=False)
    model = torch.nn.Sequential(
        torch.nn.Linear(x_std.shape[1] + 2, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 2),
    ).to(device)
    # Zero the output layer so the model starts as "leave the incumbent alone".
    torch.nn.init.zeros_(model[-1].weight)
    torch.nn.init.zeros_(model[-1].bias)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xb = torch.tensor(np.column_stack([x_std[take], p_fit[take]]), dtype=torch.float32)
    yb = torch.tensor(y_fit[take], dtype=torch.float32)
    wb = torch.tensor(w_fit[take], dtype=torch.float32)
    model.train()
    for _ in range(4):
        pred = model(xb) + xb[:, -2:]
        loss = (wb[:, None] * (pred - yb) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        ev_in = torch.tensor(np.column_stack([x_ev_std, p_ev]), dtype=torch.float32)
        mlp_pred = (model(ev_in) + ev_in[:, -2:]).clamp(-2, 2).numpy().astype(np.float32)
    mlp_wp = _wp(y_ev.astype(np.float32), mlp_pred)

    payload = {
        "cell": "OR clock quiet and age>=32, non-holdout valid",
        "scored_rows_dev": scored_rows,
        "cell_rows_dev": cell_rows,
        "cell_row_share_dev": cell_rows / scored_rows if scored_rows else 0.0,
        "fit_rows": int(x_fit.shape[0]),
        "eval_rows": int(x_ev.shape[0]),
        "class_weight_abs_sum": class_weight.tolist(),
        "class_weight_share": (class_weight / class_weight.sum()).tolist(),
        "incumbent_wp": incumbent,
        "standardized_features_only_wp": feat_only,
        "features_plus_incumbent_wp": plus,
        "delta_vs_incumbent": plus - incumbent,
        "mlp_residual_wp": mlp_wp,
        "mlp_delta_vs_incumbent": mlp_wp - incumbent,
        "residual_correlations": corrs,
        "strongest_abs_corr": max(corrs, key=lambda r: max(abs(r["t0"]), abs(r["t1"]))),
    }
    (OUT / "wave1_quiet.json").write_text(json.dumps(payload, indent=2))
    print(
        "QUIET_DONE",
        payload["delta_vs_incumbent"],
        payload["mlp_delta_vs_incumbent"],
        payload["strongest_abs_corr"],
        flush=True,
    )


if __name__ == "__main__":
    main()
