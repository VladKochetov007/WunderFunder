"""Pooled Global WP change of a candidate over a reference, on all evaluation rows and on public-like proxies.

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/de_public_proxy.py \
        --pair 0 cache/pred_REF_s3_f0_valid.npy,cache/pred_REF_s4_f0_valid.npy cache/pred_CAND_f0_valid.npy \
        [--pair 1 ...] [--proxy research/campaign_2026-10-01/data_eval/proxy.json] [--boot 2000]

Each ``--pair FOLD REF CAND`` scores the evaluation sequences of the chain-excluded manifest
research/campaign_2026-09-29/artifacts/splits/fold{FOLD}.json. REF and CAND are comma lists of
complete [1873, 20000, 2] validation exports, averaged with equal weight in float64, rounded to
float32 and clipped to [-2, 2] (the frozen campaign scorer's order). Rows: step >= 99 and is_scored.

Outputs pooled ΔWP on (i) all evaluation rows and (ii) every proxy weighting listed in proxy.json
(the recommended one first), plus each class and each fold, all with a paired component bootstrap SD. A proxy weighting multiplies
whole sequences' contributions by a weight that depends only on market class and chain metadata
(never on predictions or targets of the scored rows beyond the class label).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.gmetric import wp_from_sums  # noqa: E402

WARMUP = 99
SPLITS = ROOT / "research" / "campaign_2026-09-29" / "artifacts" / "splits"
PROXY_JSON = ROOT / "research" / "campaign_2026-10-01" / "data_eval" / "proxy.json"


def load_targets():
    """y (N, T, 2) float32 memmap-backed, mask (N, T) bool, and the per-sequence arrays."""
    with np.load(ROOT / "cache" / "valid_targets.npz") as z:
        return z["y"], z["mask"], z["row_group"].astype(int), z["seq_ix"].astype(int), z["market_class"].astype(int)


def sequence_table() -> dict[str, np.ndarray]:
    """Per validation sequence (target-cache order): class, component, chain/fold metadata."""
    with np.load(ROOT / "cache" / "valid_targets.npz") as z:
        rg, seq, cls = z["row_group"].astype(int), z["seq_ix"].astype(int), z["market_class"].astype(int)
    manifests = [json.loads((SPLITS / f"fold{k}.json").read_text()) for k in range(4)]
    comp_valid = {s["row_group"]: s["component"] for s in manifests[0]["sequences"]["valid"]}
    comp_train = np.array([s["component"] for s in manifests[0]["sequences"]["train"]])
    component = np.array([comp_valid[g] for g in rg])
    fold = np.full(len(rg), -1)
    for k, m in enumerate(manifests):
        fold[np.isin(rg, m["evaluation"]["valid"])] = k
    if (fold < 0).any():
        raise ValueError("sequence without evaluation fold")
    n_comp = max(component.max(), comp_train.max()) + 1
    n_valid = np.bincount(component, minlength=n_comp)
    n_train = np.bincount(comp_train, minlength=n_comp)
    holdout = np.array([int(hashlib.sha256(str(s).encode()).hexdigest(), 16) % 5 == 0 for s in seq])
    # historical proxy definition: holdout sequences whose component has no train or dev-valid member
    dev_components = set(component[~holdout].tolist())
    isolated = holdout & (n_train[component] == 0) & ~np.isin(component, list(dev_components))
    return {
        "row_group": rg,
        "seq_ix": seq,
        "market_class": cls,
        "component": component,
        "fold": fold,
        "component_valid": n_valid[component],
        "component_train": n_train[component],
        "component_len": n_valid[component] + n_train[component],
        "pure": n_train[component] == 0,
        "holdout": holdout,
        "isolated": isolated,
    }


def mix(paths: list[str], rows: np.ndarray) -> np.ndarray:
    """Equal float64 mean of member predictions for the given sequences, rounded to float32."""
    acc = None
    for path in paths:
        p = np.load(ROOT / path if not Path(path).is_absolute() else path, mmap_mode="r")
        part = np.asarray(p[rows], dtype=np.float64)
        acc = part if acc is None else acc + part
    return (acc / len(paths)).astype(np.float32)


def sums6(y: np.ndarray, p: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """(B, 6) raw weighted-Pearson sums over rows where sel; y, p clipped float64."""
    w = np.abs(y) * sel
    wy, wp = w * y, w * p
    return np.stack([w.sum(-1), wy.sum(-1), wp.sum(-1), (wy * y).sum(-1), (wp * p).sum(-1), (wy * p).sum(-1)], -1)


def sequence_sums(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(B, 2 [scored, unscored-required], 2 targets, 6) sums for predictions aligned to (y, mask) rows."""
    need = np.arange(y.shape[1]) >= WARMUP
    yc = np.clip(y.astype(np.float64), -2, 2)
    pc = np.clip(pred.astype(np.float64), -2, 2)
    out = np.zeros((len(y), 2, 2, 6))
    for k in range(2):
        out[:, 0, k] = sums6(yc[..., k], pc[..., k], need & mask)
        out[:, 1, k] = sums6(yc[..., k], pc[..., k], need & ~mask)
    return out


def blend_sequence_sums(paths: list[str], rows: np.ndarray, y, mask, chunk: int = 48) -> np.ndarray:
    out = np.zeros((len(rows), 2, 2, 6))
    for a in range(0, len(rows), chunk):
        r = rows[a : a + chunk]
        out[a : a + len(r)] = sequence_sums(mix(paths, r), np.asarray(y[r]), np.asarray(mask[r]))
    return out


def wp(s: np.ndarray, weights: np.ndarray | None = None, lam: float = 0.0) -> float:
    """Pooled two-target WP of (N, 2, 2, 6) sequence sums with per-sequence weights."""
    t = s[:, 0] + lam * s[:, 1]
    if weights is not None:
        t = t * weights[:, None, None]
    return float(np.mean([wp_from_sums(t[:, k].sum(0)) for k in range(2)]))


def proxy_weights(table: dict[str, np.ndarray], spec: dict) -> np.ndarray:
    """Per-sequence weight for a proxy spec (metadata only: class, component size, purity)."""
    n = len(table["market_class"])
    w = np.ones(n)
    cls = table["market_class"]
    if "class_weight" in spec:
        w = w * np.asarray(spec["class_weight"], dtype=np.float64)[cls]
    if "class_mix" in spec:  # sequence class shares of this panel moved to the mix (screen_score.py rule)
        share = np.bincount(cls, minlength=4) / n
        w = w * (np.asarray(spec["class_mix"], dtype=np.float64) / np.maximum(share, 1e-12))[cls]
    if "max_component_len" in spec:
        w = w * (table["component_len"] <= spec["max_component_len"])
    if "min_component_len" in spec:
        w = w * (table["component_len"] >= spec["min_component_len"])
    if spec.get("pure_only"):
        w = w * table["pure"]
    if "component_len_power" in spec:
        w = w * table["component_len"].astype(np.float64) ** spec["component_len_power"]
    return w


def paired_bootstrap(ref: np.ndarray, cand: np.ndarray, components: np.ndarray, weights: np.ndarray,
                     lam: float = 0.0, n_boot: int = 2000, seed: int = 271828) -> dict:
    """Resample whole components (shared across folds) and recompute both pooled weighted WPs."""
    ids, inv = np.unique(components, return_inverse=True)
    ta = (ref[:, 0] + lam * ref[:, 1]) * weights[:, None, None]
    tb = (cand[:, 0] + lam * cand[:, 1]) * weights[:, None, None]
    ca = np.zeros((len(ids), 2, 6))
    cb = np.zeros_like(ca)
    np.add.at(ca, inv, ta)
    np.add.at(cb, inv, tb)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=n_boot).astype(np.float64)

    def score(c, cnt):
        tot = np.einsum("bc,ckj->bkj", cnt, c)
        return wp_from_sums(tot).mean(-1)

    deltas = score(cb, counts) - score(ca, counts)
    observed = wp(cand, weights, lam) - wp(ref, weights, lam)
    return {"delta": observed, "boot_sd": float(deltas.std(ddof=1)),
            "boot_q025": float(np.quantile(deltas, 0.025)), "boot_q975": float(np.quantile(deltas, 0.975)),
            "p_le_0": float(np.mean(deltas <= 0))}


def compare(ref_s: np.ndarray, cand_s: np.ndarray, sub: dict, proxies: list, folds: np.ndarray,
            n_boot: int = 2000) -> dict:
    """Pooled delta on all rows, each proxy weighting, each class and each fold, with component bootstrap."""
    out = {"results": {}, "by_class": {}, "per_fold": {}}
    for item in [{"name": "all_eval_rows", "spec": {}}] + proxies:
        w = proxy_weights(sub, item["spec"])
        res = paired_bootstrap(ref_s, cand_s, sub["component"], w, n_boot=n_boot)
        res.update({"reference_wp": wp(ref_s, w), "candidate_wp": wp(cand_s, w),
                    "effective_sequences": float(w.sum() ** 2 / max((w**2).sum(), 1e-12))})
        out["results"][item["name"]] = res
    for c in range(4):
        w = (sub["market_class"] == c).astype(float)
        out["by_class"][str(c)] = paired_bootstrap(ref_s, cand_s, sub["component"], w, n_boot=n_boot)
    for f in np.unique(folds):
        w = (folds == f).astype(float)
        out["per_fold"][str(int(f))] = paired_bootstrap(ref_s, cand_s, sub["component"], w, n_boot=n_boot)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs=3, action="append", metavar=("FOLD", "REF", "CAND"), required=True)
    parser.add_argument("--proxy", default=str(PROXY_JSON))
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--output")
    args = parser.parse_args()
    y, mask, *_ = load_targets()
    table = sequence_table()
    proxies = json.loads(Path(args.proxy).read_text())["proxies_for_tool"]
    ref_parts, cand_parts, row_parts = [], [], []
    for fold, ref, cand in args.pair:
        rows = np.flatnonzero(table["fold"] == int(fold))
        ref_parts.append(blend_sequence_sums(ref.split(","), rows, y, mask))
        cand_parts.append(blend_sequence_sums(cand.split(","), rows, y, mask))
        row_parts.append(rows)
    ref_s, cand_s = np.concatenate(ref_parts), np.concatenate(cand_parts)
    rows = np.concatenate(row_parts)
    sub = {k: v[rows] for k, v in table.items()}
    report = {"folds": [int(f) for f, _, _ in args.pair], "n_sequences": int(len(rows)),
              "rows": "step>=99 AND is_scored", **compare(ref_s, cand_s, sub, proxies, sub["fold"], args.boot)}
    text = json.dumps(report, indent=1)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
