"""Blend the main head of a distilled student with its four family heads (zero extra GRU cost; foldable into the final Linear)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from crossfold_blend import load_fold  # noqa: E402
from grade_prediction_bank import score  # noqa: E402

FAMILIES = ("strong", "full_finetuned", "raw_wmse", "samefold_baseline")
COEF = np.array([0.30, 0.20, 0.25, 0.0625])


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "famdump_h230_f1_s3_ep3"
    y, mask, comp, models = load_fold(ROOT / "research/campaign_2026-10-06/bandit_continuation/prediction_bank/BANK_FOLD1.jsonl", 1,
                                      ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json", ROOT / "cache/valid_targets.npz")
    order = next(iter(models.values()))[2]
    a = np.load(ROOT / "cache" / f"allout_{name}_valid.npy", mmap_mode="r")
    A = np.asarray(a[order], dtype=np.float32)               # (N, T, 10)
    main_p, fam = A[..., :2], A[..., 2:].reshape(*A.shape[:2], 4, 2)

    def wp(p):
        r = score(y, mask, np.ascontiguousarray(p, dtype=np.float32))
        return r["wp"], r["t0"], r["t1"]
    out = {"main": wp(main_p)}
    for k, f in enumerate(FAMILIES):
        out[f"head:{f}"] = wp(fam[..., k, :])
    avg = (fam * (COEF / COEF.sum())[None, None, :, None]).sum(2)
    out["family weighted avg"] = wp(avg)
    out["family equal avg"] = wp(fam.mean(2))
    for lam in (0.25, 0.5, 0.75):
        out[f"main*(1-{lam})+family_avg*{lam}"] = wp((1 - lam) * main_p + lam * avg)
    # per-target lambda
    for l0, l1 in ((0.5, 0.0), (0.0, 0.5)):
        p = main_p.copy(); p[..., 0] = (1 - l0) * main_p[..., 0] + l0 * avg[..., 0]; p[..., 1] = (1 - l1) * main_p[..., 1] + l1 * avg[..., 1]
        out[f"lambda t0={l0} t1={l1}"] = wp(p)
    # common / difference components
    c_m, d_m = (main_p[..., 0] + main_p[..., 1]) / 2, (main_p[..., 0] - main_p[..., 1]) / 2
    c_f, d_f = (avg[..., 0] + avg[..., 1]) / 2, (avg[..., 0] - avg[..., 1]) / 2
    for tag, c, d in (("c from family, d from main", c_f, d_m), ("c from main, d from family", c_m, d_f)):
        out[tag] = wp(np.stack([c + d, c - d], -1))
    base = out["main"][0]
    for k, v in out.items():
        print(f"{v[0]:.5f} ({v[0] - base:+.5f}) t0 {v[1]:.5f} t1 {v[2]:.5f}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
