"""Cache per-sequence WP sums for the clean four-fold reference and the 11 public-scored packages.

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/de_proxy_cache.py

Reference: fold k evaluation sequences scored by the equal mean of c20260929_ft_since_s{3,4}_f{k}
(research/campaign_2026-09-30/audit/reference_fourfold_design.json). Public packages: every member
was fitted without the 368 hash-holdout validation sequences, so only those sequences are summed.
Package specs and user-reported public scores are those of scripts/campaign_public_calibration.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from de_public_proxy import blend_sequence_sums, load_targets, sequence_table  # noqa: E402

OUT = ROOT / "research" / "campaign_2026-10-01" / "data_eval"

# name -> ({member: weight}, public score); baseline predictions enter at scale 0.25
PACKAGES = {
    "baseline_x025": ({"baseline": 0.25}, 0.6503),
    "ab18": ({"ab18_wmse_struct": 1.0}, 0.6488),
    "final_tv_e2": ({"ab21_final_tv_e2": 1.0}, 0.6548),
    "blend_base_final_tv": ({"ab21_final_tv_e2": 0.5, "baseline": 0.125}, 0.6569),
    "final_s4_base35": ({"ab33_final_s4": 0.65, "baseline": 0.0875}, 0.6586),
    "final_s4ft_base35": ({"ab40_final_s4_ft": 0.65, "baseline": 0.0875}, 0.6600),
    "final_s4lpft_base35": ({"ab51_final_lpft": 0.65, "baseline": 0.0875}, 0.6601),
    "final_ema4_lpft_base35": ({"ab60_final_ema4_p1_lpft": 0.65, "baseline": 0.0875}, 0.6618),
    "final_since2_gclp": ({"ab67_final_since_s4_gcorr": 0.5, "ab67_final_since_s3_gcorr": 0.5}, 0.6622),
    "final_mixed_ema6_shared": ({"ab72_final_ema6_s3_ep5": 0.5, "ab67_final_since_s4": 0.5}, 0.6616),
    "final_ema6_shared": ({"ab72_final_ema6_s3_ep5": 0.5, "ab72_final_ema6_s4_ep5": 0.5}, 0.6612),
}
# EMA6 "single" CPU-path scenario: both EMA6 archives collapse to member 0
SINGLE = {
    "final_mixed_ema6_shared": {"ab72_final_ema6_s3_ep5": 1.0},
    "final_ema6_shared": {"ab72_final_ema6_s3_ep5": 1.0},
}


def weighted_sums(members: dict[str, float], rows, y, mask, chunk=46) -> np.ndarray:
    """Clip-after-blend sums of a fixed linear combination (float64 accumulation as historical scoring)."""
    from de_public_proxy import sequence_sums

    arrays = {n: np.load(ROOT / "cache" / f"pred_{n}_valid.npy", mmap_mode="r") for n in members}
    out = np.zeros((len(rows), 2, 2, 6))
    for a in range(0, len(rows), chunk):
        r = rows[a : a + chunk]
        p = sum(w * np.asarray(arrays[n][r], np.float64) for n, w in members.items())
        out[a : a + len(r)] = sequence_sums(p, np.asarray(y[r]), np.asarray(mask[r]))
    return out


def main() -> None:
    y, mask, *_ = load_targets()
    table = sequence_table()
    ref = np.zeros((len(y), 2, 2, 6))
    for k in range(4):
        rows = np.flatnonzero(table["fold"] == k)
        paths = [f"cache/pred_c20260929_ft_since_s{s}_f{k}_valid.npy" for s in (3, 4)]
        ref[rows] = blend_sequence_sums(paths, rows, y, mask)
        print("reference fold", k, flush=True)
    hold = np.flatnonzero(table["holdout"])
    pkg = {}
    for name, (members, _) in PACKAGES.items():
        pkg[name] = weighted_sums(members, hold, y, mask)
        print("package", name, flush=True)
    for name, members in SINGLE.items():
        pkg[f"{name}__single"] = weighted_sums(members, hold, y, mask)
    np.savez(OUT / "seqsums_reference_fourfold.npz", sums=ref)
    np.savez(OUT / "seqsums_public_packages_holdout.npz", rows=hold, names=np.array(list(pkg)),
             public=np.array([PACKAGES[n.split("__")[0]][1] for n in pkg]), sums=np.stack([pkg[n] for n in pkg]))


if __name__ == "__main__":
    main()
