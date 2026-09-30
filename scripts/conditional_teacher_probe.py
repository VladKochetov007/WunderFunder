"""Registered I1 cached teacher-mixture screen; no fitting or new inference.

    python scripts/conditional_teacher_probe.py PREREG.json --out RESULT.json

Writes RESULT.inputs.json with input hashes before reading scoring arrays. Refuses
existing artifacts. Uses <=16-sequence prediction blocks, exact pooled scorer
statistics, and fold-stratified whole-chain bootstrap of fixed model predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

# Set before numpy initialization; the command must also enforce the process CPU/RAM limit.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_var] = "2"

import fold_eval  # noqa: E402
import numpy as np  # noqa: E402
import paired_seed_eval  # noqa: E402
from fold_eval import ROOT, WARMUP, pooled, sums  # noqa: E402
from paired_seed_eval import sha256  # noqa: E402

COMPONENTS = {
    "plain3": "ft256H", "plain4": "ft256s4H",
    "distilled3": "ft256dH", "distilled4": "ft256ds4H",
    "small3": "ft128H", "small4": "ft128s4H", "chunk500": "ftc500H",
}
NAMES = ("originalC", "Q", "U", "G")
COMPARISONS = {"G_minus_U": (3, 2), "G_minus_originalC": (3, 0), "G_minus_Q": (3, 1)}


def validate_manifest(m: dict) -> None:
    if not isinstance(m, dict):
        raise ValueError("Manifest must be a JSON object")
    if (m.get("status") != "preregistered" or m.get("evidence_role") != "discovery"
            or m.get("screen_id") != "I1" or m.get("components") != COMPONENTS
            or m.get("folds") != [0, 1]
            or any(type(f) is not int for f in m["folds"])):
        raise ValueError("Requires the registered I1 discovery components and folds 0,1")
    if type(m.get("block_sequences")) is not int or not 1 <= m["block_sequences"] <= 16:
        raise ValueError("Prediction blocks must contain 1 to 16 sequences")
    if type(m.get("bootstrap_repeats")) is not int or not 100 <= m["bootstrap_repeats"] <= 2000:
        raise ValueError("Bootstrap repeats must be an integer from 100 to 2000")
    if type(m.get("bootstrap_seed")) is not int or m["bootstrap_seed"] < 0:
        raise ValueError("Bootstrap seed must be a nonnegative integer")
    if m.get("gate") != {"pooled_delta_min": 0.0004, "fold_delta_positive": True,
                         "fold_delta_originalC_min": 0.0, "weight_coverage_min": 0.05}:
        raise ValueError("Allocation gate differs from the registered I1 screen")


def mixtures(parts: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return four final ensemble predictions, then the prediction-only row gate.

    No component is clipped before averaging. All operations are float32; the
    scorer clips the final ensemble predictions and targets during accumulation.
    """
    if set(parts) != set(COMPONENTS):
        raise ValueError("Exactly the seven registered components are required")
    shape = next(iter(parts.values())).shape
    if (len(shape) != 3 or shape[-1] != 2 or any(p.shape != shape for p in parts.values())
            or any(not np.isfinite(p).all() for p in parts.values())):
        raise ValueError("Component predictions must be finite, aligned (B,T,2) arrays")
    p = np.float32(0.5) * (parts["plain3"] + parts["plain4"])
    q = np.float32(0.5) * (parts["distilled3"] + parts["distilled4"])
    dp, dq = np.float32(0.5) * (p[..., 0] - p[..., 1]), np.float32(0.5) * (q[..., 0] - q[..., 1])
    gate = np.abs(dp - dq) > np.float32(0.5) * (np.abs(dp) + np.abs(dq) + np.float32(0.05))
    u = np.float32(0.5) * (p + q)
    g = q + np.float32(0.5) * gate[..., None] * (p - q)
    other = parts["small3"] + parts["small4"] + parts["chunk500"]
    out = np.stack([np.float32(0.25) * (z + other) for z in (parts["distilled3"], q, u, g)])
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite ensemble predictions")
    return out, gate


def coverage_sums(y: np.ndarray, mask: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Per-sequence: scored rows, gated scored rows, and total/gated weights by target."""
    w = np.abs(np.clip(y.astype(np.float64), -2, 2)) * mask[..., None]
    return np.column_stack([mask.sum(1), (mask & gate).sum(1),
                            w.sum(1), (w * gate[..., None]).sum(1)])


def summarize(stats: np.ndarray, fold: np.ndarray, chain: np.ndarray, classes: np.ndarray,
              coverage: np.ndarray, repeats: int, seed: int) -> dict:
    """Fixed-artifact analogue of paired_seed_eval's stratified chain bootstrap.

    Unlike its seed-specific summarize(), these four alternatives are not training
    repetitions. No synthetic replicas or training-seed confidence claims are made.
    """
    n = len(fold)
    if (stats.shape != (4, n, 2, 6) or n == 0 or chain.shape != (n,)
            or classes.shape != (n,) or coverage.shape != (n, 6)
            or not np.isfinite(stats).all() or not np.isfinite(coverage).all()):
        raise ValueError("Malformed fixed-artifact statistics or metadata")
    if any(len(np.unique(fold[chain == ch])) != 1 for ch in np.unique(chain)):
        raise ValueError("A continuous chain cannot be split across evaluation folds")
    scopes = {f"fold_{f}": fold == f for f in np.unique(fold)}
    scopes["pooled"] = np.ones(n, dtype=bool)
    reports = {}
    for scope, selected in scopes.items():
        s = stats[:, selected].sum(1)
        c = coverage[selected].sum(0)
        total_weight = c[2:4].sum()
        if c[0] == 0 or total_weight == 0:
            raise ValueError(f"No scored rows/target weight in {scope}")
        scores = [pooled(a) for a in s]
        reports[scope] = {
            "sequences": int(selected.sum()), "chains": int(len(np.unique(chain[selected]))),
            "wp": dict(zip(NAMES, scores, strict=True)),
            "comparisons": {name: {"delta": scores[a] - scores[b]} for name, (a, b) in COMPARISONS.items()},
            "coverage": {"scored_rows": int(c[0]), "gated_rows": int(c[1]),
                         "row_fraction": float(c[1] / c[0]),
                         "target_weight": c[2:4].tolist(), "gated_target_weight": c[4:6].tolist(),
                         "metric_weight_fraction": float(c[4:6].sum() / total_weight),
                         "weight_fraction_by_target": [float(c[k + 4] / c[k + 2]) if c[k + 2] else None for k in range(2)]},
        }
    keys = sorted(set(zip(fold.tolist(), chain.tolist(), strict=True)))
    grouped = np.stack([stats[:, (fold == f) & (chain == ch)].sum(1) for f, ch in keys], axis=1)
    group_folds = np.array([f for f, _ in keys])
    draws = {scope: {name: [] for name in COMPARISONS} for scope in scopes}
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        by_fold = {f: rng.choice(ix, len(ix), replace=True) for f in np.unique(fold)
                   for ix in [np.flatnonzero(group_folds == f)]}
        picks = {f"fold_{f}": ix for f, ix in by_fold.items()}
        picks["pooled"] = np.concatenate(list(by_fold.values()))
        for scope, ix in picks.items():
            scores = [pooled(s) for s in grouped[:, ix].sum(1)]
            for name, (a, b) in COMPARISONS.items():
                draws[scope][name].append(scores[a] - scores[b])
    for scope in scopes:
        for name, boot in draws[scope].items():
            reports[scope]["comparisons"][name].update(
                conditional_chain_sd=float(np.std(boot)),
                conditional_chain_ci95=np.quantile(boot, [0.025, 0.975]).tolist())
    descriptive = {}
    for c in np.unique(classes):
        s = stats[:, classes == c].sum(1)
        scores = [pooled(a) for a in s]
        descriptive[str(int(c))] = {name: scores[a] - scores[b] for name, (a, b) in COMPARISONS.items()}
    passed = (reports["pooled"]["comparisons"]["G_minus_U"]["delta"] >= 0.0004
              and all(r["comparisons"]["G_minus_U"]["delta"] > 0
                      and r["comparisons"]["G_minus_originalC"]["delta"] >= 0
                      and r["coverage"]["metric_weight_fraction"] >= 0.05
                      for scope, r in reports.items() if scope != "pooled"))
    return {"scores": reports, "descriptive_class_deltas_pooled": descriptive,
            "allocation_gate_passed": bool(passed), "confirmation_established": False,
            "evidence_status": "exploratory_fixed_teacher_mixture_on_reused_folds",
            "uncertainty_scope": "whole-chain sampling conditional on these fitted artifacts",
            "not_established": ["isolated conditioning effect", "student improvement", "training-seed robustness",
                                "selection-adjusted significance", "public or hidden-test improvement"]}


def fingerprint(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    inventory_path = args.out.with_suffix(".inputs.json")
    if args.out.exists() or inventory_path.exists():
        raise FileExistsError("Result or pre-outcome inventory already exists; do not overwrite")
    manifest_bytes = args.manifest.read_bytes()
    m = json.loads(manifest_bytes)
    validate_manifest(m)
    paths = [ROOT / "cache/valid_targets.npz", ROOT / "cache/chain_folds.npz"]
    paths += [ROOT / "cache" / f"pred_{run}_f{fold}_valid.npy"
              for fold in m["folds"] for run in COMPONENTS.values()]
    before = {p: fingerprint(p) for p in paths}  # Missing input fails before any scoring.
    inputs = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths}
    if any(fingerprint(p) != before[p] for p in paths):
        raise RuntimeError("Input changed during hashing")
    inventory = {"screen_id": "I1", "created_utc": datetime.now(UTC).isoformat(),
                 "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                 "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                 "code_sha256": {"probe": sha256(Path(__file__)), "scorer": sha256(Path(fold_eval.__file__)),
                                 "shared_hash_helper": sha256(Path(paired_seed_eval.__file__))},
                 "inputs": inputs, "python": platform.python_version(), "numpy": np.__version__,
                 "row_alignment_basis": "predict_run.py writes full caches in valid_targets.row_group order; shape and unique metadata checked, no independent IDs embedded in .npy"}
    write_new(inventory_path, inventory)  # Immutable, before loading labels or computing any WP.
    with np.load(paths[0]) as target:
        rg, y, mask, classes = (target[k] for k in ("row_group", "y", "mask", "market_class"))
    if (y.shape != (len(rg), 20000, 2) or mask.shape != y.shape[:2]
            or classes.shape != rg.shape or len(np.unique(rg)) != len(rg)):
        raise ValueError("Malformed target cache or duplicate row groups")
    with np.load(paths[1]) as cf:
        valid = cf["split"] == "valid"
        groups = cf["row_group"][valid]
        if len(np.unique(groups)) != len(groups):
            raise ValueError("Duplicate validation row groups in chain metadata")
        info = {int(g): (int(f), int(ch), bool(pure)) for g, f, ch, pure in
                zip(groups, cf["fold"][valid], cf["chain"][valid], cf["pure"][valid], strict=True)}
    if any(int(g) not in info for g in rg):
        raise ValueError("Target row group missing from chain metadata")
    all_stats, cover, folds, chains, cls, selected = [], [], [], [], [], []
    for f in m["folds"]:
        ix = np.array([i for i, g in enumerate(rg) if info[int(g)][0] == f and info[int(g)][2]])
        if not len(ix):
            raise ValueError(f"No pure evaluation sequences for fold {f}")
        arrays = {name: np.load(ROOT / "cache" / f"pred_{run}_f{f}_valid.npy", mmap_mode="r")
                  for name, run in COMPONENTS.items()}
        if any(p.shape != y.shape or p.dtype != np.float32 for p in arrays.values()):
            raise ValueError("Prediction shape/dtype mismatch")
        for start in range(0, len(ix), m["block_sequences"]):
            rows = ix[start:start + m["block_sequences"]]
            ym, mm = y[rows], mask[rows].astype(bool, copy=True)
            mm[:, :WARMUP] = False
            if not np.isfinite(ym).all():
                raise ValueError("Nonfinite target values")
            pred, gate = mixtures({name: np.asarray(p[rows], dtype=np.float32) for name, p in arrays.items()})
            all_stats.append(np.stack([sums(ym, p, mm) for p in pred]))
            cover.append(coverage_sums(ym, mm, gate))
            selected.extend(int(rg[i]) for i in rows)
            folds.extend([f] * len(rows))
            chains.extend(info[int(rg[i])][1] for i in rows)
            cls.extend(int(classes[i]) for i in rows)
    result = summarize(np.concatenate(all_stats, axis=1), np.array(folds), np.array(chains), np.array(cls),
                       np.concatenate(cover), m["bootstrap_repeats"], m["bootstrap_seed"])
    if any(fingerprint(p) != before[p] for p in paths):
        raise RuntimeError("Input changed during evaluation; result not written")
    result["provenance"] = {"input_inventory": str(inventory_path), "inventory_sha256": sha256(inventory_path),
                            "selected_row_groups_sha256": hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
                            "finished_utc": datetime.now(UTC).isoformat()}
    write_new(args.out, result)
    print(json.dumps({"out": str(args.out), "allocation_gate_passed": result["allocation_gate_passed"],
                      "primary": result["scores"]["pooled"]["comparisons"]["G_minus_U"]}, indent=2))


if __name__ == "__main__":
    main()
