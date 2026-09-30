"""Evaluate registered candidate/control training-seed pairs on exposed pure chains.

This is a discovery report, not a significance test after model selection. Training
seed variation, fixed-model chain uncertainty, and the mean-prediction ensemble are
reported separately. Existing prediction caches are read only; output cannot overwrite.

    .venv/bin/python scripts/paired_seed_eval.py MANIFEST.json --out RESULT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fold_eval import ROOT, WARMUP, pooled, sums


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(m: dict) -> None:
    if not isinstance(m, dict):
        raise ValueError("Manifest must be a JSON object")
    if m.get("status") != "preregistered" or m.get("evidence_role") != "discovery":
        raise ValueError("This evaluator requires a preregistered discovery manifest")
    pairs = m.get("pairs")
    if not isinstance(pairs, list) or len(pairs) < 2 or not all(isinstance(p, dict) for p in pairs):
        raise ValueError("At least two matched training-seed pairs are required")
    seeds = [pair.get("seed") for pair in pairs]
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Training seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds are not independent training replications")
    folds = m.get("folds")
    if (not isinstance(folds, list) or not folds
            or any(type(fold) is not int or fold < 0 for fold in folds)
            or len(set(folds)) != len(folds)):
        raise ValueError("Fold IDs must be nonempty and unique")
    fitted_seeds = {}
    for pair in pairs:
        for side in ("candidate", "reference"):
            name = pair.get(side)
            if (not isinstance(name, str) or not name or name.strip() != name
                    or any(c in name for c in "/\\+*")):
                raise ValueError("Each side must name one cached fitted model per fold")
            if name in fitted_seeds and fitted_seeds[name] != pair["seed"]:
                raise ValueError("One cached model cannot represent different training seeds")
            fitted_seeds[name] = pair["seed"]
    if type(m.get("bootstrap_repeats")) is not int or m["bootstrap_repeats"] < 100:
        raise ValueError("At least 100 chain bootstrap draws are required")
    if type(m.get("bootstrap_seed")) is not int or m["bootstrap_seed"] < 0:
        raise ValueError("Bootstrap seed must be a nonnegative integer")


def summarize(sc: np.ndarray, sr: np.ndarray, fold: np.ndarray, chain: np.ndarray,
              repeats: int, seed: int) -> dict:
    """Stats shape: (training replicas plus ensemble, sequences, targets, six sums)."""
    if (sc.shape != sr.shape or sc.ndim != 4 or sc.shape[0] < 3
            or sc.shape[2:] != (2, 6) or not sc.shape[1]
            or fold.shape != (sc.shape[1],) or chain.shape != fold.shape):
        raise ValueError("Expected matched seed/ensemble statistics and one fold/chain per sequence")
    if not np.isfinite(sc).all() or not np.isfinite(sr).all():
        raise ValueError("Nonfinite sufficient statistics")
    if any(len(np.unique(fold[chain == ch])) != 1 for ch in np.unique(chain)):
        raise ValueError("A continuous chain cannot be split across evaluation folds")
    n_replica = len(sc) - 1
    deltas = np.array([pooled(a.sum(0)) - pooled(b.sum(0)) for a, b in zip(sc, sr, strict=True)])
    keys = sorted(set(zip(fold.tolist(), chain.tolist(), strict=True)))
    gc = np.stack([sc[:, (fold == f) & (chain == ch)].sum(1) for f, ch in keys], axis=1)
    gr = np.stack([sr[:, (fold == f) & (chain == ch)].sum(1) for f, ch in keys], axis=1)
    group_folds = np.array([key[0] for key in keys])
    rng = np.random.default_rng(seed)
    boot = np.empty((repeats, len(sc)))
    for i in range(repeats):
        pick = np.concatenate([rng.choice(idx, len(idx), replace=True)
                               for f in np.unique(fold)
                               for idx in [np.flatnonzero(group_folds == f)]])
        for j in range(len(sc)):
            boot[i, j] = pooled(gc[j, pick].sum(0)) - pooled(gr[j, pick].sum(0))
    comparisons = []
    for j in range(len(sc)):
        comparisons.append({
            "candidate_wp": pooled(sc[j].sum(0)), "reference_wp": pooled(sr[j].sum(0)),
            "delta": float(deltas[j]),
            "delta_by_fold": {str(f): pooled(sc[j, fold == f].sum(0)) - pooled(sr[j, fold == f].sum(0))
                              for f in np.unique(fold)},
            "conditional_chain_sd": float(boot[:, j].std()),
            "conditional_chain_ci95": np.quantile(boot[:, j], [0.025, 0.975]).tolist(),
            "bootstrap_nonpositive_fraction": float(np.mean(boot[:, j] <= 0)),
        })
    return {
        "sequence_count": len(fold), "chain_count": len(keys),
        "matched_pairs": comparisons[:n_replica], "mean_prediction_ensemble": comparisons[-1],
        "training_seed_summary": {
            "replications": n_replica, "matched_delta_mean": float(deltas[:n_replica].mean()),
            "matched_delta_sample_sd": float(deltas[:n_replica].std(ddof=1)),
            "matched_delta_min": float(deltas[:n_replica].min()),
            "matched_delta_max": float(deltas[:n_replica].max()),
            "positive_pairs": int(np.sum(deltas[:n_replica] > 0)),
            "conditional_chain_ci95_of_mean_delta": np.quantile(boot[:, :n_replica].mean(1), [0.025, 0.975]).tolist(),
            "caveat": "Few reused seeds describe observed optimizer variation; no population or selection-adjusted CI is claimed.",
        },
        "evidence_status": "exploratory_reused_folds",
        "not_estimated": ["adaptive selection bias", "hidden-test distribution shift"],
        "confirmation_established": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(f"Immutable result already exists: {a.out}")
    m = json.loads(a.manifest.read_text())
    validate_manifest(m)
    target_path = ROOT / "cache/valid_targets.npz"
    split_path = ROOT / "cache/chain_folds.npz"
    with np.load(target_path) as tg:
        rg, y, mask = tg["row_group"], tg["y"], tg["mask"]
    with np.load(split_path) as cf:
        valid = cf["split"] == "valid"
        info = {int(g): (int(f), int(ch), bool(p)) for g, f, ch, p in
                zip(cf["row_group"][valid], cf["fold"][valid], cf["chain"][valid], cf["pure"][valid], strict=True)}
    paths = {target_path, split_path}
    stats_c, stats_r, fold_ids, chain_ids, selected = [], [], [], [], []
    for f in m["folds"]:
        rows = np.array([i for i, g in enumerate(rg) if info[int(g)][0] == f and info[int(g)][2]])
        if not len(rows):
            raise ValueError(f"No pure evaluation sequences for fold {f}")
        arrays = []
        for pair in m["pairs"]:
            sides = []
            for side in ("candidate", "reference"):
                path = ROOT / "cache" / f"pred_{pair[side]}_f{f}_valid.npy"
                pred = np.load(path, mmap_mode="r")
                if pred.shape != y.shape:
                    raise ValueError(f"Prediction/target shape mismatch: {path}")
                paths.add(path)
                sides.append(pred)
            arrays.append(sides)
        for start in range(0, len(rows), 16):
            ix = rows[start:start + 16]
            ym, mm = y[ix], mask[ix].copy()
            mm[:, :WARMUP] = False
            cand = [np.asarray(sides[0][ix], dtype=np.float32) for sides in arrays]
            ref = [np.asarray(sides[1][ix], dtype=np.float32) for sides in arrays]
            if not np.isfinite(ym).all() or not all(np.isfinite(p).all() for p in cand + ref):
                raise ValueError("Nonfinite targets or predictions")
            cand.append(np.mean(cand, axis=0, dtype=np.float32))
            ref.append(np.mean(ref, axis=0, dtype=np.float32))
            stats_c.append(np.stack([sums(ym, p, mm) for p in cand]))
            stats_r.append(np.stack([sums(ym, p, mm) for p in ref]))
            fold_ids.extend([f] * len(ix))
            chain_ids.extend(info[int(rg[i])][1] for i in ix)
            selected.extend(int(rg[i]) for i in ix)
    result = summarize(np.concatenate(stats_c, axis=1), np.concatenate(stats_r, axis=1),
                       np.array(fold_ids), np.array(chain_ids), m["bootstrap_repeats"], m["bootstrap_seed"])
    for pair, report in zip(m["pairs"], result["matched_pairs"], strict=True):
        report.update(pair)
    result["provenance"] = {
        "manifest": str(a.manifest), "manifest_sha256": sha256(a.manifest),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": sha256(Path(__file__)), "scorer_sha256": sha256(ROOT / "scripts/fold_eval.py"),
        "python": platform.python_version(), "numpy": np.__version__,
        "finished_utc": datetime.now(UTC).isoformat(),
        "selected_row_groups_sha256": hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
        "inputs": {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(paths)},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps({"out": str(a.out), "training_seed_summary": result["training_seed_summary"],
                      "mean_prediction_ensemble": result["mean_prediction_ensemble"]}, indent=2))


if __name__ == "__main__":
    main()
