"""Grade a bank of validation predictions for one fold: singles, correlations, blends.

The bank is a JSONL registry; each row names a model, its fold, a prediction
``.npy`` with a sidecar ``.json`` listing ``row_groups``, and free-form recipe
tags.  All rows of the requested fold are aligned to that fold's fixed
evaluation groups and scored with the official ``global_wp_clip2_v1``
(scored rows >= 99).  The report adds prediction and residual correlation
matrices, every equal-weight pair blend, the all-member equal blend and a
greedy equal-weight forward selection.  Every evaluation fold used here has
been inspected before, so blend selection on it is exploratory, never a
held-out claim.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402
from wnn.gmetric import GlobalAccumulator  # noqa: E402

SCORED_FROM_ROW = 99
GROUP_CHUNK = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bank(path: Path, fold: int, skip_missing: bool = False) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if int(row["fold"]) == fold and not row.get("excluded")]
    if skip_missing:
        rows = [row for row in rows if (ROOT / row["prediction"]).is_file()]
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate model names in bank")
    return rows


def aligned(row: dict, groups: list[int], split_sha: str) -> tuple[np.ndarray, list[int]] | str:
    """Return the memmap and row order, or a reason string when identity is unverifiable."""
    path = ROOT / row["prediction"]
    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.is_file():
        return "no sidecar; row order and split ancestry unverifiable"
    sidecar = json.loads(sidecar_path.read_text())
    if "row_groups" not in sidecar:
        return "sidecar lacks row_groups"
    declared = sidecar.get("split_manifest_sha256")
    if declared is not None and declared != split_sha:
        return f"sidecar split {declared[:12]} differs from evaluation split"
    position = {int(g): i for i, g in enumerate(sidecar["row_groups"])}
    missing = [g for g in groups if g not in position]
    if missing:
        return f"missing {len(missing)} evaluation groups"
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3 or array.shape[2] != 2 or array.shape[0] != len(sidecar["row_groups"]):
        return f"unexpected prediction shape {array.shape}"
    return array, [position[g] for g in groups]


def score(targets: np.ndarray, mask: np.ndarray, prediction: np.ndarray) -> dict:
    accumulator = GlobalAccumulator()
    for start in range(0, len(targets), GROUP_CHUNK):
        part = slice(start, start + GROUP_CHUNK)
        accumulator.add(targets[part], prediction[part], mask[part])
    result = accumulator.result()
    return {"wp": result["wp"], "t0": result["t0"], "t1": result["t1"]}


def correlations(targets, mask, predictions: list[np.ndarray]) -> dict:
    """Unweighted Pearson matrices of predictions and residuals on scored rows."""
    m = len(predictions)
    out = {}
    for kind in ("prediction", "residual"):
        per_target = []
        for k in range(2):
            n = 0
            s = np.zeros(m)
            ss = np.zeros((m, m))
            for start in range(0, len(targets), GROUP_CHUNK):
                part = slice(start, start + GROUP_CHUNK)
                sel = mask[part]
                cols = []
                for p in predictions:
                    v = np.asarray(p[part][..., k][sel], dtype=np.float64)
                    if kind == "residual":
                        v = v - np.clip(targets[part][..., k][sel], -2.0, 2.0)
                    cols.append(v)
                x = np.stack(cols, axis=1)
                n += x.shape[0]
                s += x.sum(axis=0)
                ss += x.T @ x
            cov = ss / n - np.outer(s / n, s / n)
            sd = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
            per_target.append(cov / np.outer(sd, sd))
        out[kind] = 0.5 * (per_target[0] + per_target[1])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=ROOT / "cache/valid_targets.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--greedy-steps", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--skip-missing", action="store_true", help="ignore rows not produced yet")
    args = parser.parse_args()

    rows = load_bank(args.bank, args.fold, args.skip_missing)
    if len(rows) < 1:
        raise SystemExit("no bank rows for fold")
    split = json.loads(args.split.read_text())
    if int(split["fold"]) != args.fold:
        raise SystemExit("split fold differs")
    groups = [int(g) for g in split["evaluation"]["valid"]]
    component_of = {
        int(r["row_group"]): int(r["component"]) for r in split["sequences"]["valid"]
    }
    components = np.asarray([component_of[g] for g in groups])

    with np.load(args.targets, allow_pickle=False) as cache:
        position = {int(g): i for i, g in enumerate(cache["row_group"])}
        index = [position[g] for g in groups]
        targets = np.asarray(cache["y"][index], dtype=np.float32)
        mask = np.asarray(cache["mask"][index], dtype=bool)
    mask &= (np.arange(targets.shape[1]) >= SCORED_FROM_ROW)[None, :]

    predictions, identities, skipped, kept = [], {}, {}, []
    split_sha = sha256(args.split)
    for row in rows:
        result = aligned(row, groups, split_sha)
        if isinstance(result, str):
            skipped[row["name"]] = result
            continue
        array, order = result
        predictions.append(np.asarray(array[order], dtype=np.float32))
        identities[row["name"]] = sha256(ROOT / row["prediction"])
        kept.append(row)
    rows = kept
    names = [row["name"] for row in rows]
    if not names:
        raise SystemExit(f"no verifiable predictions; skipped={skipped}")

    singles = {name: score(targets, mask, p) for name, p in zip(names, predictions, strict=True)}
    corr = correlations(targets, mask, predictions)
    pairs = {}
    for i, j in itertools.combinations(range(len(names)), 2):
        blend = 0.5 * (predictions[i] + predictions[j])
        result = score(targets, mask, blend)
        result["gain_over_best_member"] = result["wp"] - max(
            singles[names[i]]["wp"], singles[names[j]]["wp"]
        )
        pairs[f"{names[i]}+{names[j]}"] = result
    all_equal = None
    if len(names) > 1:
        mean = np.zeros_like(predictions[0])
        for p in predictions:
            mean += p / len(predictions)
        all_equal = score(targets, mask, mean)
        del mean

    # Greedy equal-weight selection with replacement (Caruana); exploratory only.
    chosen, total, greedy = [], np.zeros_like(predictions[0]), []
    for _ in range(min(args.greedy_steps, 4 * len(names))):
        best = None
        for i, p in enumerate(predictions):
            trial = score(targets, mask, (total + p) / (len(chosen) + 1))
            if best is None or trial["wp"] > best[1]["wp"]:
                best = (i, trial)
        chosen.append(best[0])
        total += predictions[best[0]]
        greedy.append({"members": [names[i] for i in chosen], **best[1]})

    best_single = max(singles, key=lambda n: singles[n]["wp"])
    reference = predictions[names.index(best_single)]
    greedy_blend = total / len(chosen)
    ref_m, _ = paired_component_moments(targets, reference, reference, mask)
    blend_m, _ = paired_component_moments(targets, greedy_blend, greedy_blend, mask)
    greedy_vs_best = paired_component_bootstrap(
        blend_m, ref_m, components, n_boot=args.bootstrap, seed=271828, centered=True
    )

    report = {
        "schema": "wnn.prediction-bank-grade.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "grader_sha256": sha256(Path(__file__)),
        "fold": args.fold,
        "evaluation_groups": len(groups),
        "metric_id": "global_wp_clip2_v1",
        "exposure": "historically exposed fold; blend selection here is exploratory",
        "models": {row["name"]: {"tags": row.get("tags", {}), "prediction_sha256": identities[row["name"]]} for row in rows},
        "skipped": skipped,
        "singles": singles,
        "best_single": best_single,
        "correlation_names": names,
        "prediction_correlation": corr["prediction"].round(6).tolist(),
        "residual_correlation": corr["residual"].round(6).tolist(),
        "pairs": pairs,
        "all_equal_blend": all_equal,
        "greedy_equal_weight": greedy,
        "greedy_final_minus_best_single_bootstrap": greedy_vs_best,
        "expected_public_score": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    top = sorted(singles.items(), key=lambda kv: -kv[1]["wp"])
    print(json.dumps({"best_single": top[0], "all_equal": all_equal, "greedy_final": greedy[-1]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
