"""Score frozen paired-state heads on one disjoint outer fold.

Heads and transforms are fixed in their fit checkpoint before this command
opens evaluation states. Reports exact Global WP, paired component intervals,
class diagnostics and two softened-mask stress scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from campaign_pair_head_fit import KINDS, ResidualHead, sha256  # noqa: E402

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402
from wnn.gmetric import global_wp  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402
from wnn.splits import load_campaign_split  # noqa: E402


def soft_global_wp(targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray, lam: float) -> float:
    """Reweight required unscored rows by lambda for a transfer stress test."""
    if not np.isfinite(lam) or lam < 0 or lam > 1:
        raise ValueError("lambda must be in [0,1]")
    n, steps, _ = targets.shape
    need = np.broadcast_to(np.arange(steps)[None, :] >= 99, (n, steps))
    inclusion = np.where(mask, 1.0, lam) * need
    results: list[float] = []
    for k in range(2):
        raw_y = np.asarray(targets[..., k], dtype=np.float64)
        raw_p = np.asarray(predictions[..., k], dtype=np.float64)
        if not (np.isfinite(raw_y[need]).all() and np.isfinite(raw_p[need]).all()):
            raise ValueError("nonfinite required row")
        y = np.clip(raw_y, -2, 2)
        p = np.clip(raw_p, -2, 2)
        weight = np.abs(y) * inclusion
        positive = weight > 0
        if not positive.any():
            results.append(0.0)
            continue
        yy, pp, ww = y[positive], p[positive], weight[positive]
        dy, dp = yy - yy[0], pp - pp[0]
        total = ww.sum()
        dy -= np.sum(ww * dy) / total
        dp -= np.sum(ww * dp) / total
        vy, vp = np.sum(ww * dy * dy), np.sum(ww * dp * dp)
        results.append(
            float(np.clip(np.sum(ww * dy * dp) / np.sqrt(vy * vp), -1, 1))
            if vy / total > 1e-16 and vp / total > 1e-16
            else 0.0
        )
    return float(np.mean(results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-prefix", type=Path, required=True)
    parser.add_argument("--head-run", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch", type=int, default=65_536)
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()
    prefix = args.eval_prefix.resolve()
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    if metadata["mode"] != "eval" or metadata["selection"] != "all rows":
        raise ValueError("expected all-row evaluation export")
    for key, expected in metadata["arrays_sha256"].items():
        if sha256(prefix.with_name(f"{prefix.name}_{key}.npy")) != expected:
            raise ValueError(f"{key} evaluation artifact changed after export")
    head_path = ROOT / "research" / "campaign_2026-09-29" / "artifacts" / "pair_heads" / args.head_run
    head_provenance = json.loads((head_path / "provenance.json").read_text())
    if metadata["manifest_sha256"] != head_provenance["manifest_sha256"]:
        raise ValueError("fit and evaluation outer-fold manifests disagree")
    if metadata["members"] != head_provenance["members"]:
        raise ValueError("fit and evaluation member checkpoints disagree")
    if metadata["member_checkpoint_sha256"] != head_provenance["member_checkpoint_sha256"]:
        raise ValueError("backbone bytes differ between fitting and evaluation exports")
    split = load_campaign_split(
        ROOT / "research" / "campaign_2026-09-29" / "artifacts" / "splits" / f"fold{metadata['fold']}.json", ROOT
    )
    if split.sha256 != metadata["manifest_sha256"] or list(split.evaluation_groups) != metadata["groups"]:
        raise ValueError("evaluation groups differ from frozen split")
    states = np.load(prefix.with_name(f"{prefix.name}_states.npy"), mmap_mode="r")
    parent = np.load(prefix.with_name(f"{prefix.name}_predictions.npy"), mmap_mode="r")
    targets = np.load(prefix.with_name(f"{prefix.name}_targets.npy"), mmap_mode="r")
    mask = np.load(prefix.with_name(f"{prefix.name}_mask.npy"), mmap_mode="r")
    row_group = np.load(prefix.with_name(f"{prefix.name}_row_group.npy"), mmap_mode="r")
    groups = np.asarray(metadata["groups"], dtype=np.int32)
    n, steps = len(groups), 20_000
    if len(states) != n * steps or targets.shape != (n * steps, 2):
        raise ValueError("evaluation rows are incomplete")
    if not np.array_equal(row_group.reshape(n, steps), np.broadcast_to(groups[:, None], (n, steps))):
        raise ValueError("row-group ordering differs from the frozen evaluation panel")
    y = targets.reshape(n, steps, 2)
    m = mask.reshape(n, steps)
    baseline = np.asarray(parent.mean(axis=1), dtype=np.float32).reshape(n, steps, 2)
    component_by_group = {g: c for sp, g, c in split.components if sp == "valid"}
    components = np.array([component_by_group[int(g)] for g in groups])
    with np.load(ROOT / "cache" / "valid_targets.npz") as cache:
        classes = dict(zip(cache["row_group"].tolist(), cache["market_class"].tolist(), strict=True))
    class_id = np.array([classes[int(g)] for g in groups])
    configure_torch()
    device = torch.device("cuda")
    out_dir = ROOT / "research" / "campaign_2026-09-29" / "artifacts" / "pair_head_scores" / args.name
    out_dir.mkdir(parents=True, exist_ok=False)
    predictions: dict[str, np.ndarray] = {"average": baseline}
    checkpoint_sha: dict[str, str] = {}
    for kind in KINDS:
        path = head_path / f"{kind}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["kind"] != kind or checkpoint["fit_manifest_sha256"] != split.sha256:
            raise ValueError(f"{kind}: invalid frozen head checkpoint")
        if checkpoint["member_checkpoint_sha256"] != metadata["member_checkpoint_sha256"]:
            raise ValueError(f"{kind}: fit checkpoint used different backbone bytes")
        if checkpoint["fit_export_sha256"] != head_provenance["fit_export_sha256"]:
            raise ValueError(f"{kind}: fit export provenance differs")
        checkpoint_sha[kind] = sha256(path)
        if checkpoint_sha[kind] != head_provenance["checkpoint_sha256"][kind]:
            raise ValueError(f"{kind}: head weights changed after fit")
        state = checkpoint["model"]
        head = ResidualHead(kind, metadata["hidden"], state["mean"], state["std"]).to(device)
        head.load_state_dict(state)
        head.eval()
        output = np.lib.format.open_memmap(out_dir / f"{kind}.npy", mode="w+", dtype=np.float32, shape=(n * steps, 2))
        for start in range(0, len(states), args.batch):
            stop = min(start + args.batch, len(states))
            hidden = torch.as_tensor(np.asarray(states[start:stop], dtype=np.float32), device=device)
            parent_batch = torch.as_tensor(np.asarray(parent[start:stop]), device=device)
            with torch.no_grad():
                correction = head(hidden, parent_batch)
            result = np.asarray(parent_batch.mean(dim=1).cpu() + correction.cpu())
            if not np.isfinite(result).all():
                raise ValueError(f"{kind}: nonfinite prediction on an evaluation row")
            output[start:stop] = result
        output.flush()
        predictions[kind] = output.reshape(n, steps, 2)

    scores: dict[str, dict] = {}
    for name, values in predictions.items():
        exact = global_wp(y, values, m)
        scores[name] = {
            "exact_global_wp": exact,
            "soft_lambda002": soft_global_wp(y, values, m, 0.02),
            "soft_lambda005": soft_global_wp(y, values, m, 0.05),
            "by_class": {
                str(k): global_wp(y[class_id == k], values[class_id == k], m[class_id == k])["wp"]
                for k in np.unique(class_id)
            },
        }
    comparisons: dict[str, dict] = {}
    for name in KINDS:
        moments_a, moments_b = paired_component_moments(y, predictions[name], baseline, m)
        comparisons[f"{name}_vs_average"] = paired_component_bootstrap(
            moments_a, moments_b, components, n_boot=args.boot, seed=0, centered=True
        )
    moments_a, moments_b = paired_component_moments(y, predictions["bilinear"], predictions["unary"], m)
    comparisons["bilinear_vs_unary"] = paired_component_bootstrap(
        moments_a, moments_b, components, n_boot=args.boot, seed=0, centered=True
    )
    moments_a, moments_b = paired_component_moments(
        y, predictions["bilinear"], predictions["output_only"], m
    )
    comparisons["bilinear_vs_output_only"] = paired_component_bootstrap(
        moments_a, moments_b, components, n_boot=args.boot, seed=0, centered=True
    )
    report = {
        "name": args.name,
        "fold": metadata["fold"],
        "manifest_sha256": split.sha256,
        "fit_head_provenance_sha256": sha256(head_path / "provenance.json"),
        "frozen_head_sha256": checkpoint_sha,
        "eval_export_sha256": sha256(prefix.with_suffix(".json")),
        "n_sequences": n,
        "n_components": len(np.unique(components)),
        "selection": "need_prediction AND is_scored",
        "metric_id": "global_wp_clip2_v1",
        "scores": scores,
        "paired": comparisons,
        "promotion_screen": {
            "bilinear_delta_vs_average_ge_0005": comparisons["bilinear_vs_average"]["mean_diff"] >= 0.0005,
            "bilinear_beats_unary_exact": comparisons["bilinear_vs_unary"]["mean_diff"] > 0,
            "bilinear_beats_output_only_exact": comparisons["bilinear_vs_output_only"]["mean_diff"] > 0,
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["promotion_screen"]), flush=True)


if __name__ == "__main__":
    main()
