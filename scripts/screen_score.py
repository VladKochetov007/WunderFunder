"""Quick paired screens on clean folds with the frozen campaign scorer's arithmetic.

A screen file names prediction mixtures ("arms") per fold and the comparisons to report:

    {"folds": [0, 1],
     "arms": {"reference": {"c20260929_ft_since_s3_f{fold}": 0.5, "c20260929_ft_since_s4_f{fold}": 0.5},
              "candidate": {...}},
     "comparisons": [["candidate", "reference"]],
     "class_mix": {"isolated": [0.08, 0.11, 0.14, 0.67]}}      # optional public-likeness diagnostics

Member keys are cached prediction names (``cache/pred_<name>_valid.npy``). Every member's sidecar must carry the
fold's split-manifest hash (so no member saw the evaluation chains) and the prediction bytes must match the
sidecar hash. Moments, pooling, component bootstrap and class breakdowns come from the frozen
``campaign_score.py`` module, imported read-only.

    .venv/bin/python scripts/screen_score.py SCREEN.json --output REPORT.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.splits import file_sha256, load_campaign_split  # noqa: E402

FROZEN = ROOT / "research/campaign_2026-09-30/audit/campaign_score.py"
MANIFEST = "research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
ROW_ORDER: list[int] = []


def frozen_scorer():
    spec = importlib.util.spec_from_file_location("campaign_score", FROZEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bound_member(name: str, split, verified: dict) -> np.ndarray:
    """Memory-map one cached prediction after checking it belongs to this clean fold."""
    pred = ROOT / "cache" / f"pred_{name}_valid.npy"
    side = json.loads(pred.with_suffix(".json").read_text())
    if side.get("split_manifest_sha256") != split.sha256:
        raise ValueError(f"{name}: fitted outside clean fold {split.fold}")
    if "row_groups" in side and list(map(int, side["row_groups"])) != ROW_ORDER:
        raise ValueError(f"{name}: prediction rows are not in cache target order")
    if name not in verified:
        digest = file_sha256(pred)
        if digest != side.get("prediction_sha256"):
            raise ValueError(f"{name}: prediction bytes differ from sidecar")
        verified[name] = {"prediction_sha256": digest, "checkpoint_sha256": side.get("checkpoint_sha256"),
                          "epoch": side.get("epoch")}
    return np.load(pred, mmap_mode="r")


class HoldoutSplit:
    """Pseudo-fold for final fits: the reserved 368-sequence validation holdout (``wnn.splits.is_holdout``).

    Members must come from runs whose fit never read holdout rows (checked in ``holdout_member``). Chain
    components come from the fold-0 clean manifest, which assigns a component to every validation group.
    """

    fold = "holdout"

    def __init__(self):
        from wnn.splits import valid_row_groups
        _, held = valid_row_groups(ROOT / "datasets/valid.parquet")
        reference = load_campaign_split(ROOT / MANIFEST.format(fold=0), ROOT)
        self.evaluation_groups = tuple(int(g) for g in held)
        self.components = reference.components
        self.sha256 = None


def holdout_member(name: str, verified: dict) -> np.ndarray:
    """Memory-map a final-fit prediction after checking its run (and its init chain) never fitted on holdout."""
    import torch

    pred = ROOT / "cache" / f"pred_{name}_valid.npy"
    if name not in verified:
        side = json.loads(pred.with_suffix(".json").read_text())
        prediction_sha = file_sha256(pred)
        if side.get("run") != name:
            raise ValueError(f"{name}: prediction sidecar names run {side.get('run')!r}")
        if prediction_sha != side.get("prediction_sha256"):
            raise ValueError(f"{name}: prediction bytes differ from sidecar")
        run = name
        if not (ROOT / "artifacts" / "runs" / run).is_dir():
            run = re.sub(r"_ep\d+$", "", run)  # cached exports of a fixed epoch are named RUN_epN
        prediction_run = run
        while run:
            run_dir = ROOT / "artifacts" / "runs" / run
            snaps = sorted(run_dir.glob("model*.pt"))
            if not snaps:
                raise ValueError(f"{name}: no checkpoint for run {run}")
            if run == prediction_run:
                checkpoint_sha = side.get("checkpoint_sha256")
                matching = [p for p in snaps if file_sha256(p) == checkpoint_sha]
                if len(matching) != 1:
                    raise ValueError(f"{name}: sidecar checkpoint hash matches {len(matching)} checkpoints")
                snaps = matching
            args = torch.load(snaps[0], map_location="cpu", weights_only=False)["args"]
            if args.get("include_heldout_valid") or args.get("valid_fold", -1) not in (-1, None):
                raise ValueError(f"{name}: run {run} fitted on holdout or a fold split")
            if args.get("train_source") not in ("train", "train+valid", "valid"):
                raise ValueError(f"{name}: run {run} unknown train source")
            run = (args.get("init") or "").partition("@")[0]
        verified[name] = {"prediction_sha256": prediction_sha, "checkpoint_sha256": side.get("checkpoint_sha256"),
                          "epoch": side.get("epoch"), "holdout_safe": True}
    return np.load(pred, mmap_mode="r")


def run(screen: dict) -> dict:
    cs = frozen_scorer()
    if len(set(screen["folds"])) != len(screen["folds"]):
        raise ValueError("duplicate folds would double-count evaluation rows")
    targets = cs.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
    ROW_ORDER[:] = [int(g) for g in targets["row_group"]]
    position = {g: i for i, g in enumerate(ROW_ORDER)}
    design = {"comparisons": screen["comparisons"], "n_boot": screen.get("n_boot", 2000),
              "bootstrap_seed": 271828}
    collected, soft_all, blocks_all, classes, components = {}, {}, {}, [], []
    folds, verified = {}, {}
    for fold in screen["folds"]:
        split = HoldoutSplit() if fold == "holdout" else load_campaign_split(ROOT / MANIFEST.format(fold=fold), ROOT)
        arms = {}
        for arm, mixture in screen["arms"].items():
            total = sum(mixture.values())
            if abs(total - 1.0) > 1e-9 or min(mixture.values()) < 0:
                raise ValueError(f"{arm}: weights must be a convex mixture")
            arms[arm] = [(w, holdout_member(k, verified) if fold == "holdout"
                          else bound_member(k.format(fold=fold), split, verified)) for k, w in mixture.items()]
        component_of = {g: c for source, g, c in split.components if source == "valid"}
        rows = {arm: [] for arm in arms}
        soft = {}
        blocks = {arm: [] for arm in arms}
        fold_classes, fold_components = [], []
        for group in split.evaluation_groups:
            i = position[group]
            prediction = {}
            for arm, members in arms.items():
                mix = np.zeros((20_000, 2), dtype=np.float64)
                for w, values in members:
                    mix += w * np.asarray(values[i], dtype=np.float64)
                prediction[arm] = mix.astype(np.float32)
            row = cs.score_sequence(targets["y"][i], targets["mask"][i], prediction)
            for arm in arms:
                rows[arm].append(row["moments"][arm])
                blocks[arm].append(row["blocks"][arm])
                for lam, values in row["soft"].items():
                    soft.setdefault(lam, {}).setdefault(arm, []).append(values[arm])
            fold_classes.append(int(targets["market_class"][i]))
            fold_components.append(component_of[group])
        m = {arm: np.stack(v) for arm, v in rows.items()}
        s = {lam: {arm: np.stack(v) for arm, v in d.items()} for lam, d in soft.items()}
        for arm, route in screen.get("routed", {}).items():
            # diagnostic: per-sequence arm choice by retrospective market class (upper bound of a causal router)
            pick = np.array([route[str(c)] for c in fold_classes])
            m[arm] = np.stack([m[pick[i]][i] for i in range(len(pick))])
            for lam in s:
                s[lam][arm] = np.stack([s[lam][pick[i]][i] for i in range(len(pick))])
            blocks[arm] = [blocks[pick[i]][i] for i in range(len(pick))]
        folds[str(fold)] = cs.summary(m, s, blocks, np.array(fold_classes), np.array(fold_components), design)
        for arm, v in m.items():
            collected.setdefault(arm, []).append(v)
            blocks_all.setdefault(arm, []).extend(blocks[arm])
        for lam, d in s.items():
            for arm, v in d.items():
                soft_all.setdefault(lam, {}).setdefault(arm, []).append(v)
        classes += fold_classes
        components += fold_components
    m = {arm: np.concatenate(v) for arm, v in collected.items()}
    s = {lam: {arm: np.concatenate(v) for arm, v in d.items()} for lam, d in soft_all.items()}
    report = {"metric_id": cs.METRIC_ID, "screen": screen, "members": verified, "n_sequences": len(classes),
              "folds": folds,
              "pooled": cs.summary(m, s, blocks_all, np.array(classes), np.array(components), design)}
    for label, mix in screen.get("class_mix", {}).items():
        report.setdefault("reweighted", {})[label] = reweighted(cs, m, np.array(classes), np.array(components),
                                                                 mix, design)
    return report


def reweighted(cs, m: dict, classes: np.ndarray, components: np.ndarray, mix: list[float], design: dict) -> dict:
    """Diagnostic: rescale each sequence's centred moments so class shares of sequences match ``mix``.

    Weight, M2 and cross moments scale linearly with a sequence weight; means do not. This mimics a test set whose
    market-class composition differs from the local folds (not official; a public-likeness proxy only).
    """
    counts = np.bincount(classes, minlength=4) / len(classes)
    factor = np.asarray(mix, dtype=np.float64) / np.maximum(counts, 1e-12)
    scale = factor[classes][:, None, None] * np.array([1, 0, 0, 1, 1, 1])[None, None] + \
        np.array([0, 1, 1, 0, 0, 0])[None, None]
    mw = {arm: v * scale for arm, v in m.items()}
    exact = {arm: cs.pooled(v) for arm, v in mw.items()}
    comparisons = {}
    for cand, ctrl in design["comparisons"]:
        value = cs.paired_component_bootstrap(mw[cand], mw[ctrl], components, n_boot=design["n_boot"],
                                              seed=design["bootstrap_seed"], centered=True)
        comparisons[f"{cand}_minus_{ctrl}"] = value
    return {"mix": list(mix), "exact_global_wp": exact, "paired_component_bootstrap": comparisons}


def brief(report: dict) -> str:
    lines = []
    for name, part in [("pooled", report["pooled"])] + [(f"fold{k}", v) for k, v in report["folds"].items()]:
        for label, cmp in part["paired_component_bootstrap"].items():
            lines.append(f"{name:7s} {label:40s} {cmp['mean_diff']:+.6f} sd {cmp['bootstrap_sd']:.6f} "
                         f"ci [{cmp['ci95'][0]:+.6f}, {cmp['ci95'][1]:+.6f}]")
    for label, part in report.get("reweighted", {}).items():
        for name, cmp in part["paired_component_bootstrap"].items():
            lines.append(f"mix:{label:9s} {name:36s} {cmp['mean_diff']:+.6f} sd {cmp['bootstrap_sd']:.6f}")
    for arm, v in report["pooled"]["exact_global_wp"].items():
        by = " ".join(f"c{c}={report['pooled']['by_class'][c][arm]['wp']:.5f}" for c in sorted(report["pooled"]["by_class"]))
        lines.append(f"{arm:24s} wp {v['wp']:.6f}  {by}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("screen", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.monotonic()
    report = run(json.loads(args.screen.read_text()))
    report["screen_sha256"] = file_sha256(args.screen)
    report["wall_seconds"] = time.monotonic() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(brief(report))


if __name__ == "__main__":
    main()
