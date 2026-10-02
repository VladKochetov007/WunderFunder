"""Epoch curves of the distilled students from their saved per-epoch EMA snapshots (GPU, evaluation only).

For clean folds 0 and 1, each fold run's model_ep{0..3}.pt predicts the fold's evaluation sequences (one data pass per
fold, inputs built once per batch); exact |y|-weighted moments come from the frozen campaign scorer and are pooled per
(run, epoch). Paired component bootstrap for consecutive epochs and for the families at epoch 3 (parents, not the
head-FT mixes of the screens).
    .venv/bin/python research/campaign_2026-10-02/opt/epoch_curve.py --output OUT.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import train_rnn  # noqa: E402
from train_rnn import Sequences, build_inputs, forward_chunk, make_model  # noqa: E402

from wnn.resources import configure_torch  # noqa: E402
from wnn.splits import load_campaign_split  # noqa: E402

FROZEN = ROOT / "research/campaign_2026-09-30/audit/campaign_score.py"
MANIFEST = "research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
RUNS = {"GRU256_a05": "c20261001_d256_s3_f{fold}", "GRU256_a07": "c20261001_d256a07_s3_f{fold}",
        "LSTM192_a05": "c20261002_dlstm192a05_s3_f{fold}"}
EPOCHS = (0, 1, 2, 3)


def frozen_scorer():
    spec = importlib.util.spec_from_file_location("campaign_score", FROZEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(run: str, epoch: int, split, device):
    ck = torch.load(ROOT / "artifacts/runs" / run / f"model_ep{epoch}.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    if a.get("split_manifest_sha256") != split.sha256 or not a.get("no_eval_during_fit"):
        raise ValueError(f"{run}: not fitted on clean fold {split.fold}")
    if a.get("raw_only") or a.get("online_norm") or a.get("heuristics") or a.get("prop_input") or ck.get("readout"):
        raise ValueError(f"{run}: input or readout variant not supported here")
    model = make_model(a, ck["n_in"], ck["model"]["head.2.weight"].shape[0]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="evaluation sequences per fold (CPU smoke)")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    configure_torch()
    device = torch.device(args.device)
    cs = frozen_scorer()
    targets = cs.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
    position = {int(g): i for i, g in enumerate(targets["row_group"])}
    moments, classes, components = {}, [], []
    t0 = time.time()
    for fold in (int(f) for f in args.folds.split(",")):
        split = load_campaign_split(ROOT / MANIFEST.format(fold=fold), ROOT)
        models = {(fam, e): load(run.format(fold=fold), e, split, device) for fam, run in RUNS.items() for e in EPOCHS}
        component_of = {g: c for source, g, c in split.components if source == "valid"}
        groups = list(split.evaluation_groups)[: args.limit or None]
        loader = DataLoader(Sequences("valid", groups), batch_size=16, num_workers=args.workers,
                            prefetch_factor=1 if args.workers else None)
        done = 0
        for x, _, _ in loader:
            inputs = build_inputs(x.to(device, non_blocking=True))
            preds = {}
            for key, model in models.items():
                state, outs = None, []
                for s in range(0, inputs.shape[1], 4000):
                    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                        out, state = forward_chunk(model, inputs, slice(s, s + 4000), state)
                    outs.append(out[..., :2].float())
                preds[key] = torch.cat(outs, dim=1).cpu().numpy()
            for j in range(len(x)):
                group = groups[done + j]
                i = position[group]
                y, mask = np.asarray(targets["y"][i]), np.asarray(targets["mask"][i], bool)
                selection = mask & (np.arange(len(y)) >= 99)
                for key, p in preds.items():
                    moments.setdefault(key, []).append(
                        np.stack([cs.moments(y[:, k], p[j][:, k], selection) for k in range(2)]))
                classes.append(int(targets["market_class"][i]))
                components.append(component_of[group])
            done += len(x)
            if done % 160 == 0:
                print(f"fold {fold}: {done}/{len(groups)} sequences, {time.time() - t0:.0f} s", flush=True)
        print(f"fold {fold}: {done} sequences, {len(models)} models, {time.time() - t0:.0f} s", flush=True)
        del models
        torch.cuda.empty_cache()
    m = {f"{fam}@ep{e}": np.stack(v) for (fam, e), v in moments.items()}
    classes, components = np.array(classes), np.array(components)
    report = {"folds": args.folds, "runs": RUNS, "n_sequences": len(classes), "exact_global_wp": {}, "by_class": {},
              "paired": {}, "wall_seconds": None}
    for key, v in m.items():
        report["exact_global_wp"][key] = cs.pooled(v)["wp"]
        report["by_class"][key] = {str(c): cs.pooled(v[classes == c])["wp"] for c in np.unique(classes)}
    pairs = [(f"{fam}@ep{e + 1}", f"{fam}@ep{e}") for fam in RUNS for e in EPOCHS[:-1]]
    pairs += [("GRU256_a07@ep3", "GRU256_a05@ep3"), ("LSTM192_a05@ep3", "GRU256_a05@ep3")]
    for cand, ctrl in pairs:
        b = cs.paired_component_bootstrap(m[cand], m[ctrl], components, n_boot=args.boot, seed=271828, centered=True)
        report["paired"][f"{cand}_minus_{ctrl}"] = {k: b[k] for k in ("mean_diff", "bootstrap_sd", "ci95")}
        print(f"{cand:18s} - {ctrl:18s} {b['mean_diff']:+.5f} sd {b['bootstrap_sd']:.5f}", flush=True)
    report["wall_seconds"] = round(time.time() - t0, 1)
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
