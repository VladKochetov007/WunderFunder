"""Critic data checks (CPU only) for fold-bound distillation and the d256 recipe.

    WNN_CPU_PCT=7 WNN_MEM_MAX=4G CUDA_VISIBLE_DEVICES='' scripts/limited .venv/bin/python \
        research/campaign_2026-10-01/critic/check_teachers.py [--hash]
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from wnn.splits import file_sha256, load_campaign_split  # noqa: E402

MANIFEST = str(ROOT / "research/campaign_2026-09-29/artifacts/splits") + "/fold{fold}.json"
cli = argparse.ArgumentParser()
cli.add_argument("--hash", action="store_true", help="also re-hash prediction files (1.7 GB each for train)")
opts = cli.parse_args()
out = {}


def wp(y, p, w):
    y = np.clip(y, -2, 2); p = np.clip(p, -2, 2); w = w * np.abs(y)
    my, mp = (w * y).sum() / w.sum(), (w * p).sum() / w.sum()
    return float((w * (y - my) * (p - mp)).sum() / np.sqrt((w * (y - my) ** 2).sum() * (w * (p - mp) ** 2).sum()))


targets = {sp: np.load(ROOT / "cache" / f"{sp}_targets.npz", mmap_mode="r") for sp in ("train", "valid")}
rows = {sp: np.asarray(t["row_group"]) for sp, t in targets.items()}

# --- 1. teacher sidecars that exist: fold binding, row order, checkpoint identity, fit provenance
for fold in (0, 1, 2, 3):
    split = load_campaign_split(Path(MANIFEST.format(fold=fold)), ROOT)
    out[f"fold{fold}_fit_groups_in_target_cache"] = {
        "train": bool(set(split.train_groups) <= set(rows["train"].tolist())),
        "valid": bool(set(split.valid_groups) <= set(rows["valid"].tolist()))}
    for seed in (3, 4):
        run = f"c20260929_since_s{seed}_f{fold}"
        prov = json.loads((ROOT / "artifacts/runs" / run / "provenance.json").read_text())
        fit = prov["fit_row_groups"]
        entry = {"fit_disjoint_from_eval": not (set(fit["valid"]) & set(split.evaluation_groups)),
                 "fit_inside_manifest": set(fit["train"]) <= set(split.train_groups)
                 and set(fit["valid"]) <= set(split.valid_groups)}
        for sp in ("train", "valid"):
            side_path = ROOT / "cache" / f"pred_{run}_{sp}.json"
            if not side_path.exists():
                entry[sp] = "missing (bind_fold_teachers fails closed)"
                continue
            side = json.loads(side_path.read_text())
            ck = ROOT / "artifacts/runs" / run / f"model_ep{side.get('epoch')}.pt"
            e = {"split_hash": side.get("split_manifest_sha256") == split.sha256,
                 "row_groups_match_target_cache": side.get("row_groups") == rows[sp].tolist(),
                 "epoch": side.get("epoch"),
                 "checkpoint_sha_matches_fixed_epoch": side.get("epoch") == prov["args"]["checkpoint_epoch"]
                 and prov["checkpoint_sha256"].get(ck.name) == side.get("checkpoint_sha256")}
            if opts.hash:
                e["prediction_sha_matches"] = file_sha256(side_path.with_suffix(".npy")) == side["prediction_sha256"]
            entry[sp] = e
        out[run] = entry

# --- 2. empirical row alignment of the teacher cache as Sequences indexes it (fold 0, valid fit rows)
split = load_campaign_split(Path(MANIFEST.format(fold=0)), ROOT)
pos = {int(g): i for i, g in enumerate(rows["valid"])}
pred = np.load(ROOT / "cache/pred_c20260929_since_s3_f0_valid.npy", mmap_mode="r")
rng = np.random.default_rng(0)
fit_sample = rng.choice(np.asarray(split.valid_groups), 12, replace=False)
aligned, shifted = [], []
yv = targets["valid"]["y"]
for g in fit_sample:
    j = pos[int(g)]
    y = np.asarray(yv[j], np.float64)[99:]
    w = np.ones(len(y))
    aligned.append(np.mean([wp(y[:, k], np.asarray(pred[j], np.float64)[99:, k], w) for k in range(2)]))
    jj = (j + 1) % len(rows["valid"])
    shifted.append(np.mean([wp(y[:, k], np.asarray(pred[jj], np.float64)[99:, k], w) for k in range(2)]))
out["alignment_valid_fold0_s3"] = {"aligned_mean_wp": float(np.mean(aligned)),
                                   "off_by_one_mean_wp": float(np.mean(shifted))}

# the parquet read used by Sequences agrees with the target cache row order on a few groups
from wnn.data import SequenceFile  # noqa: E402

sf = SequenceFile(ROOT / "datasets/valid.parquet")
same = []
for g in fit_sample[:3]:
    s = sf.read(int(g))
    same.append(bool(np.allclose(np.nan_to_num(s.targets), np.nan_to_num(np.asarray(yv[pos[int(g)]])))))
out["parquet_targets_equal_cache_rows"] = same

# --- 3. d256 student args vs the clean since parent (parse exactly what d256_fold.sh passes)
import argparse as _ap  # noqa: E402

job = (ROOT / "research/campaign_2026-10-01/jobs/d256_fold.sh").read_text()
cmd = job.split(".venv/bin/python scripts/train_rnn.py", 1)[1].split("\n.venv", 1)[0].split("(( maxb", 1)[0]
cmd = cmd.replace("\\\n", " ").replace('"${extra[@]}"', "--epochs 4 --checkpoint-epoch 3")
cmd = cmd.replace('"$tag"', "c20261001_d256_s3_f0").replace('"$seed"', "3").replace("${fold}", "0")
cmd = cmd.replace('"$manifest"', MANIFEST.format(fold=0))
argv = shlex.split(cmd)
captured = {}
original = _ap.ArgumentParser.parse_args


def grab(self, args=None, namespace=None):
    ns = original(self, argv, namespace)
    captured.update(vars(ns))
    raise SystemExit(0)


_ap.ArgumentParser.parse_args = grab
import train_rnn  # noqa: E402

try:
    train_rnn.main()
except SystemExit:
    pass
_ap.ArgumentParser.parse_args = original
parent = json.loads((ROOT / "artifacts/runs/c20260929_since_s3_f0/provenance.json").read_text())["args"]
skip = {"name", "split_manifest", "split_manifest_sha256", "eval_source", "valid_fold", "n_valid_folds"}
diff = {k: {"parent": parent.get(k, "<absent>"), "d256": captured.get(k, "<absent>")}
        for k in sorted((set(parent) | set(captured)) - skip) if parent.get(k, "<absent>") != captured.get(k, "<absent>")}
out["d256_vs_since_parent_arg_diff"] = diff

print(json.dumps(out, indent=1, default=str))
