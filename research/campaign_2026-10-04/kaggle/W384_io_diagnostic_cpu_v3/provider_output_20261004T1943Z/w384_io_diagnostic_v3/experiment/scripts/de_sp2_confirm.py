"""Independent reproduction of the preregistered SP2 parent-mix confirmation (folds 2+3) and the folds 0+1 screen.

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/de_sp2_confirm.py [--folds 2 3] [--tag f23]

Arms per fold k (equal float64 means, float32 rounding, clip; scripts/de_public_proxy.py scoring):
  ft_pair            c20260929_ft_since_s{3,4}_f{k}          (two-stage FT, the reference)
  mix4               ft_pair members + parents c20260929_since_s{3,4}_f{k} @ep3
  shared_parent_head head-stage FT c20260929_ft_since_s{3,4}_f{k}h + the two parents
Every member is checked before scoring: sidecar run/epoch/split-manifest hash, prediction sha256, provenance
manifest hash, fit row groups disjoint from the fold's evaluation sequences and excluded TRAIN, and the
init chain parent@ep3 -> head stage -> two-stage FT. Gate (research/campaign_2026-10-01/screens/
parentmix_confirm_f23_PREREG.json): pooled delta >= 0.0005, each fold > 0, every class delta >= -0.0005.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from de_public_proxy import PROXY_JSON, SPLITS, blend_sequence_sums, compare, load_targets, sequence_table  # noqa: E402

OUT = ROOT / "research" / "campaign_2026-10-01" / "data_eval"
GATE = {"pooled_min": 0.0005, "each_fold_positive": True, "class_floor": -0.0005}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def members(k: int) -> dict[str, list[str]]:
    ft = [f"c20260929_ft_since_s{s}_f{k}" for s in (3, 4)]
    parents = [f"c20260929_since_s{s}_f{k}" for s in (3, 4)]
    heads = [f"c20260929_ft_since_s{s}_f{k}h" for s in (3, 4)]
    return {"ft_pair": ft, "mix4": ft + parents, "shared_parent_head": heads + parents}


def verify(k: int) -> dict:
    """Identity and leakage checks for every member of fold k; raises on any failure."""
    manifest_path = SPLITS / f"fold{k}.json"
    manifest_sha = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    eval_valid = set(manifest["evaluation"]["valid"])
    excluded_train = set(manifest["excluded"]["train"]) | set(manifest["excluded"].get("corrupt_train", []))
    expected_epoch = {"since": 3, "ft": 0}
    record = {}
    runs = sorted({m for arm in members(k).values() for m in arm})
    prov = {r: json.loads((ROOT / "artifacts" / "runs" / r / "provenance.json").read_text()) for r in runs}
    for run in runs:
        pred = ROOT / "cache" / f"pred_{run}_valid.npy"
        side = json.loads((ROOT / "cache" / f"pred_{run}_valid.json").read_text())
        p = prov[run]
        checks = {
            "sidecar_run": side["run"] == run,
            "sidecar_split_manifest": side["split_manifest_sha256"] == manifest_sha,
            "sidecar_epoch": side["epoch"] == expected_epoch["ft" if "_ft_" in run else "since"],
            "prediction_sha256": side["prediction_sha256"] == sha256(pred),
            "provenance_split_manifest": p["split_manifest_sha256"] == manifest_sha,
            "fit_valid_disjoint_from_eval": not (set(p["fit_row_groups"]["valid"]) & eval_valid),
            "fit_train_disjoint_from_excluded": not (set(p["fit_row_groups"]["train"]) & excluded_train),
            "fixed_checkpoint_policy": p["args"].get("checkpoint_policy") == "fixed",
        }
        if run.endswith("h"):  # head stage initialised from parent @ep3 checkpoint
            parent = run.replace("ft_since", "since")[:-1]
            checks["init_is_parent_ep3"] = (p["args"].get("init") == f"{parent}@ep3"
                                            and p["initial_checkpoint_sha256"] == prov[parent]["checkpoint_sha256"].get("model_ep3.pt"))
            checks["head_only_frozen_trunk"] = p["args"].get("freeze") == "inp,gru"
        elif "_ft_" in run:  # two-stage FT initialised from its head stage
            head = f"{run}h"
            hp = json.loads((ROOT / "artifacts" / "runs" / head / "provenance.json").read_text())
            checks["init_is_head_stage_ep0"] = (p["args"].get("init") == f"{head}@ep0"
                                                and p["initial_checkpoint_sha256"] == hp["checkpoint_sha256"]["model_ep0.pt"])
        else:
            checks["sidecar_checkpoint_is_ep3"] = side["checkpoint_sha256"] == p["checkpoint_sha256"].get("model_ep3.pt")
        record[run] = checks
        failed = [c for c, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"fold {k} {run}: failed {failed}")
    return {"manifest_sha256": manifest_sha, "members": record}


def gate(cmp: dict) -> dict:
    pooled = cmp["results"]["all_eval_rows"]["delta"]
    folds = {f: v["delta"] for f, v in cmp["per_fold"].items()}
    classes = {c: v["delta"] for c, v in cmp["by_class"].items()}
    ok = {"pooled": pooled >= GATE["pooled_min"], "each_fold_positive": all(v > 0 for v in folds.values()),
          "class_floor": all(v >= GATE["class_floor"] for v in classes.values())}
    return {**ok, "pass": all(ok.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--tag", default="f23")
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()
    checks = {str(k): verify(k) for k in args.folds}
    y, mask, *_ = load_targets()
    table = sequence_table()
    proxies = json.loads(PROXY_JSON.read_text())["proxies_for_tool"]
    sums = {arm: [] for arm in members(args.folds[0])}
    rows = []
    for k in args.folds:
        r = np.flatnonzero(table["fold"] == k)
        rows.append(r)
        for arm, runs in members(k).items():
            sums[arm].append(blend_sequence_sums([f"cache/pred_{m}_valid.npy" for m in runs], r, y, mask))
    rows = np.concatenate(rows)
    sub = {key: v[rows] for key, v in table.items()}
    sums = {arm: np.concatenate(v) for arm, v in sums.items()}
    report = {"folds": args.folds, "n_sequences": int(len(rows)), "n_components": int(len(np.unique(sub["component"]))),
              "rows": "step>=99 AND is_scored", "gate": GATE, "identity_checks": checks, "comparisons": {}}
    for cand, ref in (("mix4", "ft_pair"), ("shared_parent_head", "ft_pair"), ("shared_parent_head", "mix4")):
        cmp = compare(sums[ref], sums[cand], sub, proxies, sub["fold"], args.boot)
        report["comparisons"][f"{cand}_minus_{ref}"] = {**cmp, "gate": gate(cmp)}
    (OUT / f"sp2_confirm_{args.tag}.json").write_text(json.dumps(report, indent=1) + "\n")
    for name, c in report["comparisons"].items():
        r = c["results"]["all_eval_rows"]
        print(f"{name:34s} pooled {r['delta']:+.6f} sd {r['boot_sd']:.6f} | folds "
              + " ".join(f"{f}:{v['delta']:+.6f}" for f, v in c["per_fold"].items())
              + " | classes " + " ".join(f"{k}:{v['delta']:+.5f}" for k, v in c["by_class"].items())
              + f" | implied mix {c['results']['public_like_class_mix']['delta']:+.6f}"
              + f" sd {c['results']['public_like_class_mix']['boot_sd']:.6f} | gate {c['gate']['pass']}")


if __name__ == "__main__":
    main()
