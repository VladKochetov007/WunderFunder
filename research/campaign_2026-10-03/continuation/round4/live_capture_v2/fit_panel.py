"""Freeze fit membership without labels/scores; bounded real panel assembly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from live_capture import HERE, ROOT, make_environment, sha, trainer_module

PANEL = HERE / "fit_panel.json"
PROVENANCE = ROOT / "artifacts/runs/c20261003_combo_s3_f0/provenance.json"


def freeze_panel() -> None:
    """Inspect only membership, sequence identity and cache metadata."""
    if PANEL.exists():
        raise FileExistsError("fixed panel already frozen")
    provenance = json.loads(PROVENANCE.read_text())
    args = provenance["args"]
    split_path = Path(args["split_manifest"])
    manifest = json.loads(split_path.read_text())
    if sha(split_path) != provenance["split_manifest_sha256"]:
        raise ValueError("active manifest drift")
    rows = []
    bindings = {}
    for split, count in (("train", 28), ("valid", 4)):
        eligible = set(provenance["fit_row_groups"][split])
        if not eligible <= set(manifest["fit"][split]):
            raise ValueError("active fit provenance not contained in manifest")
        chosen = sorted(eligible - ({528} if split == "train" else set()))[:count]
        if split == "valid" and set(chosen) & set(manifest["evaluation"]["valid"]):
            raise ValueError("evaluation contamination")
        parquet_path = ROOT / "datasets" / f"{split}.parquet"
        parquet = pq.ParquetFile(parquet_path)
        for group in chosen:
            column = parquet.read_row_group(group, columns=["seq_ix"], use_threads=False).column(
                "seq_ix"
            )
            if len(column) != 20000:
                raise ValueError("incomplete sequence")
            rows.append({"split": split, "row_group": group, "seq_ix": int(column[0].as_py())})
        stat = parquet_path.stat()
        bindings[f"datasets/{split}.parquet"] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "identity_level": "stat plus captured exact tensor bytes; no full dataset hash",
        }
        target = ROOT / "cache" / f"{split}_targets.npz"
        order = np.load(target, allow_pickle=False)["row_group"].tolist()
        bindings[f"cache/{split}_targets.npz"] = {"sha256": sha(target)}
        for entry in args["distill_provenance"]:
            if entry["split"] != split:
                continue
            path = ROOT / "cache" / f"pred_{entry['run']}_{split}.npy"
            sidepath = path.with_suffix(".json")
            side = json.loads(sidepath.read_text())
            if (
                side["epoch"] != 3
                or side["row_groups"] != order
                or side["prediction_sha256"] != entry["prediction_sha256"]
                or side["checkpoint_sha256"] != entry["checkpoint_sha256"]
                or side["split_manifest_sha256"] != provenance["split_manifest_sha256"]
            ):
                raise ValueError("mature teacher provenance mismatch")
            bindings[str(path.relative_to(ROOT))] = {
                "sha256": entry["prediction_sha256"],
                "verification": "full array hash required at real runner entry",
                "size": path.stat().st_size,
            }
            bindings[str(sidepath.relative_to(ROOT))] = {"sha256": sha(sidepath)}
        prop = ROOT / "cache" / f"propcache_{args['propensity']}_{split}.npy"
        bindings[str(prop.with_suffix(".json").relative_to(ROOT))] = {
            "sha256": sha(prop.with_suffix(".json"))
        }
        bindings[str(prop.relative_to(ROOT))] = {
            "size": prop.stat().st_size,
            "mtime_ns": prop.stat().st_mtime_ns,
            "identity_level": "checkpoint-bound sidecar; exact selected propensity tensor captured",
        }
    payload = {
        "version": 1,
        "rule": "28 lowest permitted TRAIN + 4 lowest fit VALID; fixed prior to seq_ix read",
        "rows": rows,
        "provenance_sha256": sha(PROVENANCE),
        "manifest_path": str(split_path),
        "manifest_sha256": sha(split_path),
        "bindings": bindings,
        "scored": False,
        "teacher_epoch": 3,
        "independent_review_required": True,
    }
    PANEL.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"rows": rows, "teacher_epoch": 3, "scored": False}, indent=2))


def verify_panel() -> dict[str, Any]:
    if not PANEL.exists():
        raise ValueError("fit panel not frozen")
    panel = json.loads(PANEL.read_text())
    if (
        sha(PROVENANCE) != panel["provenance_sha256"]
        or sha(Path(panel["manifest_path"])) != panel["manifest_sha256"]
    ):
        raise ValueError("panel provenance changed")
    for name, binding in panel["bindings"].items():
        path = ROOT / name
        if "sha256" in binding and sha(path) != binding["sha256"]:
            raise ValueError("bound cache bytes changed: " + name)
        stat = path.stat()
        if "size" in binding and stat.st_size != binding["size"]:
            raise ValueError("bound file size differs")
        if "mtime_ns" in binding and stat.st_mtime_ns != binding["mtime_ns"]:
            raise ValueError("bound file stat differs")
    return panel


def real_environment() -> dict[str, Any]:
    """Fresh genuine student/EMA history on exact fixed fit-mixture inputs."""
    panel = verify_panel()
    module = trainer_module()
    provenance = json.loads(PROVENANCE.read_text())
    args = provenance["args"].copy()
    args["parity_dump"] = ""
    teachers = [(name, 1.0) for name in args["distill"].split(",")]
    datasets = {
        split: module.Sequences(
            split,
            [entry["row_group"] for entry in panel["rows"] if entry["split"] == split],
            teachers,
            module.prop_cache_path(args["propensity"], split),
        )
        for split in ("train", "valid")
    }
    # Stream one sequence at a time; no DataLoader workers/prefetch or wide read.
    batches = []
    counters = {"train": 0, "valid": 0}
    for row in panel["rows"]:
        split = row["split"]
        batches.append(datasets[split][counters[split]])
        counters[split] += 1
    x, joined_y, scored, prop = [
        torch.stack(values).to("cuda") for values in zip(*batches, strict=True)
    ]
    del batches
    cfg = {"n_in": 211, "hidden": 256, "layers": 2, "n_out": 23}
    total = (
        (provenance["fit_membership_identity"]["dataset_length"] // args["batch"])
        * 20
        * args["epochs"]
    )
    env = make_environment(module, args, cfg, "cuda", total, True)
    inputs = module.build_inputs(x)
    del x
    # Explicit panel-only normalization departure; no exported EMA substituted.
    sampled = inputs.reshape(-1, 211)[::7]
    env["model"].mean.copy_(sampled.mean(0))
    env["model"].std.copy_(sampled.std(0).clamp_min(1e-3))
    env["ema_model"].module.load_state_dict(env["model"].state_dict())
    teach, y = joined_y[..., 2:4], joined_y[..., :2]
    valid = torch.arange(20000, device="cuda")[None].expand(32, -1) >= module.WARMUP
    weights = valid.float() * (prop ** args["prop_power"] + args["prop_eps"])
    weights = module.true_mask_weights(
        weights, valid, scored, args["valid_true_mask"], args["sync_free"]
    )
    since = torch.expm1(
        inputs[..., module.N_FEATURES + module.FEATURE_NAMES.index("since_i0px")].float()
    )
    weights = weights * torch.clamp(1 / (1 + since / args["since_weight"]), min=args["since_floor"])
    sp, br, cl = module.structure_labels(y, module.sequence_class(y))
    nx_rows, nx_up, nx_ok = module.next_change_labels(y, valid)
    env.update(
        {
            "inputs": inputs,
            "y": y,
            "teach": teach,
            "main_valid": weights,
            "sw": weights.float(),
            "sp_lab": sp,
            "br_lab": br,
            "cl_lab": cl,
            "nxt_label": module.next_label_change(y),
            "nx_rows": nx_rows,
            "nx_up": nx_up,
            "nw": weights.float() * nx_ok.float(),
            "capture_metadata": {
                "evidence_kind": "real_fit_capture",
                "panel_sha256": sha(PANEL),
                "row_ids": panel["rows"],
                "input_origin": "actual build_inputs/Sequences caches and four actual loss branches",
                "departure": "fresh seed3 initialization and fixed-panel normalization, not resumed trained combo",
                "full_sequence_length": 20000,
                "cached_propensity": prop.detach().cpu(),
                "scored_mask": scored.detach().cpu(),
                "teacher_binding": panel["bindings"],
            },
        }
    )
    return env


if __name__ == "__main__":
    freeze_panel()
