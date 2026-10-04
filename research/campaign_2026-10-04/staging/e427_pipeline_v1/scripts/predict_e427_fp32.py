"""Export the fixed E427 epoch-0 checkpoint over all VALID groups in FP32."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[5]
STAGING = ROOT / "research/campaign_2026-10-04/staging/e427"
sys.path.insert(0, str(STAGING / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_e427 as trainer  # noqa: E402
from wnn.propensity import load_propensity  # noqa: E402
from wnn.resources import configure_torch, dataloader_budget  # noqa: E402
from wnn.sentinel import SentinelPreprocessor  # noqa: E402
from wnn.splits import file_sha256  # noqa: E402

STATS = ROOT / "research/campaign_2026-10-04/checks/e427_context_stats_s3_v1.json"
PARENT = "c20261004_combo_s3_final_r1"
EXPECTED_DOMAIN = list(range(1873))


def sha(path: Path) -> str:
    return file_sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--arm", choices=("aligned", "stale", "noop"), required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    ckpt_path = ROOT / "artifacts/runs" / args.run / "model_ep0.pt"
    provenance_path = ckpt_path.parent / "provenance.json"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if (ck.get("epoch") != 0 or ck.get("args", {}).get("checkpoint_policy") != "fixed"
            or ck.get("args", {}).get("checkpoint_epoch") != 0
            or ck.get("args", {}).get("no_eval_during_fit") is not True):
        raise ValueError("E427 export requires the fixed epoch-0 checkpoint with fit-time evaluation withheld")
    if ck["args"].get("arch") != "e427-volume-gate" or ck["args"].get("e427_arm") != args.arm:
        raise ValueError("checkpoint architecture/arm differs from export declaration")
    if ck.get("n_in") != 211:
        raise ValueError("E427 fixed parent input width changed")
    configure_torch()
    device = torch.device("cuda")
    model = trainer.make_model(ck["args"], ck["n_in"], 23).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    model._inference_fp32 = True
    trainer.SENTINEL_PREPROCESSOR = (
        SentinelPreprocessor(ck["sentinel_preprocessing"])
        if ck.get("sentinel_preprocessing") else None
    )
    trainer.RAW_ONLY = bool(ck["args"].get("raw_only"))
    trainer.ONLINE_NORM = bool(ck["args"].get("online_norm"))
    trainer.HEURISTICS = bool(ck["args"].get("heuristics"))
    trainer.RESEARCH_FEATURES = ck["args"].get("research_features", "none")
    input_propensity = ck["args"].get("prop_input")
    trainer.INPUT_PROPENSITY = (
        load_propensity(ROOT / "artifacts/runs" / input_propensity / "model.pt", device)
        if input_propensity else None
    )
    if ck.get("readout") is not None:
        trainer.READOUT = [value.to(device) for value in ck["readout"]]
        trainer.READOUT_TILTED = ck.get("readout_tilted", False)

    groups = np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False)["row_group"]
    typed_groups = [int(group) for group in groups]
    if typed_groups != EXPECTED_DOMAIN:
        raise ValueError("VALID target cache is not the exact ordered typed 1,873-group domain")
    dataset = trainer.Sequences("valid", groups)
    workers, prefetch = dataloader_budget(dataset.batch_tensor_bytes(16), max_workers=2)
    loader = DataLoader(dataset, batch_size=16, num_workers=workers,
                        prefetch_factor=prefetch if workers else None)
    output_path = ROOT / "cache" / f"pred_{args.output_name}_valid.npy"
    partial = output_path.with_name(output_path.name + ".partial")
    if output_path.exists() or partial.exists():
        raise FileExistsError("prediction output already exists; refusing overwrite")
    prediction = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=(1873, 20_000, 2)
    )
    offset = 0
    with torch.inference_mode():
        for x, _, _ in loader:
            values = trainer.predict(model, x, 4000, device)
            stop = offset + len(values)
            prediction[offset:stop] = values[..., :2]
            offset = stop
    prediction.flush()
    del prediction
    if offset != 1873:
        raise RuntimeError(f"export wrote {offset} of 1,873 VALID groups")
    partial.replace(output_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    parent_provenance_path = ROOT / "artifacts/runs" / PARENT / "provenance.json"
    metadata = {
        "schema": "wnn.e427-fixed-epoch0-fp32-prediction-v1",
        "run": args.run,
        "parent_run": PARENT,
        "arm": args.arm,
        "split": "valid",
        "epoch": 0,
        "checkpoint_policy": "fixed",
        "fit_time_evaluation": "withheld",
        "checkpoint_sha256": sha(ckpt_path),
        "provenance_sha256": sha(provenance_path),
        "parent_checkpoint_sha256": sha(ROOT / "artifacts/runs" / PARENT / "model_ep3.pt"),
        "parent_provenance_sha256": sha(parent_provenance_path),
        "context_stats_sha256": sha(STATS),
        "normalization_identity": provenance.get("normalization_identity"),
        "inference_precision": "fp32",
        "model_outputs": 23,
        "outputs_exported": 2,
        "prediction_sha256": sha(output_path),
        "row_groups": typed_groups,
        "split_manifest_sha256": None,
        "fit_membership_counts": {
            key: len(value) for key, value in provenance.get("fit_row_groups", {}).items()
        },
    }
    sidecar = output_path.with_suffix(".json")
    with sidecar.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"prediction": str(output_path.relative_to(ROOT)),
                      "sha256": metadata["prediction_sha256"], "rows": offset}, sort_keys=True))


if __name__ == "__main__":
    main()
