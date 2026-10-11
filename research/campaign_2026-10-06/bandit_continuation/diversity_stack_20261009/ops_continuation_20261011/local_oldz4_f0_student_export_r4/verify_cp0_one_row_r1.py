from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import resource
import runpy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run_student_export_r4 import _import_exact_trainer, _verify_bound_sources
from validate_student_export_r4 import read_json, sha256_file, write_once

LOGGER = logging.getLogger("oldz4_cp0_one_row_r1")
PACKET_ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = PACKET_ROOT / "STUDENT_EXPORT_CONTRACT_R4.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _setup_prediction_globals(
    train_rnn: Any, checkpoint: dict[str, Any], device: torch.device
) -> None:
    """Match predictor preprocessing for the frozen CP0 one-row smoke test."""
    args = checkpoint["args"]
    train_rnn.SENTINEL_PREPROCESSOR = None
    train_rnn.RAW_ONLY = bool(args.get("raw_only"))
    train_rnn.ONLINE_NORM = bool(args.get("online_norm"))
    train_rnn.INPUT_LEAD = int(args.get("input_lead", 0))
    train_rnn.RESEARCH_FEATURES = args.get("research_features", "none")
    train_rnn.INPUT_PROPENSITY = None
    train_rnn.READOUT = None
    train_rnn.READOUT_TILTED = False
    if checkpoint.get("readout") is not None:
        train_rnn.READOUT = [value.to(device) for value in checkpoint["readout"]]
        train_rnn.READOUT_TILTED = checkpoint.get("readout_tilted", False)


def main() -> None:
    """Load fitted CP0 and predict one feature-only valid group on CPU."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve(strict=True)
    contract = read_json(CONTRACT_PATH)
    source_root, predictor_path = _verify_bound_sources(project_root)
    train_rnn = _import_exact_trainer(source_root, project_root)
    import wnn.resources as wnn_resources

    dataloader_budget = wnn_resources.dataloader_budget
    loader_probe = train_rnn.Sequences("valid", [])
    loader_workers, prefetch = dataloader_budget(loader_probe.batch_tensor_bytes(16), max_workers=2)
    _require(
        os.environ.get("WNN_LOADER_WORKERS") == "0",
        "probe did not run with WNN_LOADER_WORKERS=0",
    )
    _require(
        loader_workers == 0 and prefetch == 0,
        "imported loader helper did not enforce zero workers",
    )
    predictor = runpy.run_path(str(predictor_path), run_name="oldz4_cp0_feature_probe")
    _require(predictor["ROOT"].resolve() == project_root, "predictor ROOT is not MAIN")
    _require(predictor["train_rnn"] is train_rnn, "predictor imported another trainer module")
    _require(
        predictor["FeatureOnlySequences"].__module__ == "oldz4_cp0_feature_probe",
        "one-row fixture did not use the exact predictor feature-only reader",
    )
    run = contract["run"]
    run_dir = project_root / "artifacts" / "runs" / run
    checkpoint_path = run_dir / "model_ep0.pt"
    provenance_path = run_dir / "provenance.json"
    provenance = read_json(provenance_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    _require(
        provenance.get("checkpoint_sha256", {}).get("model_ep0.pt") == checkpoint_sha256,
        "CP0 hash does not match the live run provenance",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(checkpoint.get("epoch") == 0, "producer fixture is not CP0")
    _require(checkpoint.get("args", {}).get("name") == run, "CP0 belongs to another run")
    _require(checkpoint.get("args", {}).get("checkpoint_epoch") == 1, "CP0 fit horizon differs")
    valid_path = project_root / contract["valid_parquet"]["relative_path"]
    _require(
        valid_path.stat().st_size == contract["valid_parquet"]["size_bytes"],
        "valid.parquet byte length differs from the frozen input binding",
    )
    model_state = checkpoint.get("model")
    _require(isinstance(model_state, dict), "CP0 lacks model weights")
    head_shape = list(model_state["head.2.weight"].shape)
    model = predictor["make_model"](
        checkpoint["args"], checkpoint["n_in"], model_state["head.2.weight"].shape[0]
    ).to(torch.device("cpu"))
    model.load_state_dict(model_state, strict=True)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    _setup_prediction_globals(train_rnn, checkpoint, device)
    target_index_path = project_root / "cache" / "valid_targets.npz"
    with np.load(target_index_path, allow_pickle=False) as target_index:
        row_groups = [int(group) for group in target_index["row_group"]]
    _require(len(row_groups) == 1873, "valid target row-group order is not complete")
    row_groups_sha256 = hashlib.sha256(np.asarray(row_groups, dtype="<i8").tobytes()).hexdigest()
    group = row_groups[0]
    dataset = predictor["FeatureOnlySequences"]("valid", [group])
    features = dataset[0].unsqueeze(0)
    expected_feature_count = int(train_rnn.N_FEATURES)
    _require(
        tuple(features.shape) == (1, 20000, expected_feature_count),
        "one-row feature shape differs",
    )
    started = time.monotonic()
    prediction = train_rnn.predict(model, features, 4000, device)
    elapsed = time.monotonic() - started
    _require(prediction.shape == (1, 20000, 2), "one-row prediction shape differs")
    _require(prediction.dtype == np.float32, "one-row prediction dtype differs")
    _require(bool(np.isfinite(prediction).all()), "one-row prediction contains nonfinite values")
    receipt = {
        "schema": "wnn.oldz4.student-cp0-one-row-export-probe.r1",
        "status": "pass",
        "scope": "CPU-only load/build/predict of one feature-only valid sequence; no labels, fit, full export, or grade.",
        "run": run,
        "checkpoint_epoch": 0,
        "checkpoint_sha256": checkpoint_sha256,
        "fit_attempt_id": contract["fit_attempt_id"],
        "trainer_sha256": contract["fit_trainer_sha256"],
        "trainer_module_path": str(Path(train_rnn.__file__).resolve()),
        "predictor_sha256": contract["predictor_sha256"],
        "loader_workers_environment": os.environ.get("WNN_LOADER_WORKERS"),
        "dataloader_budget_workers": loader_workers,
        "dataloader_budget_prefetch": prefetch,
        "dataloader_budget_source_path": str(Path(wnn_resources.__file__).resolve()),
        "dataloader_budget_source_sha256": sha256_file(Path(wnn_resources.__file__).resolve()),
        "train_source": checkpoint["args"].get("train_source"),
        "eval_source": checkpoint["args"].get("eval_source"),
        "split_manifest_sha256": checkpoint["args"].get("split_manifest_sha256"),
        "row_group": group,
        "valid_row_groups": len(row_groups),
        "valid_row_group_index_sha256": row_groups_sha256,
        "feature_shape": list(features.shape),
        "model_input_width": int(checkpoint["n_in"]),
        "prediction_shape": list(prediction.shape),
        "prediction_dtype": str(prediction.dtype),
        "prediction_finite": True,
        "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
        "model_head_weight_shape": head_shape,
        "elapsed_seconds": round(elapsed, 3),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "max_rss_interpretation": "Process ru_maxrss is a lifetime high-water value and exceeded independently observed limited-scope memory peaks; use the scope cgroup receipt for resource qualification.",
        "valid_parquet_size_bytes": valid_path.stat().st_size,
        "created_unix": time.time(),
    }
    write_once(args.receipt_out, receipt)
    LOGGER.info(
        json.dumps(
            {"status": "pass", "row_group": group, "elapsed_seconds": receipt["elapsed_seconds"]}
        )
    )


if __name__ == "__main__":
    main()
