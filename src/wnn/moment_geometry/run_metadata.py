"""Source, environment and exact recipe metadata for every prepared-array endpoint."""

from __future__ import annotations

import argparse
import hashlib
import platform
from pathlib import Path

import numpy as np
import torch

import wnn.gmetric as gmetric_module
import wnn.nn as nn_module
from wnn.moment_geometry.recipe import BASE_LR, BATCH, CHUNK, MAX_NORM, MIN_LR, RAW, WARMUP, WD


def endpoint_config(
    arrays: dict[str, np.ndarray], args: argparse.Namespace, arm: str, seed: int
) -> dict[str, object]:
    hashes = {
        "runner.py": Path(__file__).with_name("runner.py"),
        "run_metadata.py": Path(__file__),
        "recipe.py": Path(__file__).with_name("recipe.py"),
        "grading.py": Path(__file__).with_name("grading.py"),
        "backend.py": Path(__file__).with_name("backend.py"),
        "wnn.nn": Path(nn_module.__file__),
        "wnn.gmetric": Path(gmetric_module.__file__),
    }
    hashes.update({path.name: path for path in Path(__file__).parent.glob("*.py")})
    return {
        "evaluation_scope": args.report_label,
        "seed": seed,
        "arm": arm,
        "data_directory": str(args.data.resolve()),
        "requested_arms": args.arms,
        "data_receipt_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.data.glob("*.json")
        },
        "epochs": args.epochs,
        "model": {
            "class": "wnn.nn.RNNModel",
            "n_in": int(arrays["fit_x"].shape[-1]),
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": 0.0,
        },
        "optimizer": {"class": "AdamW", "lr_start": BASE_LR, "lr_end": MIN_LR, "weight_decay": WD},
        "schedule": "update cosine, endpoints included",
        "batch_chunk_warmup_clip": [BATCH, CHUNK, WARMUP, MAX_NORM],
        "loss": "wnn.nn.weighted_mse_loss",
        "precision": "FP32; autocast and TF32 disabled",
        "input": "fixed sequence order, chronological chunks, fit-only normalized",
        "channels": {
            "raw_event": RAW,
            "geometry_order": "mean4/std4/circle_res4/modal_occ4/causal_normalized_current4",
            "arm_treatment": {
                "control": "zero all suffix channels after211",
                "moments": "retain only first16ofsuffix",
                "scaled": "retain only first20ofsuffix",
                "dictionary": "retain only40dictionary suffixchannels after231",
                "combined": "retain all60suffixchannels",
            }[arm],
        },
        "metric": "global_wp_clip2_v1",
        "fit_sequences": int(arrays["fit_x"].shape[0]),
        "eval_sequences": int(arrays["eval_x"].shape[0]),
        "fit_component_count": int(np.unique(arrays["fit_component"]).size),
        "eval_fold_ids": np.unique(arrays["eval_fold"]).tolist(),
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in hashes.items()
        },
        "versions": [platform.python_version(), np.__version__, torch.__version__],
    }
