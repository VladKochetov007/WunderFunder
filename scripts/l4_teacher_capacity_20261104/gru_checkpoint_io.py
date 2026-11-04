"""Panel-bound loading of native causal GRU checkpoints for diagnostics and target export."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from wnn.data import N_FEATURES
from wnn.features import N_EVENT_FEATURES
from wnn.splits import file_sha256
from wnn.training.chronological_teacher import validate_chronological_teacher_membership
from wnn.training.models import make_model


def _check_native_args(args: Mapping[str, Any], n_in: int, n_out: int) -> None:
    """Admit only strict-FP32 native 211-input causal GRUs with the expected heads."""
    fixed = {"arch": "gru", "cell": "gru", "dropout": 0.0, "enc": "linear"}
    for key, expected in fixed.items():
        if args.get(key, "gru" if key == "arch" else None) != expected:
            raise ValueError(f"selected GRU checkpoint differs at {key}")
    if n_in != N_FEATURES + N_EVENT_FEATURES or n_out != 23:
        raise ValueError(f"expected native211/23 GRU; found {n_in}/{n_out}")
    if args.get("precision") != "fp32":
        raise ValueError("selected GRU checkpoint must declare strict FP32")
    unsupported = (
        "raw_only",
        "online_norm",
        "heuristics",
        "mask_weight",
        "propensity",
        "prop_cache",
        "prop_input",
        "distill",
        "distill_cache_root",
        "aux_quote",
        "aux_tier",
        "aux_mask",
        "aux_sign",
        "aux_next_k",
        "since_weight",
        "episode_weight",
        "class_weight",
        "aux",
        "gate_ce",
        "spread_cache_root",
    )
    if any(bool(args.get(key, False)) for key in unsupported):
        raise ValueError("selected GRU has unsupported input, weighting or auxiliary configuration")
    if (
        args.get("research_features", "none") != "none"
        or args.get("privileged_mode", "none") != "none"
    ):
        raise ValueError("selected GRU enables non-native inputs")


def load_gru_models(
    panel: Mapping[str, Any],
    split: Any,
    device: torch.device,
    cell_ids: Sequence[str],
    epochs: Sequence[int],
    *,
    control_recipe: bool = False,
    training_mode: bool = False,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Load exact panel cells and checkpoint epochs, preserving each model's own buffers."""
    identity = panel["identity"]
    project = Path(panel["paths"]["project_root"])
    if identity.get("split_sha256") != split.sha256:
        raise ValueError("panel and official split hashes differ")
    for key in ("source_binding_sha256", "data_binding_sha256"):
        if not isinstance(identity.get(key), str) or len(identity[key]) != 64:
            raise ValueError(f"panel lacks admitted {key}")
    if len(cell_ids) < 1 or len(set(cell_ids)) != len(cell_ids) or len(epochs) < 1:
        raise ValueError("checkpoint loader needs distinct frozen cell IDs and epochs")
    cells = {str(cell.get("cell_id")): cell for cell in panel["cells"]}
    if any(cell_id not in cells for cell_id in cell_ids):
        raise ValueError("requested teacher cell is absent from the frozen panel")
    loaded: list[dict[str, Any]] = []
    fit_order: list[int] | None = None
    for cell_id in cell_ids:
        cell = cells[cell_id]
        run = str(cell["run_name"])
        run_dir = project / "artifacts" / "runs" / run
        run_prov = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        validate_chronological_teacher_membership(run_prov, split, run)
        for epoch in epochs:
            filename = f"model_ep{epoch}.pt"
            relative = f"artifacts/runs/{run}/{filename}"
            path = project / relative
            if relative not in cell.get("artifacts", []):
                raise ValueError(f"panel does not declare {relative}")
            checkpoint_sha = file_sha256(path)
            if run_prov.get("checkpoint_sha256", {}).get(filename) != checkpoint_sha:
                raise ValueError(f"run-provenance hash differs for {relative}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("epoch") != epoch:
                raise ValueError(f"checkpoint epoch differs for {relative}")
            provenance = checkpoint.get("provenance", {})
            validate_chronological_teacher_membership(provenance, split, relative)
            fit = list(map(int, provenance["fit_row_groups"]["train"]))
            if fit_order is None:
                fit_order = fit
            args, n_in = checkpoint.get("args", {}), int(checkpoint["n_in"])
            output_weight = checkpoint.get("model", {}).get("head.2.weight")
            if output_weight is None:
                raise ValueError(f"checkpoint lacks canonical output layer: {relative}")
            n_out = int(output_weight.shape[0])
            _check_native_args(args, n_in, n_out)
            _check_native_args(run_prov["args"], n_in, n_out)
            for config in (args, run_prov["args"]):
                if (
                    config.get("checkpoint_policy") != "fixed"
                    or config.get("checkpoint_epoch") != 3
                    or config.get("no_eval_during_fit") is not False
                    or config.get("train_source") != "train"
                    or config.get("split_manifest_sha256") != split.sha256
                    or config.get("seed") != cell["seed"]
                    or config.get("init") not in (None, "")
                ):
                    raise ValueError(
                        f"checkpoint is not from its panel-bound clean run: {relative}"
                    )
            for lineage in (run_prov, provenance):
                if lineage.get("initial_checkpoint_sha256") not in (None, ""):
                    raise ValueError(f"fresh teacher has learned initializer ancestry: {relative}")
            if control_recipe:
                expected = {
                    "hidden": 256,
                    "layers": 2,
                    "loss": "wmse",
                    "loss_norm": "chunk",
                    "aux_struct": 0.3,
                    "aux_next": 0.3,
                    "aux_anneal": "none",
                }
                for config in (args, run_prov["args"]):
                    if any(
                        config.get(key, "none" if key == "aux_anneal" else None) != value
                        for key, value in expected.items()
                    ):
                        raise ValueError(f"checkpoint differs from GRU256x2 control: {relative}")
            model = make_model(dict(args), n_in, n_out)
            if any(
                value.is_floating_point() and value.dtype != torch.float32
                for value in checkpoint["model"].values()
            ):
                raise ValueError(f"checkpoint contains non-FP32 floating weights: {relative}")
            model.load_state_dict(checkpoint["model"], strict=True)
            model.to(device=device, dtype=torch.float32)
            if training_mode and any(
                isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                for module in model.modules()
            ):
                raise ValueError("training-mode gradient replay cannot include stateful BatchNorm")
            model.train(training_mode)
            loaded.append(
                {
                    "cell_id": cell_id,
                    "seed": int(cell["seed"]),
                    "epoch": epoch,
                    "run_name": run,
                    "args": args,
                    "checkpoint": str(path),
                    "checkpoint_sha256": checkpoint_sha,
                    "model": model,
                }
            )
            del checkpoint
    if fit_order is None:
        raise ValueError("no panel-bound checkpoints were loaded")
    return loaded, fit_order
