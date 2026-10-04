"""Export fixed-epoch E425 soft-route VALID predictions with staged imports."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
STAGING_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = STAGING_ROOT / "scripts" / "train_rnn.py"
SPECIALIST_PATH = STAGING_ROOT / "src" / "wnn" / "specialist.py"


def _sha256_file(path: Path) -> str:
    """Hash a file incrementally without loading its contents into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_import_roots() -> None:
    """Prioritize staged code, then retain the repository's WNN fallbacks."""
    staged_source = STAGING_ROOT / "src"
    repository_source = ROOT / "src"
    script_source = ROOT / "scripts"
    front = [str(path) for path in (staged_source, repository_source, script_source, ROOT)]
    front_set = set(front)
    sys.path[:] = [entry for entry in sys.path if entry not in front_set]
    sys.path[0:0] = front

    package = sys.modules.get("wnn")
    if package is not None:
        staged_package = str(staged_source / "wnn")
        old_paths = [path for path in package.__path__ if path != staged_package]
        package.__path__ = [staged_package, *old_paths]

    for module_name in ("wnn.specialist", "wnn.specialist_diagnostics"):
        cached = sys.modules.get(module_name)
        cached_file = getattr(cached, "__file__", None)
        if cached_file is not None and not Path(cached_file).resolve().is_relative_to(STAGING_ROOT):
            del sys.modules[module_name]
            cached = None
        if package is not None:
            attribute = module_name.rsplit(".", maxsplit=1)[-1]
            package_cached = getattr(package, attribute, None)
            package_file = getattr(package_cached, "__file__", None)
            if package_file is not None and not Path(package_file).resolve().is_relative_to(
                STAGING_ROOT
            ):
                delattr(package, attribute)
            elif cached is not None:
                setattr(package, attribute, cached)

    cached_trainer = sys.modules.get("train_rnn")
    cached_file = getattr(cached_trainer, "__file__", None)
    if cached_file is not None and Path(cached_file).resolve() != TRAINER_PATH.resolve():
        del sys.modules["train_rnn"]


def _load_staged_trainer() -> tuple[ModuleType, dict[str, object]]:
    """Load the staged trainer and prove its specialist is the staged class."""
    closure_before = _source_closure()
    _prepare_import_roots()
    spec = importlib.util.spec_from_file_location("e425_soft_staged_train_rnn", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load staged trainer from {TRAINER_PATH}")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    trainer.ROOT = ROOT

    specialist_module = sys.modules.get("wnn.specialist")
    module_file = getattr(specialist_module, "__file__", None)
    if module_file is None or Path(module_file).resolve() != SPECIALIST_PATH.resolve():
        raise RuntimeError("E425 prediction did not import the staged specialist module")
    if trainer.SpecialistGRUModel is not specialist_module.SpecialistGRUModel:
        raise RuntimeError("staged trainer and staged specialist module use different classes")
    closure_after = _source_closure()
    if closure_after["sha256"] != closure_before["sha256"]:
        raise RuntimeError("prediction source changed while staged modules were loading")
    return trainer, closure_after


def _source_closure() -> dict[str, object]:
    """Hash the staged trainer/model and all repository WNN fallbacks."""
    paths = set((ROOT / "src" / "wnn").rglob("*.py"))
    paths.update((STAGING_ROOT / "src" / "wnn").rglob("*.py"))
    paths.update((STAGING_ROOT / "scripts").glob("*.py"))
    paths.add(Path(__file__).resolve())
    root_utils = ROOT / "utils.py"
    if root_utils.is_file():
        paths.add(root_utils)
    entries = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in sorted(paths)
    ]
    payload = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "schema": "e425-prediction-source-closure-v1",
        "files": entries,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _output_width(state: Mapping[str, object]) -> int:
    """Infer and validate the 23-output specialist topology without Torch."""
    key = "universal_head.2.weight"
    weight = state.get(key)
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"specialist checkpoint lacks matrix tensor {key!r}")
    width = int(shape[0])
    heads = [
        value
        for name, value in state.items()
        if name.startswith("expert_heads.") and name.endswith(".weight")
    ]
    if not heads or any(getattr(head, "shape", (0,))[0] != width for head in heads):
        raise ValueError("specialist checkpoint expert output widths are missing or inconsistent")
    if width != 23:
        raise ValueError(f"E425 combo checkpoints require 23 outputs, got {width}")
    return width


def _predict(args: argparse.Namespace) -> None:
    """Export every fixed-epoch VALID sequence in common FP32 inference."""
    import torch
    from torch.utils.data import DataLoader

    trainer, source_closure = _load_staged_trainer()
    from wnn.data import SEQUENCE_LENGTH as DATA_SEQUENCE_LENGTH
    from wnn.propensity import load_propensity
    from wnn.resources import configure_torch, dataloader_budget
    from wnn.sentinel import SentinelPreprocessor
    from wnn.splits import file_sha256

    if args.epoch != 3 or args.split != "valid" or args.inference_precision != "fp32":
        raise ValueError("E425 predictions require fixed epoch 3, all VALID rows, and FP32")
    if not args.name.replace("_", "").isalnum():
        raise ValueError("run name must contain only letters, digits, and underscores")
    output_name = args.output_name or args.name
    if not output_name.replace("_", "").isalnum():
        raise ValueError("output name must contain only letters, digits, and underscores")

    checkpoint_path = ROOT / "artifacts" / "runs" / args.name / "model_ep3.pt"
    checkpoint_sha256 = file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args")
    state = checkpoint.get("model")
    if not isinstance(checkpoint_args, dict) or not isinstance(state, Mapping):
        raise ValueError("checkpoint lacks its training args or model state")
    if checkpoint.get("epoch") != 3:
        raise ValueError("model_ep3.pt does not contain fixed epoch 3")
    if checkpoint_args.get("checkpoint_policy") != "fixed":
        raise ValueError("E425 prediction requires the fixed checkpoint policy")
    if checkpoint_args.get("checkpoint_epoch") != 3:
        raise ValueError("E425 checkpoint metadata must fix epoch 3")
    if checkpoint_args.get("arch") != "specialist":
        raise ValueError("E425 prediction accepts only specialist checkpoints")
    if checkpoint_args.get("specialist_route") != "soft":
        raise ValueError("E425 prediction requires specialist_route='soft'")
    if checkpoint_args.get("privileged_mode", "none") != "none":
        raise ValueError("privileged teacher inputs cannot be exported as causal predictions")
    if checkpoint.get("quantization") is not None or checkpoint.get("factorization") is not None:
        raise ValueError("E425 prediction does not accept quantized or factorized checkpoints")
    if checkpoint.get("n_in") != 211:
        raise ValueError(f"E425 combo checkpoints require 211 inputs, got {checkpoint.get('n_in')}")
    output_width = _output_width(state)
    model_source_sha256 = _sha256_file(SPECIALIST_PATH)
    checkpoint_provenance = checkpoint.get("provenance") or {}
    saved_model_source = checkpoint_provenance.get("model_source_sha256")
    if saved_model_source is not None and saved_model_source != model_source_sha256:
        raise ValueError("staged specialist source differs from the fitted checkpoint provenance")

    configure_torch()
    device = torch.device("cuda")
    model = trainer.make_model(checkpoint_args, checkpoint["n_in"], output_width).to(device)
    if model.__class__ is not trainer.SpecialistGRUModel or model.route != "soft":
        raise RuntimeError("root/model factory did not construct the staged soft specialist")
    model.load_state_dict(state, strict=True)
    model._inference_fp32 = True

    trainer.SENTINEL_PREPROCESSOR = (
        SentinelPreprocessor(checkpoint["sentinel_preprocessing"])
        if checkpoint.get("sentinel_preprocessing")
        else None
    )
    trainer.RAW_ONLY = bool(checkpoint_args.get("raw_only"))
    trainer.ONLINE_NORM = bool(checkpoint_args.get("online_norm"))
    trainer.HEURISTICS = bool(checkpoint_args.get("heuristics"))
    trainer.RESEARCH_FEATURES = checkpoint_args.get("research_features", "none")
    input_propensity = checkpoint_args.get("prop_input")
    trainer.INPUT_PROPENSITY = (
        load_propensity(ROOT / "artifacts" / "runs" / input_propensity / "model.pt", device)
        if input_propensity
        else None
    )
    if checkpoint.get("readout") is not None:
        trainer.READOUT = [tensor.to(device) for tensor in checkpoint["readout"]]
        trainer.READOUT_TILTED = checkpoint.get("readout_tilted", False)

    groups = np.load(ROOT / "cache" / "valid_targets.npz", allow_pickle=False)["row_group"]
    dataset = trainer.Sequences("valid", groups)
    workers, prefetch = dataloader_budget(dataset.batch_tensor_bytes(16), max_workers=2)
    loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=workers,
        prefetch_factor=prefetch if workers else None,
    )
    final = ROOT / "cache" / f"pred_{output_name}_valid.npy"
    sidecar_path = final.with_suffix(".json")
    temporary = final.with_name(final.name + ".partial")
    if final.exists() or sidecar_path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite prediction artifact: {final}")
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(groups), DATA_SEQUENCE_LENGTH, 2),
    )
    written = 0
    for features, _, _ in loader:
        prediction = trainer.predict(model, features, 4000, device)
        if not np.isfinite(prediction).all():
            raise FloatingPointError(f"non-finite prediction values near VALID sequence {written}")
        output[written : written + len(prediction)] = prediction[..., :2]
        written += len(prediction)
    if written != len(groups):
        raise RuntimeError(f"wrote {written} of {len(groups)} VALID sequences")
    if file_sha256(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("source checkpoint changed during prediction export")
    if _source_closure()["sha256"] != source_closure["sha256"]:
        raise RuntimeError("prediction source changed during export")
    output.flush()
    del output
    temporary.replace(final)
    stale_sums = ROOT / "cache" / f"seqsums_{output_name}.npz"
    if stale_sums.exists():
        stale_sums.unlink()

    quote_cache = checkpoint_provenance.get("quote_cache")
    sidecar = {
        "run": args.name,
        "split": "valid",
        "epoch": 3,
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_sha256": file_sha256(final),
        "split_manifest_sha256": checkpoint_args.get("split_manifest_sha256"),
        "quote_cache": quote_cache,
        "gate_mode": "soft",
        "specialist_route": "soft",
        "inference_precision": "fp32",
        "quantization_preset": None,
        "recurrent_rank": None,
        "sentinel_preprocessing": checkpoint.get("sentinel_preprocessing"),
        "model_source_sha256": model_source_sha256,
        "code_source_closure": source_closure,
        "row_groups": groups.tolist(),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")


def main() -> None:
    """Parse a narrow fixed-epoch soft-route export interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("split", nargs="?", default="valid", choices=("valid",))
    parser.add_argument("--epoch", type=int, choices=(3,), default=3)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--inference-precision", choices=("fp32",), default="fp32")
    _predict(parser.parse_args())


if __name__ == "__main__":
    main()
