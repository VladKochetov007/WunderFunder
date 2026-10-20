"""Bind the predeclared H512 SWA inputs to the admitted F0 fit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT_RUN = "r01_h512_f0_s1003_local_cp5_r1"
DEST_RUN = "r01_h512_f0_s1003_local_swa345_r1"
SOURCE_ROOT = Path("/home/vlad/development/wunder/research_tmp/overnight-nll-adaptation-20261020")
TRAINING_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
TRAINER_SHA = "74530d3499bba1b5647e459105cb263bfc7205d7f3aa928684dfb279c2677345"
SPLIT_SHA = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
PLAN_SHA = "51d797d1d568d98b0d9553c934721aab2826ba5d4d61b96a24e496ce514c7ec9"
SCREEN_SHA = "f6b3b954cd2253a9eac10032c9a053f7abcd55e359c41c2062ed3bdeff2920d8"
ADMISSION_PATH = Path(
    "research/campaign_2026-10-20/submission_delivery_r1/h512_f0_followup_r1/FIT_ADMISSION_R1.json"
)
ADMISSION_SHA = "a17a468f5d0d247db92cbaac0e3d4f0800536c46d6682f2665abb4b781e776f1"
HELPER_PATH = SOURCE_ROOT / "scripts/average_trajectory_checkpoints.py"
HELPER_SHA = "8646ccc12b537a1bf2f2544d4ffd85d10287076451e5bab86a4b2d6379e39bbb3"
MAX_JSON_BYTES = 1_000_000


def sha256(path: Path) -> str:
    """Hash a file in bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one bounded JSON object."""
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"metadata exceeds {MAX_JSON_BYTES} bytes: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be a JSON object: {path}")
    return value


def load_helper() -> ModuleType:
    """Import only the hash-pinned canonical checkpoint loader and mean_state."""
    path = HELPER_PATH.resolve(strict=True)
    if sha256(path) != HELPER_SHA:
        raise ValueError("canonical trajectory helper hash differs")
    spec = importlib.util.spec_from_file_location("wnn_h512_pinned_trajectory_mean", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load pinned trajectory helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_training_source(root: Path) -> dict[str, Any]:
    """Use the producer's source-closure function from the pinned checkout."""
    sys.path.insert(0, str(root / "src"))
    from wnn.training.environment import TrainingEnvironment
    from wnn.training.source_identity import training_source_identity

    expected = root / "src/wnn/training/source_identity.py"
    if Path(training_source_identity.__code__.co_filename).resolve() != expected.resolve():
        raise ValueError("training source identity imported from a different checkout")
    return training_source_identity(TrainingEnvironment(source_root=root, root=root))


def validate_inputs(root: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify every input receipt, checkpoint identity and cold F0 ancestry."""
    root = root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    if not manifest_path.is_relative_to(root):
        raise ValueError("source manifest must be inside the project root")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != "wnn.h512-swa345-source.v1"
        or manifest.get("status") != "READY_FOR_DERIVATION"
        or manifest.get("source_run_name") != ROOT_RUN
        or manifest.get("destination_run_name") != DEST_RUN
        or manifest.get("training_source_commit") != TRAINING_COMMIT
        or manifest.get("source_split_manifest_sha256") != SPLIT_SHA
    ):
        raise ValueError("manifest differs from the predeclared H512/F0 endpoint")

    admission_path = (root / ADMISSION_PATH).resolve(strict=True)
    if not admission_path.is_relative_to(root) or sha256(admission_path) != ADMISSION_SHA:
        raise ValueError("admission receipt bytes differ")
    fit_receipt = manifest["fit_terminal_receipt"]
    fit_path = (root / fit_receipt["path"]).resolve(strict=True)
    expected_fit = root / (
        "research/campaign_2026-10-20/submission_delivery_r1/"
        "h512_f0_followup_r1/FIT_TERMINAL_ACCEPTANCE_R1.json"
    )
    if fit_path != expected_fit.resolve() or sha256(fit_path) != fit_receipt["sha256"]:
        raise ValueError("fit terminal receipt path or bytes differ")
    fit_terminal = read_json(fit_path)
    if (
        fit_terminal.get("schema") != "wnn.h512-fit-terminal.v1"
        or fit_terminal.get("status") != "FIT_COMPLETE_ACCEPTED"
        or fit_terminal.get("run_name") != ROOT_RUN
        or fit_terminal.get("source_execution_commit") != TRAINING_COMMIT
        or fit_terminal.get("plan_sha256") != PLAN_SHA
        or fit_terminal.get("screen_sha256") != SCREEN_SHA
    ):
        raise ValueError("fit has no matching accepted terminal receipt")

    provenance_item = manifest["source_provenance"]
    provenance_rel = f"artifacts/runs/{ROOT_RUN}/provenance.json"
    provenance_path = (root / provenance_item["path"]).resolve(strict=True)
    if (
        provenance_item.get("path") != provenance_rel
        or not provenance_path.is_relative_to(root)
        or sha256(provenance_path) != provenance_item["sha256"]
    ):
        raise ValueError("source run provenance path or hash differs")
    source_provenance = read_json(provenance_path)

    items = manifest.get("checkpoints")
    expected_paths = [f"artifacts/runs/{ROOT_RUN}/model_ep{epoch}.pt" for epoch in (3, 4, 5)]
    if not isinstance(items, list) or [item.get("epoch") for item in items] != [3, 4, 5]:
        raise ValueError("manifest must bind epochs 3, 4, and 5 in order")
    if [item.get("path") for item in items] != expected_paths:
        raise ValueError("manifest checkpoint paths differ from the admitted run")

    helper = load_helper()
    checkpoints = [helper.load_bound(root, item) for item in items]
    source_identity = current_training_source(SOURCE_ROOT.resolve(strict=True))
    required_fields = (
        "fit_row_groups",
        "fit_membership_identity",
        "normalization_identity",
        "trainer_implementation",
    )
    cold_fields = (
        "initial_checkpoint_sha256",
        "propensity_checkpoint_sha256",
        "distillation_cache",
    )
    identity: dict[str, Any] = {}
    arguments = checkpoints[0].get("args")
    expected_args = {
        "name": ROOT_RUN,
        "seed": 1003,
        "arch": "gru",
        "batch": 8,
        "checkpoint_epoch": 5,
        "checkpoint_policy": "fixed",
        "cell": "gru",
        "dropout": 0.0,
        "enc": "linear",
        "epochs": 6,
        "hidden": 512,
        "layers": 1,
        "loss": "wmse",
        "loss_norm": "chunk",
        "precision": "fp32",
        "train_source": "train+valid",
        "valid_fold": 0,
    }
    if not isinstance(arguments, dict) or any(
        arguments.get(key) != value for key, value in expected_args.items()
    ):
        raise ValueError("checkpoint args differ from the frozen cold FP32 H512 fit")
    if any(
        arguments.get(key) not in (None, "", False)
        for key in ("init", "resume_state", "distill", "propensity", "include_heldout_valid")
    ):
        raise ValueError("source checkpoint is not a cold, label-only F0 fit")

    for epoch, checkpoint in zip((3, 4, 5), checkpoints, strict=True):
        provenance = checkpoint.get("provenance")
        model_state = checkpoint.get("model")
        trainer = provenance.get("trainer_implementation") if isinstance(provenance, dict) else None
        if (
            checkpoint.get("epoch") != epoch
            or checkpoint.get("n_in") != 211
            or checkpoint.get("args") != arguments
            or not isinstance(model_state, dict)
            or not isinstance(provenance, dict)
            or not isinstance(trainer, dict)
            or checkpoint.get("readout") is not None
            or checkpoint.get("readout_tilted")
            or "quantization" in checkpoint
            or "factorization" in checkpoint
        ):
            raise ValueError(f"epoch {epoch} checkpoint structure or metadata differs")
        if (
            tuple(model_state["mean"].shape) != (211,)
            or tuple(model_state["std"].shape) != (211,)
            or tuple(model_state["head.2.weight"].shape) != (2, 512)
            or tuple(model_state["head.2.bias"].shape) != (2,)
        ):
            raise ValueError(f"epoch {epoch} tensor shapes differ from plain H512x1/2-output")
        if provenance.get("split_manifest_sha256") != SPLIT_SHA:
            raise ValueError(f"epoch {epoch} is not bound to fold0")
        for field in required_fields + cold_fields:
            value = provenance.get(field)
            if field in required_fields and value is None:
                raise ValueError(f"epoch {epoch} lacks required identity: {field}")
            if field in cold_fields and value not in (None, {}, ""):
                raise ValueError(f"cold label-only fit unexpectedly has {field}")
            if epoch == 3:
                identity[field] = value
            elif identity[field] != value:
                raise ValueError(f"checkpoint ancestry differs: {field}")
        files = trainer.get("files", {})
        if (
            files != source_identity["files"]
            or files.get("scripts/train_rnn.py") != TRAINER_SHA
            or trainer.get("sha256") != source_identity["sha256"]
        ):
            raise ValueError("trainer source closure differs from the admitted source tree")

    if (
        source_provenance.get("args") != arguments
        or source_provenance.get("split_manifest_sha256") != SPLIT_SHA
        or any(source_provenance.get(field) != value for field, value in identity.items())
    ):
        raise ValueError("run provenance identity differs from checkpoints")
    source_hashes = source_provenance.get("checkpoint_sha256", {})
    receipt_hashes = fit_terminal.get("checkpoint_sha256", {})
    for item in items:
        filename = Path(item["path"]).name
        if (
            source_hashes.get(filename) != item["sha256"]
            or receipt_hashes.get(filename) != item["sha256"]
        ):
            raise ValueError(f"checkpoint hash is not bound by both receipts: {filename}")
    return {
        "root": root,
        "manifest_path": manifest_path,
        "helper": helper,
        "checkpoints": checkpoints,
        "items": items,
        "fit_receipt": fit_receipt,
        "provenance_item": provenance_item,
        "source_provenance": source_provenance,
        "identity": identity,
    }
