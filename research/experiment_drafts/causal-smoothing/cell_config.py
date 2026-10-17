from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[2]
ARMS = ("raw-plus-control", "raw-plus-causal-ewma")
RECIPE = PACKET / "recipe.json"
TRAIN_SCRIPT = PACKET / "train.py"


def sha256(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically persist a receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read bounded object-shaped metadata."""
    if path.stat().st_size > 5_000_000:
        raise ValueError(f"metadata exceeds 5 MB: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def data_root() -> Path:
    """Resolve the explicitly selected dataset/cache root."""
    value = os.environ.get("WNN_PROJECT_ROOT")
    if not value:
        raise ValueError("set WNN_PROJECT_ROOT to the permitted data root")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"WNN_PROJECT_ROOT is not a directory: {root}")
    return root


def cell_key(arm: str, fold: int, seed: int, attempt: int = 1) -> str:
    """Return a unique non-resumable scientific cell attempt identity."""
    return f"{arm}-f{fold}-s{seed}-a{attempt}"


def manifest_path(root: Path, fold: int) -> Path:
    """Locate the frozen source-qualified outer-fold manifest."""
    return root / "research/campaign_2026-09-29/artifacts/splits" / f"fold{fold}.json"


def inspect_manifest(path: Path, fold: int) -> dict[str, Any]:
    """Validate membership and bind manifest, parquet, and cache identities."""
    manifest = read_json(path)
    if manifest.get("version") != 1 or manifest.get("fold") != fold or manifest.get("n_folds") != 4:
        raise ValueError("campaign split is not the expected source-qualified fold")
    fit, evaluation = manifest.get("fit"), manifest.get("evaluation")
    if not isinstance(fit, dict) or not isinstance(evaluation, dict):
        raise ValueError("campaign split lacks fitting/evaluation membership")
    train, valid, evaluated = fit.get("train"), fit.get("valid"), evaluation.get("valid")
    if not all(isinstance(items, list) and items for items in (train, valid, evaluated)):
        raise ValueError("campaign memberships must be nonempty lists")
    if any(len(set(items)) != len(items) for items in (train, valid, evaluated)):
        raise ValueError("campaign group membership contains duplicates")
    if set(valid) & set(evaluated):
        raise ValueError("fit-valid and evaluation membership overlap")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("source-qualified manifest has no input identities")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "fold": fold,
        "n_folds": 4,
        "train_groups": len(train),
        "fit_valid_groups": len(valid),
        "evaluation_groups": len(evaluated),
        "sources": sources,
        "cache_identities": {
            name: identity for name, identity in sources.items() if name.startswith("cache/")
        },
    }


def source_identity() -> dict[str, Any]:
    """Hash packet, trainer, prediction, feature, and training implementation sources."""
    paths = sorted(PACKET.glob("*.py")) + [RECIPE]
    paths += [ROOT / "scripts/train_rnn.py", ROOT / "scripts/predict_run.py"]
    paths += sorted((ROOT / "src/wnn").rglob("*.py"))
    paths += [path for path in (ROOT / "pyproject.toml", ROOT / "uv.lock") if path.is_file()]
    files = {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths if path.is_file()}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return {
        "git_commit": result.stdout.strip() if result.returncode == 0 else "unavailable",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def recipe_config() -> tuple[dict[str, Any], str]:
    """Load the recipe as the sole source of tunable training values."""
    recipe = read_json(RECIPE)
    training = recipe.get("training")
    required = {
        "epochs",
        "batch",
        "chunk",
        "update_rows",
        "detach_rows",
        "precision",
        "autocast",
        "tf32",
        "optim",
        "lr",
        "wd",
        "beta1",
        "beta2",
        "clip",
        "sched",
        "warmup_frac",
        "wsd_decay",
        "wsd_final",
        "wsd_shape",
        "hidden",
        "layers",
        "dropout",
        "loss",
        "checkpoint_policy",
        "checkpoint_epoch",
        "no_eval_during_fit",
        "save_training_state",
    }
    if not isinstance(training, dict) or required - training.keys():
        raise ValueError(f"recipe.training is missing {sorted(required - set(training or {}))}")
    if (
        recipe.get("arms") != list(ARMS)
        or recipe.get("precision") != "fp32"
        or recipe.get("tf32") is not False
        or recipe.get("autocast") is not False
    ):
        raise ValueError("recipe arm/precision contract differs from the frozen campaign")
    expected = {
        "epochs": 12,
        "batch": 8,
        "chunk": 2000,
        "update_rows": 2000,
        "detach_rows": 2000,
        "precision": "fp32",
        "autocast": False,
        "tf32": False,
        "optim": "adamw",
        "lr": 0.001,
        "wd": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "clip": 1.0,
        "sched": "wsd",
        "warmup_frac": 0.02,
        "wsd_decay": 0.2,
        "wsd_final": 0.1,
        "wsd_shape": "sqrt",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.0,
        "loss": "wmse",
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 11,
        "no_eval_during_fit": True,
        "save_training_state": True,
    }
    for name, value in expected.items():
        if training.get(name) != value:
            raise ValueError(f"recipe.training.{name} must remain {value!r}")
    return recipe, sha256(RECIPE)


def run_name(arm: str, fold: int, seed: int, attempt: int = 1, *, profile: bool = False) -> str:
    """Name model artifacts without permitting continuation or overwrite."""
    return (
        f"csm_profile_{arm.replace('-', '_')}_a{attempt}"
        if profile
        else f"csm_{arm.replace('-', '_')}_f{fold}_s{seed}_a{attempt}"
    )


def training_dir(root: Path, name: str) -> Path:
    """Resolve the canonical trainer output directory."""
    return root / "artifacts/runs" / name


def trainer_command(
    arm: str,
    fold: int,
    seed: int,
    split: dict[str, Any],
    training: dict[str, Any],
    attempt: int = 1,
    *,
    profile: bool = False,
) -> list[str]:
    """Build the fixed trainer command; profile uses bounded 5+20 batches."""
    values = {
        "--name": run_name(arm, fold, seed, attempt, profile=profile),
        "--arch": "gru",
        "--cell": "gru",
        "--enc": "linear",
        "--hidden": training["hidden"],
        "--layers": training["layers"],
        "--dropout": training["dropout"],
        "--batch": training["batch"],
        "--epochs": training["epochs"],
        "--chunk": training["chunk"],
        "--update-rows": training["update_rows"],
        "--detach-rows": training["detach_rows"],
        "--precision": training["precision"],
        "--optim": training["optim"],
        "--lr": training["lr"],
        "--wd": training["wd"],
        "--beta2": training["beta2"],
        "--clip": training["clip"],
        "--sched": training["sched"],
        "--warmup-frac": training["warmup_frac"],
        "--wsd-decay": training["wsd_decay"],
        "--wsd-final": training["wsd_final"],
        "--wsd-shape": training["wsd_shape"],
        "--loss": training["loss"],
        "--checkpoint-policy": training["checkpoint_policy"],
        "--checkpoint-epoch": training["checkpoint_epoch"],
        "--split-manifest": split["path"],
        "--train-source": "train+valid",
        "--valid-fold": fold,
        "--train-seqs": split["train_groups"],
        "--full-valid": None,
        "--seed": seed,
        "--device": "cuda",
        "--research-features": "causal_ewma",
        "--research-feature-control": "raw_zero" if arm == "raw-plus-control" else "none",
        "--no-eval-during-fit": None,
        "--save-training-state": None,
        "--grad-stats": None,
    }
    command = [sys.executable, str(TRAIN_SCRIPT)]
    for flag, value in values.items():
        command.append(flag)
        if value is not None:
            command.append(str(value))
    if profile:
        command += ["--profile-steps", "20", "--profile-warmup", "5", "--max-batches", "25"]
    return command
