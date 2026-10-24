"""Admission-gated CLI for the bounded profile and frozen age-readout screen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from .age_readout_artifacts import file_sha256
from .age_readout_portable import PortablePaths
from .age_readout_provenance import SOURCE_FILES
from .age_readout_runner import run_fit_fold, run_profile
from .age_readout_runtime import configure_cuda_memory, torch

RECIPE_SHA256 = "0a67fe12c520e56bcf14a3a1efecb5030f3aa7ce932101814c9dd1c675c930fd"
DEFAULT_PROFILE_ROOT = Path(
    "/home/vlad/development/wunder/research_disk/b300-fast-20261023/b300_scout/profiles"
)


def main() -> None:
    """Validate frozen recipe/admission, then run only the declared operation."""
    args = _arguments()
    repo_root = args.repo_root.resolve()
    recipe_path = _resolve(repo_root, args.recipe)
    recipe_sha = file_sha256(recipe_path)
    if args.recipe_sha256 != RECIPE_SHA256 or recipe_sha != RECIPE_SHA256:
        raise ValueError("machine recipe path or SHA differs from frozen recipe")
    recipe = _json(recipe_path)
    fold = next(item for item in recipe["folds"] if int(item["fold"]) == args.fold)
    admission = _json(args.admission)
    if admission.get("recipe_sha256") != recipe_sha:
        raise ValueError("admission receipt is for a different machine recipe")
    if admission.get("status") != "admitted":
        raise ValueError("profile and fit operations require a root-issued status=admitted receipt")
    source_manifest_path = _resolve(repo_root, args.source_manifest)
    _validate_source_manifest(
        repo_root,
        source_manifest_path,
        args.source_manifest_sha256,
        admission.get("source_manifest_sha256"),
        recipe_sha,
    )
    expected_mode = "profile" if args.mode == "profile" else "fit"
    if admission.get("mode") != expected_mode or admission.get("device") != args.device:
        raise ValueError("admission mode or device differs from this command")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is unavailable")
    if args.portable_config is not None and args.mode != "profile":
        raise ValueError("portable Kaggle CUDA admission permits profile mode only")
    if (
        args.portable_config is not None
        and args.device == "cuda"
        and torch.cuda.device_count() != 1
    ):
        raise RuntimeError("portable Kaggle profile requires exactly one visible CUDA device")
    portable_paths = None
    if args.portable_config is not None:
        portable_paths = PortablePaths.read(args.portable_config)
        if admission.get("transport_sha256") != portable_paths.transport_sha256:
            raise ValueError("admission does not bind this portable transport manifest")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = f"cuda:{torch.cuda.current_device()}" if args.device == "cuda" else "cpu"
    configure_cuda_memory(device)
    if args.mode == "profile":
        profile_root = args.profile_root
        if portable_paths is not None:
            if profile_root != DEFAULT_PROFILE_ROOT and profile_root != portable_paths.profile_root:
                raise ValueError("portable profile output differs from the bound path map")
            profile_root = portable_paths.profile_root
        receipt = run_profile(
            repo_root, recipe, recipe_sha, fold, device, profile_root, portable_paths
        )
    else:
        if admission.get("shared_readout_device") != device:
            raise ValueError("admission does not bind this device for all six cells")
        receipt = run_fit_fold(repo_root, recipe, recipe_sha, fold, device, portable_paths)
    print(json.dumps(receipt, sort_keys=True))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("profile", "fit-fold"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument("--portable-config", type=Path)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _validate_source_manifest(
    repo_root: Path,
    path: Path,
    supplied_sha: str,
    admitted_sha: object,
    recipe_sha: str,
) -> None:
    observed_sha = file_sha256(path)
    if observed_sha != supplied_sha or observed_sha != admitted_sha:
        raise ValueError("source manifest SHA differs from root admission")
    manifest = _json(path)
    if (
        manifest.get("schema") != "wnn.age-readout-source-manifest.v1"
        or manifest.get("recipe_sha256") != recipe_sha
    ):
        raise ValueError("source manifest schema or recipe differs from the frozen screen")
    expected = manifest.get("fit_source_sha256", {})
    if set(expected) != set(SOURCE_FILES) or manifest.get("source_sha256") != expected:
        raise ValueError("source manifest module set differs from the portable actor")
    for relative, digest in expected.items():
        if file_sha256(repo_root / relative) != digest:
            raise ValueError(f"source pin changed before admission: {relative}")
