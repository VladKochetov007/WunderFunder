from __future__ import annotations

import hashlib
import inspect
import json
import os
import runpy
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast

BASE_SOURCE_COMMIT = "fa615eb3d13a8afb9fd7cba275ef64956bdb793e"
EXPECTED_TORCH_VERSION = "2.14.0+cu130"
EXPECTED_SOURCE_HASHES = {
    "scripts/train_rnn.py": "8a29459e998f7f5d6a03b7a4b87c6f4b569002e65b8c87127c37138360909cbf",
    "src/wnn/run_identity.py": "7033ea007b9c7dcf498d3ab1e31aaeacb5725b7009fe719e9071c05b99272a76",
    "src/wnn/optim.py": "620569a752fb611adf69976f1dbed1b7882e641672c58175005f5d3f819d2991",
}
EXPECTED_TORCH_SOURCE_HASHES = {
    "torch.amp.GradScaler": "97c411da028daaf6a6ed15d06b9b20c017404846db68203be1a586e276e44039",
    "torch.optim.AdamW": "6f4a96e7bccd5e233a66f846c39f7a9da0157cf657c2275d2a269a78a6b90921",
}
EXPECTED_TRAINER_ARGS = [
    "--name",
    "e529_s4_served_head_f0_profile_r2_scalerdiag",
    "--init",
    "c20261003_combo_s4_f0@ep3",
    "--train-source",
    "valid",
    "--full-valid",
    "--mask-weight",
    "--seed",
    "3",
    "--epochs",
    "1",
    "--batch",
    "16",
    "--chunk",
    "2000",
    "--fast",
    "--inherit-init-preprocessing",
    "--loss",
    "gcorr",
    "--optim",
    "adamw",
    "--freeze",
    "inp,gru",
    "--lr",
    "0.0005",
    "--wd",
    "0.01",
    "--checkpoint-policy",
    "fixed",
    "--checkpoint-epoch",
    "0",
    "--split-manifest",
    "research/campaign_2026-09-29/artifacts/splits/fold0.json",
    "--no-eval-during-fit",
    "--profile-steps",
    "4",
    "--profile-warmup",
    "2",
    "--grad-stats",
]


class GradScalerProtocol(Protocol):
    """Methods observed or delegated by the diagnostic wrapper."""

    def is_enabled(self) -> bool: ...

    def get_scale(self) -> float: ...

    def update(self, *args: object, **kwargs: object) -> object: ...


class TorchAmpProtocol(Protocol):
    """AMP namespace needed to bind the installed GradScaler source."""

    GradScaler: type[GradScalerProtocol]


class TorchOptimProtocol(Protocol):
    """Optimizer namespace needed to bind the installed AdamW source."""

    AdamW: object


class TorchModuleProtocol(Protocol):
    """Installed torch module fields needed by runtime source verification."""

    __version__: str
    amp: TorchAmpProtocol
    optim: TorchOptimProtocol


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one pinned source file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_trainer_argv(trainer_path: Path, args: Sequence[str]) -> list[str]:
    """Build the exact argv train_rnn.py receives through runpy."""
    return [str(trainer_path), *args]


def install_update_probe(
    scaler_class: type[GradScalerProtocol], emit: Callable[[dict[str, object]], None]
) -> tuple[Callable[[], int], Callable[[], int]]:
    """Log GradScaler scale transitions while delegating every update unchanged.

    The wrapper is scoped to this diagnostic process and restores the original
    method after the profile. Calling get_scale synchronizes CUDA, so the
    instrumented profile is a correctness diagnostic and not a timing comparison.
    """
    original_update = cast(Callable[..., object], scaler_class.update)
    update_ordinal = 0

    def observed_update(scaler: GradScalerProtocol, *args: object, **kwargs: object) -> object:
        nonlocal update_ordinal
        update_ordinal += 1
        enabled = scaler.is_enabled()
        scale_before = float(scaler.get_scale())
        result = original_update(scaler, *args, **kwargs)
        scale_after = float(scaler.get_scale())
        emit(
            {
                "event": "amp_scale_update",
                "update_ordinal": update_ordinal,
                "enabled": enabled,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "scale_reduced": scale_after < scale_before,
                "scale_increased": scale_after > scale_before,
            }
        )
        return result

    scaler_class.update = observed_update

    def restore() -> int:
        """Restore GradScaler.update and report the number of observed calls."""
        scaler_class.update = cast(Callable[..., object], original_update)
        return update_ordinal

    def update_count() -> int:
        """Report the number of intercepted update calls."""
        return update_ordinal

    return restore, update_count


def emit_json(record: dict[str, object]) -> None:
    """Write one machine-readable diagnostic event to the worker log."""
    sys.stdout.write(f"AMP_SCALE_PROBE {json.dumps(record, sort_keys=True)}\n")
    sys.stdout.flush()


def verify_source(root: Path) -> dict[str, str]:
    """Fail closed unless the frozen trainer-side source files match fa615."""
    actual: dict[str, str] = {}
    for relative_path, expected_sha256 in EXPECTED_SOURCE_HASHES.items():
        path = root / relative_path
        actual[relative_path] = sha256_file(path)
        if actual[relative_path] != expected_sha256:
            raise RuntimeError(f"source pin mismatch for {relative_path}: {actual[relative_path]}")
    return actual


def verify_torch_source(torch: TorchModuleProtocol) -> dict[str, str]:
    """Bind the installed scaler and fused AdamW Python source implementations."""
    version = str(torch.__version__)
    if version != EXPECTED_TORCH_VERSION:
        raise RuntimeError(f"unexpected torch version: {version}")
    classes = {
        "torch.amp.GradScaler": torch.amp.GradScaler,
        "torch.optim.AdamW": torch.optim.AdamW,
    }
    actual: dict[str, str] = {}
    for name, cls in classes.items():
        source_path = inspect.getsourcefile(cls)
        if source_path is None:
            raise RuntimeError(f"cannot locate source for {name}")
        actual[name] = sha256_file(Path(source_path))
        if actual[name] != EXPECTED_TORCH_SOURCE_HASHES[name]:
            raise RuntimeError(f"installed source pin mismatch for {name}")
    return actual


def main() -> None:
    """Run the exact bounded profile with read-only AMP scale instrumentation."""
    root = Path(os.environ.get("WNN_PROJECT_ROOT", str(Path.cwd()))).resolve()
    if Path.cwd().resolve() != root:
        raise RuntimeError("working directory must equal WNN_PROJECT_ROOT")
    source_hashes = verify_source(root)
    trainer_path = root / "scripts/train_rnn.py"
    trainer_args = sys.argv[1:]
    if trainer_args != EXPECTED_TRAINER_ARGS:
        raise RuntimeError("trainer arguments differ from the R2 profile recipe")
    trainer_argv = build_trainer_argv(trainer_path, trainer_args)

    import torch

    torch_hashes = verify_torch_source(cast(TorchModuleProtocol, torch))
    emit_json(
        {
            "event": "probe_start",
            "base_source_commit": BASE_SOURCE_COMMIT,
            "trainer_source_sha256": source_hashes["scripts/train_rnn.py"],
            "run_identity_source_sha256": source_hashes["src/wnn/run_identity.py"],
            "optim_source_sha256": source_hashes["src/wnn/optim.py"],
            "torch_version": str(torch.__version__),
            "torch_source_sha256": torch_hashes,
            "trainer_argv_sha256": hashlib.sha256(
                json.dumps(trainer_argv, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "measurement_note": (
                "get_scale() synchronizes CUDA before and after every update; "
                "do not compare this timing with the uninstrumented profile"
            ),
        }
    )
    restore, _ = install_update_probe(torch.amp.GradScaler, emit_json)
    sys.argv = trainer_argv
    try:
        runpy.run_path(str(trainer_path), run_name="__main__")
    finally:
        observed_updates = restore()
        emit_json(
            {
                "event": "probe_summary",
                "update_calls": observed_updates,
                "expected_update_calls": 60,
                "count_matches_profile": observed_updates == 60,
            }
        )


if __name__ == "__main__":
    main()
