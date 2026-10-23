"""B300-only adapter around the canonical trainer, with paired initialization receipts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from attrs import define
from commands import fit_args, run_name
from common import ROOT, digest, load_assets, output_root, preflight, read_json, write_new

# Canonical model/runtime records deliberately retain heterogeneous values.


def runtime_identity() -> dict[str, Any]:
    """Read the actual CUDA process identity and enforce the prepared B300 policy."""
    import torch

    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError(
            "this worker requires Linux and an allocated B300; CPU fallback is forbidden"
        )
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if "B300" not in props.name:
        raise RuntimeError(
            f"expected B300, got {props.name}; amend/profile a separate hardware recipe"
        )
    if torch.is_autocast_enabled("cuda"):
        raise RuntimeError("CUDA autocast must be disabled")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 must be disabled")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("float32 matmul precision must be highest")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": props.name,
        "uuid": str(props.uuid),
        "capability": list(torch.cuda.get_device_capability()),
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "attrs", "pyarrow")
        },
        "matmul_precision": "highest",
        "tf32": False,
        "autocast": False,
    }


def state_digest(model: Any) -> str:
    """Fingerprint the full initial state, including fitting-only normalization buffers."""
    import torch

    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f"non-FP32 model state: {name}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"nonfinite model state: {name}")
        value = tensor.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


@define
class OptimizerProbe:
    """Inspect the actual training process after preprocessing and before its first update."""

    assets: dict[str, Any]
    binding: dict[str, Any]
    cell: str
    role: str
    profile: bool
    resume_sha256: str | None = None
    attempt: str | None = None

    def __call__(self, model: Any, args: argparse.Namespace) -> Any:
        """Record common initialization without altering RNG, model tensors or optimizer behavior."""

        from wnn.training_dynamics import build_optimizer

        if next(model.parameters()).device.type != "cuda" or args.precision != "fp32":
            raise ValueError("trainer process is not CUDA FP32")
        identity = runtime_identity()
        initial = state_digest(model)
        run = run_name(self.cell, self.role, self.profile)
        receipt = {
            **self.binding,
            "run": run,
            "initial_state_sha256": initial,
            "runtime": identity,
            "runtime_sha256": hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest(),
        }
        start_path = output_root(self.assets) / "start" / f"{run}.json"
        if self.resume_sha256:
            original = read_json(start_path)
            # Canonical setup has already restored model tensors, but optimizer/RNG
            # restoration happens after this hook. Preserve the original cold identity.
            receipt["initial_state_sha256"] = original["initial_state_sha256"]
            if receipt != original:
                raise ValueError("resume source/assets/runtime differs from original attempt")
        if self.role == "kd":
            control = read_json(
                output_root(self.assets)
                / "start"
                / f"{run_name(self.cell, 'labels', self.profile)}.json"
            )
            for key in (
                "initial_state_sha256",
                "runtime_sha256",
                "source_sha256",
                "recipe_sha256",
                "assets_sha256",
            ):
                if receipt[key] != control[key]:
                    raise ValueError(f"student pair differs before fitting: {key}")
        if self.resume_sha256:
            write_new(
                output_root(self.assets)
                / "attempts"
                / f"{run}_{self.resume_sha256}_{self.attempt}.json",
                {
                    **receipt,
                    "resume_state_sha256": self.resume_sha256,
                    "restored_model_state_sha256": initial,
                },
            )
        else:
            write_new(start_path, receipt)
        return build_optimizer(model, args)


def main() -> None:
    """Run one frozen profile or scientific fit; the controller invokes this subprocess."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--cell", choices=("F0", "F2"), required=True)
    parser.add_argument("--role", choices=("teacher_a", "teacher_b", "labels", "kd"), required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--attempt")
    args = parser.parse_args()
    if bool(args.resume_state) != bool(args.attempt) or (
        args.attempt and uuid.UUID(args.attempt).hex != args.attempt
    ):
        parser.error("resume requires its unique hexadecimal attempt ID")
    assets = load_assets(args.assets)
    binding = preflight(assets)
    import torch

    from wnn.resources import configure_strict_fp32
    from wnn.training.environment import TrainingEnvironment
    from wnn.training.runner import run_training

    configure_strict_fp32()
    torch.set_float32_matmul_precision("highest")
    identity = runtime_identity()
    if not args.profile:
        from receiver import require_profile

        profile = require_profile(assets, args.cell, args.role)
        if (
            profile["runtime_sha256"]
            != hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        ):
            raise ValueError("fit device/runtime differs from its admitted profile")
    if args.role in ("labels", "kd"):
        from audit import teacher_targets

        teacher_targets(assets, args.cell)
    free, total = torch.cuda.mem_get_info()
    if free < 0.1 * total:
        raise RuntimeError("less than 10% device memory available before training setup")
    run = run_name(args.cell, args.role, args.profile)
    path = Path(assets["project_root"]) / "artifacts/runs" / run
    resume_sha = None
    if args.resume_state:
        if (
            args.profile
            or args.resume_state.resolve() != (path / "training_state.pt").resolve()
            or (output_root(assets) / "terminal" / f"{run}.json").exists()
        ):
            raise ValueError("resume requires this incomplete scientific run's full epoch state")
        resume_sha = digest(args.resume_state)
    elif path.exists():
        raise FileExistsError(
            f"preserve existing attempt: {path}; see documented exact resume procedure"
        )
    environment = TrainingEnvironment(
        source_root=ROOT,
        root=Path(assets["project_root"]),
        build_optimizer=OptimizerProbe(
            assets, binding, args.cell, args.role, args.profile, resume_sha, args.attempt
        ),
    )
    started = time.monotonic()
    command = fit_args(assets, args.cell, args.role, args.profile)
    if args.resume_state:
        command += ["--resume-state", str(args.resume_state)]
    run_training(command, environment=environment)
    free, total = torch.cuda.mem_get_info()
    peak = torch.cuda.max_memory_reserved()
    if free < 0.1 * total or peak > 0.9 * total:
        raise RuntimeError(
            "profile/fit exceeded reserved memory margin; do not silently shrink batch"
        )
    write_new(
        output_root(assets) / "terminal" / f"{run}.json",
        {
            **binding,
            "run": run,
            "status": "completed",
            "profile": args.profile,
            "resume_state_sha256": resume_sha,
            "runtime": identity,
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": peak,
            "free_bytes_after": free,
            "total_bytes": total,
        },
    )


if __name__ == "__main__":
    main()
