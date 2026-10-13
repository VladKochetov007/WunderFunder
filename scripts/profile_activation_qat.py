"""Profile activation-aware QAT through the frozen trainer or synthetic smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "E-QAT-A8-BACKEND-20261013-R1"
RECIPE_RELATIVE = Path(
    "research/campaign_2026-10-13/qat_renewable_r1/delivery_r1/RUN_RECIPE_R1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _recipe_arguments(
    project_root: Path, worktree_root: Path, mode: str, device: str
) -> tuple[list[str], dict[str, Any]]:
    recipe_path = project_root / RECIPE_RELATIVE
    recipe = json.loads(recipe_path.read_text())
    fits = [
        fit
        for fit in recipe["fits"]
        if fit.get("subset") == "10pct" and fit.get("condition") == "uniform_w4"
    ]
    if len(fits) != 1:
        raise ValueError("frozen recipe must contain exactly one 10pct uniform_w4 fit")
    fit = fits[0]
    argv = list(fit["argv"])
    if len(argv) < 3 or Path(argv[1]).name != "train_rnn.py":
        raise ValueError("frozen fit argv does not name scripts/train_rnn.py")
    options = argv[2:]
    for flag in (
        "--save-training-state",
        "--grad-stats",
        "--fast",
        "--pin-memory",
        "--activation-profile-first-chunk",
    ):
        while flag in options:
            options.remove(flag)
    for flag in ("--profile-warmup", "--profile-steps", "--activation-backend"):
        while flag in options:
            position = options.index(flag)
            del options[position : position + 2]
    if "--name" not in options or "--qat" not in options:
        raise ValueError("frozen fit is missing its run name or weight-only QAT policy")
    init_position = options.index("--init") + 1
    parent_name, _, parent_epoch = options[init_position].partition("@ep")
    checkpoint_name = f"model_ep{int(parent_epoch)}.pt" if parent_epoch else "model.pt"
    checkpoint_path = project_root / "artifacts" / "runs" / parent_name / checkpoint_name
    checkpoint_sha256 = _sha256(checkpoint_path)
    expected_checkpoint_sha256 = "e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda"
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            "selected CP5 checkpoint bytes differ from the frozen activation-QAT source"
        )
    name_position = options.index("--name") + 1
    options[name_position] = f"qat_activation_profile_r1_{mode}"
    device_position = options.index("--device") + 1
    options[device_position] = device
    options.extend(
        [
            "--activation-backend",
            "scan_a8" if mode == "a8" else "scan_fp32",
            "--activation-profile-first-chunk",
            "--profile-warmup",
            "1",
            "--profile-steps",
            "1",
            "--grad-stats",
        ]
    )
    command = [sys.executable, str(worktree_root / "scripts/train_rnn.py"), *options]
    recipe_summary = {
        "condition": fit["condition"],
        "subset": fit["subset"],
        "source_name": fit["name"],
        "recipe_sha256": _sha256(recipe_path),
        "parent_checkpoint": "clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1@ep5",
        "parent_checkpoint_path": str(checkpoint_path),
        "parent_checkpoint_sha256": checkpoint_sha256,
        "weight_qat": "recurrent_w4",
        "optimizer": "fresh AdamW from frozen recipe; two bounded first-chunk updates only",
        "objective": "existing train_rnn WMSE distillation, structural and next-change losses",
        "forward_scope": "first 1000 rows from each actual fitting sequence, from sequence origin",
        "argv": command,
    }
    return command, recipe_summary


def _filesystem_type(path: Path) -> str:
    result = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    filesystem = result.stdout.strip()
    if not filesystem or filesystem in {"tmpfs", "ramfs"}:
        raise ValueError(f"Inductor cache must be on physical storage (found {filesystem!r})")
    return filesystem


def _latest_json_line(output: str, marker: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        if line.startswith(marker + " "):
            value = json.loads(line[len(marker) + 1 :])
            return value if isinstance(value, dict) else None
    return None


def _actual_profile(args: argparse.Namespace, receipt: dict[str, Any]) -> int:
    project_root = args.project_root.resolve()
    worktree_root = args.worktree_root.resolve()
    if not (worktree_root / ".git").exists() and not (worktree_root / ".git").is_file():
        raise ValueError("worktree root is not a git checkout")
    cache_setting = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not cache_setting:
        raise ValueError("set TORCHINDUCTOR_CACHE_DIR to a physical cache directory")
    cache_path = Path(cache_setting).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    filesystem = _filesystem_type(cache_path)
    recipe_hash = _sha256(project_root / RECIPE_RELATIVE)
    modes = ("a8", "no_a8") if args.mode == "both" else (args.mode,)
    receipt.update(
        {
            "profile_kind": "actual_recipe_first_chunk",
            "experiment_id": EXPERIMENT_ID,
            "recipe_path": str(project_root / RECIPE_RELATIVE),
            "recipe_sha256": recipe_hash,
            "worktree_root": str(worktree_root),
            "inductor_cache_dir": str(cache_path),
            "inductor_cache_filesystem": filesystem,
            "step_cost_bound": {
                "modes": list(modes),
                "optimizer_updates_per_mode": 2,
                "maximum_optimizer_updates": 2 * len(modes),
                "warmup_updates_per_mode": 1,
                "measured_updates_per_mode": 1,
                "causal_rows_per_update": 1000,
            },
            "activation_scope": "post-encoder/layer-input and previous-hidden recurrent matvec A8; fp32 state",
            "excluded": ["head activation A8 blocks", "raw projection and native I16 effects"],
            "quality_status": "not_evaluated",
            "backend_status": "running",
            "modes": {},
        }
    )
    _write_receipt(args.receipt, receipt)
    for mode in modes:
        stage = f"trainer_profile_{mode}"
        command, summary = _recipe_arguments(project_root, worktree_root, mode, args.device)
        environment = os.environ.copy()
        environment["WNN_PROJECT_ROOT"] = str(project_root)
        source_path = str(worktree_root / "src")
        environment["PYTHONPATH"] = ":".join(
            value for value in (source_path, environment.get("PYTHONPATH", "")) if value
        )
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache_path)
        log_path = args.receipt.with_name(f"{args.receipt.stem}_{mode}.log")
        mode_receipt: dict[str, Any] = {
            "activation_mode": mode,
            "backend_status": "running",
            "trainer_recipe": summary,
            "started_utc": datetime.now(UTC).isoformat(),
            "log_path": str(log_path),
        }
        receipt["modes"][mode] = mode_receipt
        _write_receipt(args.receipt, receipt)
        started = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                cwd=project_root,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
            )
            output = result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else "")
            log_path.write_text(output)
            mode_receipt["return_code"] = result.returncode
            mode_receipt["wall_seconds"] = time.perf_counter() - started
            mode_receipt["profile"] = _latest_json_line(result.stdout, "PROFILE")
            mode_receipt["update_evidence"] = _latest_json_line(
                result.stdout, "PROFILE_UPDATE_EVIDENCE"
            )
            mode_receipt["activation_backend_evidence"] = _latest_json_line(
                result.stdout, "PROFILE_ACTIVATION_BACKEND_EVIDENCE"
            )
            mode_receipt["run_provenance"] = _latest_json_line(
                result.stdout, "PROFILE_RUN_PROVENANCE"
            )
            mode_receipt["batch_provenance"] = _latest_json_line(
                result.stdout, "PROFILE_BATCH_PROVENANCE"
            )
            evidence = mode_receipt["activation_backend_evidence"]
            update_evidence = mode_receipt["update_evidence"]
            run_provenance = mode_receipt["run_provenance"]
            valid = (
                result.returncode == 0
                and mode_receipt["profile"] is not None
                and isinstance(update_evidence, dict)
                and update_evidence.get("valid_for_throughput") is True
                and isinstance(evidence, dict)
                and evidence.get("optimizer_updates") == 2
                and evidence.get("first_forward_compile_seconds") is not None
                and evidence.get("first_backward_compile_seconds") is not None
                and isinstance(run_provenance, dict)
                and run_provenance.get("initial_checkpoint_sha256") is not None
                and run_provenance.get("normalization_identity") is not None
                and run_provenance.get("source_sha256") is not None
                and mode_receipt["batch_provenance"] is not None
            )
            mode_receipt["backend_status"] = "profile_complete" if valid else "backend_failure"
            if not valid:
                mode_receipt["failure_stage"] = stage
                mode_receipt["failure_message"] = (
                    "trainer failed or omitted valid bounded update/profile/resource evidence; "
                    "inspect the recorded mode log"
                )
        except Exception as error:
            mode_receipt.update(
                {
                    "backend_status": "backend_failure",
                    "failure_stage": stage,
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        mode_receipt["finished_utc"] = datetime.now(UTC).isoformat()
        _write_receipt(args.receipt, receipt)
    receipt["backend_status"] = (
        "profile_complete"
        if all(mode["backend_status"] == "profile_complete" for mode in receipt["modes"].values())
        else "backend_failure"
    )
    return 0 if receipt["backend_status"] == "profile_complete" else 1


def _synthetic_smoke(args: argparse.Namespace) -> int:
    import torch
    from torch import nn

    from wnn.activation_qat import ActivationQATModel, require_finite_activation_a8
    from wnn.nn import RNNModel
    from wnn.training_qat import configure_qat, qat_weight_replacements

    class Objective(nn.Module):
        def __init__(self, model: RNNModel, quantize_activations: bool) -> None:
            super().__init__()
            self.network = ActivationQATModel(model, quantize_activations)

        def forward(
            self, inputs: torch.Tensor, replacements: dict[str, torch.Tensor]
        ) -> torch.Tensor:
            output, state = self.network(inputs, replacements, False)
            return output.square().mean() + 0.01 * state.square().mean()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for synthetic smoke but unavailable")
    modes = ("a8", "no_a8") if args.mode == "both" else (args.mode,)
    results: dict[str, Any] = {}
    for mode in modes:
        model = RNNModel(112, hidden=64, layers=2).to(device)
        configure_qat(model, "recurrent_w4")
        objective = Objective(model, mode == "a8").to(device)
        compiled = torch.compile(objective, fullgraph=True)
        inputs = torch.randn(2, 32, 112, dtype=torch.float32, device=device)
        require_finite_activation_a8(inputs, "synthetic smoke input")
        started = time.perf_counter()
        loss = compiled(inputs, qat_weight_replacements(model))
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        results[mode] = {
            "backend_status": "synthetic_smoke_complete",
            "compile_forward_backward_seconds": time.perf_counter() - started,
            "loss": float(loss.detach().cpu()),
            "quality_status": "not_evaluated",
            "optimizer_updates": 0,
        }
    print(
        json.dumps(
            {
                "profile_kind": "synthetic_smoke",
                "modes": results,
                "quality_status": "not_evaluated",
                "fit_admission": False,
            }
        ),
        flush=True,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-kind", choices=("actual_recipe", "synthetic_smoke"), required=True
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/home/vlad/development/wunder/wnn_connectome_starterpack"),
    )
    parser.add_argument("--worktree-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=("both", "a8", "no_a8"), default="both")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt: dict[str, Any] = {
        "schema": "activation_aware_qat_profile_v3",
        "source_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            cwd=args.worktree_root,
        ).stdout.strip()
        or None,
        "started_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "quality_status": "not_evaluated",
        "backend_status": "running",
        "modes": {},
    }
    _write_receipt(args.receipt, receipt)
    try:
        if args.profile_kind == "synthetic_smoke":
            sys.path.insert(0, str(args.worktree_root.resolve() / "src"))
        status = (
            _actual_profile(args, receipt)
            if args.profile_kind == "actual_recipe"
            else _synthetic_smoke(args)
        )
        if args.profile_kind == "synthetic_smoke":
            receipt["backend_status"] = (
                "synthetic_smoke_complete" if status == 0 else "backend_failure"
            )
        else:
            receipt["backend_status"] = "profile_complete" if status == 0 else "backend_failure"
    except Exception as error:
        receipt.update(
            {
                "backend_status": "backend_failure",
                "failure_stage": "profile_setup",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        status = 1
    receipt["finished_utc"] = datetime.now(UTC).isoformat()
    _write_receipt(args.receipt, receipt)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
