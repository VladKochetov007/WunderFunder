from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from kaggle_common import write_json


class _TimedLog:
    """Write a stage log and timestamp the trainer's ready-to-fit marker."""

    def __init__(self, stream: TextIO):
        self._stream = stream
        self._pending = ""
        self.loop_started_monotonic: float | None = None

    def write(self, value: str) -> int:
        self._stream.write(value)
        self._stream.flush()
        self._pending += value
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            if "loader workers " in line and self.loop_started_monotonic is None:
                self.loop_started_monotonic = time.monotonic()
        return len(value)

    def flush(self) -> None:
        self._stream.flush()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_plan(path: Path) -> dict[str, object]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "profile_argv",
        "fit_argv",
        "profile_run_name",
        "fit_run_name",
        "expected",
        "runtime_gate",
        "portable_registry_path",
        "profile_log_path",
        "fit_log_path",
        "gate_receipt_path",
    }
    if set(plan) != required or plan["schema"] != "h230-profile-gated-fit-plan-v1":
        raise ValueError("Unsupported H230 profile-fit plan schema")
    for key in ("profile_argv", "fit_argv"):
        values = plan[key]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(v, str) for v in values)
        ):
            raise ValueError(f"{key} must be a nonempty string argv list")
    return plan


def _verify_raw_inputs(root: Path, registry_path: Path) -> dict[str, object]:
    """Hash the deferred raw inputs once and retain portability's stat-keyed cache."""
    if not registry_path.is_file():
        raise FileNotFoundError(f"Portable input registry is missing: {registry_path}")
    from wnn.portability import verify_portable_source

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    identities: dict[str, object] = {}
    for name in ("datasets/train.parquet", "datasets/valid.parquet"):
        source = root / name
        if not source.is_symlink() or not source.is_file():
            raise ValueError(f"Deferred raw input is not a mounted dataset link: {name}")
        entry = registry["sources"].get(name)
        if entry is None:
            raise ValueError(f"Portable registry lacks a required raw input: {name}")
        verify_portable_source(source, name, entry["original_identity"], registry_path)
        identities[name] = {
            "bytes": source.stat().st_size,
            "sha256": entry["sha256"],
            "status": "verified_before_torch_and_cached_for_fit_contract",
        }
    return identities


def _json_records(log_path: Path) -> dict[str, object]:
    records: dict[str, object] = {"json_lines": []}
    json_lines: list[dict[str, object]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("muon_runtime "):
            records["muon_runtime"] = json.loads(line.removeprefix("muon_runtime "))
        elif line.startswith("GRAD_STATS "):
            records["grad_stats"] = json.loads(line.removeprefix("GRAD_STATS "))
        elif line.startswith("PROFILE "):
            records["profile"] = json.loads(line.removeprefix("PROFILE "))
        elif line.startswith("{"):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                json_lines.append(value)
    records["json_lines"] = json_lines
    return records


def _args_record(records: dict[str, object]) -> dict[str, object]:
    lines = records["json_lines"]
    if not isinstance(lines, list):
        raise ValueError("Profile log has no JSON line collection")
    for item in lines:
        if isinstance(item, dict) and "name" in item and "profile_steps" in item:
            return item
    raise ValueError("Profile log is missing the trainer argv receipt")


def _membership_record(records: dict[str, object]) -> dict[str, object]:
    lines = records["json_lines"]
    if not isinstance(lines, list):
        raise ValueError("Profile log has no JSON line collection")
    for item in lines:
        if isinstance(item, dict) and "source_train_sequences" in item:
            return item
    raise ValueError("Profile log is missing the training membership receipt")


def _finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _finite_tree(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    return False


def _derived_model_shape(trainer: object, args: dict[str, object]) -> dict[str, int]:
    """Derive actual model widths from the frozen trainer constants and profile argv."""
    from wnn.research_features import research_feature_names

    raw_width = trainer.N_FEATURES
    event_width = trainer.N_EVENT_FEATURES
    input_width = (
        raw_width + event_width + len(research_feature_names(str(args["research_features"])))
    )
    input_width += raw_width if args["privileged_mode"] != "none" else 0
    if args["heuristics"]:
        from wnn.heuristics import K

        input_width += len(K)
    input_width += raw_width if args["online_norm"] else 0
    input_width += 1 if args["prop_input"] else 0
    outputs = 2 + 2 * len([value for value in str(args["aux"]).split(",") if value])
    if float(args["aux_struct"]) > 0:
        outputs += trainer.N_SP_AUX + 3 + 4
    if float(args["aux_next"]) > 0:
        horizons = [value for value in str(args["aux_next_k"]).split(",") if value]
        outputs += 2 + 2 * len(horizons)
    if float(args["aux_tier"]) > 0:
        outputs += 4
    if float(args["aux_mask"]) > 0:
        outputs += 1
    if float(args["aux_sign"]) > 0:
        outputs += 1
    if float(args["aux_quote"]) > 0:
        outputs += 2
    if args["distill_family_head"] or float(args["distill_family_weight"]) > 0:
        outputs += 8
    return {
        "raw_feature_width": raw_width,
        "event_feature_width": event_width,
        "model_input_width": input_width,
        "model_output_width": outputs,
    }


def evaluate_profile(
    plan: dict[str, object],
    records: dict[str, object],
    *,
    device_name: str,
    derived_shape: dict[str, int],
    allocator_peak_bytes: int,
    allocator_reserved_peak_bytes: int,
    profile_wall_seconds: float,
    profile_loop_seconds: float,
    raw_preflight_seconds: float,
) -> dict[str, object]:
    """Fail closed on device, shape, gradient, timing, membership or budget drift."""
    expected = plan["expected"]
    runtime_gate = plan["runtime_gate"]
    if not isinstance(expected, dict) or not isinstance(runtime_gate, dict):
        raise ValueError("Profile plan expected/runtime sections must be objects")
    runtime = records.get("muon_runtime")
    stats = records.get("grad_stats")
    profile = records.get("profile")
    if (
        not isinstance(runtime, dict)
        or not isinstance(stats, dict)
        or not isinstance(profile, dict)
    ):
        raise ValueError("Profile output must include Muon runtime, GRAD_STATS and PROFILE records")
    args = _args_record(records)
    membership = _membership_record(records)

    failures: list[str] = []
    if str(expected["device_name_substring"]).lower() not in device_name.lower():
        failures.append("device_name_mismatch")
    if runtime.get("device_type") != "cuda":
        failures.append("device_type_not_cuda")
    if runtime.get("device_capability") != expected["device_capability"]:
        failures.append("device_capability_mismatch")
    if runtime.get("cuda_bf16_supported") is not expected["cuda_bf16_supported"]:
        failures.append("cuda_bf16_support_mismatch")
    if runtime.get("policy") != expected["orthogonalization_policy"]:
        failures.append("orthogonalization_policy_mismatch")
    if runtime.get("selected_dtype") != expected["orthogonalization_dtype"]:
        failures.append("orthogonalization_dtype_mismatch")
    if (
        runtime.get("scope") != expected["muon_scope"]
        or runtime.get("gatewise") != expected["muon_gatewise"]
    ):
        failures.append("muon_geometry_mismatch")

    for key in (
        "name",
        "seed",
        "batch",
        "chunk",
        "epochs",
        "profile_warmup",
        "profile_steps",
        "save_training_state",
    ):
        if args.get(key) != expected["profile_args"][key]:
            failures.append(f"profile_arg_mismatch:{key}")
    if args.get("max_batches") != expected["profile_batch_count"]:
        failures.append("profile_batch_count_mismatch")
    if args.get("device") != "cuda" or args.get("precision") != expected["model_precision"]:
        failures.append("trainer_device_or_precision_mismatch")
    if args.get("optim") != "muon" or args.get("sched") != "onecycle":
        failures.append("optimizer_or_schedule_mismatch")
    if args.get("muon_scope") != expected["muon_scope"]:
        failures.append("muon_scope_mismatch")
    if args.get("muon_gatewise") != expected["muon_gatewise"]:
        failures.append("muon_geometry_mismatch")
    if args.get("muon_lr_mult") != expected["muon_lr_multiplier"]:
        failures.append("muon_lr_multiplier_mismatch")
    if args.get("distill_cache_variant") != expected["distill_cache_variant"]:
        failures.append("distillation_variant_mismatch")
    if args.get("distill_cache_all_data") is not True:
        failures.append("all_data_cache_mode_missing")
    if args.get("include_heldout_valid") is not True or args.get("no_eval_during_fit") is not True:
        failures.append("all_data_no_evaluation_scope_mismatch")

    if membership.get("source_train_sequences") != expected["train_sequences"]:
        failures.append("train_membership_count_mismatch")
    if membership.get("eligible_valid_train_sequences") != expected["valid_sequences"]:
        failures.append("eligible_valid_membership_count_mismatch")
    if membership.get("used_valid_train_sequences") != expected["valid_sequences"]:
        failures.append("used_valid_membership_count_mismatch")
    if membership.get("eval_sequences") != 0:
        failures.append("profile_created_evaluation_loader")
    if derived_shape != expected["model_shape"]:
        failures.append("derived_model_shape_mismatch")
    if allocator_peak_bytes > int(runtime_gate["allocator_memory_bytes"]):
        failures.append("allocator_peak_exceeds_profile_cap")
    if allocator_reserved_peak_bytes > int(runtime_gate["allocator_memory_bytes"]):
        failures.append("allocator_reserved_peak_exceeds_profile_cap")

    if stats.get("steps") != expected["gradient_observations"]:
        failures.append("gradient_observation_count_mismatch")
    if stats.get("nonfinite_steps") != 0:
        failures.append("nonfinite_gradient_observed")
    for key in ("total_q05_q50_q95", "family_median", "family_sq_share_mean"):
        if key not in stats:
            failures.append(f"gradient_stat_missing:{key}")
    if not _finite_tree(stats):
        failures.append("gradient_stat_nonfinite")
    quantiles = stats.get("total_q05_q50_q95")
    if (
        not isinstance(quantiles, list)
        or len(quantiles) != 3
        or not all(_finite_positive(value) for value in quantiles)
    ):
        failures.append("gradient_norm_quantiles_invalid")
    profile_step_ms = profile.get("step_wall")
    profile_rate = profile.get("seq_per_s")
    if not _finite_positive(profile_step_ms) or not _finite_positive(profile_rate):
        failures.append("timed_batch_interval_missing_or_nonfinite")
    for key, value in profile.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            failures.append(f"profile_metric_nonfinite:{key}")
    if not _finite_positive(profile_wall_seconds) or not _finite_positive(profile_loop_seconds):
        failures.append("profile_wall_measurement_missing")

    total_batches = int(expected["batches_per_epoch"]) * int(expected["epochs"])
    loop_batch_seconds = profile_loop_seconds / int(expected["profile_batch_count"])
    profile_setup_seconds = max(profile_wall_seconds - profile_loop_seconds, 0.0)
    fit_setup_reserve = max(
        profile_setup_seconds, float(runtime_gate["minimum_fit_setup_reserve_seconds"])
    ) + float(runtime_gate["extra_fit_contract_reserve_seconds"])
    projected_seconds = (
        raw_preflight_seconds
        + profile_wall_seconds
        + fit_setup_reserve
        + loop_batch_seconds * total_batches * float(runtime_gate["timing_margin_multiplier"])
        + float(runtime_gate["checkpoint_reserve_seconds"])
    )
    if projected_seconds > int(runtime_gate["worker_timeout_seconds"]):
        failures.append("projected_profile_plus_fit_exceeds_worker_timeout")

    return {
        "status": "passed" if not failures else "failed_closed",
        "failures": failures,
        "device_name": device_name,
        "derived_model_shape": derived_shape,
        "allocator_peak_bytes": allocator_peak_bytes,
        "allocator_reserved_peak_bytes": allocator_reserved_peak_bytes,
        "runtime": runtime,
        "profile_args": args,
        "membership": membership,
        "gradient_statistics": stats,
        "profile_metrics": profile,
        "profile_wall_seconds": profile_wall_seconds,
        "profile_loop_seconds": profile_loop_seconds,
        "profile_loop_seconds_per_batch": loop_batch_seconds,
        "profile_setup_seconds": profile_setup_seconds,
        "fit_setup_reserve_seconds": fit_setup_reserve,
        "raw_preflight_seconds": raw_preflight_seconds,
        "projected_profile_plus_fit_seconds": projected_seconds,
        "projection_method": (
            "raw input hash once + observed profile runtime + repeated profile setup reserve + "
            "observed four-batch loop wall rate times 779 batches by four epochs with 15% margin "
            "+ fixed-checkpoint reserve"
        ),
        "amp_skip_limitation": (
            "The frozen profile branch returns before EpochUpdateCounter.finish and does not emit "
            "an explicit scaler skip count. The gate requires 80 gradient observations and zero "
            "nonfinite gradient observations; it does not claim a directly emitted AMP skip count."
        ),
    }


def _invoke_trainer(module: object, argv: list[str], log_path: Path) -> tuple[float, float | None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    old_argv = sys.argv
    started = time.monotonic()
    sink: _TimedLog | None = None
    try:
        sys.argv = ["scripts/train_rnn.py", *argv]
        with log_path.open("w", encoding="utf-8") as stream:
            sink = _TimedLog(stream)
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                module.main()
    finally:
        sys.argv = old_argv
    return time.monotonic() - started, None if sink is None else sink.loop_started_monotonic


def _validate_fit_output(
    root: Path, run_name: str, expected: dict[str, object]
) -> dict[str, object]:
    run_dir = root / "artifacts/runs" / run_name
    required = ("model_ep3.pt", "model.pt", "provenance.json", "training_state.pt", "history.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Fit completed without required terminal artifacts: {missing}")
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("selected_epoch") != 3 or provenance.get("args", {}).get("epochs") != 4:
        raise ValueError("Fit output does not prove the frozen fixed epoch-3 policy")
    if provenance.get("args", {}).get("name") != run_name:
        raise ValueError("Fit provenance run identity mismatch")
    membership = provenance.get("fit_membership_identity", {})
    if membership.get("train", {}).get("count") != expected["train_sequences"]:
        raise ValueError("Terminal fit train membership differs from the approved recipe")
    if membership.get("valid", {}).get("count") != expected["valid_sequences"]:
        raise ValueError("Terminal fit valid membership differs from the approved recipe")
    hashes = {
        name: {"bytes": (run_dir / name).stat().st_size, "sha256": _hash_file(run_dir / name)}
        for name in required
    }
    recorded_hashes = provenance.get("checkpoint_sha256", {})
    for checkpoint_name in ("model_ep3.pt", "model.pt"):
        if recorded_hashes.get(checkpoint_name) != hashes[checkpoint_name]["sha256"]:
            raise ValueError(f"Provenance checkpoint hash mismatch: {checkpoint_name}")
    return {
        "run_name": run_name,
        "selected_epoch": 3,
        "fit_membership": membership,
        "artifacts": hashes,
        "checkpoint_hashes_match_provenance": True,
    }


def _initialize_cuda_context(torch: Any, expected: dict[str, object]) -> dict[str, object]:
    """Initialize the fresh child CUDA context and fail closed before profiling."""
    cuda = torch.cuda
    if not cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the profile child process")
    if cuda.device_count() != 1:
        raise RuntimeError("the profile child requires exactly one visible CUDA device")

    # CUDA state is process-local: the worker parent cannot initialize this child.
    cuda.init()
    cuda.set_device(0)
    device_name = cuda.get_device_name(0)
    capability = list(cuda.get_device_capability(0))
    bf16_supported = bool(cuda.is_bf16_supported())
    if str(expected["device_name_substring"]).lower() not in device_name.lower():
        raise RuntimeError(f"profile child device differs from the frozen gate: {device_name}")
    if capability != expected["device_capability"]:
        raise RuntimeError("profile child CUDA capability differs from the frozen gate")
    if bf16_supported is not expected["cuda_bf16_supported"]:
        raise RuntimeError("profile child BF16 support differs from the frozen gate")

    cuda.reset_peak_memory_stats(0)
    return {
        "initialized_in_child": True,
        "selected_device": 0,
        "device_name": device_name,
        "device_capability": capability,
        "cuda_bf16_supported": bf16_supported,
        "peak_memory_stats_reset_after_init": True,
    }


def run(plan_path: Path) -> None:
    """Verify deferred raw inputs, profile the exact recipe, then conditionally fit."""
    plan = _read_plan(plan_path)
    root = Path(os.environ["WNN_PROJECT_ROOT"]).resolve()
    receipt_path = root / str(plan["gate_receipt_path"])
    profile_log_path = root / str(plan["profile_log_path"])
    fit_log_path = root / str(plan["fit_log_path"])
    if receipt_path.exists() or profile_log_path.exists() or fit_log_path.exists():
        raise FileExistsError("Profile-gated run destinations already exist")

    raw_started = time.monotonic()
    raw_inputs = _verify_raw_inputs(root, root / str(plan["portable_registry_path"]))
    raw_preflight_seconds = time.monotonic() - raw_started

    sys.path.insert(0, str(root / "scripts"))
    import torch
    import train_rnn

    cuda_context = _initialize_cuda_context(torch, plan["expected"])
    print(json.dumps({"cuda_context_initialization": cuda_context}), flush=True)
    try:
        profile_wall, loop_started = _invoke_trainer(
            train_rnn, plan["profile_argv"], profile_log_path
        )
    except SystemExit as error:
        write_json(
            receipt_path,
            {
                "status": "profile_failed_system_exit",
                "failure": {"type": "SystemExit", "code": error.code},
                "profile_log_path": str(profile_log_path.relative_to(root)),
                "profile_log_sha256": _hash_file(profile_log_path),
                "raw_input_identity": raw_inputs,
            },
        )
        raise
    except Exception as error:
        write_json(
            receipt_path,
            {
                "status": "profile_failed",
                "failure": {"type": type(error).__name__, "message": str(error)},
                "profile_log_path": str(profile_log_path.relative_to(root)),
                "profile_log_sha256": _hash_file(profile_log_path)
                if profile_log_path.is_file()
                else None,
                "raw_input_identity": raw_inputs,
            },
        )
        raise
    records = _json_records(profile_log_path)
    device_name = torch.cuda.get_device_name(0)
    loop_seconds = 0.0 if loop_started is None else max(0.0, time.monotonic() - loop_started)
    args = _args_record(records)
    derived_shape = _derived_model_shape(train_rnn, args)
    gate = evaluate_profile(
        plan,
        records,
        device_name=device_name,
        derived_shape=derived_shape,
        allocator_peak_bytes=torch.cuda.max_memory_allocated(0),
        allocator_reserved_peak_bytes=torch.cuda.max_memory_reserved(0),
        profile_wall_seconds=profile_wall,
        profile_loop_seconds=loop_seconds,
        raw_preflight_seconds=raw_preflight_seconds,
    )
    gate["raw_input_identity"] = raw_inputs
    gate["shape_evidence"] = (
        "Widths are recomputed from the frozen V4 trainer's feature/output constants and the "
        "actual parsed profile argv; profile execution then exercises the corresponding forward "
        "and gradient path for all 80 chunk updates."
    )
    gate["profile_log_path"] = str(profile_log_path.relative_to(root))
    gate["profile_log_sha256"] = _hash_file(profile_log_path)
    gate["fit_started"] = False
    gate["source_manifest_sha256"] = plan["expected"]["source_manifest_sha256"]
    gate["recipe_sha256"] = plan["expected"]["recipe_sha256"]

    if gate["status"] != "passed":
        gate["status"] = "fit_not_started_profile_gate_failed"
        write_json(receipt_path, gate)
        raise RuntimeError(f"H230 profile gate failed: {gate['failures']}")

    if gate["projected_profile_plus_fit_seconds"] > plan["runtime_gate"]["worker_timeout_seconds"]:
        gate["status"] = "fit_not_started_projected_runtime_exceeds_cap"
        write_json(receipt_path, gate)
        return

    torch.cuda.empty_cache()
    gc.collect()
    fit_started = time.monotonic()
    gate["fit_started"] = True
    gate["fit_started_monotonic"] = fit_started
    gate["status"] = "fit_running"
    write_json(receipt_path, gate)
    try:
        fit_wall, _ = _invoke_trainer(train_rnn, plan["fit_argv"], fit_log_path)
        fit = _validate_fit_output(root, str(plan["fit_run_name"]), plan["expected"])
    except SystemExit as error:
        gate["status"] = "fit_failed_system_exit"
        gate["fit_wall_seconds"] = time.monotonic() - fit_started
        gate["failure"] = {"type": "SystemExit", "code": error.code}
        if fit_log_path.exists():
            gate["fit_log_sha256"] = _hash_file(fit_log_path)
        write_json(receipt_path, gate)
        raise
    except Exception as error:
        gate["status"] = "fit_failed"
        gate["fit_wall_seconds"] = time.monotonic() - fit_started
        gate["failure"] = {"type": type(error).__name__, "message": str(error)}
        if fit_log_path.exists():
            gate["fit_log_sha256"] = _hash_file(fit_log_path)
        write_json(receipt_path, gate)
        raise
    gate.update(
        {
            "status": "fit_complete",
            "fit_wall_seconds": fit_wall,
            "fit_log_path": str(fit_log_path.relative_to(root)),
            "fit_log_sha256": _hash_file(fit_log_path),
            "fit_output": fit,
            "fit_started_monotonic": fit_started,
            "fit_finished_monotonic": time.monotonic(),
        }
    )
    write_json(receipt_path, gate)


def main() -> None:
    """Run the profile gate and the root-authorized conditional fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan)


if __name__ == "__main__":
    main()
