"""Validate a terminal amplitude continuation boundary before admitting its successor."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    """Hash a file incrementally to keep artifact reads bounded."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value with the campaign's canonical argv encoding."""
    payload = json.dumps(value, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_bounded_json(path: Path, maximum_bytes: int) -> Any:
    """Read a small JSON receipt only after checking its file size."""
    size = path.stat().st_size
    if size > maximum_bytes:
        raise ValueError(f"JSON receipt exceeds bound: {path} is {size} bytes")
    return json.loads(path.read_text())


def walk_tensors(value: Any, path: str = "state"):
    """Yield nested tensors with stable field paths for finite-value checks."""
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from walk_tensors(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from walk_tensors(item, f"{path}[{index}]")


def finite_tensor_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Check every floating/complex tensor without materializing a flat copy."""
    floating_count = 0
    non_finite: list[str] = []
    for path, tensor in walk_tensors(state):
        if tensor.is_floating_point() or tensor.is_complex():
            floating_count += 1
            if not bool(torch.isfinite(tensor).all().item()):
                non_finite.append(path)
    return {
        "floating_or_complex_tensor_count": floating_count,
        "all_float_tensors_finite": not non_finite,
        "non_finite_tensor_paths": non_finite,
    }


def optimizer_step_summary(optimizer_state: dict[str, Any]) -> dict[str, Any]:
    """Report Adam step counts for each initialized parameter state."""
    states = optimizer_state.get("state")
    if not isinstance(states, dict):
        raise ValueError("optimizer state dictionary is missing its parameter state map")
    step_values: list[int] = []
    missing_steps: list[str] = []
    for parameter_id, values in states.items():
        if not isinstance(values, dict) or "step" not in values:
            missing_steps.append(str(parameter_id))
            continue
        step = values["step"]
        if isinstance(step, torch.Tensor):
            if step.numel() != 1 or not bool(torch.isfinite(step).all().item()):
                raise ValueError(f"invalid optimizer step tensor for parameter {parameter_id}")
            step_values.append(int(step.item()))
        else:
            step_values.append(int(step))
    return {
        "parameter_state_count": len(states),
        "step_values": step_values,
        "unique_step_values": sorted(set(step_values)),
        "missing_step_parameter_ids": missing_steps,
    }


def generator_state_sha256(value: Any) -> str | None:
    """Hash an explicit generator tensor, or preserve the global-RNG None mode."""
    if value is None:
        return None
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
        raise ValueError("loader generator state must be None or a byte tensor")
    payload = value.cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    """Validate actual attempt, artifacts, ancestry and fixed schedule clocks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--source-preflight", type=Path, required=True)
    parser.add_argument("--prior-gate", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--expected-next-epoch", type=int, required=True)
    parser.add_argument("--expected-attempts", type=int, required=True)
    parser.add_argument("--expected-completed-updates", type=int, required=True)
    parser.add_argument("--expected-amp-skips", type=int, required=True)
    parser.add_argument("--expected-total-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve(strict=True)
    packet_path = args.packet.resolve(strict=True)
    source_path = args.source_preflight.resolve(strict=True)
    prior_gate_path = args.prior_gate.resolve(strict=True)
    packet = read_bounded_json(packet_path, 2_000_000)
    source_preflight = read_bounded_json(source_path, 4_000_000)
    prior_gate = read_bounded_json(prior_gate_path, 2_000_000)

    stages = {stage["part"]: stage for stage in packet["stages"]}
    stage = next(
        (candidate for candidate in stages.values() if candidate["attempt_id"] == args.attempt_id),
        None,
    )
    if stage is None:
        raise ValueError(f"attempt is absent from the frozen packet: {args.attempt_id}")
    if args.expected_next_epoch != stage["inclusive_stop_after_epoch"] + 1:
        raise ValueError("requested epoch boundary does not follow the stage stop epoch")

    runtime = root / "research/campaign_2026-10-06/bandit_continuation/runtime"
    attempt_path = runtime / "attempts" / f"{args.attempt_id}.json"
    attempt = read_bounded_json(attempt_path, 2_000_000)
    if attempt.get("status") != "finished" or attempt.get("phase") != "terminal":
        raise ValueError("resource attempt is not a terminal successful worker")
    if attempt.get("exit_status") != 0 or attempt.get("failure") is not None:
        raise ValueError("resource attempt did not finish with exit status 0")
    if attempt.get("command") != stage["worker_argv"]:
        raise ValueError("actual worker command differs from the frozen stage vector")
    if attempt.get("contract_sha256") != hashlib.sha256(packet_path.read_bytes()).hexdigest():
        raise ValueError("attempt contract hash differs from the frozen packet")

    vector_binding = source_preflight["exact_vectors"][stage["part"]]
    if attempt.get("source_manifest_sha256") != vector_binding["source_manifest_sha256"]:
        raise ValueError("actual attempt source manifest differs from the exact preflight")
    manifest = attempt.get("source_sha256")
    expected_manifest = source_preflight["source_manifest_file_hashes"]
    if manifest != expected_manifest:
        raise ValueError("actual attempt source file hashes differ from the sealed preflight")
    manifest_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest_digest != attempt["source_manifest_sha256"]:
        raise ValueError("actual attempt manifest digest does not match its file-hash map")
    if sha256_json(attempt["command"]) != vector_binding["argv_sha256"]:
        raise ValueError("actual attempt worker argv hash differs from the source preflight")

    output_dir = root / "artifacts/runs" / stage["run_name"]
    state_path = output_dir / "training_state.pt"
    provenance_path = output_dir / "provenance.json"
    history_path = output_dir / "history.json"
    checkpoint_path = output_dir / f"model_ep{stage['inclusive_stop_after_epoch']}.pt"
    for required in (state_path, provenance_path, history_path, checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(f"required terminal artifact is absent: {required}")

    state_size = state_path.stat().st_size
    maximum_state_bytes = 64_000_000
    if state_size > maximum_state_bytes:
        raise ValueError(f"training state exceeds the bounded local validator limit: {state_size}")
    state_hash = sha256_file(state_path)
    provenance = read_bounded_json(provenance_path, 2_000_000)
    history = read_bounded_json(history_path, 1_000_000)

    if int(provenance.get("seed", -1)) != 3:
        raise ValueError("provenance seed differs from the frozen seed-3 recipe")
    if provenance.get("split_manifest_sha256") != source_preflight[
        "source_manifest_file_hashes"
    ].get(
        "/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-09-29/artifacts/splits/fold0.json"
    ):
        raise ValueError("provenance split manifest hash differs from the source preflight")
    if (
        provenance.get("initial_checkpoint_sha256")
        != prior_gate["p1"]["training_state"]["initializer_checkpoint_sha256"]
    ):
        raise ValueError(
            "original seed-3 initializer ancestry differs from the validated P1 receipt"
        )

    resume_input = stage.get("resume_state_input")
    if not resume_input:
        raise ValueError("this continuation stage must bind an actual predecessor state")
    resume_arg_index = attempt["command"].index("--resume-state")
    actual_resume_path = Path(attempt["command"][resume_arg_index + 1]).resolve(strict=True)
    expected_resume_path = Path(resume_input).resolve(strict=True)
    if actual_resume_path != expected_resume_path:
        raise ValueError("actual resume-state path differs from the frozen stage contract")
    expected_resume_hash = prior_gate["p1"]["training_state"]["sha256"]
    actual_resume_hash = sha256_file(actual_resume_path)
    if actual_resume_hash != expected_resume_hash:
        raise ValueError("predecessor state hash differs from the validated P1 receipt")
    resumed_from = provenance.get("resumed_from")
    if not isinstance(resumed_from, dict) or (
        Path(resumed_from.get("path", "")).resolve() != expected_resume_path
        or resumed_from.get("sha256") != actual_resume_hash
        or resumed_from.get("next_epoch") != 2
    ):
        raise ValueError("actual trainer provenance does not attest the expected P1 resume")

    args_used = provenance.get("args", {})
    expected_args = {
        "seed": 3,
        "epochs": 6,
        "batch": 16,
        "chunk": 2000,
        "update_rows": 0,
        "precision": "fp16",
        "distill_alpha": 0.9,
        "distill_cache_scale": packet["fit_schedule"]["scale"],
        "aux_struct": 0.3,
        "aux_next": 0.3,
        "valid_fold": 0,
        "train_source": "train+valid",
        "init": "c20261003_combo_s3_f0@ep3",
        "no_eval_during_fit": True,
    }
    mismatched_args = {
        key: {"actual": args_used.get(key), "expected": value}
        for key, value in expected_args.items()
        if args_used.get(key) != value
    }
    if mismatched_args:
        raise ValueError(f"trainer provenance recipe fields differ: {mismatched_args}")
    cache = provenance.get("distillation_cache", {})
    if cache.get("main_target_scale") != packet["fit_schedule"]["scale"]:
        raise ValueError("cache scaling provenance differs from the scale-only treatment")
    if cache.get("variant") != "homogeneous" or not cache.get(
        "main_target_scale_applied_before_training_clip"
    ):
        raise ValueError("target provenance does not establish homogeneous pre-clip scaling")

    # The state file is a trusted, locally generated PyTorch producer artifact; its size
    # is checked before loading, and all tensor inspection stays in this bounded worker.
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("format") != "wnn.epoch-training-state" or state.get("version") != 1:
        raise ValueError("training-state format/version is not the expected full-resume format")
    if state.get("boundary") != "epoch" or state.get("next_epoch") != args.expected_next_epoch:
        raise ValueError("training state is not the required completed epoch boundary")
    contract = state.get("contract", {})
    options = contract.get("options", {})
    if (
        options.get("epochs") != 6
        or options.get("distill_cache_scale") != packet["fit_schedule"]["scale"]
    ):
        raise ValueError("saved resume contract has the wrong schedule horizon or target scale")
    if contract.get("total_steps") != args.expected_total_steps:
        raise ValueError("saved resume contract has the wrong full OneCycle step horizon")
    if state.get("next_epoch") != stage["inclusive_stop_after_epoch"] + 1:
        raise ValueError("training state next_epoch is not exactly after the admitted stage stop")

    progress = state.get("progress", {})
    updates = progress.get("cumulative_updates", {})
    expected_updates = {
        "optimizer_attempts": args.expected_attempts,
        "optimizer_completed_updates": args.expected_completed_updates,
        "amp_skipped_updates": args.expected_amp_skips,
        "completed_batches": args.expected_attempts // 10,
    }
    update_mismatches = {
        key: {"actual": updates.get(key), "expected": value}
        for key, value in expected_updates.items()
        if updates.get(key) != value
    }
    if update_mismatches:
        raise ValueError(f"saved cumulative optimizer clocks differ: {update_mismatches}")
    if progress.get("step_count") != args.expected_attempts:
        raise ValueError("saved attempted-update counter differs from the schedule")

    scheduler = state.get("scheduler", {})
    scheduler_summary = {
        "last_epoch": scheduler.get("last_epoch"),
        "step_count": scheduler.get("_step_count"),
        "total_steps": scheduler.get("total_steps"),
    }
    expected_scheduler = {
        "last_epoch": args.expected_attempts,
        "step_count": args.expected_attempts + 1,
        "total_steps": args.expected_total_steps,
    }
    if scheduler_summary != expected_scheduler:
        raise ValueError(f"scheduler state differs: {scheduler_summary} != {expected_scheduler}")

    optimizer_summary = optimizer_step_summary(state.get("optimizer", {}))
    if optimizer_summary["missing_step_parameter_ids"]:
        raise ValueError("one or more initialized optimizer states lack a step counter")
    if optimizer_summary["unique_step_values"] != [args.expected_completed_updates]:
        raise ValueError("per-parameter Adam steps differ from completed non-skipped updates")

    rng = state.get("rng", {})
    cuda_rng = rng.get("cuda", [])
    if not isinstance(cuda_rng, list) or len(cuda_rng) != 1:
        raise ValueError("saved CUDA RNG state count differs from the single-GPU recipe")
    loader = state.get("loader", {})
    expected_loader = {
        "batch_size": 16,
        "drop_last": True,
        "num_workers": 0,
        "dataset_length": 11940,
    }
    loader_mismatches = {
        key: {"actual": loader.get(key), "expected": value}
        for key, value in expected_loader.items()
        if loader.get(key) != value
    }
    if loader_mismatches:
        raise ValueError(f"saved loader state differs: {loader_mismatches}")
    loader_summary = {
        **expected_loader,
        "sampler_type": loader.get("sampler_type"),
        "generator_mode": (
            "global CPU RNG"
            if loader.get("loader_generator") is None and loader.get("sampler_generator") is None
            else "explicit loader/sampler generator state"
        ),
        "loader_generator_state_sha256": generator_state_sha256(loader.get("loader_generator")),
        "sampler_generator_state_sha256": generator_state_sha256(loader.get("sampler_generator")),
        "sampler_epoch": loader.get("sampler_epoch"),
    }

    finite_summary = finite_tensor_summary(state)
    if not finite_summary["all_float_tensors_finite"]:
        raise ValueError("one or more saved floating tensors are non-finite")
    checkpoint_hash = sha256_file(checkpoint_path)
    expected_checkpoint_hash = provenance.get("checkpoint_sha256", {}).get(checkpoint_path.name)
    if checkpoint_hash != expected_checkpoint_hash:
        raise ValueError("terminal checkpoint file hash differs from its provenance")
    if not history or history[-1].get("epoch") != stage["inclusive_stop_after_epoch"]:
        raise ValueError("history does not end at the actual inclusive stop epoch")
    if any(item.get("evaluation") != "withheld" for item in history):
        raise ValueError(
            "fit history contains evaluation evidence despite the frozen no-eval recipe"
        )

    receipt = {
        "schema": "wnn.strategy-amplitude-resume-boundary.r1",
        "created_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "attempt_id": args.attempt_id,
        "stage": stage["part"],
        "run_name": stage["run_name"],
        "worker_attempt": {
            "status": attempt["status"],
            "phase": attempt["phase"],
            "exit_status": attempt["exit_status"],
            "finished_utc": attempt["finished_utc"],
            "contract_sha256": attempt["contract_sha256"],
            "source_manifest_sha256": attempt["source_manifest_sha256"],
            "argv_sha256": sha256_json(attempt["command"]),
            "worker_peak_ram_bytes": attempt["peak_ram_bytes"],
            "aggregate_peak_ram_bytes": attempt["peak_parent_ram_bytes"],
            "peak_observed_vram_bytes": attempt["peak_observed_vram_bytes"],
            "worker_memory_event_deltas": attempt.get("final_memory_event_deltas"),
            "parent_local_memory_event_deltas": attempt.get("final_parent_memory_event_deltas"),
            "parent_hierarchical_memory_event_deltas": attempt.get(
                "final_parent_hierarchical_memory_event_deltas"
            ),
        },
        "input_resume_state": {
            "path": str(actual_resume_path),
            "sha256": actual_resume_hash,
            "matches_validated_p1_state": True,
            "trainer_provenance_next_epoch": resumed_from["next_epoch"],
        },
        "output_training_state": {
            "path": str(state_path),
            "bytes": state_size,
            "sha256": state_hash,
            "format": state["format"],
            "version": state["version"],
            "boundary": state["boundary"],
            "next_epoch": state["next_epoch"],
            "contract_epochs": options["epochs"],
            "contract_total_steps": contract["total_steps"],
            "progress_step_count": progress["step_count"],
            "cumulative_updates": updates,
            "scheduler": scheduler_summary,
            "optimizer": optimizer_summary,
            "rng": {
                "python_present": "python" in rng,
                "numpy_present": "numpy" in rng,
                "torch_state_bytes": int(rng["torch"].numel()),
                "cuda_device_states": len(cuda_rng),
                "cuda_state_bytes": [int(item.numel()) for item in cuda_rng],
            },
            "loader": loader_summary,
            **finite_summary,
        },
        "terminal_checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_hash,
            "matches_provenance": True,
        },
        "provenance": {
            "path": str(provenance_path),
            "sha256": sha256_file(provenance_path),
            "initializer_checkpoint_sha256": provenance["initial_checkpoint_sha256"],
            "split_manifest_sha256": provenance["split_manifest_sha256"],
            "distillation_cache_metadata_sha256": cache["metadata_sha256"],
            "target_scale": cache["main_target_scale"],
            "evaluation_withheld": True,
        },
        "history_epochs": [int(item["epoch"]) for item in history],
        "successor_stage": "p3",
        "successor_resume_state_path": str(state_path),
        "successor_resume_state_sha256": state_hash,
        "p3_launch_authorized_by_this_receipt": False,
        "fresh_shared_resource_admission_required": True,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing evidence receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
