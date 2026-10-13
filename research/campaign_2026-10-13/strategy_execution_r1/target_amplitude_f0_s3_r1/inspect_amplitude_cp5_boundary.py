"""Validate the frozen amplitude P3 continuation and fixed CP5 artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import torch
from inspect_amplitude_resume_boundary import (
    finite_tensor_summary,
    generator_state_sha256,
    optimizer_step_summary,
    read_bounded_json,
    sha256_file,
    sha256_json,
)


def main() -> int:
    """Validate P3's actual source, resume ancestry, clocks and CP5 checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--source-preflight", type=Path, required=True)
    parser.add_argument("--p2-gate", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--expected-total-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve(strict=True)
    packet_path = args.packet.resolve(strict=True)
    source_path = args.source_preflight.resolve(strict=True)
    p2_gate_path = args.p2_gate.resolve(strict=True)
    packet = read_bounded_json(packet_path, 2_000_000)
    source_preflight = read_bounded_json(source_path, 4_000_000)
    p2_gate = read_bounded_json(p2_gate_path, 2_000_000)
    stage = next(
        (
            stage
            for stage in packet["stages"]
            if stage.get("part") == "p3" and stage.get("attempt_id") == args.attempt_id
        ),
        None,
    )
    if stage is None:
        raise ValueError(f"frozen packet has no P3 stage for {args.attempt_id}")
    expected_next_epoch = stage["inclusive_stop_after_epoch"] + 1
    if expected_next_epoch != 6:
        raise ValueError("frozen P3 stop epoch does not yield the expected CP5 boundary")
    if args.expected_total_steps != packet["fit_schedule"]["total_steps"]:
        raise ValueError("requested schedule horizon differs from the frozen packet")

    runtime = root / "research/campaign_2026-10-06/bandit_continuation/runtime"
    attempt_path = runtime / "attempts" / f"{args.attempt_id}.json"
    attempt = read_bounded_json(attempt_path, 2_000_000)
    if attempt.get("status") != "finished" or attempt.get("phase") != "terminal":
        raise ValueError("P3 resource attempt is not terminal")
    if attempt.get("exit_status") != 0 or attempt.get("failure") is not None:
        raise ValueError("P3 worker did not finish successfully")
    if attempt.get("command") != stage["worker_argv"]:
        raise ValueError("actual P3 worker command differs from the sealed vector")
    packet_hash = sha256_file(packet_path)
    if attempt.get("contract_sha256") != packet_hash:
        raise ValueError("P3 attempt contract hash differs from the frozen packet")

    vector_binding = source_preflight["exact_vectors"][stage["part"]]
    if attempt.get("source_manifest_sha256") != vector_binding["source_manifest_sha256"]:
        raise ValueError("P3 source manifest digest differs from the exact-vector preflight")
    source_hashes = attempt.get("source_sha256")
    if source_hashes != source_preflight["source_manifest_file_hashes"]:
        raise ValueError("P3 source file hashes differ from the sealed preflight")
    source_digest = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if source_digest != attempt["source_manifest_sha256"]:
        raise ValueError("P3 source file map does not produce the recorded manifest digest")
    if sha256_json(attempt["command"]) != vector_binding["argv_sha256"]:
        raise ValueError("P3 worker argv hash differs from the exact-vector preflight")

    p2_state = p2_gate["output_training_state"]
    if p2_gate.get("stage") != "p2" or p2_state.get("next_epoch") != 4:
        raise ValueError("P3 predecessor gate is not the validated P2 epoch-4 boundary")
    p2_state_path = Path(p2_state["path"]).resolve(strict=True)
    p2_state_hash = p2_state["sha256"]
    if sha256_file(p2_state_path) != p2_state_hash:
        raise ValueError("P2 predecessor state no longer matches its validated hash")
    resume_index = attempt["command"].index("--resume-state")
    actual_resume_path = Path(attempt["command"][resume_index + 1]).resolve(strict=True)
    frozen_resume_path = Path(stage["resume_state_input"]).resolve(strict=True)
    if actual_resume_path != frozen_resume_path or actual_resume_path != p2_state_path:
        raise ValueError("P3 resumed from a path other than the validated P2 state")
    actual_resume_hash = sha256_file(actual_resume_path)
    if actual_resume_hash != p2_state_hash:
        raise ValueError("P3 input state hash differs from the validated P2 boundary")

    output_dir = root / "artifacts/runs" / stage["run_name"]
    state_path = output_dir / "training_state.pt"
    provenance_path = output_dir / "provenance.json"
    history_path = output_dir / "history.json"
    checkpoint_path = output_dir / f"model_ep{stage['inclusive_stop_after_epoch']}.pt"
    for required_path in (state_path, provenance_path, history_path, checkpoint_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"required P3 terminal artifact is absent: {required_path}")

    state_size = state_path.stat().st_size
    if state_size > 64_000_000:
        raise ValueError(f"P3 training state exceeds the bounded validator limit: {state_size}")
    provenance = read_bounded_json(provenance_path, 2_000_000)
    history = read_bounded_json(history_path, 1_000_000)
    if int(provenance.get("seed", -1)) != 3:
        raise ValueError("P3 provenance seed differs from the frozen recipe")
    split_hash = source_preflight["source_manifest_file_hashes"].get(
        "/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-09-29/artifacts/splits/fold0.json"
    )
    if provenance.get("split_manifest_sha256") != split_hash:
        raise ValueError("P3 provenance split hash differs from the frozen source preflight")
    initializer_hash = p2_gate["provenance"]["initializer_checkpoint_sha256"]
    if provenance.get("initial_checkpoint_sha256") != initializer_hash:
        raise ValueError("P3's original seed-3 initializer ancestry changed")
    resumed_from = provenance.get("resumed_from")
    if not isinstance(resumed_from, dict) or (
        Path(resumed_from.get("path", "")).resolve() != actual_resume_path
        or resumed_from.get("sha256") != actual_resume_hash
        or resumed_from.get("next_epoch") != p2_state["next_epoch"]
    ):
        raise ValueError("P3 provenance does not attest the validated P2 resume state")

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
    argument_mismatches = {
        key: {"actual": args_used.get(key), "expected": expected_value}
        for key, expected_value in expected_args.items()
        if args_used.get(key) != expected_value
    }
    if argument_mismatches:
        raise ValueError(f"P3 recipe provenance differs from the packet: {argument_mismatches}")
    cache = provenance.get("distillation_cache", {})
    if (
        cache.get("variant") != "homogeneous"
        or cache.get("main_target_scale") != packet["fit_schedule"]["scale"]
        or not cache.get("main_target_scale_applied_before_training_clip")
        or cache.get("metadata_sha256")
        != p2_gate["provenance"]["distillation_cache_metadata_sha256"]
    ):
        raise ValueError("P3 target cache ancestry or pre-clip amplitude differs from P2")

    # This is a locally produced, trusted PyTorch state. Bound its size before loading.
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("format") != "wnn.epoch-training-state" or state.get("version") != 1:
        raise ValueError("P3 state format/version differs from the full-resume format")
    if state.get("boundary") != "epoch" or state.get("next_epoch") != expected_next_epoch:
        raise ValueError("P3 state is not the validated epoch-5 terminal boundary")
    contract = state.get("contract", {})
    options = contract.get("options", {})
    if (
        options.get("epochs") != 6
        or options.get("distill_cache_scale") != packet["fit_schedule"]["scale"]
    ):
        raise ValueError("P3 saved resume contract changed epoch horizon or target scale")
    if contract.get("total_steps") != args.expected_total_steps:
        raise ValueError("P3 saved contract changed the full OneCycle horizon")

    progress = state.get("progress", {})
    updates = progress.get("cumulative_updates", {})
    expected_attempts = args.expected_total_steps
    if (
        updates.get("optimizer_attempts") != expected_attempts
        or progress.get("step_count") != expected_attempts
        or updates.get("completed_batches") != expected_attempts // 10
        or updates.get("optimizer_completed_updates", -1) + updates.get("amp_skipped_updates", -1)
        != expected_attempts
    ):
        raise ValueError("P3 cumulative optimizer clocks do not cover the frozen schedule")
    scheduler = state.get("scheduler", {})
    scheduler_summary = {
        "last_epoch": scheduler.get("last_epoch"),
        "step_count": scheduler.get("_step_count"),
        "total_steps": scheduler.get("total_steps"),
    }
    expected_scheduler = {
        "last_epoch": expected_attempts,
        "step_count": expected_attempts + 1,
        "total_steps": args.expected_total_steps,
    }
    if scheduler_summary != expected_scheduler:
        raise ValueError(f"P3 scheduler state differs from the full schedule: {scheduler_summary}")
    optimizer_summary = optimizer_step_summary(state.get("optimizer", {}))
    if optimizer_summary["missing_step_parameter_ids"]:
        raise ValueError("P3 optimizer has an initialized parameter without a step counter")
    if optimizer_summary["unique_step_values"] != [updates["optimizer_completed_updates"]]:
        raise ValueError("P3 Adam steps differ from completed non-skipped optimizer updates")

    rng = state.get("rng", {})
    cuda_rng = rng.get("cuda", [])
    if not all(key in rng for key in ("python", "numpy", "torch")):
        raise ValueError("P3 state is missing Python, NumPy or CPU Torch RNG state")
    if not isinstance(cuda_rng, list) or len(cuda_rng) != 1:
        raise ValueError("P3 CUDA RNG state count differs from the single-GPU recipe")
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
        raise ValueError(f"P3 loader contract differs from the frozen fit: {loader_mismatches}")
    finite_summary = finite_tensor_summary(state)
    if not finite_summary["all_float_tensors_finite"]:
        raise ValueError("P3 training state contains non-finite floating tensors")

    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != provenance.get("checkpoint_sha256", {}).get(checkpoint_path.name):
        raise ValueError("P3 epoch-5 checkpoint differs from the model provenance hash")
    if not history or history[-1].get("epoch") != stage["inclusive_stop_after_epoch"]:
        raise ValueError("P3 history does not terminate at the frozen epoch-5 checkpoint")
    if [item.get("epoch") for item in history] != list(range(6)):
        raise ValueError("P3 history does not include the full six-epoch lineage")
    if any(item.get("evaluation") != "withheld" for item in history):
        raise ValueError("P3 training history contains evaluation feedback")

    receipt = {
        "schema": "wnn.strategy-amplitude-cp5-boundary.r1",
        "created_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "attempt_id": args.attempt_id,
        "run_name": stage["run_name"],
        "source_packet": {"path": str(packet_path), "sha256": packet_hash},
        "source_preflight": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "source_manifest_sha256": attempt["source_manifest_sha256"],
            "argv_sha256": vector_binding["argv_sha256"],
        },
        "p2_predecessor_gate": {
            "path": str(p2_gate_path),
            "sha256": sha256_file(p2_gate_path),
            "training_state_path": str(p2_state_path),
            "training_state_sha256": p2_state_hash,
            "next_epoch": p2_state["next_epoch"],
        },
        "worker_attempt": {
            "status": attempt["status"],
            "phase": attempt["phase"],
            "exit_status": attempt["exit_status"],
            "started_utc": attempt["started_utc"],
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
            "matches_validated_p2_state": True,
            "trainer_provenance_next_epoch": resumed_from["next_epoch"],
        },
        "training_state": {
            "path": str(state_path),
            "bytes": state_size,
            "sha256": sha256_file(state_path),
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
                "python_present": True,
                "numpy_present": True,
                "torch_state_bytes": int(rng["torch"].numel()),
                "cuda_device_states": len(cuda_rng),
                "cuda_state_bytes": [int(item.numel()) for item in cuda_rng],
            },
            "loader": {
                **expected_loader,
                "sampler_type": loader.get("sampler_type"),
                "generator_mode": (
                    "global CPU RNG"
                    if loader.get("loader_generator") is None
                    and loader.get("sampler_generator") is None
                    else "explicit loader/sampler generator state"
                ),
                "loader_generator_state_sha256": generator_state_sha256(
                    loader.get("loader_generator")
                ),
                "sampler_generator_state_sha256": generator_state_sha256(
                    loader.get("sampler_generator")
                ),
                "sampler_epoch": loader.get("sampler_epoch"),
            },
            **finite_summary,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_hash,
            "matches_provenance": True,
            "fixed_checkpoint_epoch": stage["inclusive_stop_after_epoch"],
        },
        "provenance": {
            "path": str(provenance_path),
            "sha256": sha256_file(provenance_path),
            "initializer_checkpoint_sha256": initializer_hash,
            "split_manifest_sha256": provenance["split_manifest_sha256"],
            "distillation_cache_metadata_sha256": cache["metadata_sha256"],
            "target_scale": cache["main_target_scale"],
            "evaluation_withheld": True,
        },
        "history_epochs": [int(item["epoch"]) for item in history],
        "scientific_score": None,
        "fit_result_only": True,
    }
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing validation receipt: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"receipt": str(output_path), "sha256": sha256_file(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
