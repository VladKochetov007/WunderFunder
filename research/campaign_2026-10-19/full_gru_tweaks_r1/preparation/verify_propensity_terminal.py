"""Verify terminal metadata and checkpoint integrity for the clean F0 prerequisite."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from propensity_terminal_checks import (
    _check_fit_identity,
    _check_model_pair,
    _expect_progress,
    _read_jsonl,
    _summarize_events,
)
from propensity_terminal_support import (
    EXPECTED_BINDING_SHA256,
    EXPECTED_COMMIT,
    EXPECTED_EPOCHS,
    EXPECTED_INVOCATION,
    EXPECTED_RUN_NAME,
    EXPECTED_SPLIT_SHA256,
    EXPECTED_UPDATES_PER_EPOCH,
    MAX_JSON_BYTES,
    MAX_TORCH_BYTES,
    _hash_file,
    _json_file,
    _load_torch_file,
    _numpy_safe_globals,
    _state_tensor_summary,
    _verify_binding,
    run_loader_self_test,
)


def _summary_value(value: object) -> dict[str, object]:
    """Select safe, scalar normalization metadata and omit all fitted values."""
    if not isinstance(value, dict):
        raise ValueError("normalization summary is missing")
    keys = (
        "method",
        "accumulator_dtype",
        "buffer_dtype",
        "count",
        "std_floor",
        "mean_sha256",
        "std_sha256",
    )
    summary = {key: value.get(key) for key in keys}
    if (
        summary["method"] != "ordered_fit_population_welford_float64_v1"
        or summary["accumulator_dtype"] != "float64"
        or summary["buffer_dtype"] != "float32"
        or summary["count"] != 28_100_000
        or summary["std_floor"] != 0.001
        or not all(
            isinstance(summary[key], str)
            and len(summary[key]) == 64
            and all(character in "0123456789abcdef" for character in summary[key])
            for key in ("mean_sha256", "std_sha256")
        )
    ):
        raise ValueError("normalization recipe differs")
    return summary


def _verify_checkpoint_files(
    run_dir: Path, provenance: dict[str, object]
) -> dict[str, dict[str, object]]:
    """Hash every selected and per-epoch checkpoint against final provenance."""
    recorded = provenance.get("checkpoint_sha256")
    names = [*(f"model_ep{epoch}.pt" for epoch in range(4)), "model.pt"]
    if not isinstance(recorded, dict) or set(recorded) != set(names):
        raise ValueError("final provenance checkpoint inventory differs")
    result: dict[str, dict[str, object]] = {}
    for name in names:
        digest, metadata = _hash_file(run_dir / name, MAX_TORCH_BYTES)
        if digest != recorded[name]:
            raise ValueError(f"checkpoint digest differs: {name}")
        result[name] = {"sha256": digest, "bytes": metadata["bytes"]}
    return result


def _check_optimizer_state(
    optimizer: object, torch: object, expected_params: int
) -> dict[str, object]:
    """Require one AdamW group, completed per-parameter steps and finite moments."""
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
        raise ValueError("saved AdamW state is missing")
    groups = optimizer.get("param_groups")
    states = optimizer["state"]
    if not isinstance(groups, list) or len(groups) != 1 or len(states) != expected_params:
        raise ValueError("saved AdamW parameter inventory differs")
    group = groups[0]
    if (
        group.get("lr") != 0.002
        or group.get("weight_decay") != 0.01
        or len(group.get("params", [])) != expected_params
    ):
        raise ValueError("saved AdamW recipe differs")
    for state in states.values():
        step = state.get("step")
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or float(step.item()) != 3520.0:
            raise ValueError("an AdamW parameter did not complete all expected steps")
    metadata = _state_tensor_summary(optimizer["state"], torch, np)
    return {
        "parameter_count": expected_params,
        "step_count_each_parameter": 3520,
        "tensors": metadata,
    }


def verify_terminal_run(
    *,
    run_dir: Path,
    source_root: Path,
    binding_path: Path,
    split_path: Path,
    output_path: Path,
    invocation_id: str,
    native_exit_status: int,
    binding_sha256: str,
    split_sha256: str,
    torch: object,
    np: object,
) -> dict[str, object]:
    """Verify a stopped successful run and write no weights or normalization values."""
    if native_exit_status != 0 or invocation_id != EXPECTED_INVOCATION:
        raise ValueError("native unit must have exited zero for the admitted invocation")
    if binding_sha256 != EXPECTED_BINDING_SHA256 or split_sha256 != EXPECTED_SPLIT_SHA256:
        raise ValueError("caller-supplied expected packet/split hash differs")
    if run_dir.name != EXPECTED_RUN_NAME or run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("run directory identity is invalid")
    binding, actual_binding_sha = _json_file(binding_path)
    if actual_binding_sha != binding_sha256:
        raise ValueError("binding bytes differ from the pinned packet")
    source_summary = _verify_binding(binding, source_root)
    split_doc, actual_split_sha = _json_file(split_path)
    if actual_split_sha != split_sha256 or binding.get("split_sha256") != actual_split_sha:
        raise ValueError("split manifest bytes differ from the pinned fold")
    split_doc["_path"] = str(split_path.resolve())

    provenance, provenance_sha = _json_file(run_dir / "provenance.json")
    history_doc, history_sha = _json_file(run_dir / "history.json")
    if provenance.get("fit_status") != "completed":
        raise ValueError("run provenance is not completed")
    output_files = _verify_checkpoint_files(run_dir, provenance)
    journal_events, journal_sha, journal_bytes = _read_jsonl(run_dir / "progress.jsonl")

    safe_names = run_loader_self_test(torch, np)
    state, state_sha, state_stat = _load_torch_file(
        run_dir / "training_state.pt", torch, allowlist=_numpy_safe_globals(np)
    )
    checkpoint, model_sha, model_stat = _load_torch_file(run_dir / "model.pt", torch)
    if not isinstance(state, dict) or not isinstance(checkpoint, dict):
        raise ValueError("loaded checkpoint payload is not a mapping")
    if (
        state.get("format") != "wnn.strict-propensity-training-state"
        or state.get("version") != 1
        or state.get("boundary") != "completed_epoch"
    ):
        raise ValueError("saved training state format differs")
    if (
        state.get("next_epoch") != 4
        or checkpoint.get("epoch") != 3
        or checkpoint.get("hidden") != 64
        or checkpoint.get("n_in") != 211
    ):
        raise ValueError("terminal checkpoint architecture or epoch differs")
    if state.get("scheduler_state") is not None:
        raise ValueError("unexpected scheduler state in fixed-learning-rate run")
    if history_doc.get("epochs") != state.get("history"):
        raise ValueError("history JSON differs from the durable training state")
    attempt_id = state.get("attempt_id")
    if not isinstance(attempt_id, str):
        raise ValueError("training-state attempt id is missing")

    identity = _check_fit_identity(
        provenance, state, binding, split_doc, actual_split_sha, split_path
    )
    model_tensors = _check_model_pair(checkpoint.get("model"), state.get("model"), torch)
    if len(model_tensors) != 10:
        raise ValueError("PropensityNet tensor inventory differs")
    if checkpoint.get("args", {}).get("name") != EXPECTED_RUN_NAME:
        raise ValueError("selected checkpoint run name differs")
    if checkpoint.get("provenance", {}).get("source_sha256") != provenance.get("source_sha256"):
        raise ValueError("selected checkpoint source provenance differs")

    normalization = _summary_value(provenance.get("normalization"))
    contract = state["contract"]
    if contract.get("normalization_summary") != provenance.get("normalization"):
        raise ValueError("durable normalization summary differs from final provenance")
    for name, digest_key in (("mean", "mean_sha256"), ("std", "std_sha256")):
        tensor_meta = model_tensors[name]
        if tensor_meta["sha256"] != normalization[digest_key] or tensor_meta["shape"] != [211]:
            raise ValueError(f"normalizer metadata differs for {name}")
        identity_meta = contract.get("normalization_identity", {}).get(name, {})
        if identity_meta.get("bytes_sha256") != tensor_meta["sha256"] or identity_meta.get(
            "shape"
        ) != [211]:
            raise ValueError(f"normalizer identity differs for {name}")

    epoch_summaries = []
    for epoch, record in enumerate(state["history"]):
        if record.get("epoch") != epoch or record.get("evaluation") != "withheld":
            raise ValueError("epoch history differs from withheld four-epoch schedule")
        progress = _expect_progress(
            record.get("optimizer_progress"), EXPECTED_UPDATES_PER_EPOCH, f"epoch {epoch}"
        )
        epoch_summaries.append(
            {"epoch": epoch, "evaluation": "withheld", "optimizer_progress": progress}
        )
    total = EXPECTED_EPOCHS * EXPECTED_UPDATES_PER_EPOCH
    state_progress = state.get("progress")
    if not isinstance(state_progress, dict):
        raise ValueError("aggregate training progress is missing")
    aggregate_progress = _expect_progress(provenance.get("optimizer_progress"), total, "aggregate")
    for key in (
        "optimizer_attempts",
        "optimizer_step_calls",
        "optimizer_completed_updates",
        "nonfinite_updates",
    ):
        if state_progress.get(key) != provenance["optimizer_progress"].get(key):
            raise ValueError(f"training-state and provenance progress differ: {key}")
    if (
        state_progress.get("loss_steps") != total
        or state_progress.get("preclip_grad_norm_steps") != total
    ):
        raise ValueError("aggregate optimizer diagnostics do not cover every update")
    checkpoint_model = checkpoint.get("model")
    if not isinstance(checkpoint_model, dict):
        raise ValueError("selected checkpoint model state is missing")
    optimizer_summary = _check_optimizer_state(
        state.get("optimizer"), torch, len(checkpoint_model) - 2
    )
    journal_summary = _summarize_events(
        journal_events, attempt_id, int(state["event_sequence"]), state_sha
    )
    if journal_summary["event_counts"].get("optimizer_update_completed") != total:
        raise ValueError("journal completed-update count differs")
    if journal_bytes > MAX_JSON_BYTES:
        raise ValueError("progress journal exceeded the configured bound")

    model_artifact = {"sha256": model_sha, "bytes": model_stat["bytes"]}
    if output_files["model.pt"] != model_artifact:
        raise ValueError("model checkpoint changed between hash and safe load")
    state_artifact = {"sha256": state_sha, "bytes": state_stat["bytes"]}
    output_files["training_state.pt"] = state_artifact
    output_files["provenance.json"] = {
        "sha256": provenance_sha,
        "bytes": (run_dir / "provenance.json").stat().st_size,
    }
    output_files["history.json"] = {
        "sha256": history_sha,
        "bytes": (run_dir / "history.json").stat().st_size,
    }
    output_files["progress.jsonl"] = {"sha256": journal_sha, "bytes": journal_bytes}
    return {
        "schema": "fg19.propensity-terminal-verification.v1",
        "status": "VERIFIED",
        "scope": "integrity and optimizer accounting only; no inference, evaluation, refit, or quality grade",
        "native_unit": {"invocation_id": invocation_id, "exit_status": native_exit_status},
        "binding": {"sha256": actual_binding_sha, **source_summary},
        "split": {"sha256": actual_split_sha, **identity},
        "run": {
            "name": EXPECTED_RUN_NAME,
            "source_commit": EXPECTED_COMMIT,
            "precision_profile": "strict_fp32_v1",
            "fit_status": "completed",
            "selected_epoch": 3,
            "epochs": EXPECTED_EPOCHS,
            "args": {
                key: provenance["args"][key]
                for key in (
                    "hidden",
                    "epochs",
                    "lr",
                    "seed",
                    "checkpoint_policy",
                    "checkpoint_epoch",
                    "no_eval_during_fit",
                    "loader_workers",
                )
            },
            "normalization": normalization,
        },
        "files": output_files,
        "model": {"hidden": 64, "n_in": 211, "epoch": 3, "tensor_metadata": model_tensors},
        "training_state": {
            "format": state["format"],
            "version": state["version"],
            "boundary": state["boundary"],
            "next_epoch": state["next_epoch"],
            "attempt_id": attempt_id,
            "event_sequence": state["event_sequence"],
            "model_matches_selected_checkpoint": True,
            "optimizer": optimizer_summary,
            "rng_metadata_only": _state_tensor_summary(state.get("rng"), torch, np),
            "loader_metadata_only": _state_tensor_summary(state.get("loader"), torch, np),
        },
        "progress": {
            "aggregate": aggregate_progress,
            "epochs": epoch_summaries,
            "journal": journal_summary,
        },
        "restricted_loader_self_test": {"status": "passed", "allowed_globals": safe_names},
        "verification_runtime": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "cuda_initialized_after_load": bool(torch.cuda.is_initialized()),
            "torch_threads": torch.get_num_threads(),
            "payloads_exported": False,
        },
    }


def _write_create_only(path: Path, report: dict[str, object]) -> None:
    """Create one immutable compact JSON result in an existing directory."""
    if not path.parent.is_dir() or path.parent.is_symlink() or path.exists():
        raise FileExistsError("verification output parent must exist and output must be new")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--invocation-id")
    parser.add_argument("--native-exit-status", type=int)
    parser.add_argument("--expected-binding-sha256")
    parser.add_argument("--expected-split-sha256")
    parser.add_argument(
        "--self-test", action="store_true", help="test restricted NumPy RNG loading only"
    )
    return parser


def main() -> None:
    """Run the isolated self-test or one pinned terminal verification."""
    args = _parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("verification process unexpectedly initialized CUDA")
    if args.self_test:
        result = {"status": "passed", "allowed_globals": run_loader_self_test(torch, np)}
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return
    required = (
        args.run_dir,
        args.source_root,
        args.binding,
        args.split,
        args.output,
        args.invocation_id,
        args.native_exit_status,
        args.expected_binding_sha256,
        args.expected_split_sha256,
    )
    if any(value is None for value in required):
        raise SystemExit(
            "full verification requires run/source/binding/split/output and exact exit/invocation/hash arguments"
        )
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before terminal verification")
    report = verify_terminal_run(
        run_dir=args.run_dir,
        source_root=args.source_root,
        binding_path=args.binding,
        split_path=args.split,
        output_path=args.output,
        invocation_id=args.invocation_id,
        native_exit_status=args.native_exit_status,
        binding_sha256=args.expected_binding_sha256,
        split_sha256=args.expected_split_sha256,
        torch=torch,
        np=np,
    )
    if report["verification_runtime"]["cuda_initialized_after_load"]:
        raise RuntimeError("terminal verification initialized CUDA")
    _write_create_only(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
