"""Terminal integrity check for the admitted clean-F0 propensity prerequisite."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from propensity_terminal_support import (
    EXPECTED_EPOCHS,
    EXPECTED_RUN_NAME,
    EXPECTED_SPLIT_SHA256,
    EXPECTED_UPDATES_PER_EPOCH,
    MAX_JSON_BYTES,
    _group_identity,
    _stable_file,
    _tensor_summary,
)


def _expect_progress(progress: object, expected: int, scope: str) -> dict[str, object]:
    """Fail closed unless update and finite counters match the frozen recipe."""
    if not isinstance(progress, dict):
        raise ValueError(f"{scope} progress is missing")
    counter_keys = (
        "optimizer_attempts",
        "optimizer_step_calls",
        "optimizer_completed_updates",
    )
    if any(progress.get(key) != expected for key in counter_keys):
        raise ValueError(f"{scope} update counts differ from the frozen recipe")
    finite_keys = (
        "nonfinite_updates",
        "nonfinite_inputs",
        "nonfinite_logits",
        "nonfinite_losses",
        "nonfinite_gradients",
        "nonfinite_gradient_norms",
        "nonfinite_weights",
        "nonfinite_optimizer_states",
        "optimizer_rejected_nonfinite",
    )
    if any(progress.get(key, 0) != 0 for key in finite_keys):
        raise ValueError(f"{scope} contains a rejected/non-finite update")
    scalar_keys = (
        "loss_mean",
        "preclip_grad_norm_mean",
        "preclip_grad_norm_max",
        "learning_rate_first",
        "learning_rate_last",
        "learning_rate_min",
        "learning_rate_max",
    )
    scalars = {key: progress.get(key) for key in scalar_keys}
    for key, value in scalars.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{scope} has non-finite scalar metadata: {key}")
    if scalars["learning_rate_first"] != 0.002 or any(
        scalars[key] != 0.002
        for key in ("learning_rate_last", "learning_rate_min", "learning_rate_max")
    ):
        raise ValueError(f"{scope} effective learning rate differs")
    if (
        scalars["preclip_grad_norm_mean"] < 0
        or scalars["preclip_grad_norm_max"] < scalars["preclip_grad_norm_mean"]
    ):
        raise ValueError(f"{scope} gradient norm summary is inconsistent")
    return {
        **{key: progress[key] for key in counter_keys},
        **scalars,
        **{key: progress.get(key, 0) for key in finite_keys},
    }


def _summarize_events(
    events: list[dict[str, object]],
    attempt_id: str,
    state_cursor: int,
    state_sha: str,
    *,
    epochs: int = EXPECTED_EPOCHS,
    per_epoch: int = EXPECTED_UPDATES_PER_EPOCH,
) -> dict[str, object]:
    """Check ordered update, checkpoint and terminal events without retaining payloads."""
    counts: dict[str, int] = {}
    starts: set[tuple[int, int, int]] = set()
    completes: set[tuple[int, int, int]] = set()
    finished_epochs: set[int] = set()
    resume_events: dict[int, dict[str, object]] = {}
    terminal: list[dict[str, object]] = []
    for sequence, event in enumerate(events, 1):
        if event.get("sequence") != sequence or event.get("attempt_id") != attempt_id:
            raise ValueError("progress sequence or attempt identity differs")
        kind = event.get("event")
        if not isinstance(kind, str):
            raise ValueError("progress event name is missing")
        allowed = {
            "run_started",
            "initial_resume_checkpoint_saved",
            "optimizer_update_started",
            "optimizer_update_completed",
            "epoch_completed",
            "resume_checkpoint_saved",
            "run_completed",
        }
        if kind not in allowed:
            raise ValueError(f"unexpected progress event: {kind}")
        counts[kind] = counts.get(kind, 0) + 1
        if kind in {"optimizer_update_started", "optimizer_update_completed"}:
            key = (int(event["epoch"]), int(event["batch"]), int(event["chunk_start"]))
            epoch, batch, chunk_start = key
            if not (
                0 <= epoch < epochs
                and 0 <= batch < 88
                and 0 <= chunk_start < 20_000
                and chunk_start % 2_000 == 0
            ):
                raise ValueError("optimizer update event lies outside the fixed training grid")
            table = starts if kind.endswith("started") else completes
            if key in table:
                raise ValueError("duplicate optimizer update event")
            table.add(key)
            if kind.endswith("completed"):
                for scalar in (event.get("loss"), event.get("preclip_grad_norm")):
                    if not isinstance(scalar, (int, float)) or not math.isfinite(scalar):
                        raise ValueError("completed update has non-finite diagnostics")
                if event.get("learning_rates") != [0.002]:
                    raise ValueError("completed update learning rate differs")
            elif event.get("learning_rates") != [0.002]:
                raise ValueError("started update learning rate differs")
        elif kind == "run_started":
            if event.get("next_epoch") != 0:
                raise ValueError("fresh run did not start at epoch zero")
        elif kind == "initial_resume_checkpoint_saved":
            if (
                event.get("next_epoch") != 0
                or 0 in resume_events
                or not isinstance(event.get("sha256"), str)
            ):
                raise ValueError("initial resume checkpoint boundary is invalid")
            resume_events[0] = event
        elif kind == "epoch_completed":
            epoch = int(event["epoch"])
            if not 0 <= epoch < epochs or epoch in finished_epochs:
                raise ValueError("duplicate or invalid completed epoch event")
            finished_epochs.add(epoch)
        elif kind == "resume_checkpoint_saved":
            next_epoch = int(event["next_epoch"])
            if (
                not 1 <= next_epoch <= epochs
                or next_epoch in resume_events
                or not isinstance(event.get("sha256"), str)
            ):
                raise ValueError("duplicate checkpoint boundary event")
            resume_events[next_epoch] = event
        elif kind == "run_completed":
            if event.get("next_epoch") != epochs:
                raise ValueError("successful terminal event has an invalid epoch cursor")
            terminal.append(event)
    total = epochs * per_epoch
    if len(starts) != total or starts != completes:
        raise ValueError("journal optimizer attempts and completions differ")
    if any(sum(key[0] == epoch for key in starts) != per_epoch for epoch in range(epochs)):
        raise ValueError("journal per-epoch update count differs")
    if finished_epochs != set(range(epochs)) or set(resume_events) != set(range(epochs + 1)):
        raise ValueError("journal is missing an epoch/checkpoint boundary")
    final = resume_events[epochs]
    if final.get("sha256") != state_sha or final.get("sequence") != state_cursor + 1:
        raise ValueError("final resume checkpoint hash/cursor differs")
    if len(terminal) != 1 or terminal[0].get("event") != "run_completed":
        raise ValueError("journal lacks one successful completion event")
    if terminal[0].get("sequence") != final.get("sequence", 0) + 1:
        raise ValueError("completion event does not immediately follow final checkpoint")
    if counts.get("run_started") != 1 or counts.get("initial_resume_checkpoint_saved") != 1:
        raise ValueError("journal initial boundary is incomplete")
    return {"event_count": len(events), "event_counts": counts, "attempt_id": attempt_id}


def _read_jsonl(path: Path) -> tuple[list[dict[str, object]], str, int]:
    """Read bounded JSONL metadata and retain only parsed event objects."""
    raw, digest, metadata = _stable_file(path, MAX_JSON_BYTES)
    if not raw.endswith(b"\n"):
        raise ValueError("progress journal ends with an incomplete line")
    lines = raw.splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError("progress journal contains an empty line")
    events = [json.loads(line) for line in lines]
    if any(not isinstance(event, dict) for event in events):
        raise ValueError("progress journal contains a non-object line")
    return events, digest, metadata["bytes"]


def _check_model_pair(
    model_checkpoint: object, saved_model: object, torch: object
) -> dict[str, object]:
    """Require exact model-state equality and return only per-tensor hashes/metadata."""
    if not isinstance(model_checkpoint, dict) or not isinstance(saved_model, dict):
        raise ValueError("model checkpoint state is missing")
    if set(model_checkpoint) != set(saved_model):
        raise ValueError("checkpoint and resume model parameter keys differ")
    expected_shapes = {
        "mean": [211],
        "std": [211],
        "inp.weight": [64, 211],
        "inp.bias": [64],
        "gru.weight_ih_l0": [192, 64],
        "gru.weight_hh_l0": [192, 64],
        "gru.bias_ih_l0": [192],
        "gru.bias_hh_l0": [192],
        "out.weight": [1, 128],
        "out.bias": [1],
    }
    if set(model_checkpoint) != set(expected_shapes):
        raise ValueError("PropensityNet parameter-name inventory differs")
    summaries: dict[str, object] = {}
    for name in sorted(model_checkpoint):
        first = _tensor_summary(model_checkpoint[name], torch)
        second = _tensor_summary(saved_model[name], torch)
        if (
            first != second
            or first["dtype"] != "torch.float32"
            or first["shape"] != expected_shapes[name]
        ):
            raise ValueError(f"model state differs or is not strict FP32: {name}")
        summaries[name] = first
    return summaries


def _check_fit_identity(
    provenance: dict[str, object],
    state: dict[str, object],
    binding: dict[str, object],
    split_doc: dict[str, object],
    split_sha: str,
    split_path: Path,
) -> dict[str, object]:
    """Validate fixed F0 options, source/split identities and withheld membership."""
    args = provenance.get("args")
    contract = state.get("contract")
    if not isinstance(args, dict) or not isinstance(contract, dict):
        raise ValueError("run arguments or resume contract are missing")
    static = contract.get("static")
    if not isinstance(static, dict):
        raise ValueError("static resume contract is missing")
    expected = {
        "name": EXPECTED_RUN_NAME,
        "hidden": 64,
        "epochs": EXPECTED_EPOCHS,
        "lr": 0.002,
        "seed": 0,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "no_eval_during_fit": True,
        "precision_profile": "strict_fp32_v1",
        "loader_workers": 0,
        "resume": False,
        "heldout_fold": 0,
        "dev_all": False,
    }
    if any(args.get(key) != value for key, value in expected.items()):
        raise ValueError("strict propensity arguments differ from the admitted run")
    if (
        args.get("split_manifest_sha256") != split_sha
        or Path(str(args.get("split_manifest"))).resolve() != split_path.resolve()
    ):
        raise ValueError("run arguments do not identify the admitted split manifest")
    options = static.get("options")
    if not isinstance(options, dict) or any(
        options.get(key) != value for key, value in expected.items() if key != "resume"
    ):
        raise ValueError("resume static options differ from the admitted recipe")
    runtime = static.get("runtime")
    model = static.get("model")
    optimizer = static.get("optimizer")
    loader = static.get("loader")
    if not all(isinstance(value, dict) for value in (runtime, model, optimizer, loader)):
        raise ValueError("resume static runtime/model/optimizer/loader metadata is missing")
    if runtime.get("device") != "cuda" or runtime.get("cuda_device_count") != 1:
        raise ValueError("training state was not captured from the admitted single-GPU runtime")
    if runtime.get("matmul_tf32") is not False or runtime.get("cudnn_tf32") is not False:
        raise ValueError("training runtime did not record TF32 disabled")
    if model != {"class": "PropensityNet", "hidden": 64, "n_in": 211}:
        raise ValueError("static model architecture differs")
    if (
        optimizer.get("class") != "AdamW"
        or optimizer.get("defaults", {}).get("lr") != 0.002
        or optimizer.get("defaults", {}).get("weight_decay") != 0.01
    ):
        raise ValueError("static optimizer recipe differs")
    if (
        loader.get("batch_size") != 16
        or loader.get("drop_last") is not False
        or loader.get("num_workers") != 0
        or loader.get("dataset_length") != 1405
    ):
        raise ValueError("static fit loader differs")
    source = provenance.get("source_sha256")
    if source != binding.get("profile_source_hashes") or static.get("source_sha256") != source:
        raise ValueError("strict propensity source pins differ from the binding")
    if (
        provenance.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA256
        or static.get("split", {}).get("sha256") != split_sha
    ):
        raise ValueError("strict propensity split identity differs")
    if static.get("split", {}).get("fold") != 0 or split_doc.get("fold") != 0:
        raise ValueError("strict propensity run is not F0")
    fit_group_map = provenance.get("fit_row_groups")
    fit_group_hashes = provenance.get("fit_row_groups_sha256")
    if not isinstance(fit_group_map, dict) or not isinstance(fit_group_hashes, dict):
        raise ValueError("recorded fit group membership is missing")
    fit_ids = fit_group_map.get("valid")
    eval_ids = provenance.get("evaluation_row_groups")
    fit_section, eval_section = split_doc.get("fit"), split_doc.get("evaluation")
    if not isinstance(fit_section, dict) or not isinstance(eval_section, dict):
        raise ValueError("split fit/evaluation sections are missing")
    split_fit, split_eval = fit_section.get("valid"), eval_section.get("valid")
    if fit_ids != split_fit or eval_ids != split_eval:
        raise ValueError("recorded fit/evaluation memberships differ from the split")
    fit_identity = _group_identity(fit_ids, np)
    eval_identity = _group_identity(eval_ids, np)
    static_split = static.get("split", {})
    if fit_identity["count"] != 1405 or fit_identity["count"] != static_split.get(
        "fit_valid_count"
    ):
        raise ValueError("strict propensity fit membership count differs")
    if fit_identity["ordered_int64_le_sha256"] != static_split.get(
        "fit_valid_ordered_int64_le_sha256"
    ):
        raise ValueError("strict propensity fit membership digest differs")
    if fit_identity["ordered_int64_le_sha256"] != fit_group_hashes.get("valid"):
        raise ValueError("provenance fit membership digest differs")
    if eval_identity["count"] != static_split.get("evaluation_count") or eval_identity[
        "ordered_int64_le_sha256"
    ] != static_split.get("evaluation_ordered_int64_le_sha256"):
        raise ValueError("strict propensity evaluation membership digest differs")
    if (
        provenance.get("precision_profile") != "strict_fp32_v1"
        or provenance.get("fit_status") != "completed"
        or provenance.get("current_attempt_status") != "completed"
    ):
        raise ValueError("run provenance does not report completed strict fitting")
    attempts = provenance.get("training_attempts")
    if (
        not isinstance(attempts, list)
        or len(attempts) != 1
        or attempts[0].get("attempt_id") != state.get("attempt_id")
        or attempts[0].get("resumed_from_epoch") != 0
        or attempts[0].get("status") != "completed"
    ):
        raise ValueError("run is not the single fresh completed fit attempt")
    if any(record.get("evaluation") != "withheld" for record in state.get("history", [])):
        raise ValueError("run history includes evaluation during fitting")
    if not isinstance(state.get("attempt_id"), str) or provenance.get(
        "current_fit_attempt_id"
    ) != state.get("attempt_id"):
        raise ValueError("terminal training attempt identity differs")
    return {
        "fit_groups": fit_identity,
        "evaluation_groups": eval_identity,
        "fold": 0,
        "evaluation_during_fit": "withheld",
    }
