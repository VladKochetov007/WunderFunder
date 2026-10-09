#!/usr/bin/env python3
"""Independently audit the admitted native GPU FP16 terminal artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[6]
NATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "tpu_gpu_benchmark/native_precision_gpu_admitted_r1"
)
ATTEMPT = NATIVE / "terminal_collect_r1/attempt_001"
OUTPUTS = ATTEMPT / "outputs"
OUT = OUTPUTS / "out"
DISPATCH_SHA256 = "f0750930dc3afcd5573bcc89b0aefe2b50843885b7254358ffa1144a09981231"
PACKET_SHA256 = "52ef723bc159773627d0d89ff6923fd62f8615bf012cc7ed14de148aa54807ac"
CONTROL_REVIEW_SHA256 = "60e9770de432a6b39d3ae41c818d60b2d47b5298f630b33ee3483a0e7fecc7f9"
REVIEW_JSON = NATIVE / "NATIVE_FP16_TERMINAL_REVIEW_LUNA_R1.json"
REVIEW_MD = NATIVE / "NATIVE_FP16_TERMINAL_REVIEW_LUNA_R1.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_artifact_records(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    verified = 0
    total_bytes = 0
    for item in records:
        relative = Path(item["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"{label}: unsafe artifact path {relative}")
        path = OUT / relative
        require(path.is_file() and not path.is_symlink(), f"{label}: missing/nonregular artifact {relative}")
        require(path.stat().st_size == int(item["size_bytes"]), f"{label}: size mismatch {relative}")
        require(sha256(path) == item["sha256"], f"{label}: hash mismatch {relative}")
        verified += 1
        total_bytes += path.stat().st_size
    return {"verified_files": verified, "verified_bytes": total_bytes}


def _walk_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _walk_tensors(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_tensors(nested)


def _walk_named(value: Any, key_name: str):
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key == key_name:
                yield nested
            yield from _walk_named(nested, key_name)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_named(nested, key_name)


def load_checkpoint(path: Path) -> dict[str, Any]:
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(checkpoint, dict), f"checkpoint is not a mapping: {path}")
    return checkpoint


def verify_arm(arm_name: str, arm: dict[str, Any], eval_groups: list[int]) -> dict[str, Any]:
    record_summary = verify_artifact_records(arm["artifacts"], arm_name)
    by_path = {item["path"]: item for item in arm["artifacts"]}
    run_dir = OUT / "arms" / arm_name
    provenance_path = run_dir / "model/provenance.json"
    provenance = read_json(provenance_path)
    expected_loss = "wmse" if arm_name == "wmse" else "huber"
    args = provenance["args"]
    require(args.get("optim") == "adamw", f"{arm_name}: optimizer differs from FP32 control");
    require(args.get("loss") == expected_loss, f"{arm_name}: loss differs")
    require(args.get("huber_delta") == 1.0, f"{arm_name}: Huber delta differs")
    require(args.get("precision") == "fp16", f"{arm_name}: trainer did not request FP16 autocast")
    require(args.get("seed") == 20261009 and provenance.get("seed") == 20261009,
            f"{arm_name}: seed differs")
    require(args.get("checkpoint_epoch") == 1 and provenance.get("selected_epoch") == 1,
            f"{arm_name}: selected epoch differs")
    require(provenance.get("split_manifest_sha256") == "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
            f"{arm_name}: fold0 manifest hash differs")
    require(provenance.get("cumulative_updates", {}).get("optimizer_attempts") == 220,
            f"{arm_name}: optimizer attempts differ")
    require(provenance.get("cumulative_updates", {}).get("optimizer_completed_updates") == 220,
            f"{arm_name}: completed update count differs")
    require(provenance.get("cumulative_updates", {}).get("amp_skipped_updates") == 0,
            f"{arm_name}: AMP skipped updates are nonzero")

    model_path = run_dir / "model/model.pt"
    ep1_path = run_dir / "model/model_ep1.pt"
    state_path = run_dir / "model/training_state.pt"
    model = load_checkpoint(model_path)
    ep1 = load_checkpoint(ep1_path)
    require(model.get("epoch") == ep1.get("epoch") == 1, f"{arm_name}: checkpoint epoch mismatch")
    model_state = model.get("model")
    ep1_state = ep1.get("model")
    require(isinstance(model_state, dict) and isinstance(ep1_state, dict),
            f"{arm_name}: checkpoint model map missing")
    require(set(model_state) == set(ep1_state), f"{arm_name}: checkpoint tensor names differ")
    require(all(torch.equal(model_state[name], ep1_state[name]) for name in model_state),
            f"{arm_name}: model.pt and model_ep1.pt tensors differ")
    require(all(torch.isfinite(tensor).all().item() for tensor in _walk_tensors(model_state)),
            f"{arm_name}: nonfinite model tensor")
    training_state = load_checkpoint(state_path)
    require(any(value == 2 for value in _walk_named(training_state, "next_epoch")),
            f"{arm_name}: training state did not advance to next_epoch 2")
    state_tensor_count = 0
    for tensor in _walk_tensors(training_state):
        if tensor.is_floating_point():
            require(torch.isfinite(tensor).all().item(), f"{arm_name}: nonfinite training-state tensor")
            state_tensor_count += 1

    prediction_path = run_dir / "predictions_valid.npy"
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    require(prediction.dtype == np.float32 and prediction.shape == (468, 20_000, 2),
            f"{arm_name}: prediction shape/dtype mismatch")
    require(bool(np.isfinite(prediction).all()), f"{arm_name}: nonfinite prediction output")
    sidecar_path = run_dir / "predictions_valid.json"
    sidecar = read_json(sidecar_path)
    model_sha = by_path[f"arms/{arm_name}/model/model.pt"]["sha256"]
    prediction_sha = by_path[f"arms/{arm_name}/predictions_valid.npy"]["sha256"]
    require(sidecar.get("checkpoint_sha256") == model_sha, f"{arm_name}: sidecar does not bind model.pt")
    require(sidecar.get("prediction_sha256") == prediction_sha, f"{arm_name}: sidecar prediction hash differs")
    require(list(map(int, sidecar.get("row_groups", []))) == eval_groups,
            f"{arm_name}: sidecar membership/order differs from frozen fold0 evaluation")
    require(sidecar.get("epoch") == 1 and sidecar.get("split") == "valid",
            f"{arm_name}: sidecar split/epoch differs")
    require(sidecar.get("inference_precision") == "fp32", f"{arm_name}: output inference precision differs")

    update_path = run_dir / "update_trace.jsonl"
    update_rows = [json.loads(line) for line in update_path.read_text().splitlines() if line.strip()]
    require(len(update_rows) == 220, f"{arm_name}: update trace row count differs")
    require([row.get("attempt_index") for row in update_rows] == list(range(220)),
            f"{arm_name}: update trace attempts are not contiguous")
    require(all(row.get("gradient_finite") is True for row in update_rows),
            f"{arm_name}: nonfinite gradient recorded")
    require(all(row.get("scaler_skipped") is False for row in update_rows),
            f"{arm_name}: scaler skipped an update")
    require(all(row.get("amp_scale_before") == 1024.0 and row.get("amp_scale_after") == 1024.0
                for row in update_rows), f"{arm_name}: AMP scale changed unexpectedly")

    timing = arm["timing"]
    return {
        "artifact_inventory": record_summary,
        "run_name": provenance["args"]["name"],
        "provenance_sha256": by_path[f"arms/{arm_name}/model/provenance.json"]["sha256"],
        "model_sha256": model_sha,
        "model_ep1_sha256": by_path[f"arms/{arm_name}/model/model_ep1.pt"]["sha256"],
        "model_pt_epoch1_tensor_equality": True,
        "model_floating_tensor_count": sum(1 for tensor in _walk_tensors(model_state) if tensor.is_floating_point()),
        "training_state_sha256": by_path[f"arms/{arm_name}/model/training_state.pt"]["sha256"],
        "training_state_next_epoch": 2,
        "training_state_floating_tensor_count_finite": state_tensor_count,
        "prediction": {
            "path": str(prediction_path.relative_to(ROOT)),
            "sha256": prediction_sha,
            "shape": list(prediction.shape),
            "dtype": str(prediction.dtype),
            "finite": True,
            "sidecar_sha256": by_path[f"arms/{arm_name}/predictions_valid.json"]["sha256"],
            "sidecar_checkpoint_target": "model.pt",
            "row_group_count": len(eval_groups),
            "inference_precision": sidecar["inference_precision"],
        },
        "updates": {
            "optimizer_attempts": len(update_rows),
            "completed": 220,
            "skipped": 0,
            "all_gradients_finite": True,
            "amp_scale_min": min(float(row["amp_scale_before"]) for row in update_rows),
            "amp_scale_max": max(float(row["amp_scale_after"]) for row in update_rows),
            "update_trace_sha256": by_path[f"arms/{arm_name}/update_trace.jsonl"]["sha256"],
        },
        "timing": timing,
    }


def main() -> None:
    require(not REVIEW_JSON.exists() and not REVIEW_MD.exists(), "refusing to overwrite native terminal review")
    dispatch_path = ROOT / NATIVE / "ROOT_NATIVE_GPU_FP16_DISPATCH_R1.json"
    dispatch_sha = sha256(dispatch_path)
    require(dispatch_sha == DISPATCH_SHA256, "root native dispatch receipt hash changed")
    dispatch = read_json(dispatch_path)
    require(dispatch.get("packet_sha256") == PACKET_SHA256, "dispatch packet pin changed")
    attempt_result = read_json(ROOT / ATTEMPT / "attempt_result.json")
    provider = read_json(ROOT / ATTEMPT / "provider_status.json")
    final_status = read_json(ROOT / OUTPUTS / "final_status.json")
    require(attempt_result.get("output", {}).get("exit_code") == 0,
            "terminal output collection did not exit successfully")
    require(provider.get("status") == "KernelWorkerStatus.COMPLETE" and not provider.get("failure_message"),
            "provider did not report successful terminal state")
    require(provider.get("kernel") == "vladkochetov07/wnn-native-gpu-fp16-20261009-r1",
            "provider terminal belongs to another kernel")
    require(final_status.get("state") == "complete", "worker terminal status is not complete")

    contract_path = OUT / "training_contract.json"
    contract_sha = sha256(contract_path)
    contract = read_json(contract_path)
    pair_receipt_path = OUT / "gpu_pair_run_receipt.json"
    pair_receipt = read_json(pair_receipt_path)
    require(pair_receipt.get("state") == "complete", "GPU pair worker receipt did not complete")
    require(pair_receipt.get("training_contract_sha256") == contract_sha,
            "GPU pair receipt does not bind the collected training contract")
    require(contract.get("state") == "complete", "training contract is not complete")
    require(contract.get("native_admission", {}).get("root_admission_sha256")
            == "571b43d16bec28eed4ac025de45d0f26e6ad7880e0f56013733b4190c70dcfee",
            "worker contract does not bind the embedded root admission")
    require(contract.get("native_control_binding", {}).get("producer_review_sha256") == CONTROL_REVIEW_SHA256,
            "worker control binding differs from reviewed R5 FP32 source")

    inventory = read_json(OUT / "gpu_inventory.json")
    devices = inventory.get("devices", [])
    require(len(devices) == 2 and all(device.get("name") == "Tesla T4" for device in devices),
            "actual GPU inventory is not two Tesla T4 devices")
    fit_results = pair_receipt.get("fit_results", [])
    require(len(fit_results) == 2 and all(result.get("returncode") == 0 for result in fit_results),
            "one or more paired fit processes failed")
    gate = read_json(OUT / "shared/device_profile_gate.json")
    lifecycle = read_json(OUT / "shared/native_fp16_lifecycle_gate.json")
    native_precision = gate.get("native_precision", {})
    require(gate.get("conditional_fit_headroom", {}).get("state") == "passed",
            "conditional fit headroom gate failed")
    require(lifecycle.get("state") == "passed" and lifecycle.get("lifecycle", {}).get("all_outputs_and_states_finite"),
            "native FP16 lifecycle gate did not pass")
    require(lifecycle.get("lifecycle", {}).get("same_model_reset_is_bitwise_repeatable") is True,
            "native FP16 reset test did not pass")
    require(lifecycle.get("lifecycle", {}).get("serialized_state_reload_is_bitwise_repeatable") is True,
            "native FP16 reload test did not pass")
    require(native_precision.get("compute_precision") == "fp16 autocast"
            and native_precision.get("master_parameter_dtype") == "float32"
            and native_precision.get("optimizer_state_dtype") == "float32",
            "native AMP/master-weight precision contract mismatch")
    require(native_precision.get("profile_attempts") == 20
            and native_precision.get("profile_completed_updates") == 20
            and native_precision.get("profile_skipped_updates") == 0,
            "20-update native precision profile did not complete without skips")
    require(native_precision.get("amp_initial_scale") == 1024.0,
            "native precision scale differs from predeclared 1024")
    require(gate.get("device", {}).get("name") == "Tesla T4",
            "profile did not run on a Tesla T4")

    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split = read_json(split_path)
    eval_groups = list(map(int, split["evaluation"]["valid"]))
    require(len(eval_groups) == 468 and len(set(eval_groups)) == 468,
            "frozen fold0 evaluation membership differs")
    arm_summaries = {
        arm_name: verify_arm(arm_name, arm, eval_groups)
        for arm_name, arm in contract.get("arms", {}).items()
    }
    require(set(arm_summaries) == {"wmse", "huber_d1"}, "training contract does not contain the exact loss pair")
    shared_records = contract.get("artifacts", [])
    shared_summary = verify_artifact_records(shared_records, "shared")

    phase_path = OUTPUTS / "phase.jsonl"
    phases = [json.loads(line) for line in phase_path.read_text().splitlines() if line.strip()]
    require(phases and phases[-1].get("phase") == "worker_complete",
            "last recorded phase is not worker_complete")
    require(phases[-1].get("contract_sha256") == contract_sha,
            "worker_complete phase does not bind training contract")
    result = {
        "schema": "wnn.native-gpu-fp16-terminal-independent-review.r1",
        "created_utc": datetime.now(UTC).isoformat(),
        "reviewer": "Luna/max independent terminal reviewer",
        "verdict": "terminal_integrity_pass",
        "scope": "Provider/worker terminal, actual T4 profile/lifecycle, paired FP16 update traces, FP32 checkpoint/state, prediction artifacts, and source-qualified fold0 output membership; no new fit or provider action.",
        "admission": {
            "dispatch_path": str(dispatch_path.relative_to(ROOT)),
            "dispatch_sha256": dispatch_sha,
            "packet_sha256": dispatch["packet_sha256"],
            "provider_kernel": provider["kernel"],
            "provider_status": provider["status"],
            "provider_observed_utc": provider["observed_utc"],
            "provider_failure_message": provider["failure_message"],
        },
        "terminal_collection": {
            "attempt_result_path": str((ROOT / ATTEMPT / "attempt_result.json").relative_to(ROOT)),
            "attempt_result_sha256": sha256(ROOT / ATTEMPT / "attempt_result.json"),
            "output_collection_exit_code": attempt_result["output"]["exit_code"],
            "output_collection_elapsed_seconds": attempt_result["output"]["elapsed_seconds"],
            "collected_bytes": attempt_result["output"]["bytes_in_monitor_dir"],
            "terminal_state": final_status["state"],
            "training_contract_sha256": contract_sha,
            "worker_pair_receipt_sha256": sha256(pair_receipt_path),
            "phase_count": len(phases),
            "final_phase": phases[-1],
        },
        "actual_device": {
            "profile_device": gate["device"],
            "visible_devices": devices,
            "lifecycle_device": lifecycle["device"],
            "native_profile": native_precision,
            "lifecycle": lifecycle["lifecycle"],
            "conditional_headroom": gate["conditional_fit_headroom"],
            "resource_peak_cgroup_bytes": pair_receipt.get("memory_peak"),
        },
        "paired_fit": {
            "arm_count": 2,
            "fit_process_receipts": fit_results,
            "arms": arm_summaries,
            "shared_artifact_inventory": shared_summary,
            "model": contract["model"],
            "fit_contract": contract["fit"],
        },
        "comparison_limit": "This terminal integrity review has no official metric grade. Fixed fold0 is exposed; R5 FP32 source review pins controls. Report native-vs-control quality only after the frozen native metric grade.",
        "expected_public_score": None,
    }
    REVIEW_JSON.parent.mkdir(parents=True, exist_ok=True)
    json_tmp = REVIEW_JSON.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(json_tmp, REVIEW_JSON)
    md = [
        "# Native GPU FP16 terminal review R1",
        "",
        "Verdict: terminal integrity pass. The admitted Kaggle worker completed on two Tesla T4 devices; its actual FP16 lifecycle gate passed before the pair fit.",
        "",
        "Both WMSE and Huber-d1 arms completed 220/220 updates with zero skipped updates, finite gradient traces, and a fixed GradScaler value of 1024. Model parameters and optimizer accumulators remained FP32. Both epoch-1 checkpoints, resumable states, and FP32 validation prediction arrays are pinned and finite; each prediction sidecar binds the actual `model.pt` checkpoint and the exact 468-group fold0 order.",
        "",
        f"WMSE measured fit wall time: {arm_summaries['wmse']['timing']['fit_process_wall_seconds_including_startup_and_checkpoint_export']:.3f}s; post-warmup median update CUDA event: {arm_summaries['wmse']['timing']['median_update_cuda_event_seconds_after_first_five']:.6f}s; prediction export: {arm_summaries['wmse']['timing']['prediction_export_wall_seconds']:.3f}s.",
        f"Huber-d1 measured fit wall time: {arm_summaries['huber_d1']['timing']['fit_process_wall_seconds_including_startup_and_checkpoint_export']:.3f}s; post-warmup median update CUDA event: {arm_summaries['huber_d1']['timing']['median_update_cuda_event_seconds_after_first_five']:.6f}s; prediction export: {arm_summaries['huber_d1']['timing']['prediction_export_wall_seconds']:.3f}s.",
        "",
        "The separate pre-fit gates recorded 20/20 finite FP32 shape updates and 20/20 native FP16 updates with no skips. The native reset/reload gate passed on the actual T4. Setup, first-update lazy cost, fit wall time, update-event time, prediction export, and collection are kept separate in the JSON receipt.",
        "",
        "This is an artifact and execution-integrity review, not a score or performance win claim. The fold0 panel is exposed, and no public score is inferred.",
        "",
        f"Machine-readable review: `{REVIEW_JSON.relative_to(ROOT)}`.",
    ]
    md_tmp = REVIEW_MD.with_suffix(".md.tmp")
    md_tmp.write_text("\n".join(md) + "\n")
    os.replace(md_tmp, REVIEW_MD)
    print(json.dumps({
        "review_json": str(REVIEW_JSON),
        "review_sha256": sha256(REVIEW_JSON),
        "review_markdown": str(REVIEW_MD),
        "review_markdown_sha256": sha256(REVIEW_MD),
        "verdict": result["verdict"],
        "arms": {name: {"updates": arm["updates"], "prediction": arm["prediction"]} for name, arm in arm_summaries.items()},
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
