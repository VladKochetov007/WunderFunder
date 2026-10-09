#!/usr/bin/env python3
"""Run one finite six-arm AdamW cohort against the admitted parsed-sequence cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

CODE_ROOT = Path(
    os.environ.get("WNN_SAMPLE_SOURCE_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
if not (CODE_ROOT / "scripts/loss_pair_gpu_worker.py").is_file():
    CODE_ROOT = (
        Path(__file__).resolve().parents[1]
        / "d02_muon_loss_successor_r5_profile_export_r2"
        / "code"
    )
SCRIPTS = CODE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(CODE_ROOT / "src"))

import loss_pair_gpu_worker as base  # noqa: E402

SPEC_PATH = Path(__file__).with_name("cohort_spec_r1.json")
SPEC_SHA256 = "3c18a981109fb8711227db73ff80e42b530954023dff77bfac73745f6e2c526c"
CONTROL_CONTRACT_SHA256 = "5b63ed39aa6046e9f02878e7dcb4b48982240bffa45ec09b7466dfb9eeac63f7"
CANDIDATE_POOL_SHA256 = "e5d37d2e5b427fee3648bb6cae1aebcd189bf9b4db642a1c07b47c64743935ee"
ROOT_ADMISSION_ENV = "WNN_SAMPLED_ROOT_ADMISSION_JSON"


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha256(path: Path) -> str:
    return base._sha256_file(path)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    """Load the frozen six-arm manifest and reject changes to the finite cohort."""
    if not path.is_file() or _sha256(path) != SPEC_SHA256:
        raise base.WorkerRejected("sampled cohort spec is missing or differs from the reviewed hash")
    spec = _json(path)
    if (
        spec.get("schema") != "wnn.diversity.sampled-cohort-spec.v1"
        or spec.get("format_version") != 1
        or spec.get("candidate_pool", {}).get("sha256") != CANDIDATE_POOL_SHA256
        or spec.get("parsed_cache", {}).get("cache_manifest_sha256")
        != base.EXPECTED_CACHE_MANIFEST_SHA256
        or spec.get("parsed_cache", {}).get("cache_key") != base.EXPECTED_CACHE_KEY
        or spec.get("controls", {}).get("training_contract_sha256") != CONTROL_CONTRACT_SHA256
        or spec.get("parsed_cache", {}).get("source_stage_receipt_sha256")
        != base.EXPECTED_SOURCE_STAGE_RECEIPT_SHA256
        or spec.get("execution", {}).get("worker_wall_seconds") != base.WALL_SECONDS
    ):
        raise base.WorkerRejected("sampled cohort spec identity or limits differ")
    arms = spec.get("arms")
    expected_ids = [
        "R_adamw_huber_d4",
        "Q_adamw_huber_d2",
        "I_gru256_fixedmix",
        "U_fixedmix_huber_d1",
        "J_adamw_wmse_wsd",
        "K_adamw_wmse_sgdr",
    ]
    waves = spec.get("execution", {}).get("waves")
    if (
        not isinstance(arms, list)
        or [row.get("candidate_id") for row in arms] != expected_ids
        or waves != [
            expected_ids[0:2],
            expected_ids[2:4],
            expected_ids[4:6],
        ]
        or spec.get("fit_allowed") is not False
        or spec.get("queue_activation_allowed") is not False
    ):
        raise base.WorkerRejected("sampled cohort rows/waves are not the exact finite six")
    seen_queue: set[str] = set()
    seen_runs: set[str] = set()
    for arm in arms:
        queue_id, run_name = arm.get("queue_id"), arm.get("run_name")
        if (
            not isinstance(queue_id, str)
            or not queue_id.startswith("SAMPLED_R3_")
            or queue_id in seen_queue
            or not isinstance(run_name, str)
            or run_name in seen_runs
            or arm.get("loss") not in {"wmse", "huber"}
            or arm.get("activation") not in {"gelu", "fixed_mix"}
            or arm.get("scheduler") not in {"onecycle", "wsd", "sgdr"}
        ):
            raise base.WorkerRejected("sampled recipe contains a duplicate or unsupported field")
        seen_queue.add(queue_id)
        seen_runs.add(run_name)
    return spec


def build_train_argv(
    arm: dict[str, Any], manifest_path: Path, validation_path: Path, validation_sha256: str
) -> list[str]:
    """Map a reviewed row to the actual R5 trainer CLI without unsupported options."""
    if arm.get("loss") == "huber":
        delta = arm.get("huber_delta")
        if not isinstance(delta, (int, float)) or not math.isfinite(float(delta)) or delta <= 0:
            raise base.WorkerRejected("Huber arm has an invalid delta")
    elif arm.get("loss") != "wmse" or arm.get("huber_delta") is not None:
        raise base.WorkerRejected("WMSE arm unexpectedly declares a Huber delta")
    command = [
        sys.executable,
        str(base.ROOT / "scripts/train_rnn.py"),
        "--name",
        str(arm["run_name"]),
        "--device",
        "cuda",
        "--arch",
        "gru",
        "--cell",
        "gru",
        "--enc",
        "linear",
        "--hidden",
        "256",
        "--layers",
        "2",
        "--train-source",
        "train",
        "--valid-fold",
        "0",
        "--full-valid",
        "--split-manifest",
        str(base.ROOT / base.SPLIT_RELATIVE),
        "--parsed-cache-manifest",
        str(manifest_path),
        "--parsed-cache-validation-receipt",
        str(validation_path),
        "--expected-parsed-cache-manifest-sha256",
        base.EXPECTED_CACHE_MANIFEST_SHA256,
        "--expected-parsed-cache-validation-sha256",
        validation_sha256,
        "--expected-parsed-cache-key",
        base.EXPECTED_CACHE_KEY,
        "--seed",
        "20261009",
        "--batch-order",
        "seeded",
        "--batch",
        "32",
        "--chunk",
        "2000",
        "--epochs",
        "2",
        "--optim",
        "adamw",
        "--warmup-frac",
        "0.05",
        "--beta2",
        "0.999",
        "--head-lr-mult",
        "1.0",
        "--readout-activation",
        str(arm["activation"]),
        "--dropout",
        "0",
        "--sched",
        str(arm["scheduler"]),
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--clip",
        "1.0",
        "--precision",
        "fp32",
        "--loss-norm",
        "chunk",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "1",
        "--no-eval-during-fit",
        "--save-training-state",
        "--grad-stats",
        "--deterministic",
        "--sync-free",
        "--pin-memory",
        "--loss",
        str(arm["loss"]),
    ]
    parameters = arm.get("scheduler_parameters", {})
    if arm["scheduler"] == "wsd":
        command += ["--wsd-decay", str(parameters["decay_share"]), "--wsd-final", str(parameters["final_lr_fraction"])]
    elif arm["scheduler"] == "sgdr":
        command += ["--restarts", str(parameters["restarts"]), "--wsd-final", str(parameters["final_lr_fraction"])]
    elif arm["scheduler"] != "onecycle":
        raise base.WorkerRejected(f"unsupported scheduler: {arm['scheduler']}")
    if arm["loss"] == "huber":
        command += ["--huber-delta", str(float(arm["huber_delta"]))]
    return command


def _same_npz(left: Path, right: Path) -> None:
    import numpy as np

    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise base.WorkerRejected("paired prefit NPZ fields differ")
        for key in a.files:
            if a[key].dtype != b[key].dtype or a[key].shape != b[key].shape or not np.array_equal(a[key], b[key]):
                raise base.WorkerRejected(f"paired prefit tensors differ at {key}")


def verify_prefit_arms(
    arm_dirs: dict[str, Path], spec: dict[str, Any], ready_candidate_ids: set[str] | None = None
) -> None:
    """Require reset model, normalization, and sampler identity within each declared pair."""
    by_candidate = {row["candidate_id"]: row for row in spec["arms"]}
    pair_ids = (("R_adamw_huber_d4", "Q_adamw_huber_d2"),
                ("I_gru256_fixedmix", "U_fixedmix_huber_d1"),
                ("J_adamw_wmse_wsd", "K_adamw_wmse_sgdr"))
    for left_id, right_id in pair_ids:
        if ready_candidate_ids is not None and not {left_id, right_id}.issubset(ready_candidate_ids):
            continue
        left = arm_dirs[by_candidate[left_id]["queue_id"]] / "prefit"
        right = arm_dirs[by_candidate[right_id]["queue_id"]] / "prefit"
        for name in ("initial_state.npz", "normalization.npz"):
            _same_npz(left / name, right / name)
        for name in ("normalization.json", "sample_orders.json"):
            if _json(left / name) != _json(right / name):
                raise base.WorkerRejected(f"paired fresh-state metadata differs: {name}")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _verify_checkpoint_selection(arm_dir: Path, run_name: str) -> dict[str, str]:
    """Bind the exported checkpoint to epoch 1 and verify matching model tensors."""
    import torch

    model_dir = arm_dir / "model"
    selected = model_dir / "model.pt"
    fixed_epoch = model_dir / "model_ep1.pt"
    provenance = _json(model_dir / "provenance.json")
    args = provenance.get("args", {})
    selected_sha, epoch_sha = _sha256(selected), _sha256(fixed_epoch)
    if (
        provenance.get("selected_epoch") != 1
        or args.get("name") != run_name
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != 1
        or provenance.get("checkpoint_sha256", {}).get("model.pt") != selected_sha
        or provenance.get("checkpoint_sha256", {}).get("model_ep1.pt") != epoch_sha
    ):
        raise base.WorkerRejected("provenance does not bind the selected fixed epoch-1 checkpoint")
    selected_checkpoint = torch.load(selected, map_location="cpu", weights_only=False)
    epoch_checkpoint = torch.load(fixed_epoch, map_location="cpu", weights_only=False)
    if selected_checkpoint.get("epoch") != 1 or epoch_checkpoint.get("epoch") != 1:
        raise base.WorkerRejected("selected/epoch checkpoint metadata does not name epoch 1")
    selected_state = selected_checkpoint.get("model")
    epoch_state = epoch_checkpoint.get("model")
    if not isinstance(selected_state, dict) or not isinstance(epoch_state, dict) or set(selected_state) != set(epoch_state):
        raise base.WorkerRejected("selected and fixed-epoch model tensor keys differ")
    for name in selected_state:
        if not torch.equal(selected_state[name], epoch_state[name]):
            raise base.WorkerRejected(f"selected checkpoint tensors differ from fixed epoch 1 at {name}")
    return {"selected_model_sha256": selected_sha, "fixed_epoch_checkpoint_sha256": epoch_sha}


def _process_io_telemetry(label: str) -> dict[str, Any]:
    rows = [
        sample["children"][label]["io"]
        for sample in base._SAMPLES
        if label in sample.get("children", {})
        and isinstance(sample["children"][label].get("io"), dict)
    ]
    if not rows:
        return {"sample_count": 0, "state": "unavailable"}
    first, last = rows[0], rows[-1]
    delta = {
        key: max(0, int(last[key]) - int(first[key]))
        for key in ("rchar", "read_bytes")
        if key in first and key in last
    }
    return {
        "sample_count": len(rows),
        "start": {key: first.get(key) for key in ("rchar", "read_bytes")},
        "last_live_sample": {key: last.get(key) for key in ("rchar", "read_bytes")},
        "delta": delta,
        "interpretation": "process /proc I/O observations during fit; read_bytes is physical I/O, rchar includes cached reads",
    }


def _command_wall_seconds(label: str) -> float:
    rows = [row for row in base._COMMANDS if row.get("label") == label]
    if len(rows) != 1:
        raise base.WorkerRejected(f"expected one process timing record for {label}")
    row = rows[0]
    started, finished = row.get("started_monotonic_seconds"), row.get("finished_monotonic_seconds")
    if not _finite_number(started) or not _finite_number(finished) or finished < started:
        raise base.WorkerRejected(f"process timing record is incomplete for {label}")
    return float(finished) - float(started)


def interpret_schedule_trace(arm: dict[str, Any], raw_path: Path, output_path: Path) -> dict[str, Any]:
    """Add the recipe scheduler identity without rewriting the trainer's raw trace."""
    raw_bytes = raw_path.read_bytes()
    raw_schedule = json.loads(raw_bytes)
    if raw_schedule.get("total_steps") != 220 or len(raw_schedule.get("rows", [])) != 220:
        raise base.WorkerRejected("schedule trace is not complete for all 220 attempts")
    interpretation = {
        "schema": "wnn.diversity.scheduler-trace-interpretation.v1",
        "format_version": 1,
        "scheduler": arm["scheduler"],
        "scheduler_parameters": arm["scheduler_parameters"],
        "producer_schema": raw_schedule.get("schema"),
        "producer_schema_note": (
            "trainer currently labels all loss-pair scheduler traces onecycle; interpret the observed rows using this recipe binding"
            if arm["scheduler"] != "onecycle" else "trainer OneCycle schema confirmed against the exact D01 control trace"
        ),
        "beta1_policy": "OneCycleLR observed per update" if arm["scheduler"] == "onecycle" else "constant AdamW beta1=0.9",
        "rows": raw_schedule["rows"],
        "raw_schedule_trace_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    output_path.write_bytes(_canonical_json(interpretation))
    if raw_path.read_bytes() != raw_bytes:
        raise base.WorkerRejected("raw trainer schedule trace changed while writing interpretation")
    return interpretation


def _expected_nononecycle_rows(arm: dict[str, Any]) -> list[dict[str, Any]]:
    """Exercise the actual trainer scheduler builder with the frozen arm settings."""
    if arm.get("scheduler") not in {"wsd", "sgdr"}:
        raise base.WorkerRejected("curve preflight is only for WSD/SGDR arms")
    from types import SimpleNamespace

    import torch

    from wnn.training_dynamics import build_scheduler

    parameters = arm["scheduler_parameters"]
    args = SimpleNamespace(
        sched=arm["scheduler"],
        warmup_frac=0.05,
        wsd_decay=float(parameters["decay_share"]) if arm["scheduler"] == "wsd" else 0.2,
        wsd_final=float(parameters["final_lr_fraction"]),
        restarts=int(parameters["restarts"]) if arm["scheduler"] == "sgdr" else 1,
        lr=0.001,
        head_lr_mult=1.0,
        optim="adamw",
    )
    parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = build_scheduler(optimizer, args, 220)
    rows = []
    for attempt_index in range(220):
        rows.append(
            {
                "attempt_index": attempt_index,
                "scheduler_step_pre": int(scheduler.last_epoch),
                "optimizer_groups": [
                    {
                        "group_index": group_index,
                        "lr": float(group["lr"]),
                        "beta1": float(group["betas"][0]),
                    }
                    for group_index, group in enumerate(optimizer.param_groups)
                ],
            }
        )
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    if len(rows) != 220 or rows[-1]["scheduler_step_pre"] != 219:
        raise base.WorkerRejected("trainer scheduler builder did not produce 220 pre-update rows")
    return rows


def verify_arm_fit(
    arm: dict[str, Any],
    arm_dir: Path,
    control_schedule: dict[str, Any],
    expected_scheduler_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Check real trainer outputs, update traces, checkpoints and scheduler telemetry."""
    required = (
        "model_ep0.pt",
        "model_ep1.pt",
        "model.pt",
        "training_state.pt",
        "provenance.json",
        "history.json",
        "checkpoint_timing.jsonl",
    )
    missing = [name for name in required if not (arm_dir / "model" / name).is_file()]
    if missing:
        raise base.WorkerRejected(f"fit output is missing checkpoints/provenance: {missing}")
    missing = [
        name
        for name in ("update_trace.jsonl", "schedule_trace.json", "prefit/initial_state.npz",
                     "prefit/normalization.npz", "prefit/normalization.json", "prefit/sample_orders.json",
                     "prefit/profile_inputs.npz")
        if not (arm_dir / name).is_file()
    ]
    if missing:
        raise base.WorkerRejected(f"fit output is missing trace/prefit artifacts: {missing}")
    provenance = _json(arm_dir / "model" / "provenance.json")
    args = provenance.get("args", {})
    expected_args = {
        "name": arm["run_name"],
        "loss": arm["loss"],
        "readout_activation": arm["activation"],
        "sched": arm["scheduler"],
        "optim": "adamw",
        "precision": "fp32",
        "seed": 20261009,
        "epochs": 2,
        "batch": 32,
        "chunk": 2000,
        "hidden": 256,
        "layers": 2,
        "train_source": "train",
        "valid_fold": 0,
        "lr": 0.001,
        "wd": 0.01,
        "warmup_frac": 0.05,
        "beta2": 0.999,
        "clip": 1.0,
        "dropout": 0.0,
        "loss_norm": "chunk",
        "full_valid": True,
        "no_eval_during_fit": True,
        "grad_stats": True,
        "deterministic": True,
        "sync_free": True,
        "pin_memory": True,
        "device": "cuda",
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 1,
    }
    for key, expected in expected_args.items():
        if args.get(key) != expected:
            raise base.WorkerRejected(f"trainer provenance differs for {arm['queue_id']}: args.{key}")
    expected_delta = float(arm["huber_delta"]) if arm["loss"] == "huber" else 0.5
    if args.get("huber_delta") != expected_delta:
        raise base.WorkerRejected(f"trainer Huber delta differs for {arm['queue_id']}")
    if arm["scheduler"] == "wsd" and (
        args.get("wsd_decay") != arm["scheduler_parameters"]["decay_share"]
        or args.get("wsd_final") != arm["scheduler_parameters"]["final_lr_fraction"]
    ):
        raise base.WorkerRejected("WSD scheduler provenance differs from the finite recipe")
    if arm["scheduler"] == "sgdr" and (
        args.get("restarts") != arm["scheduler_parameters"]["restarts"]
        or args.get("wsd_final") != arm["scheduler_parameters"]["final_lr_fraction"]
    ):
        raise base.WorkerRejected("SGDR scheduler provenance differs from the finite recipe")
    diagnostics = provenance.get("update_diagnostics", [])
    attempts = sum(int(row.get("epoch", {}).get("optimizer_attempts", -1)) for row in diagnostics)
    completed = sum(int(row.get("epoch", {}).get("optimizer_completed_updates", -1)) for row in diagnostics)
    skipped = sum(int(row.get("epoch", {}).get("amp_skipped_updates", -1)) for row in diagnostics)
    if attempts != 220 or completed != 220 or skipped != 0 or len(diagnostics) != 2:
        raise base.WorkerRejected(f"fit update counts do not equal 220 complete attempts: {attempts}/{completed}/{skipped}")
    update_rows = [json.loads(line) for line in (arm_dir / "update_trace.jsonl").read_text().splitlines() if line.strip()]
    if len(update_rows) != 220:
        raise base.WorkerRejected(f"fit update trace has {len(update_rows)} rows, expected 220")
    for index, row in enumerate(update_rows):
        if (
            row.get("attempt_index") != index
            or row.get("scaler_skipped") is not False
            or row.get("gradient_finite") is not True
            or not _finite_number(row.get("unclipped_global_grad_norm"))
            or not _finite_number(row.get("elapsed_seconds"))
            or not row.get("optimizer_groups")
            or any(not _finite_number(group.get("lr")) or not _finite_number(group.get("beta1"))
                   for group in row["optimizer_groups"])
        ):
            raise base.WorkerRejected(f"fit update diagnostics are invalid at attempt {index}")
    raw_schedule = _json(arm_dir / "schedule_trace.json")
    if raw_schedule.get("total_steps") != 220 or len(raw_schedule.get("rows", [])) != 220:
        raise base.WorkerRejected("schedule trace is not complete for all 220 attempts")
    for index, row in enumerate(raw_schedule["rows"]):
        if row.get("attempt_index") != index or row.get("scheduler_step_pre") != index:
            raise base.WorkerRejected(f"schedule trace position differs at attempt {index}")
    if arm["scheduler"] == "onecycle":
        if raw_schedule != control_schedule:
            raise base.WorkerRejected("OneCycle pre-update LR/beta1 trace differs from the D01 AdamW control")
    else:
        if expected_scheduler_rows is None or len(expected_scheduler_rows) != 220:
            raise base.WorkerRejected("non-OneCycle arm lacks its actual-builder schedule preflight")
        for index, row in enumerate(raw_schedule["rows"]):
            groups = row.get("optimizer_groups", [])
            if len(groups) != 1 or groups[0].get("group_index") != 0 or groups[0].get("beta1") != 0.9:
                raise base.WorkerRejected(f"{arm['scheduler']} must retain constant AdamW beta1=0.9 at {index}")
            expected = expected_scheduler_rows[index]
            if (
                row.get("attempt_index") != expected["attempt_index"]
                or row.get("scheduler_step_pre") != expected["scheduler_step_pre"]
                or groups[0].get("lr") != expected["optimizer_groups"][0]["lr"]
                or groups[0].get("beta1") != expected["optimizer_groups"][0]["beta1"]
            ):
                raise base.WorkerRejected(
                    f"observed {arm['scheduler']} curve differs from trainer builder at update {index}"
                )
    sidecar_path = arm_dir / "schedule_trace_interpreted.json"
    interpret_schedule_trace(arm, arm_dir / "schedule_trace.json", sidecar_path)
    provenance_path = arm_dir / "model" / "provenance.json"
    update_seconds = [float(row["elapsed_seconds"]) for row in update_rows]
    post_warm = sorted(update_seconds[5:])
    checkpoint_rows = [
        json.loads(line)
        for line in (arm_dir / "model" / "checkpoint_timing.jsonl").read_text().splitlines()
        if line.strip()
    ]
    checkpoint_seconds = [
        float(row["checkpoint_and_state_export_seconds"])
        for row in checkpoint_rows
        if _finite_number(row.get("checkpoint_and_state_export_seconds"))
    ]
    if len(checkpoint_seconds) != 2:
        raise base.WorkerRejected("expected two measured checkpoint/state exports")
    checkpoint_binding = _verify_checkpoint_selection(arm_dir, arm["run_name"])
    return {
        "optimizer_attempts": attempts,
        "optimizer_completed_updates": completed,
        "amp_skipped_updates": skipped,
        "fit_update_cuda_event_seconds_sum": sum(update_seconds),
        "first_update_cuda_event_seconds": update_seconds[0],
        "median_update_cuda_event_seconds_after_first_five": post_warm[len(post_warm) // 2],
        "checkpoint_and_state_export_seconds_sum": sum(checkpoint_seconds),
        "checkpoint_binding": checkpoint_binding,
        "measured_fit_process_io": _process_io_telemetry(f"fit_{arm['queue_id']}"),
        "update_trace": base._artifact(arm_dir / "update_trace.jsonl", base.OUT),
        "schedule_trace_raw": base._artifact(arm_dir / "schedule_trace.json", base.OUT),
        "schedule_trace_interpreted": base._artifact(sidecar_path, base.OUT),
        "provenance": base._artifact(provenance_path, base.OUT),
    }


def verify_prefit_against_control(
    arm_dirs: dict[str, Path],
    arms: list[dict[str, Any]],
    control_roles: dict[str, Path],
    ready_candidate_ids: set[str] | None = None,
) -> None:
    """Require every fresh arm to reproduce D01 initialization, normalization, and order."""
    from wnn.d02_controls import assert_same_npz

    by_candidate = {arm["candidate_id"]: arm for arm in arms}
    for candidate_id, arm in by_candidate.items():
        if ready_candidate_ids is not None and candidate_id not in ready_candidate_ids:
            continue
        prefit = arm_dirs[arm["queue_id"]] / "prefit"
        assert_same_npz(prefit / "initial_state.npz", control_roles["initial_state"])
        assert_same_npz(prefit / "normalization.npz", control_roles["normalization"])
        assert_same_npz(prefit / "profile_inputs.npz", control_roles["profile_inputs"])
        actual = _json(prefit / "sample_orders.json")
        expected = _json(control_roles["sample_orders"])
        for field in (
            "schema",
            "seed",
            "batch_size",
            "drop_last",
            "membership_sha256",
            "epoch_seeds",
            "epochs",
            "evaluation_ids",
        ):
            if actual.get(field) != expected.get(field):
                raise base.WorkerRejected(f"fresh arm differs from D01 sample order at {field}")


def _arm_artifacts(arm_dir: Path) -> list[dict[str, Any]]:
    return [
        base._artifact(path, base.OUT)
        for path in sorted(arm_dir.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def _find_control_contract() -> tuple[dict[str, Any], dict[str, Path], Path]:
    from wnn.d02_controls import load_controls

    paths = [path for path in base._find_files(base.INPUT, "training_contract.json")
             if _sha256(path) == CONTROL_CONTRACT_SHA256]
    if len(paths) != 1:
        raise base.WorkerRejected(f"expected one exact D01 AdamW control contract, found {len(paths)}")
    contract, roles = load_controls(paths[0], CONTROL_CONTRACT_SHA256)
    return contract, roles, paths[0]


def _validate_admission(spec: dict[str, Any], bootstrap: dict[str, Any]) -> tuple[dict[str, Any], int]:
    raw = os.environ.get(ROOT_ADMISSION_ENV)
    if not raw:
        raise base.WorkerRejected("root sampled-cohort admission payload is absent")
    admission = json.loads(raw)
    overlay_sha = bootstrap.get("runner_overlay_manifest_sha256")
    if not isinstance(overlay_sha, str) or len(overlay_sha) != 64:
        raise base.WorkerRejected("boot receipt lacks the exact runner overlay manifest SHA")
    required = {
        "schema": "wnn.sampled-cohort.root-admission.v1",
        "kernel_id": spec["kernel_id"],
        "push_allowed": True,
        "conditional_fit_allowed": True,
        "source_commit": spec["runtime_source"]["source_commit"],
        "source_dataset": spec["runtime_source"]["source_dataset"],
        "candidate_pool_sha256": CANDIDATE_POOL_SHA256,
        "cohort_spec_sha256": SPEC_SHA256,
        "source_code_manifest_sha256": bootstrap.get("code_manifest_sha256"),
        "cache_manifest_sha256": base.EXPECTED_CACHE_MANIFEST_SHA256,
        "cache_key": base.EXPECTED_CACHE_KEY,
        "control_contract_sha256": CONTROL_CONTRACT_SHA256,
        "runner_overlay_manifest_sha256": bootstrap.get("runner_overlay_manifest_sha256"),
        "runner_worker_sha256": _sha256(Path(__file__).resolve()),
        "runner_overlay_dataset": spec["runtime_source"]["runner_overlay_dataset"],
    }
    for key, expected in required.items():
        if admission.get(key) != expected:
            raise base.WorkerRejected(f"root sampled-cohort admission differs at {key}")
    runtime = admission.get("runtime_seconds")
    if type(runtime) is not int or not 0 < runtime <= int(spec["execution"]["worker_wall_seconds"]):
        raise base.WorkerRejected("root runtime exceeds the bounded six-arm worker wall")
    return admission, runtime


def _copy_control_artifacts(control_contract: dict[str, Any], control_roles: dict[str, Path], control_path: Path) -> dict[str, Any]:
    shared = base.OUT / "shared"
    copies: dict[str, Path] = {}
    for role in ("initial_state", "normalization", "sample_orders", "onecycle_trace", "profile_inputs"):
        source = control_roles[role]
        destination = shared / f"d01_control_{source.name}"
        base._copy_required(source, destination)
        copies[role] = destination
    control_copy = shared / "d01_control_training_contract.json"
    base._copy_required(control_path, control_copy)
    return {role: base._artifact(path, base.OUT) for role, path in copies.items()} | {
        "training_contract": base._artifact(control_copy, base.OUT)
    }


def _collect_shared(
    spec: dict[str, Any],
    bootstrap: dict[str, Any],
    source_receipt: dict[str, Any],
    manifest_path: Path,
    runtime_validation: Path,
    runtime_profile: Path,
    control_contract: dict[str, Any],
    control_roles: dict[str, Path],
    control_path: Path,
    arm_dirs: dict[str, Path],
    arm_run_names: dict[str, str],
    fit_results: list[dict[str, Any]],
    prediction_results: dict[str, dict[str, Any]],
    fit_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    shared = base.OUT / "shared"
    first = arm_dirs[spec["arms"][0]["queue_id"]] / "prefit"
    _copy = base._copy_required
    _copy(first / "initial_state.npz", shared / "initial_state.npz")
    _copy(first / "normalization.npz", shared / "normalization.npz")
    _copy(first / "normalization.json", shared / "normalization.json")
    _copy(first / "sample_orders.json", shared / "sample_orders.json")
    _copy(first / "profile_inputs.npz", shared / "profile_inputs.npz")
    _copy(runtime_validation, shared / "cache_validation_runtime.json")
    _copy(runtime_profile, shared / "cache_reader_profile_runtime.json")
    _copy(manifest_path, shared / "cache_manifest.json")
    _copy(manifest_path.parent / "source_verification.json", shared / "source_verification.json")
    _copy(manifest_path.parent.parent / "source_stage_receipt.json", shared / "source_stage_receipt.json")
    fold0_receipt = Path(bootstrap["fold0_asset_validation_receipt"])
    _copy(fold0_receipt, shared / "fold0_assets_validation.json")
    control_roles_out = _copy_control_artifacts(control_contract, control_roles, control_path)

    profile_receipt = {
        "schema": "wnn.loss-pair.profile-inputs-receipt.v1",
        "format_version": 1,
        "source": "first actual fit batch from SAMPLED_R3_HUBER_D4, before update attempt zero",
        "feature_semantics": "112 raw float32 features plus 99 causal event features; not normalized; model applies shared normalization",
        "target_semantics": "raw float32 target labels; trainers clip to [-2,2] and use abs clipped label weights",
        "row_mask_semantics": "row index >= 99; not the official is_scored mask",
        "cache_manifest_sha256": base.EXPECTED_CACHE_MANIFEST_SHA256,
        "cache_key": base.EXPECTED_CACHE_KEY,
        "initial_state_sha256": _sha256(shared / "initial_state.npz"),
        "sample_orders_sha256": _sha256(shared / "sample_orders.json"),
        "profile_inputs_sha256": _sha256(shared / "profile_inputs.npz"),
    }
    base._write_json(shared / "profile_inputs_receipt.json", profile_receipt)
    profile_path = shared / "profile_inputs.npz"
    with __import__("numpy").load(profile_path, allow_pickle=False) as arrays:
        expected = {
            "inputs": ((32, 2000, 211), "<f4"),
            "targets": ((32, 2000, 2), "<f4"),
            "row_mask": ((32, 2000), "|b1"),
        }
        if set(arrays.files) != set(expected):
            raise base.WorkerRejected("actual cohort profile fixture has unexpected NPZ fields")
        for name, (shape, dtype) in expected.items():
            if arrays[name].shape != shape or arrays[name].dtype.str != dtype:
                raise base.WorkerRejected(f"actual cohort profile fixture differs at {name}")

    cache_root = manifest_path.parent
    artifact_paths = {
        "device_profile_gate": shared / "device_profile_gate.json",
        "fold0_asset_validation_receipt": shared / "fold0_assets_validation.json",
        "cache_manifest": shared / "cache_manifest.json",
        "cache_validation_receipt": shared / "cache_validation_runtime.json",
        "cache_reader_profile": shared / "cache_reader_profile_runtime.json",
        "source_stage_receipt": shared / "source_stage_receipt.json",
        "source_verification": shared / "source_verification.json",
        "initial_state": shared / "initial_state.npz",
        "normalization": shared / "normalization.npz",
        "normalization_metadata": shared / "normalization.json",
        "sample_orders": shared / "sample_orders.json",
        "onecycle_trace": shared / "d01_control_onecycle_trace.json",
        "profile_inputs": shared / "profile_inputs.npz",
        "profile_inputs_receipt": shared / "profile_inputs_receipt.json",
        "d01_control_contract": shared / "d01_control_training_contract.json",
        "scheduler_curve_preflight_wsd": shared / "scheduler_curve_SAMPLED_R3_WSD_WMSE.json",
        "scheduler_curve_preflight_sgdr": shared / "scheduler_curve_SAMPLED_R3_SGDR_WMSE.json",
    }
    # A passed R5 real-T4 gate is immutable predecessor evidence; this cohort does not rerun it.
    gate_paths = [
        path for path in base._find_files(base.INPUT, "device_profile_gate.json")
        if path.is_file() and "d02" in str(path).lower()
    ]
    if not gate_paths:
        raise base.WorkerRejected("the reviewed R5 actual-T4 shape gate is not mounted")
    gate_path = next((path for path in gate_paths if _sha256(path) == "a5445694b64988811b3dbf9b1dd5ea50a94ef2b7ee3f02c337c2647dd23d88d9"), None)
    if gate_path is None:
        raise base.WorkerRejected("no mounted R5 actual-T4 gate matches the pinned SHA")
    _copy(gate_path, artifact_paths["device_profile_gate"])
    _copy(control_roles["onecycle_trace"], artifact_paths["onecycle_trace"])
    artifact_roles = {name: base._artifact(path, base.OUT) for name, path in artifact_paths.items()}

    arms: dict[str, Any] = {}
    for arm in spec["arms"]:
        queue_id = arm["queue_id"]
        arm_dir = arm_dirs[queue_id]
        fit_process_wall = _command_wall_seconds(f"fit_{queue_id}")
        measured_fit_wall = next(
            (row.get("wall_seconds") for row in fit_results if row.get("label") == f"fit_{queue_id}"),
            None,
        )
        if measured_fit_wall is not None and abs(float(measured_fit_wall) - fit_process_wall) > 1.0:
            raise base.WorkerRejected(f"fit process timing sources disagree for {queue_id}")
        arms[queue_id] = {
            "queue_id": queue_id,
            "candidate_id": arm["candidate_id"],
            "candidate_recipe_sha256": arm["candidate_recipe_sha256"],
            "parent_candidates": arm["parent_candidates"],
            "run_name": arm_run_names[queue_id],
            "optimizer": "AdamW",
            "loss": arm["loss"],
            "huber_delta": arm["huber_delta"],
            "activation": arm["activation"],
            "scheduler": arm["scheduler"],
            "scheduler_parameters": arm["scheduler_parameters"],
            "updates_attempted": 220,
            "fit_result": next(row for row in fit_results if row["label"] == f"fit_{queue_id}"),
            "prediction_result": prediction_results[queue_id],
            "fit_validation": fit_summaries[queue_id],
            "artifact_roles": {
                "selected_checkpoint": base._artifact(arm_dir / "model" / "model.pt", base.OUT),
                "fixed_epoch_checkpoint": base._artifact(arm_dir / "model" / "model_ep1.pt", base.OUT),
                "resume_state": base._artifact(arm_dir / "model" / "training_state.pt", base.OUT),
                "provenance": base._artifact(arm_dir / "model" / "provenance.json", base.OUT),
                "predictions": base._artifact(arm_dir / "predictions_valid.npy", base.OUT),
                "prediction_sidecar": base._artifact(arm_dir / "predictions_valid.json", base.OUT),
                "prediction_membership": base._artifact(arm_dir / "prediction_membership.json", base.OUT),
                "gradient_diagnostics": fit_summaries[queue_id]["update_trace"],
                "raw_scheduler_trace": fit_summaries[queue_id]["schedule_trace_raw"],
                "interpreted_scheduler_trace": fit_summaries[queue_id]["schedule_trace_interpreted"],
            },
            "timing": {
                "fit_process_wall_seconds_including_import_setup_data_mapping_training_and_checkpoint_export": fit_process_wall,
                "fit_setup_data_access_residual_seconds_including_import_and_loader_setup": max(
                    0.0,
                    fit_process_wall
                    - float(fit_summaries[queue_id]["fit_update_cuda_event_seconds_sum"])
                    - float(fit_summaries[queue_id]["checkpoint_and_state_export_seconds_sum"]),
                ),
                "fit_update_cuda_event_seconds_sum": fit_summaries[queue_id]["fit_update_cuda_event_seconds_sum"],
                "checkpoint_and_state_export_seconds_sum": fit_summaries[queue_id]["checkpoint_and_state_export_seconds_sum"],
                "measured_fit_process_io": fit_summaries[queue_id]["measured_fit_process_io"],
                "prediction_export_wall_seconds": _command_wall_seconds(f"predict_{queue_id}"),
            },
            "artifacts": _arm_artifacts(arm_dir),
        }

    contract = {
        "schema": "wnn.diversity.sampled-training-contract.v1",
        "format_version": 1,
        "state": "complete",
        "parent_campaign": {
            "candidate_pool_sha256": CANDIDATE_POOL_SHA256,
            "sampled_cohort_spec_sha256": SPEC_SHA256,
            "source_commit": spec["runtime_source"]["source_commit"],
            "source_code_manifest_sha256": spec["runtime_source"]["code_manifest_sha256"],
            "bootstrap_receipt_sha256": _sha256(base.OUT / "bootstrap_receipt.json"),
        },
        "parsed_cache": {
            "cache_key": base.EXPECTED_CACHE_KEY,
            "manifest_artifact_name": "shared/cache_manifest.json",
            "manifest_sha256": base.EXPECTED_CACHE_MANIFEST_SHA256,
            "validation_receipt_sha256": _sha256(shared / "cache_validation_runtime.json"),
            "scope": "cohort",
            "parse_once": True,
            "parse_once_semantics": "the pinned CPU producer built the shared cache once; each fit subprocess independently reopens read-only memmaps, so this does not claim one persistent parsed-array process",
            "parse_once_evidence_artifact": "shared/source_stage_receipt.json",
            "parse_once_evidence_sha256": _sha256(shared / "source_stage_receipt.json"),
            "producer_parse_evidence": {
                "kernel": base.SOURCE_KERNEL,
                "receipt_state": source_receipt.get("state"),
                "cache_key": source_receipt.get("cache_key"),
                "build_command": source_receipt.get("build_command"),
                "raw_source_bytes_expected": source_receipt.get("raw_source_bytes_expected"),
                "receipt_sha256": _sha256(shared / "source_stage_receipt.json"),
                "runtime_parse_count_asserted": False,
            },
            "mapping": "each fit subprocess independently maps the same CPU-produced arrays read-only; no per-arm Parquet parse or cache rewrite",
            "mount_path": str(cache_root),
            "membership_sha256": base.EXPECTED_MEMBERSHIP_SHA256,
            "split_manifest_sha256": base.EXPECTED_SPLIT_SHA256,
            "portable_registry_sha256": base.EXPECTED_REGISTRY_SHA256,
            "fit_train_count": 350,
            "fit_valid_count_not_used_for_fit": 28,
            "evaluation_valid_count": 468,
            "array_payload_bytes": 7_732_440_000,
        },
        "source": {
            "dataset": spec["runtime_source"]["source_dataset"],
            "source_commit": spec["runtime_source"]["source_commit"],
            "code_manifest_sha256": spec["runtime_source"]["code_manifest_sha256"],
            "trainer_sha256": spec["runtime_source"]["trainer_sha256"],
            "kernel_id": spec["kernel_id"],
            "fold0_asset_dataset": base.FOLD0_ASSET_DATASET,
            "fold0_asset_manifest_sha256": base.EXPECTED_FOLD0_ASSET_MANIFEST_SHA256,
            "fold0_asset_validation_receipt_sha256": _sha256(shared / "fold0_assets_validation.json"),
            "source_stage_kernel": base.SOURCE_KERNEL,
            "source_stage_receipt_sha256": _sha256(shared / "source_stage_receipt.json"),
        },
        "control": {
            "kernel": spec["controls"]["kernel"],
            "training_contract_sha256": CONTROL_CONTRACT_SHA256,
            "training_contract": control_contract,
            "artifact_roles": control_roles_out,
            "onecycle_policy": "OneCycle rows are checked exactly against this producer; WSD/SGDR traces retain their native observed rows and are labeled by recipe-specific sidecars.",
        },
        "model": {
            "family": "two-layer GRU",
            "input_width": 211,
            "input_layout": {"raw_features": 112, "causal_event_features": 99},
            "hidden_size": 256,
            "layers": 2,
            "outputs": 2,
            "precision": "float32",
            "tf32": False,
        },
        "fit": spec["common_recipe"],
        "arms": arms,
        "artifacts": artifact_roles,
        "runtime": {
            "worker_wall_seconds": time.monotonic() - base._STARTED,
            "commands": base._COMMANDS,
            "resource_sample_count": len(base._SAMPLES),
            "max_cgroup_current_bytes": max(
                (row["memory"].get("cgroup_current_bytes", 0) for row in base._SAMPLES), default=None
            ),
            "minimum_mem_available_bytes": min(
                (row["memory"].get("mem_available_bytes", 0) for row in base._SAMPLES), default=None
            ),
            "maximum_working_output_bytes": max(
                (row.get("working_output_bytes", 0) for row in base._SAMPLES), default=0
            ),
            "scientific_result_claim": "none until independent grading; fold-0 validation membership is historically exposed",
        },
    }
    base._write_json(base.OUT / "training_contract.json", contract)
    return contract


def run_cohort(bootstrap: dict[str, Any], spec_path: Path = SPEC_PATH) -> dict[str, Any]:
    spec = load_spec(spec_path)
    admission, runtime_seconds = _validate_admission(spec, bootstrap)
    base.WALL_SECONDS = int(spec["execution"]["worker_wall_seconds"])
    deadline = base._STARTED + runtime_seconds
    if base.OUT.exists():
        present = {path.name for path in base.OUT.iterdir()}
        if present != {"bootstrap_receipt.json", "logs"}:
            raise base.WorkerRejected("cohort output directory contains files outside the boot receipt")
    else:
        base.OUT.mkdir(parents=True)
        (base.OUT / "logs").mkdir()
    (base.OUT / "arms").mkdir(exist_ok=False)
    (base.OUT / "shared").mkdir(exist_ok=False)

    fold0_receipt = Path(bootstrap.get("fold0_asset_validation_receipt", ""))
    fold0_sha = bootstrap.get("fold0_asset_validation_receipt_sha256")
    if (
        bootstrap.get("fold0_asset_dataset") != base.FOLD0_ASSET_DATASET
        or bootstrap.get("fold0_asset_manifest_sha256") != base.EXPECTED_FOLD0_ASSET_MANIFEST_SHA256
        or not fold0_receipt.is_file()
        or not isinstance(fold0_sha, str)
        or _sha256(fold0_receipt) != fold0_sha
    ):
        raise base.WorkerRejected("boot receipt lacks the hash-bound fold-0 auxiliary validation")
    copied_fold0 = base.OUT / "shared" / "fold0_assets_validation.json"
    base._copy_required(fold0_receipt, copied_fold0)
    base._FOLD0_ASSET_RECEIPT_SHA256 = _sha256(copied_fold0)
    from wnn.fold0_assets import validate_preverified_fold0_assets

    validate_preverified_fold0_assets(
        base.ROOT,
        base.ROOT / base.SPLIT_RELATIVE,
        base.ROOT / base.REGISTRY_RELATIVE,
        copied_fold0,
        base._FOLD0_ASSET_RECEIPT_SHA256,
    )
    base._append_phase("fold0_auxiliary_assets_verified", receipt_sha256=base._FOLD0_ASSET_RECEIPT_SHA256)
    base._append_phase("cohort_worker_start", queue_ids=[row["queue_id"] for row in spec["arms"]], worker_wall_seconds=runtime_seconds)

    manifest_path, source_receipt_path = base._mounted_cache()
    source_receipt = base._validate_source_stage(manifest_path, source_receipt_path)
    cache_manifest = _json(manifest_path)
    if _sha256(manifest_path) != base.EXPECTED_CACHE_MANIFEST_SHA256:
        raise base.WorkerRejected("mounted parsed-cache manifest SHA differs")
    if cache_manifest.get("cache_key") != base.EXPECTED_CACHE_KEY:
        raise base.WorkerRejected("mounted parsed-cache key differs")
    if base._memory_snapshot().get("cgroup_current_bytes") is None:
        raise base.WorkerRejected("cgroup memory is unavailable before cohort resource checks")
    startup_sample = base._sample("cohort_startup_guard")
    startup_reason = base._guard(startup_sample, deadline, startup=True)
    if startup_reason:
        raise base.WorkerRejected(f"cohort startup resource guard failed: {startup_reason}")
    if os.statvfs(base.WORK).f_bavail * os.statvfs(base.WORK).f_frsize < 4_000_000_000:
        raise base.WorkerRejected("working disk is below the four-gigabyte reserve")

    inventory = base._gpu_inventory()
    gate_sha = "a5445694b64988811b3dbf9b1dd5ea50a94ef2b7ee3f02c337c2647dd23d88d9"
    gate_paths = [path for path in base._find_files(base.INPUT, "device_profile_gate.json") if _sha256(path) == gate_sha]
    if len(gate_paths) != 1:
        raise base.WorkerRejected("the exact R5 actual-T4 profile gate is not mounted")
    base._write_json(base.OUT / "gpu_inventory.json", {"schema": "wnn.diversity.sampled-gpu-inventory.v1", "devices": inventory})
    base._append_phase("actual_t4_gate_reused", predecessor_gate_sha256=gate_sha, observed_devices=inventory)

    runtime_validation = base.OUT / "shared" / "cache_validation_runtime.json"
    runtime_profile = base.OUT / "shared" / "cache_reader_profile_runtime.json"
    profile_result = base._wait_one(
        [
            sys.executable,
            str(base.ROOT / "scripts/profile_parsed_sequence_cache.py"),
            "--manifest",
            str(manifest_path),
            "--expected-manifest-sha256",
            base.EXPECTED_CACHE_MANIFEST_SHA256,
            "--validation-receipt",
            str(runtime_validation),
            "--profile-output",
            str(runtime_profile),
            "--per-role",
            "2",
            "--seed",
            "20261009",
        ],
        "cache_profile_verify_once",
        base.OUT / "logs" / "cache_profile_verify_once.log",
        deadline,
        base._environment(),
        poll_seconds=1.0,
    )
    if profile_result.get("returncode") != 0:
        raise base.WorkerRejected("the one cohort cache profile/validation failed")
    validation_sha = _sha256(runtime_validation)
    profile_value = _json(runtime_profile)
    if profile_value.get("cache_key") != base.EXPECTED_CACHE_KEY or profile_value.get("validation_receipt_sha256") != validation_sha:
        raise base.WorkerRejected("cohort cache profile does not bind its validation receipt")

    control_contract, control_roles, control_path = _find_control_contract()
    control_schedule = _json(control_roles["onecycle_trace"])
    if (
        control_schedule.get("schema") != "wnn.loss-pair.onecycle-trace.v1"
        or control_schedule.get("total_steps") != 220
        or len(control_schedule.get("rows", [])) != 220
    ):
        raise base.WorkerRejected("the D01 AdamW OneCycle trace is incomplete")
    control_schedule = {"schema": control_schedule["schema"], "format_version": control_schedule.get("format_version"),
                        "total_steps": control_schedule["total_steps"], "rows": control_schedule["rows"]}
    d01_profile_gate = _json(gate_paths[0])
    if d01_profile_gate.get("state") != "passed":
        raise base.WorkerRejected("the predecessor actual-T4 profile gate did not pass")
    base._copy_required(gate_paths[0], base.OUT / "shared" / "device_profile_gate.json")

    arms = spec["arms"]
    arm_dirs = {row["queue_id"]: base.OUT / "arms" / row["queue_id"] for row in arms}
    for path in arm_dirs.values():
        path.mkdir(parents=True, exist_ok=False)
    run_names = {row["queue_id"]: row["run_name"] for row in arms}
    fit_results: list[dict[str, Any]] = []
    prediction_results: dict[str, dict[str, Any]] = {}
    fit_summaries: dict[str, dict[str, Any]] = {}
    scheduler_preflights: dict[str, list[dict[str, Any]]] = {}
    progress_path = base.OUT / "cohort_progress.json"
    base._write_json(progress_path, {"schema": "wnn.diversity.sampled-cohort-progress.v1", "state": "running", "completed_waves": [], "arm_states": {row["queue_id"]: "pending" for row in arms}})
    id_to_arm = {row["candidate_id"]: row for row in arms}
    for wave_index, wave_ids in enumerate(spec["execution"]["waves"]):
        wave_arms = [id_to_arm[candidate_id] for candidate_id in wave_ids]
        before = base._sample(f"wave_{wave_index + 1}_startup")
        reason = base._guard(before, deadline, startup=True)
        if reason:
            raise base.WorkerRejected(f"wave startup resource guard failed: {reason}")
        for arm in wave_arms:
            if arm["scheduler"] not in {"wsd", "sgdr"}:
                continue
            expected_rows = _expected_nononecycle_rows(arm)
            scheduler_preflights[arm["queue_id"]] = expected_rows
            preflight_path = base.OUT / "shared" / f"scheduler_curve_{arm['queue_id']}.json"
            base._write_json(
                preflight_path,
                {
                    "schema": "wnn.diversity.scheduler-curve-preflight.v1",
                    "format_version": 1,
                    "queue_id": arm["queue_id"],
                    "candidate_id": arm["candidate_id"],
                    "scheduler": arm["scheduler"],
                    "scheduler_parameters": arm["scheduler_parameters"],
                    "optimizer": "AdamW",
                    "learning_rate": 0.001,
                    "beta1": 0.9,
                    "total_steps": 220,
                    "trainer_builder": "wnn.training_dynamics.build_scheduler",
                    "trainer_builder_sha256": base._sha256_file(base.ROOT / "src/wnn/training_dynamics.py"),
                    "rows": expected_rows,
                },
            )
            base._append_phase(
                "scheduler_curve_preflight_passed",
                queue_id=arm["queue_id"],
                scheduler=arm["scheduler"],
                rows=220,
                artifact_path=preflight_path.relative_to(base.OUT).as_posix(),
                artifact_sha256=_sha256(preflight_path),
            )
        fit_specs = []
        for gpu_index, arm in enumerate(wave_arms):
            run_dir = base.ROOT / "artifacts/runs" / arm["run_name"]
            if run_dir.exists() or run_dir.is_symlink():
                raise base.WorkerRejected(f"refusing to reuse an existing run directory: {run_dir}")
            arm_dir = arm_dirs[arm["queue_id"]]
            env = base._environment(gpu=str(gpu_index), arm_dir=arm_dir, run_name=arm["run_name"])
            env.pop("WNN_D02_CONTROL_CONTRACT", None)
            env.pop("WNN_D02_CONTROL_CONTRACT_SHA256", None)
            command = build_train_argv(arm, manifest_path, runtime_validation, validation_sha)
            fit_specs.append((command, f"fit_{arm['queue_id']}", base.OUT / "logs" / f"fit_{arm['queue_id']}.log", env))
        base._append_phase("fit_wave_start", wave_index=wave_index, queue_ids=wave_ids, devices={wave_arms[0]["queue_id"]: "0", wave_arms[1]["queue_id"]: "1"})
        wave_fit_results = base._wait_parallel(fit_specs, deadline)
        fit_results.extend(wave_fit_results)
        for arm in wave_arms:
            queue_id = arm["queue_id"]
            base._copy_arm_outputs(arm["run_name"], arm_dirs[queue_id])
        if any(row.get("returncode") != 0 or row.get("guard_stop") for row in wave_fit_results):
            base._write_json(progress_path, {"schema": "wnn.diversity.sampled-cohort-progress.v1", "state": "failed_fit", "completed_waves": _json(progress_path).get("completed_waves", []), "fit_results": fit_results})
            return {"state": "failed_fit", "fit_results": fit_results}
        for arm in wave_arms:
            queue_id = arm["queue_id"]
            missing = _json(arm_dirs[queue_id] / "artifact_collection.json").get("missing", [])
            if missing:
                raise base.WorkerRejected(f"successful fit is missing required artifacts for {queue_id}: {missing}")
            fit_summaries[queue_id] = verify_arm_fit(
                arm,
                arm_dirs[queue_id],
                control_schedule,
                scheduler_preflights.get(queue_id),
            )
        verify_prefit_arms(arm_dirs, {**spec, "arms": arms}, ready_candidate_ids=set(wave_ids))
        verify_prefit_against_control(arm_dirs, arms, control_roles, ready_candidate_ids=set(wave_ids))

        prediction_specs = []
        prediction_by_label = {}
        for gpu_index, arm in enumerate(wave_arms):
            queue_id = arm["queue_id"]
            prediction = base._export_predictions(
                queue_id,
                arm["run_name"],
                str(gpu_index),
                manifest_path,
                runtime_validation,
                validation_sha,
                arm_dirs[queue_id],
            )
            prediction_by_label[prediction["label"]] = (prediction, arm)
            prediction_specs.append((prediction["command"], prediction["label"], prediction["log_path"], prediction["env"]))
        base._append_phase("prediction_wave_start", wave_index=wave_index, queue_ids=wave_ids)
        wave_prediction_results = base._wait_parallel(prediction_specs, deadline)
        for result in wave_prediction_results:
            prediction, arm = prediction_by_label[result["label"]]
            queue_id = arm["queue_id"]
            if result.get("returncode") != 0 or result.get("guard_stop"):
                raise base.WorkerRejected(f"prediction export failed for {queue_id}: {result}")
            prediction.update(result)
            prediction["manifest_path"] = str(manifest_path)
            prediction["manifest"] = cache_manifest
            base._collect_prediction(prediction, arm["run_name"], arm_dirs[queue_id])
            sidecar = _json(arm_dirs[queue_id] / "predictions_valid.json")
            prediction_path = arm_dirs[queue_id] / "predictions_valid.npy"
            checkpoint_sha = fit_summaries[queue_id]["checkpoint_binding"]["selected_model_sha256"]
            if (
                sidecar.get("run") != arm["run_name"]
                or sidecar.get("epoch") != 1
                or sidecar.get("checkpoint_sha256") != checkpoint_sha
                or sidecar.get("prediction_sha256") != _sha256(prediction_path)
            ):
                raise base.WorkerRejected(
                    f"prediction sidecar is not bound to the selected checkpoint and array for {queue_id}"
                )
            prediction["validated_checkpoint_sha256"] = checkpoint_sha
            prediction["validated_prediction_sha256"] = _sha256(prediction_path)
            prediction_results[queue_id] = result
        completed = _json(progress_path).get("completed_waves", [])
        completed.append({"wave_index": wave_index, "candidate_ids": wave_ids})
        base._write_json(progress_path, {"schema": "wnn.diversity.sampled-cohort-progress.v1", "state": "running", "completed_waves": completed, "fit_results": fit_results, "arm_states": {arm["queue_id"]: ("predictions_complete" if arm["queue_id"] in prediction_results else "pending") for arm in arms}})
        base._append_phase("wave_complete", wave_index=wave_index, queue_ids=wave_ids)

    verify_prefit_arms(arm_dirs, {**spec, "arms": arms})
    verify_prefit_against_control(arm_dirs, arms, control_roles)
    _collect_shared(
        spec,
        bootstrap,
        source_receipt,
        manifest_path,
        runtime_validation,
        runtime_profile,
        control_contract,
        control_roles,
        control_path,
        arm_dirs,
        run_names,
        fit_results,
        prediction_results,
        fit_summaries,
    )
    contract_sha = _sha256(base.OUT / "training_contract.json")
    base._write_json(base.OUT / "commands.json", {"commands": base._COMMANDS})
    result = {
        "schema": "wnn.diversity.sampled-cohort-run-receipt.v1",
        "format_version": 1,
        "state": "complete",
        "kernel_id": spec["kernel_id"],
        "source_commit": spec["runtime_source"]["source_commit"],
        "spec_sha256": SPEC_SHA256,
        "candidate_pool_sha256": CANDIDATE_POOL_SHA256,
        "training_contract_sha256": contract_sha,
        "cache_manifest_sha256": base.EXPECTED_CACHE_MANIFEST_SHA256,
        "cache_validation_sha256": validation_sha,
        "control_contract_sha256": CONTROL_CONTRACT_SHA256,
        "completed_arm_count": len(prediction_results),
        "fit_results": fit_results,
        "prediction_results": prediction_results,
    }
    base._write_json(base.OUT / "sampled_cohort_run_receipt.json", result)
    base._write_json(progress_path, {"schema": "wnn.diversity.sampled-cohort-progress.v1", "state": "complete", "completed_waves": _json(progress_path).get("completed_waves", []), "training_contract_sha256": contract_sha})
    base._append_phase("cohort_complete", training_contract_sha256=contract_sha, completed_arm_count=len(prediction_results))
    return result


def main() -> int:
    status_path = base.WORK / "final_status.json"
    try:
        bootstrap = _json(base.OUT / "bootstrap_receipt.json")
        result = run_cohort(bootstrap)
        state = result.get("state", "failed")
        base._write_json(status_path, {"state": state, "elapsed_seconds": time.monotonic() - base._STARTED})
        return 0 if state == "complete" else 2
    except BaseException as error:
        failure = {
            "state": "failed",
            "elapsed_seconds": time.monotonic() - base._STARTED,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        try:
            base._write_json(base.WORK / "failure.json", failure)
            base._write_json(status_path, {"state": "failed", "elapsed_seconds": failure["elapsed_seconds"]})
            base._append_phase("sampled_cohort_failed", error_type=failure["error_type"], error=failure["error"])
        except OSError:
            print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
