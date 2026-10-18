#!/usr/bin/env python3
"""Bounded structural verification of a completed rank1 E6 endpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import cast

PLAN_SHA = "dcafeddb5d1d744ea0d6f981f7d70154403c5f6e4cc199bbd8060b1c961f3177"
R4_SHA = "302d3c1a8bd5059b6d6047c76a6f85dd71a7a049cdb3d3da6fa5940735fc54c3"
SOURCE_REVISION = "a266b036"
REMOTE = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
STEPS, TOTAL = 14_920, 89_520
ARTIFACTS = ("model_ep5.pt", "model.pt", "provenance.json", "training_state.pt", "history.json")
SOURCE_FILES = (*ARTIFACTS, "attempt.json", "training.log")


@dataclass(frozen=True)
class Arm:
    """Frozen F0 cell identity and audited architecture shape."""

    run: str
    base: str
    attempt: str
    hidden: int
    layers: int
    parameters: int
    tensors: int
    optimizer_states: int


ARMS = {
    "wide": Arm(
        "rr18_r01_wide_f0_s1001_e6",
        "rr18_r01_wide_f0_s1001",
        "computer-vision-homework-r01-wide-f0s1001-ep25-v1",
        512,
        1,
        2_210_306,
        12,
        10,
    ),
    "deep": Arm(
        "rr18_r01_deep_f0_s1001_e6",
        "rr18_r01_deep_f0_s1001",
        "computer-vision-homework-r01-deep-f0s1001-ep25-v1",
        384,
        2,
        2_151_554,
        16,
        14,
    ),
}


def _need(ok: bool, label: str) -> None:
    """Fail closed on one violated terminal evidence invariant."""
    if not ok:
        raise ValueError(f"{label} differs")


def _eq(value: object, expected: object, label: str) -> None:
    """Compare a receipt value without truthy/default coercion."""
    _need(value == expected, label)


def _map(value: object, label: str) -> Mapping[str, object]:
    """Require a mapping while retaining producer field types."""
    _need(isinstance(value, Mapping), f"{label} is not a mapping")
    return cast(Mapping[str, object], value)


def _json(path: Path) -> dict[str, object]:
    """Read one JSON object."""
    return cast(dict[str, object], json.loads(path.read_text()))


def _sha(path: Path) -> str:
    """Hash a file in fixed-size chunks before trusting the helper dependency."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _r4(path: Path) -> ModuleType:
    """Load only the hash-pinned source, registry and tensor helpers."""
    _eq(_sha(path), R4_SHA, "R4 helper SHA256")
    spec = importlib.util.spec_from_file_location("verified_rank1_r4", path)
    _need(spec is not None and spec.loader is not None, "R4 helper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cmd(command: Sequence[str], flag: str) -> str:
    """Read a required valued CLI option from the frozen attempt."""
    _need(flag in command and command.index(flag) + 1 < len(command), f"command lacks {flag}")
    return command[command.index(flag) + 1]


def _terminal(record: Mapping[str, object], arm: Arm, command: list[str]) -> None:
    """Require explicit successful completion and exact frozen command identity."""
    for key, value in {
        "id": arm.attempt,
        "status": "finished",
        "exit_status": 0,
        "command": command,
        "unit": f"wnn-bandit-{arm.attempt}.service",
    }.items():
        _eq(record.get(key), value, f"attempt {key}")
    _need(
        str(record["working_directory"]).endswith(f"/source-{SOURCE_REVISION}"),
        "attempt source path",
    )
    expected = {
        "--name": arm.run,
        "--epochs": "6",
        "--checkpoint-epoch": "5",
        "--resume-state": str(REMOTE / "data/artifacts/runs" / arm.base / "training_state.pt"),
    }
    for flag, value in expected.items():
        _eq(_cmd(command, flag), value, f"command {flag}")
    _need("--no-eval-during-fit" in command, "fit validation mode")


def _attempt(arm: Arm, path: Path, plan_path: Path, r4: ModuleType) -> dict[str, object]:
    """Bind one successful terminal attempt to its frozen sequence step."""
    plan, record = _json(plan_path), _json(path)
    steps = cast(list[dict[str, object]], plan["steps"])
    matches = [step for step in steps if step["attempt"] == arm.attempt]
    _need(len(matches) == 1 and _sha(plan_path) == PLAN_SHA, "frozen plan/step identity")
    command = cast(list[str], matches[0]["expected_command"])
    _terminal(record, arm, command)
    return {
        "attempt": arm.attempt,
        "record_sha256": r4.sha256_file(path),
        "status": "finished",
        "exit_status": 0,
        "plan_sha256": PLAN_SHA,
        "unit": record["unit"],
    }


def _collection(
    arm: Arm, snap: Path, path: Path, record_path: Path, r4: ModuleType
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Hash-check five final artifacts and separately copied attempt/log evidence."""
    receipt = _json(path)
    fields = {
        "run": arm.run,
        "epoch_index": 5,
        "source_directory": str(REMOTE / "data/artifacts/runs" / arm.run),
        "snapshot_directory": str(snap),
    }
    for key, value in fields.items():
        _eq(receipt.get(key), value, f"collection {key}")
    _need(
        receipt["source_snapshot_hash_match"] is True
        and receipt["producer_model_hash_matches"] is True,
        "collection hash guards",
    )
    before, after = (
        _map(receipt["source_stats_before"], "source stats"),
        _map(receipt["source_stats_after"], "post-copy stats"),
    )
    _need(before == after and set(before) == set(SOURCE_FILES), "stable source/collected-file set")
    manifest = _map(receipt["artifacts"], "artifact manifest")
    _eq(set(manifest), set(ARTIFACTS), "terminal artifacts")
    artifacts: dict[str, dict[str, object]] = {}
    for name in ARTIFACTS:
        row, file = _map(manifest[name], name), snap / name
        actual = {"path": str(file), "bytes": file.stat().st_size, "sha256": r4.sha256_file(file)}
        _eq(dict(row), actual, f"artifact {name}")
        artifacts[name] = actual
    for field, name in (("attempt_evidence", "attempt.json"), ("log_evidence", "training.log")):
        evidence, file = _map(receipt[field], field), snap / name
        actual = {"path": str(file), "bytes": file.stat().st_size, "sha256": r4.sha256_file(file)}
        for key, value in actual.items():
            _eq(evidence.get(key), value, f"{field} {key}")
    terminal = _map(receipt["attempt_evidence"], "attempt evidence")
    _eq(
        (terminal.get("status"), terminal.get("exit_status")),
        ("finished", 0),
        "collected attempt terminal status",
    )
    _eq(_json(record_path), _json(snap / "attempt.json"), "copied attempt record")
    return artifacts, receipt


def _history(value: object, parent: object) -> list[dict[str, object]]:
    """Check six fit-only summaries and preserve verified stage1 history."""
    _need(isinstance(value, list) and len(value) == 6, "six-epoch history")
    _need(isinstance(parent, list) and len(parent) == 2, "two-epoch parent history")
    rows = cast(list[dict[str, object]], value)
    _eq([r["epoch"] for r in rows], list(range(6)), "history indices")
    _eq(rows[:2], parent, "stage1 history prefix")
    for i, row in enumerate(rows):
        _eq(row["evaluation"], "withheld", f"epoch {i} eval mode")
        _need(
            math.isfinite(float(row["train_seconds"])) and row["train_seconds"] > 0,
            f"epoch {i} train time",
        )
        d, fit = (
            _map(row["update_diagnostics"], "diagnostics"),
            _map(row["fit_summary"], "fit summary"),
        )
        epoch = {
            "optimizer_attempts": STEPS,
            "amp_skipped_updates": 0,
            "optimizer_completed_updates": STEPS,
            "ema_updates": 0,
            "completed_batches": 1492,
        }
        _eq(d["epoch"], epoch, f"epoch {i} counters")
        total = (i + 1) * STEPS
        _eq(
            d["cumulative"],
            {
                "optimizer_attempts": total,
                "amp_skipped_updates": 0,
                "optimizer_completed_updates": total,
                "ema_updates": 0,
                "completed_batches": (i + 1) * 1492,
            },
            f"epoch {i} cumulative",
        )
        _eq(
            _map(d["grad_stats"], "gradient stats")["nonfinite_steps"],
            0,
            f"epoch {i} nonfinite gradients",
        )
        _need(math.isfinite(float(fit["objective_mean"])), f"epoch {i} loss")
        _eq(
            (fit["optimizer_windows"], fit["optimizer_updates"]),
            (STEPS, d["epoch"]),
            f"epoch {i} optimizer summary",
        )
        rates = cast(list[dict[str, object]], fit["learning_rate_by_group"])
        _need(
            len(rates) == 1
            and all(math.isfinite(float(rates[0][k])) for k in ("first", "last", "mean")),
            f"epoch {i} learning rate",
        )
    return rows


def verify(args: argparse.Namespace) -> dict[str, object]:
    """Validate terminal counters, ancestry, source IDs and finite FP32 tensors."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    r4 = _r4(args.r4_helper)
    import torch

    arm, snap, source = ARMS[args.arm], args.snapshot_dir.resolve(), args.source_root.resolve()
    attempt = _attempt(arm, args.attempt_record, args.plan, r4)
    artifacts, collection = _collection(arm, snap, args.collection_receipt, args.attempt_record, r4)
    sys.path.insert(0, str(source / "src"))
    from wnn.training_state import load_training_state

    state = load_training_state(snap / "training_state.pt")
    parent_path = args.stage1_snapshot_dir / "training_state.pt"
    parent, stage1 = load_training_state(parent_path), _json(args.stage1_verification)
    contract = _map(state["contract"], "contract")
    _eq((state["scaler"], state["ema"], state["swa"]), ({}, None, None), "FP32 auxiliary state")
    _eq((stage1["status"], stage1["next_epoch"]), ("passed", 2), "stage1 receipt")
    _eq(
        _map(stage1["artifacts"], "stage1 artifacts")["training_state.pt"]["sha256"],
        r4.sha256_file(parent_path),
        "stage1 hash",
    )
    _eq(
        (parent["next_epoch"], state["next_epoch"], contract),
        (2, 6, parent["contract"]),
        "continuation parent/contract",
    )
    _eq((contract["epochs"], contract["total_steps"]), (6, TOTAL), "fixed schedule")
    source_count = r4.verify_sources(source, contract)
    archive = _json(args.archive_manifest)
    _eq(archive["source_revision"], SOURCE_REVISION, "source revision")
    _need(
        any(
            x["sha256"] == r4.SOURCE_ARCHIVE_SHA256
            for x in cast(list[dict[str, object]], archive["archives"])
        ),
        "source archive SHA",
    )
    registry_sha = r4.verify_registry(contract, args.data_registry)
    for key in ("sources", "fit_membership", "normalization"):
        _eq(contract[key], parent["contract"][key], f"stage1 {key}")

    parent_sha = r4.sha256_file(parent_path)
    resume = {
        "path": str(REMOTE / "data/artifacts/runs" / arm.base / "training_state.pt"),
        "sha256": parent_sha,
        "next_epoch": 2,
    }
    provenance = _json(snap / "provenance.json")
    terminal_counts = {
        "optimizer_attempts": TOTAL,
        "amp_skipped_updates": 0,
        "optimizer_completed_updates": TOTAL,
        "ema_updates": 0,
        "completed_batches": 8_952,
    }
    for actual, expected, label in (
        (collection["resumed_from"], resume, "collection ancestry"),
        (provenance["resumed_from"], resume, "provenance ancestry"),
        (provenance["fit_membership_identity"], contract["fit_membership"], "membership"),
        (provenance["normalization_identity"], contract["normalization"], "normalization"),
        (provenance["cumulative_updates"], terminal_counts, "provenance update counts"),
        (provenance["selected_epoch"], 5, "selected epoch"),
    ):
        _eq(actual, expected, label)

    model = _map(state["model"], "resume model")
    raw, selected = (
        torch.load(snap / name, map_location="cpu", weights_only=False)
        for name in ("model_ep5.pt", "model.pt")
    )
    for checkpoint in (raw, selected):
        _eq(
            (checkpoint["epoch"], checkpoint["args"]["name"]),
            (5, arm.run),
            "inference checkpoint identity",
        )
        _eq(
            (
                checkpoint["args"]["hidden"],
                checkpoint["args"]["layers"],
                checkpoint["args"]["seed"],
                checkpoint["args"]["valid_fold"],
                checkpoint["args"]["precision"],
            ),
            (arm.hidden, arm.layers, 1001, 0, "fp32"),
            "cell/architecture",
        )
        _eq(set(checkpoint["model"]), set(model), "model tensor names")
        _need(
            all(torch.equal(checkpoint["model"][k], model[k]) for k in model),
            "inference weights differ from resume model",
        )
        embedded = _map(checkpoint["provenance"], "embedded checkpoint provenance")
        for field in (
            "args",
            "trainer_implementation",
            "fit_membership_identity",
            "normalization_identity",
            "resumed_from",
        ):
            _eq(embedded.get(field), provenance[field], f"embedded provenance {field}")
    _need(
        all(torch.equal(raw["model"][k], selected["model"][k]) for k in model),
        "model.pt differs from epoch5",
    )
    hashes = _map(provenance["checkpoint_sha256"], "checkpoint hashes")
    for name in ("model_ep5.pt", "model.pt"):
        _eq(hashes[name], artifacts[name]["sha256"], f"provenance {name}")

    tensors = r4.tensor_values(model)
    _eq(len(tensors), arm.tensors, "model tensor count")
    _need(
        all(
            torch.is_tensor(t) and t.dtype == torch.float32 and bool(torch.isfinite(t).all())
            for t in tensors
        ),
        "model is not finite FP32",
    )
    parameters = sum(t.numel() for k, t in model.items() if k not in {"mean", "std"})
    _eq(parameters, arm.parameters, "parameter count")
    optimizer = _map(state["optimizer"], "optimizer")
    ids = [p for group in optimizer["param_groups"] for p in group["params"]]
    states = _map(optimizer["state"], "optimizer states")
    _need(len(ids) == len(set(ids)) and set(ids) == set(states), "optimizer state coverage")
    _eq(len(states), arm.optimizer_states, "optimizer state count")
    opt_tensors = [t for value in states.values() for t in r4.tensor_values(value)]
    floating = [t for t in opt_tensors if torch.is_tensor(t) and t.is_floating_point()]
    _need(
        all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in floating),
        "optimizer state is not finite FP32",
    )

    progress, scheduler = _map(state["progress"], "progress"), _map(state["scheduler"], "scheduler")
    counts = terminal_counts
    _eq(
        (progress["cumulative_updates"], progress["step_count"]),
        (counts, TOTAL),
        "terminal update counts",
    )
    _eq(
        (scheduler["last_epoch"], scheduler["_step_count"]),
        (TOTAL, TOTAL + 1),
        "scheduler endpoint",
    )
    history = _history(progress["history"], parent["progress"]["history"])
    _eq(json.loads((snap / "history.json").read_text()), history, "history JSON")
    _need(
        not torch.cuda.is_initialized()
        and all(t.device.type == "cpu" for t in tensors + opt_tensors),
        "CPU-only verification",
    )
    return {
        "kind": "terminal_epoch5_checkpoint_structural_verification",
        "status": "passed",
        "created_utc": datetime.now(UTC).isoformat(),
        "arm": args.arm,
        "run": arm.run,
        "attempt": attempt,
        "epoch_index": 5,
        "next_epoch": 6,
        "collection_receipt_sha256": r4.sha256_file(args.collection_receipt),
        "r4_helper_sha256": R4_SHA,
        "verification_helper_sha256": r4.sha256_file(Path(__file__)),
        "artifacts": artifacts,
        "source_revision": SOURCE_REVISION,
        "source_files_verified": source_count,
        "source_archive_manifest_sha256": r4.sha256_file(args.archive_manifest),
        "dataset_registry_sha256": registry_sha,
        "stage1_training_state_sha256": parent_sha,
        "stage1_verification_sha256": r4.sha256_file(args.stage1_verification),
        "plan_sha256": PLAN_SHA,
        "resumed_from": resume,
        "model_tensors_verified": len(tensors),
        "trainable_parameter_count": parameters,
        "optimizer_states_verified": len(states),
        "optimizer_floating_tensors_verified": len(floating),
        "completed_update_counts": counts,
        "scheduler_last_epoch": TOTAL,
        "scheduler_step_count": TOTAL + 1,
        "fit_history": [
            {
                "epoch": row["epoch"],
                "train_seconds": row["train_seconds"],
                "fit_summary": row["fit_summary"],
            }
            for row in history
        ],
        "limitations": [
            "Structural checks only; no measured GPU resume parity.",
            "Raw parquets were not rehashed; registry identities were checked.",
            "Fixed six epochs is not a convergence, quality or CP30 result.",
        ],
    }


def main() -> None:
    """Run one bounded CPU check and write a structural receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    for name in (
        "snapshot-dir",
        "source-root",
        "stage1-snapshot-dir",
        "stage1-verification",
        "collection-receipt",
        "attempt-record",
        "plan",
        "data-registry",
        "archive-manifest",
        "r4-helper",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _need(not args.output.exists(), "output receipt already exists")
    result = verify(args)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
