from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from admission import SPLIT_SHA, authority, digest

PLAN_SHA256 = "b40ba65c7c2495f1a08a42331800ce8090bf9bbe63e7cb5514af401e784cf1c0"
AUTHORITY_SHA256 = "24e2847879db39016cfa952eed292a6c6dfa1c5dcf3f5154a09eeca19bca44a5"
SOURCE_COMMIT = "5d73bc2b10ae20e607e0d4b4533da85e76e27bca"
SOURCE_MANIFEST_SHA256 = "a1e16d24ab53032643807edb5471b8bc65b06bdd96da516944bcf36736339937"
TRAINING_SOURCE_SHA256 = "569123615f69cc8adae3ff362ce5db720dbd068dec3a5b0f94e2d6dc3201be6c"
ORIGINAL_GRADER_SHA256 = "29314bdac0cda0904737ca18c8e563d374c190c621e077735d119b46c9ce52b4"
RECEIVER_PID = 1_818_502
RECEIVER_BIRTH = 158_011_868
FIRST_PROFILE_EPOCH = 1_794_168_764.8507051
FIXED_DEADLINE_EPOCH = 1_794_197_564.8507051
WMSE_REL_TOLERANCE = 1e-6
WMSE_ABS_TOLERANCE = 1e-7
MAX_JSON_BYTES = 32 * 1024 * 1024
CELL_IDENTITIES = (
    ("C0_muon256_a4_s203101", "C0", 203101),
    ("C1_adamw256_a4_s203101", "C1", 203101),
    ("C2_muon512_a2_s203101", "C2", 203101),
    ("C0_muon256_a4_s203102", "C0", 203102),
    ("C1_adamw256_a4_s203102", "C1", 203102),
    ("C2_muon512_a2_s203102", "C2", 203102),
)
RECEIVER_EXIT_SCHEMA = "wnn.transformer-optimizer-horizon-receiver-exit.v1"
REPAIR_SCHEMA = "wnn.transformer-optimizer-horizon-comparison-repair.v1"
JsonObject = dict[str, Any]
ConfigLoader = Callable[[Path], JsonObject]


def _regular_bytes(path: Path, *, limit: int = MAX_JSON_BYTES) -> bytes:
    """Read one stable bounded regular file without following symlinks."""
    if path.is_symlink():
        raise ValueError(f"artifact is a symlink: {path.name}")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError(f"artifact is not a bounded regular file: {path.name}")
    with path.open("rb") as stream:
        first = os.fstat(stream.fileno())
        payload = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if (
        len(payload) != before.st_size
        or (first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError(f"artifact changed during read: {path.name}")
    return payload


def _read_object(path: Path) -> JsonObject:
    """Read a strict finite-JSON object from one bounded artifact."""
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(_regular_bytes(path), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"artifact must be a JSON object: {path.name}")
    return value


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_authority(
    plan: JsonObject, authority_path: Path, expected_authority_sha256: str
) -> JsonObject:
    """Recheck the exact ROOT authority, including its unchanged collection deadline."""
    if expected_authority_sha256 != AUTHORITY_SHA256 or digest(authority_path) != AUTHORITY_SHA256:
        raise ValueError("authority bytes differ from the frozen source-bound deadline")
    if Path(plan["authority"]).resolve() != authority_path.resolve():
        raise ValueError("plan does not name the supplied authority file")
    value = _read_object(authority_path)
    authority(value)
    if (
        value.get("source_commit") != SOURCE_COMMIT
        or value.get("first_profile_epoch") != FIRST_PROFILE_EPOCH
        or value.get("deadline_epoch") != FIXED_DEADLINE_EPOCH
    ):
        raise ValueError("authority differs from the original six-fit deadline")
    return value


def _valid_plan(path: Path, expected_sha256: str) -> JsonObject:
    """Bind postprocessing to the exact source/config/split six-cell plan."""
    if path.is_symlink() or expected_sha256 != PLAN_SHA256 or digest(path) != PLAN_SHA256:
        raise ValueError("receiver plan bytes differ from the exact frozen six-cell plan")
    plan = _read_object(path)
    if (
        plan.get("schema") != "wnn.transformer-optimizer-horizon-panel.v1"
        or plan.get("source_commit") != SOURCE_COMMIT
        or plan.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
        or plan.get("expected_training_source_sha256") != TRAINING_SOURCE_SHA256
        or plan.get("split_sha256") != SPLIT_SHA
        or [(item.get("cell"), item.get("arm"), item.get("seed")) for item in plan.get("cells", [])]
        != list(CELL_IDENTITIES)
    ):
        raise ValueError("plan identity differs from the approved panel")
    if digest(Path(plan["source_manifest"])) != SOURCE_MANIFEST_SHA256:
        raise ValueError("source capsule manifest bytes differ")
    manifest = _read_object(Path(plan["source_manifest"]))
    if manifest.get("commit") != SOURCE_COMMIT:
        raise ValueError("source capsule commit differs")
    return plan


def _valid_receiver_exit(
    path: Path, expected_sha256: str, plan: JsonObject, authority_value: JsonObject
) -> JsonObject:
    """Require ROOT's exit observation for the exact receiver identity and clock."""
    if path.is_symlink() or digest(path) != expected_sha256:
        raise ValueError("receiver-exit receipt digest differs")
    receipt = _read_object(path)
    if (
        receipt.get("schema") != RECEIVER_EXIT_SCHEMA
        or receipt.get("observed_absent") is not True
        or receipt.get("receiver_pid") != RECEIVER_PID
        or receipt.get("receiver_birth") != RECEIVER_BIRTH
        or receipt.get("plan_sha256") != PLAN_SHA256
        or receipt.get("source_commit") != SOURCE_COMMIT
        or receipt.get("authority_sha256") != AUTHORITY_SHA256
        or receipt.get("observed_status")
        not in ("absent", "pid_reused_different_birth", "exited_zombie")
        or not _finite(receipt.get("observed_at_epoch"))
        or not FIRST_PROFILE_EPOCH <= receipt["observed_at_epoch"] <= authority_value["deadline_epoch"]
        or receipt.get("observed_at_epoch") > authority_value["deadline_epoch"]
    ):
        raise ValueError("ROOT receiver-exit receipt is not bound to this run or deadline")
    return receipt


def _last_monitor_event(path: Path) -> JsonObject:
    """Parse the last complete monitor event without accepting malformed JSON lines."""
    payload = _regular_bytes(path)
    lines = payload.splitlines()
    if not lines:
        raise ValueError("monitor evaluation log is empty")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise ValueError("last monitor evaluation is not an object")
    return value


def _validate_monitor(event: JsonObject, cell_name: str) -> JsonObject:
    """Check fixed-E4 full-valid raw-z monitoring with bounded WMSE roundoff."""
    metrics = event.get("metrics", {})
    observed_wmse = metrics.get("valid/loss")
    raw_z_wmse = metrics.get("valid/z/wmse")
    if (
        event.get("kind") != "evaluation"
        or event.get("epoch") != 4
        or event.get("step") != 3280
        or metrics.get("valid/sequence_count") != 1873
        or metrics.get("valid/selected_rows") != 3_528_273
        or metrics.get("valid/wp") != metrics.get("valid/z/wp")
        or not all(_finite(metrics.get(key)) for key in ("valid/wp", "valid/wp/t0", "valid/wp/t1"))
        or not _finite(observed_wmse)
        or not _finite(raw_z_wmse)
        or not math.isclose(
            observed_wmse,
            raw_z_wmse,
            rel_tol=WMSE_REL_TOLERANCE,
            abs_tol=WMSE_ABS_TOLERANCE,
        )
    ):
        raise ValueError(f"full-valid raw-z monitor failed fixed-E4 checks: {cell_name}")
    return metrics


def _validate_cell(
    runtime: Path, plan: JsonObject, cell: JsonObject, load_config: ConfigLoader
) -> tuple[JsonObject, JsonObject]:
    """Recheck one endpoint's full-valid score, ancestry, model bytes and monitor record."""
    name = cell["cell"]
    fit = runtime / "runs" / name / "fit"
    terminal = _read_object(fit / "terminal_status.json")
    score = _read_object(fit / "score.json")
    model_path = fit / "model.pt"
    if model_path.is_symlink() or not stat.S_ISREG(model_path.stat().st_mode):
        raise ValueError(f"model checkpoint is not a regular file: {name}")
    checkpoint_sha = digest(model_path)
    producer, official = score.get("producer", {}), score.get("score", {})
    identity = terminal.get("identity", {})
    config_path = Path(cell["config_path"])
    if config_path.is_symlink() or digest(config_path) != cell["config_sha256"]:
        raise ValueError(f"frozen config bytes differ: {name}")
    config = load_config(config_path)
    provenance = producer.get("provenance", {})
    if (
        terminal.get("status") != "completed"
        or terminal.get("completed_epochs") != 4
        or terminal.get("global_updates") != 3280
        or terminal.get("model_sha256") != checkpoint_sha
        or producer.get("checkpoint_sha256") != checkpoint_sha
        or producer.get("config") != config
        or producer.get("teacher") is not None
        or "reference" in producer
        or identity.get("source_sha256") != plan["expected_training_source_sha256"]
        or identity.get("config_sha256") != cell["normalized_config_sha256"]
        or identity.get("world_size") != 2
        or provenance.get("identity") != identity
        or score.get("split_sha256") != SPLIT_SHA
        or score.get("sequences") != 1873
        or official.get("selected_rows") != 3_528_273
        or official.get("metric_id") != "global_wp_clip2_v1"
        or not all(_finite(official.get(key)) for key in ("wp", "t0", "t1"))
    ):
        raise ValueError(f"completed endpoint score/ancestry checks failed: {name}")
    metrics = _validate_monitor(_last_monitor_event(fit / "monitor_evaluations.jsonl"), name)
    record = {
        "arm": cell["arm"],
        "seed": cell["seed"],
        "optimizer": cell["optimizer"],
        "chunk_rows": config["chunk_rows"],
        "official_raw_z": official,
        "valid_z_same_forward": {
            key: value for key, value in metrics.items() if key.startswith("valid/z/")
        },
        "checkpoint_sha256": checkpoint_sha,
        "score_sha256": digest(fit / "score.json"),
    }
    wmse_check = {
        "valid_loss": metrics["valid/loss"],
        "valid_z_wmse": metrics["valid/z/wmse"],
        "absolute_difference": abs(metrics["valid/loss"] - metrics["valid/z/wmse"]),
    }
    return record, wmse_check


def _paired_deltas(cells: dict[str, JsonObject]) -> list[JsonObject]:
    """Compute the four predeclared descriptive paired WP deltas."""
    deltas = []
    for seed in (203101, 203102):
        control = cells[f"C0_muon256_a4_s{seed}"]["official_raw_z"]
        for arm, question, suffix in (
            ("C1", "AdamW recipe vs Muon reference", "adamw256_a4"),
            ("C2", "512x2 horizon vs 256x4", "muon512_a2"),
        ):
            candidate = cells[f"{arm}_{suffix}_s{seed}"]["official_raw_z"]
            deltas.append(
                {"seed": seed, "contrast": question, "arm": arm, "delta_wp": candidate["wp"] - control["wp"]}
            )
    return deltas


def build_comparison(
    plan: JsonObject,
    load_config: ConfigLoader,
    *,
    script_sha256: str,
    original_grader_sha256: str,
    receiver_exit_sha256: str,
) -> JsonObject:
    """Build canonical grade payload only after all six independent endpoint checks pass."""
    runtime = Path(plan["runtime"])
    validated = {
        cell["cell"]: _validate_cell(runtime, plan, cell, load_config) for cell in plan["cells"]
    }
    cells = {name: value[0] for name, value in validated.items()}
    wmse_roundoff = {name: value[1] for name, value in validated.items()}
    return {
        "schema": "wnn.transformer-optimizer-horizon-comparison.v1",
        "status": "complete_exposed_fixed4E_screen",
        "metric_id": "global_wp_clip2_v1",
        "split_sha256": SPLIT_SHA,
        "cells": cells,
        "paired_deltas": _paired_deltas(cells),
        "interpretation": (
            "Two paired fitting seeds on repeatedly exposed official validation; descriptive only, "
            "with no significance or public-gain claim. C1 is an AdamW recipe contrast including "
            "its decay coverage, not an isolated optimizer-only effect."
        ),
        "automatic_followup": False,
        "comparison_repair": {
            "schema": REPAIR_SCHEMA,
            "script_sha256": script_sha256,
            "original_grader_sha256": original_grader_sha256,
            "reason": (
                "The frozen R1 grader required exact equality between valid/loss and "
                "valid/z/wmse. Actual same-forward full-valid reductions differ by floating-point "
                "roundoff; R2 retains full coverage, raw-z WP alias, fixed endpoint and byte checks, "
                "and uses the declared finite WMSE tolerance only."
            ),
            "wmse_tolerance": {"relative": WMSE_REL_TOLERANCE, "absolute": WMSE_ABS_TOLERANCE},
            "wmse_roundoff_by_cell": wmse_roundoff,
            "receiver_exit_receipt_sha256": receiver_exit_sha256,
        },
    }


def _existing_matches(existing: JsonObject, proposed: JsonObject) -> bool:
    """Allow no-op only when an existing canonical comparison matches all six current grades."""
    keys = ("schema", "status", "metric_id", "split_sha256", "cells", "paired_deltas")
    if not all(existing.get(key) == proposed.get(key) for key in keys):
        return False
    return "comparison_repair" not in existing or existing.get("comparison_repair") == proposed.get(
        "comparison_repair"
    )


def write_comparison_once(path: Path, value: JsonObject) -> bool:
    """Create canonical output exclusively, or verify an existing comparison without overwrite."""
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if path.is_symlink():
        raise ValueError("comparison output cannot be a symlink")
    if path.exists():
        existing = _read_object(path)
        if _existing_matches(existing, value):
            return False
        raise FileExistsError("comparison.json exists but does not match all six verified endpoints")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        existing = _read_object(path)
        if _existing_matches(existing, value):
            return False
        raise FileExistsError(
            "comparison.json appeared and differs from the verified endpoints"
        ) from None
    return True


def run(
    plan_path: Path,
    plan_sha256: str,
    authority_path: Path,
    authority_sha256: str,
    receiver_exit_path: Path,
    receiver_exit_sha256: str,
    *,
    now: float,
    load_config: ConfigLoader,
) -> tuple[Path, bool, str]:
    """Verify fixed authority/exit/plan, then build and create-or-no-op canonical grade."""
    if now > FIXED_DEADLINE_EPOCH:
        raise ValueError("original ROOT collection deadline has passed")
    plan = _valid_plan(plan_path, plan_sha256)
    auth = _valid_authority(plan, authority_path, authority_sha256)
    _valid_receiver_exit(receiver_exit_path, receiver_exit_sha256, plan, auth)
    original_grader_sha256 = digest(Path(__file__).with_name("compare.py"))
    if original_grader_sha256 != ORIGINAL_GRADER_SHA256:
        raise ValueError("frozen R1 grader bytes differ from the recorded repair basis")
    payload = build_comparison(
        plan,
        load_config,
        script_sha256=digest(Path(__file__)),
        original_grader_sha256=original_grader_sha256,
        receiver_exit_sha256=receiver_exit_sha256,
    )
    if time.time() > auth["deadline_epoch"]:
        raise ValueError("original ROOT collection deadline elapsed during R2 grading")
    output = Path(plan["runtime"]) / "comparison.json"
    created = write_comparison_once(output, payload)
    return output, created, digest(output)


def main() -> None:
    """Run CPU-only R2 postprocessing; ROOT owns the exit receipt and invocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--receiver-exit", type=Path, required=True)
    parser.add_argument("--receiver-exit-sha256", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    plan = _valid_plan(args.plan, args.plan_sha256)
    source = Path(plan["source_root"])
    sys.path.insert(0, str(source / "src"))
    from wnn.training.sequence_config import load_config

    output, created, output_sha256 = run(
        args.plan,
        args.plan_sha256,
        args.authority,
        args.authority_sha256,
        args.receiver_exit,
        args.receiver_exit_sha256,
        now=time.time(),
        load_config=load_config,
    )
    print(
        json.dumps(
            {"status": "created" if created else "verified_noop", "path": str(output), "sha256": output_sha256},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
