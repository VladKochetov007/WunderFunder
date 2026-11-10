"""Publish bounded, identity-checked local L4 progress to PR 77."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

SOURCE_SHA256 = "b05a73fc8a8a7a4f3f46962e31fd81ee6b401db715109e36d4f0d6e32bf35140"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
CELLS = (
    "shared_control_s203201",
    "shared_control_s203202",
    "t0_specialist_s203201",
    "t0_specialist_s203202",
    "t1_specialist_s203201",
    "t1_specialist_s203202",
)
GPU_BINDINGS = {
    "shared_control_s203201": (0, "222adc44-ecd3-a645-cf7e-ddda9d495f7a"),
    "shared_control_s203202": (1, "bce838c4-097e-0623-7b67-6a97f3778510"),
    "t0_specialist_s203201": (2, "9fd565e2-ebb2-35be-c628-d971a575aa7e"),
    "t0_specialist_s203202": (3, "02d8270f-e4ee-1363-8232-15c0726fdab1"),
    "t1_specialist_s203201": (4, "9943f86e-f085-2d36-2c12-8d2a22111ef0"),
    "t1_specialist_s203202": (5, "48b463a4-5b0a-e277-9ef1-ee37be8015b1"),
}
EXPECTED_UPDATES = 38_760
UPDATES_PER_EPOCH = 6_460
VALID_SEQUENCES = 1_873
VALID_ROWS = 3_528_273
POLL_SECONDS = 30
MAX_WALL_SECONDS = 43_200
STATE_SCHEMA = "wnn.l4-next70-pr77-publisher-state.v1"
CLIENT_PATH = Path(
    r"C:\Users\admin\Documents\Codex\reviews\WunderFunder-architecture-execution-20261107\scripts\sequence_progress_github.py"
)
SOURCE_ROOT = "/data/lc-epanov_s/tabdata-next70-20261110-r1/source-specialist-r2"

Cell = dict[str, object]
Snapshot = dict[str, object]


def canonical_sha256(value: object) -> str:
    """Hash repository JSON identities using its stable serialization."""
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, limit: int = 2_000_000) -> dict[str, object]:
    """Read one bounded regular JSON file without following a symbolic link."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or linked evidence file: {path.name}")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError(f"evidence file is not regular or exceeds {limit} bytes: {path.name}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"evidence file must contain a JSON object: {path.name}")
    return value


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _read_tail_jsonl(path: Path, maximum: int = 262_144) -> dict[str, object] | None:
    """Read the last complete JSON object from a bounded tail of a growing log."""
    if path.is_symlink() or not path.is_file():
        return None
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        stream.seek(max(0, size - maximum))
        data = stream.read(maximum)
    lines = data.splitlines()
    if size > maximum and lines:
        lines = lines[1:]
    for line in reversed(lines):
        if len(line) > 131_072:
            continue
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _read_evaluations(path: Path) -> tuple[list[dict[str, object]], str | None]:
    """Read the small six-epoch full-validation journal with strict line bounds."""
    if path.is_symlink() or not path.is_file():
        return [], "evaluation journal is missing"
    size = path.stat().st_size
    if size > 1_000_000:
        return [], "evaluation journal exceeds the 1 MB local read bound"
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if index == len(lines) - 1:
                continue
            return records, "evaluation journal contains a partial interior line"
        if len(line) > 32_768:
            return records, "evaluation event exceeds its 32 KB line bound"
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return records, "evaluation journal contains malformed JSON"
        if not isinstance(value, dict):
            return records, "evaluation journal contains a non-object record"
        records.append(value)
        if len(records) > 10:
            return records, "evaluation journal exceeds the fixed-epoch record bound"
    return records, None


def load_admission(artifact_root: Path) -> dict[str, object]:
    """Require the exact ROOT-approved six-cell source and chronological split."""
    admission = _read_json(artifact_root / "ROOT-fit-admission-r1.json", 128_000)
    profiles = admission.get("profiles")
    if (
        admission.get("schema") != "wnn.l4-next70-root-fit-admission.v1"
        or admission.get("approved") is not True
        or admission.get("source_sha256") != SOURCE_SHA256
        or admission.get("split_sha256") != SPLIT_SHA256
        or admission.get("source_root") != SOURCE_ROOT
        or admission.get("max_fresh_fits") != len(CELLS)
        or not isinstance(profiles, dict)
        or set(profiles) != set(CELLS)
    ):
        raise ValueError("ROOT admission does not match the frozen six-cell reporter contract")
    for cell in CELLS:
        entry = profiles[cell]
        expected_gpu, expected_uuid = GPU_BINDINGS[cell]
        if (
            not isinstance(entry, dict)
            or entry.get("gpu") != expected_gpu
            or entry.get("gpu_uuid") != expected_uuid
            or not _is_sha256(entry.get("config_sha256"))
        ):
            raise ValueError(f"ROOT admission binding is malformed for {cell}")
    return admission


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _verify_metadata(cell: str, directory: Path, admission: dict[str, object]) -> dict[str, object]:
    """Bind local run metadata to source, config, split, data membership and assigned GPU."""
    metadata = _read_json(directory / "run_metadata.json", 1_000_000)
    profile = admission["profiles"][cell]
    assert isinstance(profile, dict)
    config, source, provenance = (
        metadata.get("config"),
        metadata.get("source"),
        metadata.get("provenance"),
    )
    if (
        not isinstance(config, dict)
        or not isinstance(source, dict)
        or not isinstance(provenance, dict)
    ):
        raise ValueError("run metadata lacks config, source or provenance")
    if canonical_sha256(config) != profile["config_sha256"]:
        raise ValueError("normalized config hash differs from ROOT admission")
    if source.get("sha256") != SOURCE_SHA256:
        raise ValueError("source capsule hash differs from ROOT admission")
    if (
        config.get("epochs") != 6
        or config.get("objective_mode") != "standalone_labels_wmse_v2"
        or config.get("required_device") != "L4"
        or config.get("required_world_size") != 1
        or config.get("checkpoint_endpoint_policy") != "configured_final_epoch"
        or config.get("supervised_target") not in (None, 0, 1)
        or not isinstance(config.get("model"), dict)
        or config["model"].get("architecture") != "gru"
        or config["model"].get("hidden") != 256
        or config["model"].get("layers") != 2
    ):
        raise ValueError("run config differs from the frozen standalone six-epoch recipe")
    expected_target = (
        0 if cell.startswith("t0_specialist_") else 1 if cell.startswith("t1_specialist_") else None
    )
    expected_seed = 203201 if cell.endswith("203201") else 203202
    if (
        config.get("supervised_target") != expected_target
        or config.get("seed") != expected_seed
        or config.get("checkpoint_retention") != "all_epochs"
        or config.get("normalization_scope") != "all_fit"
        or config.get("normalization_sequences") != 10_325
        or config.get("training_mask") != "all_after_warmup"
    ):
        raise ValueError(
            "cell target, seed, normalization or endpoint differs from its frozen plan"
        )
    data = provenance.get("data")
    if not isinstance(data, dict):
        raise ValueError("run metadata lacks data provenance")
    if provenance.get("data_sha256") != canonical_sha256(data):
        raise ValueError("data provenance digest does not verify")
    normalization = data.get("normalization")
    objective_contract = data.get("objective_contract")
    if (
        data.get("split_sha256") != SPLIT_SHA256
        or data.get("teacher") is not None
        or not isinstance(normalization, dict)
        or normalization.get("scope") != "all_fit"
        or normalization.get("sequence_count") != 10_325
        or data.get("training_mask") != "all_after_warmup"
        or not isinstance(objective_contract, dict)
        or objective_contract.get("mode") != "standalone_labels_wmse_v2"
        or objective_contract.get("reference_training_ancestry") != "none"
    ):
        raise ValueError("split or teacher ancestry differs from the approved standalone recipe")
    fit_members, evaluation_members = data.get("fit_members"), data.get("evaluation_members")
    if (
        not isinstance(fit_members, list)
        or len(fit_members) != 10_325
        or any(
            not isinstance(item, list) or len(item) != 2 or item[0] != "train"
            for item in fit_members
        )
        or len(
            {
                (item[0], item[1])
                for item in fit_members
                if isinstance(item, list) and len(item) == 2
            }
        )
        != 10_325
        or evaluation_members != [["valid", index] for index in range(VALID_SEQUENCES)]
    ):
        raise ValueError("fit or full-validation membership differs from the official split")
    monitor = data.get("report_only_monitor")
    expected_eval_hash = hashlib.sha256(
        json.dumps(evaluation_members, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        not isinstance(monitor, dict)
        or monitor.get("split_sha256") != SPLIT_SHA256
        or monitor.get("evaluation_member_count") != VALID_SEQUENCES
        or monitor.get("evaluation_member_sha256") != expected_eval_hash
        or monitor.get("endpoint") != "raw standalone z; no reference or blend"
    ):
        raise ValueError("report-only monitor provenance is incomplete or mismatched")
    runtime, devices = provenance.get("runtime"), None
    if isinstance(runtime, dict):
        devices = runtime.get("devices")
    expected_uuid = profile["gpu_uuid"]
    if (
        provenance.get("world_size") != 1
        or not isinstance(devices, list)
        or len(devices) != 1
        or not isinstance(devices[0], dict)
        or devices[0].get("uuid") != expected_uuid
        or devices[0].get("name") != "NVIDIA L4"
    ):
        raise ValueError("runtime device identity differs from ROOT's assigned L4")
    return {"metadata": metadata, "data": data, "config": config, "provenance": provenance}


def _validate_eval_event(
    event: dict[str, object], cell_data: dict[str, object], epoch: int
) -> tuple[bool, str | None]:
    """Check complete official-valid coverage and its source-bound monitoring context."""
    metrics, context = event.get("metrics"), event.get("context")
    if (
        event.get("schema") != "wnn.monitor.v1"
        or event.get("kind") != "evaluation"
        or type(event.get("epoch")) is not int
        or event.get("epoch") != epoch
        or type(event.get("step")) is not int
        or event.get("step") != epoch * UPDATES_PER_EPOCH
        or not isinstance(metrics, dict)
        or not isinstance(context, dict)
    ):
        return False, "event schema, epoch or fixed update boundary is invalid"
    members = cell_data["evaluation_members"]
    assert isinstance(members, list)
    member_sha = hashlib.sha256(json.dumps(members, separators=(",", ":")).encode()).hexdigest()
    if (
        context.get("split_sha256") != SPLIT_SHA256
        or context.get("evaluation_member_sha256") != member_sha
        or context.get("evaluation_member_count") != VALID_SEQUENCES
        or context.get("endpoint") != "raw standalone z; no reference or blend"
    ):
        return False, "evaluation context does not bind the approved complete validation cohort"
    if (
        type(metrics.get("valid/sequence_count")) is not int
        or metrics.get("valid/sequence_count") != VALID_SEQUENCES
        or type(metrics.get("valid/selected_rows")) is not int
        or metrics.get("valid/selected_rows") != VALID_ROWS
    ):
        return False, "full-validation sequence or selected-row coverage is incomplete"
    if not all(_finite_number(metrics.get(f"valid/wp/t{target}")) for target in (0, 1)):
        return False, "one or both selected-target weighted-Pearson values are missing"
    return True, None


def inspect_cell(artifact_root: Path, cell: str, admission: dict[str, object]) -> Cell:
    """Build one bounded cell view; invalid identity evidence is quarantined."""
    directory = artifact_root / "live-dashboard" / "native" / f"l4-{cell}-fit"
    result: Cell = {
        "cell": cell,
        "directory": directory,
        "identity_ok": False,
        "phase": "unknown",
        "step": None,
        "epoch": None,
        "train": None,
        "evaluations": [],
        "endpoint_ok": False,
        "terminal": None,
        "error": None,
    }
    try:
        owner = _read_json(directory / ".sequence_mirror_owner.json", 8_192)
        receipt = _read_json(directory / "collection_receipt.json", 64_000)
        source_root = str(admission["source_root"])
        run_root = source_root.rsplit("/", 1)[0] + "/runs"
        expected_remote = f"{run_root}/{cell}/fit"
        if (
            owner.get("host") != "l4"
            or owner.get("source_directory") != expected_remote
            or receipt.get("host") != "l4"
            or receipt.get("source_directory") != expected_remote
        ):
            raise ValueError("native mirror owner or collection receipt path mismatches admission")
        bound = _verify_metadata(cell, directory, admission)
        result["identity_ok"] = True
        progress = _read_json(directory / "monitor_progress.json", 16_384)
        result["phase"] = progress.get("phase", "unknown")
        raw_step = progress.get("last_successful_update", progress.get("step"))
        result["step"] = raw_step if type(raw_step) is int and raw_step >= 0 else None
        raw_epoch = progress.get("epoch")
        result["epoch"] = raw_epoch if _finite_number(raw_epoch) else None
        rank = _read_tail_jsonl(directory / "rank0.jsonl")
        if rank is not None:
            result["train"] = {
                "step": rank.get("step"),
                "loss": rank.get("total_loss"),
                "lr": rank.get("learning_rate"),
                "grad_preclip": rank.get("gradient_norm_preclip"),
                "gradient_clipped": rank.get("gradient_clipped"),
            }
        evaluations, journal_error = _read_evaluations(directory / "monitor_evaluations.jsonl")
        result["journal_error"] = journal_error
        checked: list[dict[str, object]] = []
        seen_epochs: set[int] = set()
        for event in evaluations:
            epoch = event.get("epoch")
            if type(epoch) is not int or epoch in seen_epochs:
                checked.append({"ok": False, "epoch": epoch, "error": "duplicate or invalid epoch"})
                continue
            seen_epochs.add(epoch)
            ok, error = _validate_eval_event(event, bound["data"], epoch)
            checked.append(
                {"ok": ok, "epoch": epoch, "event": event if ok else None, "error": error}
            )
        result["evaluations"] = checked
        terminal_path = directory / "terminal_status.json"
        terminal: dict[str, object] | None = None
        if terminal_path.exists():
            # Actual terminal receipts embed full membership provenance (~703 KB).
            value = _read_json(terminal_path, 2_000_000)
            identity = value.get("identity")
            if not isinstance(identity, dict):
                raise ValueError("terminal status lacks its run identity")
            prov = bound["provenance"]
            if (
                identity.get("source_sha256") != SOURCE_SHA256
                or identity.get("config_sha256") != admission["profiles"][cell]["config_sha256"]
                or identity.get("data_sha256") != prov.get("data_sha256")
                or identity.get("world_size") != 1
            ):
                raise ValueError("terminal status identity differs from the admitted run")
            status = value.get("status")
            if status == "completed" and (
                type(value.get("completed_epochs")) is not int
                or value.get("completed_epochs") != 6
                or type(value.get("global_updates")) is not int
                or value.get("global_updates") != EXPECTED_UPDATES
                or not _is_sha256(value.get("model_sha256"))
            ):
                raise ValueError("completed terminal record lacks exact six-epoch/model evidence")
            terminal = {
                "status": status,
                "completed_epochs": value.get("completed_epochs"),
                "global_updates": value.get("global_updates"),
                "model_sha256": value.get("model_sha256"),
                "record_sha256": hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
                "native_byte_verified": False,
            }
        else:
            observer = _read_tail_jsonl(directory / "monitor_observer.jsonl", 1_048_576)
            context = observer.get("context") if isinstance(observer, dict) else None
            returncode = context.get("worker_returncode") if isinstance(context, dict) else None
            if type(returncode) is int:
                terminal = {
                    "status": "worker-exit-0-unverified"
                    if returncode == 0
                    else f"worker-exit-{returncode}",
                    "returncode": returncode,
                    "record_sha256": hashlib.sha256(
                        json.dumps(observer, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "native_byte_verified": False,
                }
        result["terminal"] = terminal
        events = result["evaluations"]
        endpoint_ok = (
            terminal is not None
            and terminal.get("status") == "completed"
            and len(events) == 6
            and [item.get("epoch") for item in events] == [1, 2, 3, 4, 5, 6]
            and all(item.get("ok") is True for item in events)
            and isinstance(events[-1].get("event"), dict)
            and events[-1]["event"].get("step") == EXPECTED_UPDATES
        )
        result["endpoint_ok"] = endpoint_ok
        return result
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        json.JSONDecodeError,
    ) as error:
        result["error"] = str(error)
        return result


def inspect_snapshot(artifact_root: Path) -> Snapshot:
    """Read admission and six local mirror views; make no SSH, GPU or network calls."""
    admission = load_admission(artifact_root)
    cells = [inspect_cell(artifact_root, cell, admission) for cell in CELLS]
    return {"admission": admission, "cells": cells, "clock": _source_clock(artifact_root)}


def _source_clock(artifact_root: Path) -> dict[str, object]:
    """Measure mirror transport freshness in the local clock domain used by its bridge."""
    live = artifact_root / "live-dashboard"
    result: dict[str, object] = {
        "local_now": time.time(),
        "last_applied": None,
        "age_seconds": None,
    }
    try:
        bridge = _read_json(live / "bridge-state.json", 8_000_000)
        host = bridge["hosts"]["l4"]
        if not isinstance(host, dict):
            return result
        result["last_applied"] = host.get("last_applied_unix")
        if not _finite_number(host.get("last_applied_unix")):
            return result
        local_now = time.time()
        age = local_now - float(host["last_applied_unix"])
        result["local_now"] = local_now
        result["age_seconds"] = age if age >= 0 else None
        result["last_errors"] = host.get("last_errors", [])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return result
    return result


def _format_metric(value: object) -> str:
    return f"{float(value):.6f}" if _finite_number(value) else "unknown"


def _valid_metric(cell: Cell, target: int) -> object:
    events = cell.get("evaluations")
    if not isinstance(events, list):
        return None
    valid = [
        item for item in events if item.get("ok") is True and isinstance(item.get("event"), dict)
    ]
    if not valid:
        return None
    return valid[-1]["event"]["metrics"].get(f"valid/wp/t{target}")


def _cell_line(cell: Cell) -> str:
    name = str(cell["cell"])
    if cell.get("error"):
        return f"- `{name}` - evidence not verified: {cell['error']}"
    if cell.get("identity_ok") is not True:
        return f"- `{name}` - identity not verified; metrics withheld."
    train = cell.get("train")
    train = train if isinstance(train, dict) else {}
    terminal = cell.get("terminal")
    if isinstance(terminal, dict):
        phase = f"terminal `{terminal.get('status', 'unknown')}`"
    else:
        phase = f"{cell.get('phase', 'unknown')} {cell.get('epoch', 'unknown')}/6 E"
    val = " / ".join(_format_metric(_valid_metric(cell, target)) for target in (0, 1))
    model = (
        f"; trainer-reported model SHA `{str(terminal['model_sha256'])[:12]}...` (not byte-verified locally)"
        if isinstance(terminal, dict) and _is_sha256(terminal.get("model_sha256"))
        else ""
    )
    return (
        f"- `{name}` GPU{GPU_BINDINGS[name][0]} - {phase}; step "
        f"{cell.get('step', 'unknown')}/{EXPECTED_UPDATES}; train row "
        f"{train.get('step', 'unknown')} loss {_format_metric(train.get('loss'))}, "
        f"LR {_format_metric(train.get('lr'))}, pre-clip grad "
        f"{_format_metric(train.get('grad_preclip'))}, clipped {train.get('gradient_clipped', 'unknown')}; "
        f"latest full-valid WP t0/t1 {val}{model}"
    )


def grade_rows(cells: list[Cell]) -> list[dict[str, object]] | None:
    """Derive the fixed baseline and three declared pairings per seed from E6 only."""
    by_cell = {str(cell["cell"]): cell for cell in cells}
    if len(by_cell) != len(CELLS):
        return None
    for name in CELLS:
        terminal = by_cell[name].get("terminal")
        if (
            by_cell[name].get("identity_ok") is not True
            or not isinstance(terminal, dict)
            or terminal.get("status") != "completed"
            or by_cell[name].get("endpoint_ok") is not True
        ):
            return None

    def e6(cell: str, target: int) -> float:
        events = by_cell[cell]["evaluations"]
        event = events[-1]["event"]
        value = event["metrics"][f"valid/wp/t{target}"]
        if not _finite_number(value):
            raise ValueError("E6 target score is not finite")
        return float(value)

    recipes = (
        ("baseline shared control", "shared_control", "shared_control"),
        ("t0 specialist / shared t1", "t0_specialist", "shared_control"),
        ("shared t0 / t1 specialist", "shared_control", "t1_specialist"),
        ("t0 specialist / t1 specialist", "t0_specialist", "t1_specialist"),
    )
    rows: list[dict[str, object]] = []
    for seed in (203201, 203202):
        for label, left_arm, right_arm in recipes:
            left = f"{left_arm}_s{seed}"
            right = f"{right_arm}_s{seed}"
            t0, t1 = e6(left, 0), e6(right, 1)
            rows.append({"seed": seed, "recipe": label, "t0": t0, "t1": t1, "wp": 0.5 * (t0 + t1)})
    return rows


def render_report(snapshot: Snapshot, heading: str, source_now: float | None) -> str:
    """Render one bounded human-readable progress or completion comment."""
    admission = snapshot["admission"]
    cells = snapshot["cells"]
    clock = snapshot["clock"]
    timestamp = (
        datetime.fromtimestamp(source_now, UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        if source_now is not None
        else "local clock time unknown"
    )
    age = clock.get("age_seconds")
    freshness = f"{float(age):.0f}s" if _finite_number(age) else "unknown"
    lines = [
        f"## L4-NEXT70 PR77 - {heading}",
        f"Observed on the local clock {timestamp}; mirror last-applied age: {freshness}. Native event timestamps are left as recorded.",
        f"Admission source `{str(admission['source_sha256'])[:16]}...`; chronological split `{str(admission['split_sha256'])[:16]}...`.",
        "",
        "### Six admitted fits",
    ]
    errors = clock.get("last_errors")
    if isinstance(errors, list) and errors:
        lines.insert(
            2,
            f"Mirror collector reports {len(errors)} recent error(s); freshness is not a proof of complete per-file application.",
        )
    lines.extend(_cell_line(cell) for cell in cells)
    all_terminal = all(cell.get("terminal") is not None for cell in cells)
    rows = grade_rows(cells)
    lines.extend(["", "### Endpoint grading"])
    if rows is not None:
        lines.append(
            "All six fixed-E6 endpoints and six full-valid events pass the local identity and coverage gates. "
            "The trainer-reported model SHA values have not been byte-verified by this local reporter."
        )
        lines.append(
            "Scores are `0.5 * (left model valid/wp/t0 + right model valid/wp/t1)`; rows are per seed."
        )
        lines.extend(
            f"- Seed {row['seed']}, {row['recipe']}: t0={row['t0']:.6f}, t1={row['t1']:.6f}, complete-output WP={row['wp']:.6f}."
            for row in rows
        )
    elif all_terminal:
        lines.append(
            "All six workers have terminal evidence, but fixed-E6/full-valid coverage is incomplete or a worker did not complete; no complete-output scores are computed."
        )
    else:
        lines.append(
            "Endpoint scores are pending fixed-E6 terminal artifacts and all six exact-coverage full-valid events."
        )
    lines.extend(
        [
            "",
            "Official validation has been repeatedly exposed as development evidence. No validation-fitted weights, summed gains, significance claim, or public-score claim is used here.",
        ]
    )
    body = "\n".join(lines)
    if len(body.encode("utf-8")) > 30_000:
        raise ValueError("PR progress comment exceeds the 30 KB reporter bound")
    return body


def _load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"schema": STATE_SCHEMA, "reports": {}}
    state = _read_json(path, 2_000_000)
    if state.get("schema") != STATE_SCHEMA or not isinstance(state.get("reports"), dict):
        raise ValueError("publisher state file is malformed")
    return state


def _save_state(path: Path, state: dict[str, object]) -> None:
    """Atomically persist a pending marker before network POST for retry-safe deduplication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    payload = json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    if len(payload) > 2_000_000:
        raise ValueError("publisher state exceeds its 2 MB bound")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def publish_once(
    client: object,
    state_path: Path,
    marker: str,
    body: str,
    prepared_at: float,
) -> str:
    """Reconcile a marker, persist pending before POST, and record only confirmed success."""
    state = _load_state(state_path)
    reports = state["reports"]
    assert isinstance(reports, dict)
    saved = reports.get(marker)
    if isinstance(saved, dict) and saved.get("status") == "posted":
        return "already-posted"
    if isinstance(saved, dict) and saved.get("status") == "pending":
        body = str(saved["body"])
        prepared_at = float(saved["prepared_at"])
    found = client.find_report(77, marker, prepared_at)  # type: ignore[attr-defined]
    if found is not None:
        reports[marker] = {
            "status": "posted",
            "body": body,
            "prepared_at": prepared_at,
            "comment_id": found.get("id"),
        }
        _save_state(state_path, state)
        return "reconciled-existing"

    def before_post() -> None:
        reports[marker] = {"status": "pending", "body": body, "prepared_at": prepared_at}
        _save_state(state_path, state)

    result = client.post_report(77, body, before_post=before_post)  # type: ignore[attr-defined]
    reports[marker] = {
        "status": "posted",
        "body": body,
        "prepared_at": prepared_at,
        "comment_id": result.get("id"),
        "html_url": result.get("html_url"),
    }
    _save_state(state_path, state)
    return "posted"


def _load_client(deadline: float) -> object:
    """Load the existing narrowly allowlisted GitHub client only for live posting."""
    spec = importlib.util.spec_from_file_location("sequence_progress_github", CLIENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"existing GitHub progress client is unavailable: {CLIENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ProgressGitHub({77}, deadline=deadline)


def _terminal_markers(cells: list[Cell]) -> list[tuple[str, Cell]]:
    result: list[tuple[str, Cell]] = []
    for cell in cells:
        terminal = cell.get("terminal")
        if not isinstance(terminal, dict):
            continue
        digest = str(terminal.get("record_sha256", "unknown"))[:16]
        marker = f"<!-- wnn-l4-next70-pr77-completion-{cell['cell']}-{digest} -->"
        result.append((marker, cell))
    return result


def _output_recipes_marker(cells: list[Cell]) -> str:
    digest = canonical_sha256(
        [(cell.get("cell"), cell.get("terminal", {}).get("record_sha256")) for cell in cells]
    )[:16]
    return f"<!-- wnn-l4-next70-pr77-cohort-endpoints-{digest} -->"


def run(
    artifact_root: Path,
    state_path: Path,
    max_seconds: int,
    *,
    dryrun: bool = False,
    once: bool = False,
    client_factory: Callable[[float], object] = _load_client,
) -> int:
    """Publish on the current hour, each terminal event and completed cohort; stop at 12h."""
    if not 1 <= max_seconds <= MAX_WALL_SECONDS:
        raise ValueError(f"--maxseconds must be in [1, {MAX_WALL_SECONDS}]")
    deadline = time.monotonic() + max_seconds
    client = None if dryrun else client_factory(deadline)
    seen_errors: set[str] = set()
    while True:
        snapshot = inspect_snapshot(artifact_root)
        cells = snapshot["cells"]
        source_now = snapshot["clock"].get("local_now")
        prepared_at = float(source_now) if _finite_number(source_now) else time.time()
        interval_epoch = int(prepared_at // 1_800) * 1_800
        interval_bucket = datetime.fromtimestamp(interval_epoch, UTC).strftime("%Y%m%dT%H%M")
        hour_marker = f"<!-- wnn-l4-next70-pr77-30m-{interval_bucket}Z -->"
        due: list[tuple[str, str, float]] = []
        report = render_report(
            snapshot,
            "30-minute progress",
            float(source_now) if _finite_number(source_now) else None,
        )
        due.append((hour_marker, report, prepared_at))
        for marker, cell in _terminal_markers(cells):
            body = render_report(
                snapshot,
                f"terminal event: {cell['cell']} ({cell['terminal']['status']})",
                float(source_now) if _finite_number(source_now) else None,
            )
            body += f"\n\n{marker}"
            due.append((marker, body, prepared_at))
        rows = grade_rows(cells)
        all_terminal = all(cell.get("terminal") is not None for cell in cells)
        if rows is not None or (
            all_terminal
            and any(cell.get("terminal", {}).get("status") != "completed" for cell in cells)
        ):
            marker = _output_recipes_marker(cells)
            body = render_report(
                snapshot,
                "fixed-E6 cohort endpoints"
                if rows is not None
                else "cohort terminal status; endpoint grading incomplete",
                float(source_now) if _finite_number(source_now) else None,
            )
            body += f"\n\n{marker}"
            due.append((marker, body, prepared_at))

        for marker, body, created in due:
            body = (
                body + ("\n\n" if not body.endswith("\n") else "") + marker
                if marker not in body
                else body
            )
            if dryrun:
                print(body)
                continue
            try:
                outcome = publish_once(client, state_path, marker, body, created)
                print(f"{outcome}: {marker}")
            except (
                Exception
            ) as error:  # network/response uncertainty is retried via persistent marker
                message = f"{marker}: {type(error).__name__}: {error}"
                if message not in seen_errors:
                    print(message, file=sys.stderr)
                    seen_errors.add(message)
        terminal_failures = all_terminal and any(
            not isinstance(cell.get("terminal"), dict)
            or cell["terminal"].get("status") != "completed"
            for cell in cells
        )
        if once or (all_terminal and (rows is not None or terminal_failures)):
            return 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 0
        time.sleep(min(POLL_SECONDS, remaining))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="l4-next70-20261110 artifact root")
    parser.add_argument("--pr77", action="store_true", required=True, help="allow only PR 77")
    parser.add_argument("--maxseconds", type=int, default=MAX_WALL_SECONDS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dryrun",
        action="store_true",
        help="read local evidence and print reports without writing or network",
    )
    mode.add_argument("--once", action="store_true", help="publish one local snapshot and exit")
    parser.add_argument(
        "--state-file", type=Path, default=Path(__file__).with_name("publisher_state_pr77.json")
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.pr77:
        raise ValueError("--pr77 is required")
    return run(
        args.root,
        args.state_file,
        args.maxseconds,
        dryrun=args.dryrun,
        once=args.once or args.dryrun,
    )


if __name__ == "__main__":
    raise SystemExit(main())
