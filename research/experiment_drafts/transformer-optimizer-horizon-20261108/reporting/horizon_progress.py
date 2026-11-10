from __future__ import annotations

import hashlib
import json
import math
import re
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from collect_readouts import (
    EXPECTED_CELLS,
    MANIFEST_SHA256,
    MAX_TOTAL_BYTES,
    PLAN_SHA256,
    SOURCE_COMMIT,
    SPLIT_SHA256,
)

SOURCE_SHA256 = "569123615f69cc8adae3ff362ce5db720dbd068dec3a5b0f94e2d6dc3201be6c"
EXPECTED_CONFIGS = {
    "C0_muon256_a4_s203101": "5881e2facddc48a0023f588c7e47dfd02e065e823bc94d565faf36ec3d7594c9",
    "C1_adamw256_a4_s203101": "6cd1f8637c55a9c329acaa6bf1bcbd9f42ce3e585bc3e477d8f9e550c26adda6",
    "C2_muon512_a2_s203101": "6833575632e06baa04bdcd526df41f938423a1dbd0d11a2726ed35f35c96c5f7",
    "C0_muon256_a4_s203102": "7714bc8950d9d3ec881ddc993ec94842827217c54f5bb9dcd80b663751b811ab",
    "C1_adamw256_a4_s203102": "8ee20d8f8341ed198fa8e49acaba37c1d47dc30754495b8037151c8eadeb51e7",
    "C2_muon512_a2_s203102": "232c3b9ed836f841640feecb41c18cb4ad29cb0cf2d689c55dddf9632814bc3e",
}
MAX_JSON_BYTES = 4 * 1_048_576
MAX_TAIL_BYTES = 2 * 1_048_576
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
JsonObject = dict[str, Any]


def canonical_sha(value: object) -> str:
    """Hash JSON identities with the producer's stable encoding."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        .encode()
    ).hexdigest()


def finite(value: object) -> bool:
    """Accept only finite JSON metric scalars."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def object_record(path: Path, max_bytes: int = MAX_JSON_BYTES) -> tuple[JsonObject, str]:
    """Read one bounded native JSON object, refusing symlinks and malformed roots."""
    if path.is_symlink() or not path.exists():
        return {}, ""
    before = path.stat()
    if not path.is_file() or before.st_size > max_bytes:
        raise ValueError("native report artifact is not a bounded regular file")
    payload = path.read_bytes()
    after = path.stat()
    if len(payload) != before.st_size or (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("native report artifact changed during read")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("native report artifact must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def object_file(path: Path, max_bytes: int = MAX_JSON_BYTES) -> JsonObject:
    """Return one bounded native JSON object without retaining its byte digest."""
    return object_record(path, max_bytes)[0]


def event_tail(path: Path) -> list[JsonObject]:
    """Read complete recent JSONL records under a strict byte and line cap."""
    if path.is_symlink() or not path.exists():
        return []
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        start = max(0, size - MAX_TAIL_BYTES)
        stream.seek(start)
        payload = stream.read(MAX_TAIL_BYTES)
    lines = payload.split(b"\n")[:-1]
    if start:
        lines = lines[1:]
    result = []
    for line in lines:
        if len(line) > 131_072:
            continue
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def load_panel(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Bind every report row to ROOT's exact source-bound receiver plan."""
    if path.is_symlink() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("frozen receiver plan is not a bounded regular file")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    plan = json.loads(payload)
    cells = plan.get("cells", [])
    if (
        expected_sha256 != PLAN_SHA256
        or digest != expected_sha256
        or plan.get("schema") != "wnn.transformer-optimizer-horizon-panel.v1"
        or plan.get("source_commit") != SOURCE_COMMIT
        or plan.get("source_manifest_sha256") != MANIFEST_SHA256
        or plan.get("expected_training_source_sha256") != SOURCE_SHA256
        or plan.get("split_sha256") != SPLIT_SHA256
        or [cell.get("cell") for cell in cells] != list(EXPECTED_CELLS)
        or any(cell.get("config_sha256") != EXPECTED_CONFIGS[cell["cell"]] for cell in cells)
    ):
        raise ValueError("plan differs from ROOT's exact six-cell receiver panel")
    return {"sha256": digest, "plan": plan, "cells": {c["cell"]: c for c in cells}}


def _metadata_ids(metadata: JsonObject, cell: dict[str, Any]) -> dict[str, str]:
    """Check label-only recipe, source/data identity and complete chronological membership."""
    config = metadata["config"]
    provenance = metadata["provenance"]
    data = provenance["data"]
    fit_members, evaluation = data["fit_members"], data["evaluation_members"]
    if (
        metadata["source"].get("sha256") != SOURCE_SHA256
        or canonical_sha(config) != cell["normalized_config_sha256"]
        or provenance.get("world_size") != 2
        or provenance.get("data_sha256") != canonical_sha(data)
        or data.get("split_sha256") != SPLIT_SHA256
        or data.get("teacher") is not None
        or config.get("objective_mode") != "standalone_labels_wmse_v2"
        or config.get("optimizer") != cell["optimizer"]
        or config.get("seed") != cell["seed"]
        or config.get("epochs") != 4
        or config.get("batch_size") != 128
        or config.get("chunk_rows") != (512 if cell["arm"] == "C2" else 256)
        or config.get("accumulate_chunks") != (2 if cell["arm"] == "C2" else 4)
        or data.get("training_mask") != "all_after_warmup"
        or data.get("normalization", {}).get("scope") != "all_fit"
        or data.get("normalization", {}).get("sequence_count") != 10325
        or data.get("objective_contract", {}).get("reference_training_ancestry") != "none"
        or data.get("report_only_monitor", {}).get("generalization_gap_comparable") is not False
        or data.get("report_only_monitor", {}).get("evaluation_mask") != "is_scored after warmup"
        or data.get("report_only_monitor", {}).get("evaluation_member_count") != 1873
        or len(fit_members) != 10325
        or len({tuple(member) for member in fit_members}) != 10325
        or any(member[0] != "train" for member in fit_members)
        or len(evaluation) != 1873
        or len({tuple(member) for member in evaluation}) != 1873
        or any(member[0] != "valid" for member in evaluation)
    ):
        raise ValueError("producer run metadata differs from the frozen train-only recipe")
    return {
        "source": SOURCE_SHA256,
        "config": canonical_sha(config),
        "data": provenance["data_sha256"],
        "split": SPLIT_SHA256,
    }


def _full_valid_events(events: list[JsonObject], metadata: JsonObject) -> list[JsonObject]:
    """Keep only complete official valid raw-z events from the bound v2 monitor."""
    context = metadata["provenance"]["data"]["report_only_monitor"]
    valid = []
    for event in events:
        metrics = event.get("metrics", {})
        if (
            event.get("kind") == "evaluation"
            and type(event.get("epoch")) is int
            and 1 <= event["epoch"] <= 4
            and event.get("step") == event["epoch"] * 820
            and event.get("context") == context
            and context.get("split_sha256") == SPLIT_SHA256
            and context.get("evaluation_member_count") == 1873
            and metrics.get("valid/sequence_count") == 1873
            and metrics.get("valid/selected_rows") == 3_528_273
            and metrics.get("valid/wp") == metrics.get("valid/z/wp")
            and all(finite(metrics.get(key)) for key in ("valid/wp", "valid/wp/t0", "valid/wp/t1"))
            and all(finite(value) for value in metrics.values())
        ):
            valid.append(event)
    return valid


def _collected_object(readouts: dict[str, Any], relative: str) -> tuple[dict[str, Any], str] | None:
    """Return a readout-only RPC artifact and its byte digest when present."""
    files = readouts.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("readout bundle files must be a mapping")
    entry = files.get(relative, {})
    if not isinstance(entry, dict):
        raise ValueError("readout bundle entry must be a mapping")
    if entry.get("status") != "read":
        return None
    digest = entry.get("sha256")
    value = entry.get("value")
    if (
        not isinstance(digest, str)
        or not SHA256_RE.fullmatch(digest)
        or type(entry.get("bytes")) is not int
        or not 0 <= entry["bytes"] <= MAX_JSON_BYTES
        or not isinstance(value, dict)
    ):
        raise ValueError("readout collector entry is malformed")
    return value, digest


def _readout_object(
    folder: Path, filename: str, readouts: dict[str, Any], cell: str
) -> tuple[dict[str, Any], str] | None:
    """Prefer the collector's exact bytes; fall back to the existing local mirror."""
    relative = f"runs/{cell}/fit/{filename}"
    collected = _collected_object(readouts, relative)
    if collected is not None:
        return collected
    path = folder / filename
    if not path.exists():
        return None
    return object_record(path)


def _canonical_grade(
    terminal: JsonObject | None,
    score_record: tuple[JsonObject, str] | None,
    ids: dict[str, str],
    metadata: JsonObject,
) -> dict[str, Any] | None:
    """Require fixed E4/full-valid score with matching producer and terminal hashes."""
    if terminal is None or score_record is None:
        return None
    score, score_sha = score_record
    producer, metrics = score.get("producer", {}), score.get("score", {})
    identity = terminal.get("identity", {})
    checkpoint_sha = terminal.get("model_sha256")
    if (
        terminal.get("status") != "completed"
        or terminal.get("completed_epochs") != 4
        or terminal.get("global_updates") != 3280
        or not isinstance(checkpoint_sha, str)
        or not SHA256_RE.fullmatch(checkpoint_sha)
        or not all(identity.get(f"{key}_sha256") == ids[key] for key in ("source", "config", "data"))
        or identity.get("world_size") != 2
        or score.get("split_sha256") != SPLIT_SHA256
        or score.get("sequences") != 1873
        or metrics.get("metric_id") != "global_wp_clip2_v1"
        or metrics.get("selected_rows") != 3_528_273
        or not all(finite(metrics.get(key)) for key in ("wp", "t0", "t1"))
        or producer.get("checkpoint_sha256") != checkpoint_sha
        or producer.get("config") != metadata["config"]
        or producer.get("provenance", {}).get("identity") != identity
        or producer.get("teacher") is not None
        or "reference" in producer
    ):
        return None
    return {
        "wp": metrics["wp"],
        "t0": metrics["t0"],
        "t1": metrics["t1"],
        "checkpoint_sha256": checkpoint_sha,
        "score_sha256": score_sha,
        "status": "grade_producer_verified_pending_final_byte_audit",
    }


def _comparison_valid(
    comparison: tuple[JsonObject, str] | None, rows: list[dict[str, Any]]
) -> bool:
    """Require ROOT's all-six comparison and byte hashes before claiming final audit."""
    if comparison is None or len(rows) != 6:
        return False
    value, _ = comparison
    cells, deltas = value.get("cells", {}), value.get("paired_deltas", [])
    expected_deltas = {(seed, arm) for seed in (203101, 203102) for arm in ("C1", "C2")}
    if not (
        value.get("schema") == "wnn.transformer-optimizer-horizon-comparison.v1"
        and value.get("status") == "complete_exposed_fixed4E_screen"
        and value.get("metric_id") == "global_wp_clip2_v1"
        and value.get("split_sha256") == SPLIT_SHA256
        and isinstance(cells, dict)
        and set(cells) == set(EXPECTED_CELLS)
        and isinstance(deltas, list)
        and len(deltas) == 4
        and all(isinstance(entry, dict) for entry in deltas)
        and {(entry.get("seed"), entry.get("arm")) for entry in deltas} == expected_deltas
        and all(finite(entry.get("delta_wp")) for entry in deltas)
    ):
        return False
    for row in rows:
        grade, record = row.get("grade"), cells.get(row["cell"], {})
        if (
            not grade
            or record.get("checkpoint_sha256") != grade["checkpoint_sha256"]
            or record.get("score_sha256") != grade["score_sha256"]
            or record.get("official_raw_z", {}).get("wp") != grade["wp"]
        ):
            return False
    return True


def _source_clock(progress: JsonObject, observer: list[JsonObject]) -> float | None:
    """Return the latest producer/observer timestamp, never controller status time."""
    values = []
    heartbeat = progress.get("heartbeat_at")
    if isinstance(heartbeat, str):
        with suppress(ValueError):
            values.append(datetime.fromisoformat(heartbeat.replace("Z", "+00:00")).timestamp())
    if observer and isinstance(observer[-1].get("wall_time"), str):
        with suppress(ValueError):
            values.append(
                datetime.fromisoformat(observer[-1]["wall_time"].replace("Z", "+00:00")).timestamp()
            )
    return max(values, default=None)


def observe_cell(
    native_root: Path,
    cell: dict[str, Any],
    readouts: dict[str, Any],
    comparison: tuple[JsonObject, str] | None,
    now: float,
) -> dict[str, Any]:
    """Observe one exact producer directory; never read controller liveness state."""
    name = cell["cell"]
    folder = native_root / f"b300-{name}-fit"
    row: dict[str, Any] = {
        "cell": name,
        "arm": cell["arm"],
        "seed": cell["seed"],
        "optimizer": cell["optimizer"],
        "chunk_rows": 512 if cell["arm"] == "C2" else 256,
        "pair": cell["pair"],
        "phase": "producer artifacts pending",
        "fit_complete": False,
        "grade_status": "pending",
        "byte_audit": False,
    }
    if folder.is_symlink() or not folder.is_dir():
        return row
    metadata = object_file(folder / "run_metadata.json")
    if not metadata:
        return row | {"phase": "producer metadata pending"}
    ids = _metadata_ids(metadata, cell)
    progress = object_file(folder / "monitor_progress.json")
    observer = event_tail(folder / "monitor_observer.jsonl")
    train = event_tail(folder / "rank0.jsonl")
    evaluations = _full_valid_events(event_tail(folder / "monitor_evaluations.jsonl"), metadata)
    terminal_record = _readout_object(folder, "terminal_status.json", readouts, name)
    score_record = _readout_object(folder, "score.json", readouts, name)
    terminal = terminal_record[0] if terminal_record else None
    identity = terminal.get("identity", {}) if terminal else {}
    complete = bool(
        terminal
        and terminal.get("status") == "completed"
        and terminal.get("completed_epochs") == 4
        and terminal.get("global_updates") == 3280
        and isinstance(terminal.get("model_sha256"), str)
        and SHA256_RE.fullmatch(terminal["model_sha256"])
        and all(identity.get(f"{key}_sha256") == ids[key] for key in ("source", "config", "data"))
        and identity.get("world_size") == 2
    )
    grade = _canonical_grade(terminal, score_record, ids, metadata)
    source_time = _source_clock(progress, observer)
    age = None if source_time is None else now - source_time
    phase = str(progress.get("phase", "producer phase unavailable"))
    if terminal and terminal.get("status") in ("failed", "stopped", "timed_out", "early_stopped"):
        phase = f"producer {terminal['status']}; endpoint incomplete"
    elif complete:
        phase = "fixed4E fit complete"
    elif age is not None and age > 300:
        phase = f"stale last producer observation ({age / 60:.0f}m): {phase}"
    elif observer and observer[-1].get("metrics", {}).get("observer/process_alive_count") == 0:
        phase = "observed process exit; terminal receipt pending"
    else:
        phase = f"latest producer phase: {phase}"
    latest = max(evaluations, key=lambda event: event["epoch"], default={})
    sample = next(
        (event for event in reversed(train) if isinstance(event.get("optimizer_update_diagnostics"), dict)),
        {},
    )
    observer_event = observer[-1] if observer else {}
    resource_metrics = observer_event.get("metrics", {})
    devices = [int(index) for index in cell["pair"].split(",")]
    utilization = [resource_metrics.get(f"gpu/device_{index}/utilization_pct") for index in devices]
    memory_gib = [resource_metrics.get(f"gpu/device_{index}/memory_used_bytes") for index in devices]
    utilization = [float(value) for value in utilization if finite(value)]
    memory_gib = [float(value) / (1024**3) for value in memory_gib if finite(value)]
    return row | {
        "ids": ids,
        "phase": phase,
        "fit_complete": complete,
        "terminal_observation": terminal.get("status") if terminal else None,
        "updates": progress.get("last_successful_update"),
        "epoch": progress.get("last_successful_update", 0) / 820,
        "rank0_step": train[-1].get("step") if train else None,
        "latest_valid_epoch": latest.get("epoch"),
        "latest_valid": latest.get("metrics", {}),
        "latest_valid_is_provisional": bool(latest),
        "grade": grade,
        "grade_status": grade["status"] if grade else "canonical official score pending or unverified",
        "byte_audit": False,
        "source_age_seconds": age,
        "probe": sample.get("optimizer_update_diagnostics", {}),
        "probe_step": sample.get("step"),
        "train_sample_count": len(train),
        "latest_train": train[-1] if train else {},
        "gpu_utilization_pct": utilization,
        "gpu_memory_gib": memory_gib,
    }


def snapshot(
    native_root: Path,
    panel: dict[str, Any],
    readout_path: Path | None,
    now: float,
) -> dict[str, Any]:
    """Read the six exact mirrored runs and optional ROOT readout-only collection."""
    readouts: dict[str, Any] = {}
    if readout_path is not None and readout_path.exists():
        readouts = object_file(readout_path, MAX_TOTAL_BYTES + MAX_JSON_BYTES)
        if (
            readouts.get("schema") != "wnn.transformer-optimizer-horizon-readouts.v1"
            or readouts.get("plan_sha256") != panel["sha256"]
            or readouts.get("source_commit") != SOURCE_COMMIT
            or type(readouts.get("total_bytes")) is not int
            or readouts["total_bytes"] > MAX_TOTAL_BYTES
        ):
            raise ValueError("ROOT readout bundle differs from the frozen six-cell panel")
    comparison = _collected_object(readouts, "comparison.json")
    rows = []
    for cell in panel["plan"]["cells"]:
        try:
            rows.append(observe_cell(native_root, cell, readouts, comparison, now))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            rows.append(
                {
                    "cell": cell["cell"],
                    "arm": cell["arm"],
                    "seed": cell["seed"],
                    "phase": f"native evidence unavailable/inconsistent: {type(error).__name__}",
                    "fit_complete": False,
                    "grade_status": "pending",
                    "byte_audit": False,
                }
            )
    comparison_valid = _comparison_valid(comparison, rows)
    for row in rows:
        row["byte_audit"] = bool(comparison_valid and row.get("grade"))
        if row["byte_audit"]:
            row["grade_status"] = "fixed E4 checkpoint byte audit verified by ROOT comparison"
    completion_signature = [
        {
            "cell": row["cell"],
            "failure": row.get("terminal_observation")
            if row.get("terminal_observation") in ("failed", "stopped", "timed_out", "early_stopped")
            else None,
            "grade": row.get("grade"),
            "byte_audit": row["byte_audit"],
        }
        for row in rows
    ]
    compare_sha = comparison[1] if comparison_valid and comparison else None
    return {
        "host": "b300-transformer-optimizer-horizon-20261108",
        "observed_at_utc": datetime.fromtimestamp(now, UTC).isoformat(),
        "panel": {"manifest_sha256": panel["sha256"], "source_commit": SOURCE_COMMIT},
        "cells": rows,
        "comparison_sha256": compare_sha,
        "comparison_present": bool(comparison_valid),
        "paired_deltas": comparison[0].get("paired_deltas", []) if comparison_valid else [],
        "terminal_signature": canonical_sha(
            {"cells": completion_signature, "comparison_sha256": compare_sha}
        ),
    }


def _metric(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    return f"{value:.6f}" if finite(value) else "pending"


def _range(values: list[float], precision: str = ".2g") -> str:
    """Format metric ranges without rounding small LR/weight ratios to zero."""
    if not values:
        return "pending"
    return f"{format(min(values), precision)}–{format(max(values), precision)}"


def render(value: dict[str, Any]) -> str:
    """Render provisional monitors and producer-verified fixed grades distinctly."""
    rows = value["cells"]
    fits = sum(row["fit_complete"] for row in rows)
    grades = sum(bool(row.get("grade")) for row in rows)
    audits = sum(row["byte_audit"] for row in rows)
    lines = [
        f"Transformer optimizer/horizon screen: {fits}/6 fixed4E fits complete; {grades}/6 producer-verified official grades; {audits}/6 final checkpoint byte audits.",
        f"Read-only snapshot at {value['observed_at_utc']}. Frozen receiver plan SHA256 `{value['panel']['manifest_sha256']}`; source `{SOURCE_COMMIT}`.",
        "The receiver controller status file is not used for liveness; phase and freshness come from producer progress and observer records.",
        "",
        "| Cell | Recipe | Producer phase | Update / epoch | Latest full-valid monitor (provisional WP) | Fixed E4 official WP | Status / freshness |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in rows:
        monitor = row.get("latest_valid", {})
        monitor_text = "pending"
        if row.get("latest_valid_is_provisional"):
            monitor_text = (
                f"E{row['latest_valid_epoch']} {_metric(monitor, 'valid/wp')} "
                f"({_metric(monitor, 'valid/wp/t0')}, {_metric(monitor, 'valid/wp/t1')})"
            )
        grade = row.get("grade")
        grade_text = (
            f"{grade['wp']:.6f} ({grade['t0']:.6f}, {grade['t1']:.6f})"
            if grade
            else "pending"
        )
        updates = row.get("updates")
        update_text = str(updates) if type(updates) is int else "unknown"
        epoch = row.get("epoch")
        epoch_text = f"{epoch:.3f}/4" if finite(epoch) else "unknown/4"
        age = row.get("source_age_seconds")
        freshness = "unknown" if age is None else f"{age:.0f}s old"
        if age is not None and age < 0:
            freshness = f"source clock ahead {-age:.0f}s"
        if grade and row.get("byte_audit"):
            status = "checkpoint bytes verified by comparison"
        elif grade:
            status = "grade producer hash verified; final byte audit pending"
        else:
            status = row["grade_status"]
        recipe = f"{row['optimizer']} / chunk{row['chunk_rows']} / seed{row['seed']}"
        lines.append(
            f"| {row['cell']} | {recipe} | {row['phase']} | {update_text} / {epoch_text} | {monitor_text} | {grade_text} | {status}; {freshness} |"
        )
    lines.extend(["", "Latest sampled optimizer/resource diagnostics:"])
    for row in rows:
        probe = row.get("probe", {})
        groups = probe.get("groups", {}) if isinstance(probe, dict) else {}
        lr_values = [
            item.get("applied_lr") for item in groups.values() if finite(item.get("applied_lr"))
        ]
        ratios = [
            item.get("delta_weight_over_pre_weight_norm")
            for item in groups.values()
            if finite(item.get("delta_weight_over_pre_weight_norm"))
        ]
        train = row.get("latest_train", {})
        update_text = row.get("probe_step") or "pending"
        lines.append(
            f"- {row['cell']}: probe update {update_text}; applied LR range {_range(lr_values, '.3e')}; "
            f"group ΔW/W range {_range(ratios)}; preclip norm {_metric(probe, 'preclip_gradient_norm')}; "
            f"assigned-pair GPU utilization {_range(row.get('gpu_utilization_pct', []), '.0f')}%; "
            f"memory {_range(row.get('gpu_memory_gib', []), '.1f')} GiB; "
            f"rank0 peak allocated {(_metric({'m': train.get('peak_memory_bytes', 0) / 1024**3}, 'm') if finite(train.get('peak_memory_bytes')) else 'pending')} GiB."
        )
    if value.get("comparison_present"):
        lines.extend(
            [
                "",
                f"ROOT comparison readout is validated (SHA256 `{value['comparison_sha256']}`); paired deltas:",
            ]
        )
        for delta in value["paired_deltas"]:
            lines.append(
                f"- Seed {delta['seed']} {delta['contrast']}: ΔWP {delta['delta_wp']:+.6f}."
            )
    else:
        lines.extend(["", "Paired-seed deltas are pending ROOT's completed comparison readout."])
    lines.extend(
        [
            "",
            "Full-valid monitor values are provisional, same-forward raw-z diagnostics. Canonical grades require the fixed-E4 official score, 1,873 sequences / 3,528,273 scored rows, and matching terminal/producer checkpoint hashes. Before the ROOT comparison readout arrives, no independent model-file byte audit is claimed.",
            "C1 is an AdamW recipe contrast that also changes decoupled weight-decay coverage; it does not isolate optimizer family alone.",
            "Official validation is repeatedly exposed development evidence. This progress report makes no significance, public-gain, or promotion claim.",
        ]
    )
    return "\n".join(lines)
