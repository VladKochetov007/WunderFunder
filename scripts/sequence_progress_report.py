from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.sequence_progress_panel import validate_l4_continuation, validate_recipe

# Native metadata/events contain heterogeneous nested JSON; fields are checked before use.
JsonObject = dict[str, Any]
EPOCHS = {"l4": 12, "b300": 30}
CELLS = {
    "l4": tuple(
        f"{family}_s{seed}"
        for family in ("gru", "gru_local", "mamba2", "ema_tabm_style")
        for seed in (101001, 101002)
    ),
    "b300": tuple(
        f"{family}_s{seed}" for family in ("labels", "complement") for seed in (101001, 101002)
    ),
}
SPLIT = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
MAX_TAIL_BYTES = 2_097_152


def canonical_sha(value: object) -> str:
    """Hash a JSON identity with the canonical producer encoding."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _object(path: Path) -> JsonObject:
    if path.is_symlink() or not path.exists():
        return {}
    if not path.is_file() or path.stat().st_size > 16 * 1_048_576:
        raise ValueError("artifact is not a bounded regular JSON file")
    value = json.loads(path.read_bytes())
    return value if isinstance(value, dict) else {}


def _events(path: Path) -> list[JsonObject]:
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
    rows = []
    for line in lines:
        if len(line) > 131_072:
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _sha(value: object) -> str | None:
    return (
        value.lower()
        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value)
        else None
    )


def _time(value: object) -> float | None:
    try:
        result = (
            datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else None
        )
        return result.timestamp() if result is not None and result.tzinfo is not None else None
    except (ValueError, OverflowError):
        return None


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _budget(
    metadata: JsonObject, host: str, panel: JsonObject | None = None
) -> tuple[int, dict[str, str]]:
    config, provenance = metadata["config"], metadata["provenance"]
    data = provenance["data"]
    members, evaluation = data["fit_members"], data["evaluation_members"]
    if (
        config["epochs"] != EPOCHS[host]
        or len(members) != 10325
        or len(evaluation) != 1873
        or any(member[0] != "train" for member in members)
        or data["split_sha256"] != SPLIT
    ):
        raise ValueError("declared fit/evaluation membership or epoch budget differs")
    if (
        len({tuple(member) for member in members}) != 10325
        or len({tuple(member) for member in evaluation}) != 1873
        or any(member[0] != "valid" for member in evaluation)
    ):
        raise ValueError("native membership is duplicated or has another evaluation source")
    if canonical_sha(data) != provenance.get("data_sha256"):
        raise ValueError("native data identity differs from its metadata")
    world, batch, chunk, accumulation = (
        provenance["world_size"],
        config["batch_size"],
        config["chunk_rows"],
        config["accumulate_chunks"],
    )
    if any(type(value) is not int or value < 1 for value in (world, batch, chunk, accumulation)):
        raise ValueError("invalid native update schedule")
    updates = (
        EPOCHS[host]
        * math.ceil(math.ceil(len(members) / world) / batch)
        * math.ceil(math.ceil(20000 / chunk) / accumulation)
    )
    if host == "b300" and panel is None and updates != 774600:
        raise ValueError("B300 production update budget differs from 774600")
    source = _sha(metadata.get("source", {}).get("sha256"))
    if source is None:
        raise ValueError("source identity is missing")
    return updates, {"source": source, "config": canonical_sha(config), "split": SPLIT}


def _fit_complete(
    terminal: JsonObject, metadata: JsonObject, ids: dict[str, str], host: str, updates: int
) -> bool:
    identity = terminal.get("identity", {})
    return (
        terminal.get("status") == "completed"
        and terminal.get("completed_epochs") == EPOCHS[host]
        and terminal.get("global_updates") == updates
        and _sha(terminal.get("model_sha256")) is not None
        and identity.get("source_sha256") == ids["source"]
        and identity.get("config_sha256") == ids["config"]
        and identity.get("data_sha256") == metadata["provenance"].get("data_sha256")
        and _sha(identity.get("data_sha256")) is not None
        and identity.get("world_size") == metadata["provenance"].get("world_size")
    )


def _full_evaluation(event: JsonObject) -> bool:
    metrics = event.get("metrics", {})
    context = event.get("context", {})
    split = context.get("valid", {}).get("split_sha256", context.get("split_sha256"))
    return (
        event.get("kind") == "evaluation"
        and metrics.get("valid/sequence_count") == 1873
        and metrics.get("valid/selected_rows") == 3528273
        and split == SPLIT
        and all(_finite(metrics.get(key)) for key in ("valid/wp", "valid/wp/t0", "valid/wp/t1"))
    )


def _fixed_grade(score: JsonObject, terminal: JsonObject, ids: dict[str, str]) -> bool:
    metrics, producer = score.get("score", {}), score.get("producer", {})
    composition, candidate = producer.get("composition", {}), producer.get("candidate", {})
    return (
        score.get("split_sha256") == ids["split"]
        and score.get("sequences") == 1873
        and metrics.get("metric_id") == "global_wp_clip2_v1"
        and metrics.get("selected_rows") == 3528273
        and all(_finite(metrics.get(key)) for key in ("wp", "t0", "t1"))
        and producer.get("kind") == "sequence-reference-blend-v1"
        and composition.get("method") == "linear_fp32_no_clip"
        and composition.get("alpha") == 0.1
        and composition.get("candidate_checkpoint_sha256") == terminal.get("model_sha256")
        and producer.get("composition_sha256") == canonical_sha(composition)
        and _sha(composition.get("reference_sha256")) is not None
        and producer.get("reference", {}).get("reference_sha256")
        == composition.get("reference_sha256")
        and candidate.get("checkpoint_sha256") == terminal.get("model_sha256")
        and canonical_sha(candidate.get("config", {})) == ids["config"]
    )


def observe_cell(
    root: Path, host: str, cell: str, now: float, panel: JsonObject | None = None
) -> JsonObject:
    """Summarize only a declared fit directory, never a profile or unbound endpoint."""
    continuation = (panel or {}).get("continuations", {}).get(cell) if host == "l4" else None
    folder = root / (
        continuation["run_id"]
        if continuation
        else f"{panel['run_prefix'] if panel else host}-{cell}-fit"
    )
    row: JsonObject = {
        "cell": cell,
        "phase": "fit not observed",
        "expected_epochs": EPOCHS[host],
        "fit_complete": False,
        "endpoint_complete": False,
        "updates": None,
        "epoch": None,
        "continuation": bool(continuation),
    }
    if folder.is_symlink() or not folder.is_dir():
        return row
    metadata = _object(folder / "run_metadata.json")
    if not metadata:
        return row | {"phase": "fit metadata pending"}
    budget, ids = _budget(metadata, host, panel)
    if panel is not None and host == "b300":
        validate_recipe(panel, cell, ids, budget)
    if continuation:
        validate_l4_continuation(continuation, cell, metadata, ids, budget)
    terminal, progress = (
        _object(folder / "terminal_status.json"),
        _object(folder / "monitor_progress.json"),
    )
    observer = _events(folder / "monitor_observer.jsonl")
    latest_observer = observer[-1] if observer else {}
    evaluations = [
        event for event in _events(folder / "monitor_evaluations.jsonl") if _full_evaluation(event)
    ]
    latest = evaluations[-1] if evaluations else {}
    train = [
        event
        for event in _events(folder / "monitor_events.jsonl")
        if event.get("kind") == "train_step"
    ]
    if host == "b300":
        train = _events(folder / "rank0.jsonl")
    clips = [
        event.get("metrics", {}).get("optimizer/was_clipped", event.get("gradient_clipped"))
        for event in train
    ]
    clips = [float(value) for value in clips if _finite(value)]
    fit_complete = _fit_complete(terminal, metadata, ids, host, budget)
    grade = _object(folder / "score.json")
    endpoint = (
        (fit_complete and latest.get("epoch") == 12 and latest.get("step") == budget)
        if host == "l4"
        else (fit_complete and _fixed_grade(grade, terminal, ids))
    )
    if (
        host == "l4"
        and endpoint
        and (continuation or latest.get("context", {}).get("run_identity_sha256"))
    ):
        endpoint = latest.get("context", {}).get("run_identity_sha256") == canonical_sha(
            terminal["identity"]
        )
    heartbeat = _time(progress.get("heartbeat_at"))
    observed = _time(latest_observer.get("wall_time"))
    times = [value for value in (heartbeat, observed) if value is not None]
    source_time = max(times) if times else None
    nonfinite = {}
    for event in train:
        values = event.get("metrics", event)
        nonfinite.update(
            {
                key: value
                for key, value in values.items()
                if ("nonfinite" in key or "skipped" in key) and _finite(value)
            }
        )
    phase = str(progress.get("phase", "source phase unavailable"))
    if fit_complete:
        phase = (
            "fit complete; endpoint recorded" if endpoint else "fit complete; final grade pending"
        )
    elif terminal.get("status") == "completed" or phase == "completed":
        phase = "terminal receipt awaiting identity/budget proof"
    elif terminal.get("status") in ("failed", "stopped", "timed_out", "early_stopped"):
        phase = f"fit incomplete; producer {terminal['status']}"
    elif latest_observer.get("metrics", {}).get("observer/process_alive_count") == 0:
        phase = f"observed process exit {latest_observer.get('context', {}).get('worker_returncode')}; fit incomplete"
    row.update(
        phase=phase,
        ids=ids,
        fit_complete=fit_complete,
        endpoint_complete=bool(endpoint),
        terminal_observation=terminal.get("status"),
        checkpoint_sha256=terminal.get("model_sha256") if fit_complete else None,
        updates=terminal.get("global_updates")
        if fit_complete
        else progress.get("last_successful_update", progress.get("step")),
        epoch=terminal.get("completed_epochs") if fit_complete else progress.get("epoch"),
        budget_updates=budget,
        latest_valid=latest.get("metrics", {}),
        latest_valid_epoch=latest.get("epoch"),
        fixed_grade=grade.get("score") if endpoint and host == "b300" else None,
        source_observed_utc=datetime.fromtimestamp(source_time, UTC).isoformat()
        if source_time
        else None,
        source_age_seconds=now - source_time
        if source_time is not None and source_time <= now
        else None,
        source_clock_ahead_seconds=source_time - now
        if source_time is not None and source_time > now
        else None,
        clipping=f"{sum(value > 0 for value in clips)}/{len(clips)} observed tail samples"
        if clips
        else "not recorded",
        nonfinite=", ".join(f"{key}={value:g}" for key, value in sorted(nonfinite.items()))
        or "not separately recorded",
        collection_utc=_object(folder / "collection_receipt.json").get("collected_at_unix"),
    )
    return row


def observe_host(root: Path, host: str, now: float, panel: JsonObject | None = None) -> JsonObject:
    """Collect a fixed host panel without fitting, scoring or scientific selection."""
    rows = []
    for cell in CELLS[host]:
        try:
            rows.append(observe_cell(root, host, cell, now, panel))
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            rows.append(
                {
                    "cell": cell,
                    "phase": "artifact evidence unavailable or inconsistent",
                    "fit_complete": False,
                    "endpoint_complete": False,
                    "expected_epochs": EPOCHS[host],
                }
            )
    terminal = [
        {
            "cell": row["cell"],
            "fit": row["fit_complete"],
            "endpoint": row["endpoint_complete"],
            "sha": row.get("checkpoint_sha256"),
            "producer_status": row.get("terminal_observation"),
            "exit": row["phase"] if "exit" in row["phase"] else None,
        }
        for row in rows
    ]
    snapshot = {
        "host": host,
        "observed_at_utc": datetime.fromtimestamp(now, UTC).isoformat(),
        "cells": rows,
        "terminal_signature": canonical_sha(terminal),
    }
    if panel is not None:
        snapshot["panel"] = {
            "run_prefix": panel["run_prefix"],
            "manifest_sha256": panel["manifest_sha256"],
        }
        snapshot["terminal_signature"] = canonical_sha(
            {"terminal": terminal, "panel": snapshot["panel"]}
        )
    return snapshot


def render_report(snapshot: JsonObject) -> str:
    """Render concise English progress with source ages and explicit completion limits."""
    host, rows = snapshot["host"], snapshot["cells"]
    completed = sum(row["fit_complete"] for row in rows)
    endpoints = sum(row["endpoint_complete"] for row in rows)
    lines = [
        f"{host.upper()} execution progress: {completed}/{len(rows)} fits complete; {endpoints}/{len(rows)} final endpoints recorded.",
        f"Report generated {snapshot['observed_at_utc']}. Cached source observations are shown below.",
        "",
        "| Cell | Last observed phase | Updates | Epochs | Latest full-valid WP (t0, t1) | Source age |",
        "|---|---|---:|---:|---|---:|",
    ]
    if snapshot.get("panel"):
        panel = snapshot["panel"]
        lines[1:1] = [
            f"Recipe panel `{panel['run_prefix']}`; frozen manifest SHA256 `{panel['manifest_sha256']}`. Earlier recipe reports remain historical."
        ]
        if host == "l4":
            lines[2:2] = [
                "The two GRU continuation rows retain the S1 epoch/log prefix; later updates use the pinned continuation source. Source IDs below identify the continuation producers. No quality gain is inferred."
            ]
    for row in rows:
        metrics = row.get("fixed_grade") or row.get("latest_valid", {})
        prefix = "" if row.get("fixed_grade") else "valid/"
        score = "pending"
        if _finite(metrics.get(prefix + "wp")):
            score = (
                f"{metrics[prefix + 'wp']:.6f} ({metrics[prefix + 'wp/t0']:.6f}, {metrics[prefix + 'wp/t1']:.6f})"
                if prefix
                else f"{metrics['wp']:.6f} ({metrics['t0']:.6f}, {metrics['t1']:.6f})"
            )
            score += (
                " fixed blend grade"
                if row.get("fixed_grade")
                else f" at epoch {row.get('latest_valid_epoch')}"
            )
        age = (
            f"{row['source_age_seconds']:.0f}s"
            if row.get("source_age_seconds") is not None
            else "unknown"
        )
        if row.get("source_clock_ahead_seconds") is not None:
            age = f"source clock ahead {row['source_clock_ahead_seconds']:.0f}s"
        epoch = f"{row['epoch']:.3f}" if _finite(row.get("epoch")) else "unknown"
        updates = row.get("updates") if type(row.get("updates")) is int else "unknown"
        lines.append(
            f"| {row['cell']} | {row['phase']} | {updates} | {epoch}/{EPOCHS[host]} | {score} | {age} |"
        )
    lines.extend(["", "Source/config/split identities and observed diagnostics:"])
    for row in rows:
        if row.get("ids"):
            ids = row["ids"]
            lines.append(
                f"- {row['cell']}: `{ids['source'][:12]}` / `{ids['config'][:12]}` / `{ids['split'][:12]}`; clipping {row['clipping']}; nonfinite counters {row['nonfinite']}; source UTC {row.get('source_observed_utc') or 'unknown'}."
            )
        if row.get("checkpoint_sha256"):
            lines.append(f"  Producer checkpoint SHA256: `{row['checkpoint_sha256']}`.")
    lines.extend(
        [
            "",
            "Fit completion records the producer checkpoint SHA; artifact upload and independent byte verification remain pending.",
            "Official valid is development-exposed. Scientific interpretation remains with ROOT.",
        ]
    )
    return "\n".join(lines)
