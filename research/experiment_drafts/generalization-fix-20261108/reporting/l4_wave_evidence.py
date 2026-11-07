from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

from scripts.sequence_progress_report import _events, _finite, _object, _sha, _time, canonical_sha

# These dictionaries contain heterogeneous, checked native JSON artifacts.
JsonObject = dict[str, object]
SOURCE = "5bbbf68eccdc8c8773ff4b4eda2c4026d9c3f31b145dfc0d683b17d20a1987d0"
SPLIT = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
REFERENCE = "1863e5e6f844c8f172f3fd54879ff908e0e415183766c354482b1ac9c5bfd985"
POSTFIT_SOURCE = "0042a76f80f685666da7f62f263b2d355d0c35c6eca18a1e0d1fe39097660bab"
PANEL = "b4f27c0addeff1125842da4bb067de51a5f6896c657832fbf58c2a0d7ac268d4"
INVENTORY = "2b8bfe5454d03e716757bb65f1ecc280a0a748c216ae2cf4c0876bfb15d9de40"
CELLS = tuple(
    f"{arm}_s{seed}"
    for arm in ("raw112_d0", "raw112_d02", "event211_d02")
    for seed in (101101, 101102)
)
EPOCHS, UPDATES, ROWS = 6, 38760, 3528273


def inventory() -> dict[str, dict[str, str]]:
    """Read the frozen actor inventory without importing model/training code."""
    value = _object(Path(__file__).parents[1] / "l4/CONFIG_INVENTORY_R1.json")
    if set(value) != set(CELLS) or canonical_sha(value) != INVENTORY:
        raise ValueError("six-cell frozen config inventory differs")
    return value


def identity(metadata: dict, cell: str) -> dict:
    """Bind the actual normalized config, source, membership and six-epoch schedule."""
    config, data = metadata["config"], metadata["provenance"]["data"]
    provenance, source = metadata["provenance"], metadata["source"]
    expected = inventory()[cell]["normalized"]
    if (
        canonical_sha(config) != expected
        or config["seed"] != int(cell[-6:])
        or config["epochs"] != EPOCHS
        or config["batch_size"] != 32
        or config["chunk_rows"] != 1000
        or config["accumulate_chunks"] != 1
        or provenance["world_size"] != 1
        or source["sha256"] != SOURCE
        or canonical_sha(source["files"]) != SOURCE
        or data["split_sha256"] != SPLIT
        or canonical_sha(data) != provenance["data_sha256"]
        or data["teacher"] is not None
        or data["normalization"]["scope"] != "all_fit"
        or data["normalization"]["sequence_count"] != 10325
    ):
        raise ValueError("native recipe/source/normalization differs from the fixed wave")
    for role, count, origin in (
        ("fit_members", 10325, "train"),
        ("evaluation_members", 1873, "valid"),
    ):
        members = data[role]
        if (
            len(members) != count
            or len({tuple(m) for m in members}) != count
            or any(m[0] != origin for m in members)
        ):
            raise ValueError("native whole-sequence membership differs")
    if EPOCHS * math.ceil(10325 / 32) * math.ceil(20000 / 1000) != UPDATES:
        raise ValueError("six-epoch update derivation differs")
    return {"source": SOURCE, "config": expected, "split": SPLIT, "data": provenance["data_sha256"]}


def full_valid(event: dict) -> bool:
    """Accept only the native complete official validation event, never a sampled view."""
    metrics, context = event.get("metrics", {}), event.get("context", {})
    valid = context.get("valid", {})
    return (
        event.get("kind") == "evaluation"
        and type(event.get("epoch")) is int
        and 1 <= event["epoch"] <= EPOCHS
        and event.get("step") == event["epoch"] * (UPDATES // EPOCHS)
        and valid.get("split_sha256", context.get("split_sha256")) == SPLIT
        and metrics.get("valid/sequence_count") == 1873
        and metrics.get("valid/selected_rows") == ROWS
        and all(_finite(metrics.get(k)) for k in ("valid/wp", "valid/wp/t0", "valid/wp/t1"))
        and all(_finite(value) for value in metrics.values())
    )


def fit_complete(terminal: dict, ids: dict, evaluations: list[dict]) -> bool:
    """Record producer completion only with exact epochs/updates/SHA and all six grades."""
    native = terminal.get("identity", {})
    identity_matches = all(
        native.get(f"{key}_sha256") == ids[key] for key in ("source", "config", "data")
    )
    return (
        terminal.get("status") == "completed"
        and terminal.get("completion_kind") == "max_epochs"
        and terminal.get("completed_epochs") == EPOCHS
        and terminal.get("requested_epochs") == EPOCHS
        and terminal.get("global_updates") == UPDATES
        and _sha(terminal.get("model_sha256")) is not None
        and identity_matches
        and native.get("world_size") == 1
        and len(evaluations) == EPOCHS
        and {e["epoch"] for e in evaluations} == set(range(1, EPOCHS + 1))
        and all(
            e.get("context", {}).get("run_identity_sha256") == canonical_sha(native)
            for e in evaluations
        )
    )


def observe_fit(root: Path, cell: str, now: float) -> dict:
    """Observe only the exact new fit ID; never infer death from stale log timestamps."""
    folder = root / f"l4-{cell}-fit"
    row = {"cell": cell, "fit_complete": False, "grades": {}, "phase": "fit not observed"}
    if folder.is_symlink() or not folder.is_dir():
        return row
    metadata = _object(folder / "run_metadata.json")
    if not metadata:
        return row | {"phase": "fit metadata pending"}
    ids = identity(metadata, cell)
    progress, terminal = (
        _object(folder / "monitor_progress.json"),
        _object(folder / "terminal_status.json"),
    )
    observer, train = (
        _events(folder / "monitor_observer.jsonl"),
        _events(folder / "monitor_events.jsonl"),
    )
    evaluations = [e for e in _events(folder / "monitor_evaluations.jsonl") if full_valid(e)]
    complete = fit_complete(terminal, ids, evaluations)
    phase = str(progress.get("phase", "source phase unavailable"))
    if complete:
        phase = "fit complete; producer checkpoint SHA recorded"
    elif terminal:
        phase = f"producer {terminal.get('status', 'status unavailable')}; completion proof pending"
    elif observer and observer[-1].get("metrics", {}).get("observer/process_alive_count") == 0:
        phase = "observed process exit; fit incomplete"
    times = [
        _time(progress.get("heartbeat_at")),
        _time(observer[-1].get("wall_time")) if observer else None,
    ]
    source_time = max((v for v in times if v is not None), default=None)
    clips, nonfinite = [], {}
    for event in train:
        metrics = event.get("metrics", {})
        clip = metrics.get("optimizer/was_clipped")
        if _finite(clip):
            clips.append(float(clip))
        nonfinite.update(
            {
                k: v
                for k, v in metrics.items()
                if ("nonfinite" in k or "skipped" in k) and _finite(v)
            }
        )
    latest = max(evaluations, key=lambda e: e["epoch"], default={})
    row.update(
        ids=ids,
        phase=phase,
        fit_complete=complete,
        updates=terminal.get("global_updates")
        if complete
        else progress.get("last_successful_update"),
        epoch=EPOCHS if complete else progress.get("epoch"),
        valid_count=len(evaluations),
        latest_valid=latest.get("metrics", {}),
        latest_valid_epoch=latest.get("epoch"),
        terminal_observation=terminal.get("status"),
        terminal=terminal,
        checkpoint_sha256=terminal.get("model_sha256") if complete else None,
        source_observed_utc=datetime.fromtimestamp(source_time, UTC).isoformat()
        if source_time is not None
        else None,
        source_age_seconds=now - source_time if source_time is not None else None,
        clipping=f"{sum(v > 0 for v in clips)}/{len(clips)} observed tail samples"
        if clips
        else "not recorded",
        nonfinite=nonfinite,
        collection_utc=_object(folder / "collection_receipt.json").get("collected_at_unix"),
    )
    return row
