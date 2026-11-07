from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from scripts.sequence_progress_report import (
    _events,
    _finite,
    _fixed_grade,
    _object,
    _sha,
    _time,
    canonical_sha,
)

SOURCE = "88ad6a8637390f39906b7ada30792a822f2172298770d9f27476d10028cc0a28"
MANIFEST = "d9f9baf4ca9cf21fc3e7fbe5ec73fc7d20dcb0424ef02d5edb581e2d3ddc805e"
SPLIT = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
RAW_CONFIGS = {
    "labels_s101101": "bd8254d1c13751e39c1896b6cf3bae95358ac347e7da89798609ab44884c8a7c",
    "mixture_corr_s101101": "ae003edc2b6941b6e497995a84c32aaaf25b1b33a2265bb46d04dce8fa088f07",
    "labels_s101102": "d48d24bcc784ac1e7081e75a6390555b71324a1e11673c9e0b3bd07ed6d24630",
    "mixture_corr_s101102": "70ed155e968423cf1fed75a6441d4067defbc508f20798123be0978d3a81b65a",
}
NORMALIZED_CONFIGS = {
    "labels_s101101": "d3ef8df541ec5ce9ac4f68b816e5ac61a59e7c1d0a181954f0fb6bdd4f8409ef",
    "labels_s101102": "4ba224ac9d5b850a1589a1b98796883696b7440c7d8f5de8c6e34f04d562babc",
    "mixture_corr_s101101": "2d9e589d65335e8231a7949902abac67d437f64886e4f2b75b22cf610076b879",
    "mixture_corr_s101102": "a2df03662fb3ebaf59b4ddad08408ec57083d08dfaa4a8d4d3f11ee3c90ae34b",
}


def load_panel(path: Path, expected_sha: str) -> dict:
    """Bind ROOT's actual receiver plan; no arbitrary cells/source/recipe aliases."""
    raw = path.read_bytes()
    if path.is_symlink() or len(raw) > 2_097_152 or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("ROOT receiver-plan source bytes/digest differ")
    plan = json.loads(raw)
    cells = plan["cells"]
    if (
        plan.get("expected_training_source_sha256") != SOURCE
        or plan.get("source_manifest_sha256") != MANIFEST
        or len(cells) != 4
        or {cell["cell"] for cell in cells} != set(RAW_CONFIGS)
        or any(
            cell["config_sha256"] != RAW_CONFIGS[cell["cell"]]
            or cell.get("normalized_config_sha256") != NORMALIZED_CONFIGS[cell["cell"]]
            for cell in cells
        )
    ):
        raise ValueError("receiver plan differs from exact four-cell fixed4E panel")
    return {"sha256": expected_sha, "cells": {cell["cell"]: cell for cell in cells}}


def metadata_identity(metadata: dict, cell: str, panel: dict) -> dict:
    """Check actual full-precision recipe schedule and official source-qualified membership."""
    config, provenance, source = metadata["config"], metadata["provenance"], metadata["source"]
    data = provenance["data"]
    ids = {
        "source": SOURCE,
        "config": canonical_sha(config),
        "data": canonical_sha(data),
        "split": SPLIT,
    }
    if (
        ids["config"] != panel["cells"][cell]["normalized_config_sha256"]
        or source.get("sha256") != SOURCE
        or canonical_sha(source["files"]) != SOURCE
        or config["seed"] != int(cell[-6:])
        or config["epochs"] != 4
        or config["batch_size"] != 128
        or config["accumulate_chunks"] != 4
        or config["chunk_rows"] != 256
        or provenance["world_size"] != 2
        or data["split_sha256"] != SPLIT
        or provenance["data_sha256"] != ids["data"]
        or data["teacher"] is not None
        or data["normalization"]["scope"] != "all_fit"
        or data["normalization"]["sequence_count"] != 10325
    ):
        raise ValueError("new B300 metadata differs from pinned source/config/4E recipe")
    for key, origin, count in (
        ("fit_members", "train", 10325),
        ("evaluation_members", "valid", 1873),
    ):
        members = data[key]
        if (
            len(members) != count
            or len({tuple(m) for m in members}) != count
            or any(m[0] != origin for m in members)
        ):
            raise ValueError("official whole-sequence memberships differ")
    if 4 * math.ceil(math.ceil(10325 / 2) / 128) * math.ceil(math.ceil(20000 / 256) / 4) != 3280:
        raise ValueError("four-epoch update derivation differs")
    return ids


def full_valid(event: dict, report_context: dict) -> bool:
    """Bind same-forward mixture/z metrics to the actual report-only reference context."""
    metrics = event.get("metrics", {})
    return (
        event.get("kind") == "evaluation"
        and type(event.get("epoch")) is int
        and 1 <= event["epoch"] <= 4
        and event.get("step") == 820 * event["epoch"]
        and event.get("context") == report_context
        and report_context.get("split_sha256") == SPLIT
        and report_context.get("evaluation_member_count") == 1873
        and metrics.get("valid/sequence_count") == 1873
        and metrics.get("valid/selected_rows") == 3528273
        and all(
            _finite(metrics.get(k))
            for k in (
                "valid/wp",
                "valid/wp/t0",
                "valid/wp/t1",
                "valid/z/wp",
                "valid/z/wp/t0",
                "valid/z/wp/t1",
            )
        )
        and all(_finite(v) for v in metrics.values())
    )


def completion(terminal: dict, ids: dict, evaluations: list[dict]) -> bool:
    """Require the actual producer's full4E/3280 terminal manifest and all official looks."""
    identity = terminal.get("identity", {})
    return (
        terminal.get("status") == "completed"
        and terminal.get("completed_epochs") == 4
        and terminal.get("global_updates") == 3280
        and _sha(terminal.get("model_sha256")) is not None
        and all(identity.get(f"{key}_sha256") == ids[key] for key in ("source", "config", "data"))
        and identity.get("world_size") == 2
        and len(evaluations) == 4
        and {e["epoch"] for e in evaluations} == {1, 2, 3, 4}
    )


def observe_cell(root: Path, cell: str, panel: dict, now: float) -> dict:
    """Read only the new b300-host fit ID; old seeds/panels/profiles remain historical."""
    folder = root / f"b300-{cell}-fit"
    row = {
        "cell": cell,
        "phase": "fit not observed",
        "fit_complete": False,
        "endpoint_complete": False,
    }
    if folder.is_symlink() or not folder.is_dir():
        return row
    metadata = _object(folder / "run_metadata.json")
    if not metadata:
        return row | {"phase": "fit metadata pending"}
    ids = metadata_identity(metadata, cell, panel)
    context = metadata["provenance"]["data"]["report_only_monitor"]
    evaluations = [
        e for e in _events(folder / "monitor_evaluations.jsonl") if full_valid(e, context)
    ]
    terminal, progress, score = (
        _object(folder / "terminal_status.json"),
        _object(folder / "monitor_progress.json"),
        _object(folder / "score.json"),
    )
    complete = completion(terminal, ids, evaluations)
    endpoint = complete and _fixed_grade(score, terminal, ids)
    if endpoint:
        endpoint = (
            score["producer"]["candidate"].get("provenance", {}).get("identity")
            == terminal["identity"]
            and score["producer"]["composition"]["reference_sha256"] == context["reference_sha256"]
        )
    observer = _events(folder / "monitor_observer.jsonl")
    times = [
        _time(progress.get("heartbeat_at")),
        _time(observer[-1].get("wall_time")) if observer else None,
    ]
    observed = max((v for v in times if v is not None), default=None)
    phase = str(progress.get("phase", "source phase unavailable"))
    if complete:
        phase = (
            "fit complete; fixed4E mixture grade recorded"
            if endpoint
            else "fit complete; fixed4E mixture grade pending"
        )
    elif terminal:
        phase = f"producer {terminal.get('status')}; completion proof pending"
    elif observer and observer[-1].get("metrics", {}).get("observer/process_alive_count") == 0:
        phase = "observed process exit; fit incomplete"
    train, clips, counters = _events(folder / "rank0.jsonl"), [], {}
    for event in train:
        metrics = event.get("metrics", event)
        value = metrics.get("gradient_clipped", metrics.get("optimizer/was_clipped"))
        if type(value) in (bool, int, float) and math.isfinite(value):
            clips.append(float(value))
        counters.update(
            {
                k: v
                for k, v in metrics.items()
                if ("nonfinite" in k or "skipped" in k) and _finite(v)
            }
        )
    latest = max(evaluations, key=lambda e: e["epoch"], default={})
    return row | {
        "ids": ids,
        "phase": phase,
        "fit_complete": complete,
        "endpoint_complete": bool(endpoint),
        "updates": terminal.get("global_updates")
        if complete
        else progress.get("last_successful_update"),
        "epoch": 4 if complete else progress.get("epoch"),
        "valid_count": len(evaluations),
        "latest_valid": latest.get("metrics", {}),
        "latest_valid_epoch": latest.get("epoch"),
        "checkpoint_sha256": terminal.get("model_sha256") if complete else None,
        "terminal_observation": terminal.get("status"),
        "fixed_grade": score.get("score") if endpoint else None,
        "source_observed_utc": datetime.fromtimestamp(observed, UTC).isoformat()
        if observed is not None
        else None,
        "source_age_seconds": now - observed if observed is not None else None,
        "clipping": f"{sum(v > 0 for v in clips)}/{len(clips)} observed tail samples"
        if clips
        else "not recorded",
        "nonfinite": counters,
    }


def snapshot(root: Path, panel: dict, now: float) -> dict:
    """Keep one fixed panel and truthful terminal transitions for durable batch publication."""
    rows = []
    for cell in RAW_CONFIGS:
        try:
            rows.append(observe_cell(root, cell, panel, now))
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            rows.append(
                {
                    "cell": cell,
                    "phase": "artifact evidence unavailable or inconsistent",
                    "fit_complete": False,
                    "endpoint_complete": False,
                }
            )
    terminal = [
        {
            "cell": r["cell"],
            "fit": r["fit_complete"],
            "grade": r["endpoint_complete"],
            "sha": r.get("checkpoint_sha256"),
            "status": r.get("terminal_observation"),
            "exit": "exit" in r["phase"],
        }
        for r in rows
    ]
    return {
        "host": "b300-objective-20261108",
        "observed_at_utc": datetime.fromtimestamp(now, UTC).isoformat(),
        "cells": rows,
        "panel": {"run_prefix": "b300-objective-20261108", "manifest_sha256": panel["sha256"]},
        "terminal_signature": canonical_sha(terminal),
    }
