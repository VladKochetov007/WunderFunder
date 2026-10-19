"""Bind changed R01 CPU placement to the already sealed density successor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from r01_recovery_common import read_json, sha256_file

DENSITY_MANIFEST_SHA256 = "e9927b5970c630b24284b4bc6d0fe5788f23546daa1755859f39573f6a4bcb6f"


def verify_placement(
    project_root: Path, job: dict[str, Any], resource: dict[str, Any]
) -> dict[str, Any]:
    """Accept initial placement or an exact PID/argv-bound successor launch event."""
    if resource.get("visible_devices") != job.get("gpu_uuid"):
        raise ValueError("R01 fit moved away from its panel GPU")
    if resource.get("cpu_affinity") == job.get("cpu_ids_initial"):
        return {"authority": "frozen_panel_initial_placement"}
    control = project_root / "runtime/r01-density4-v1"
    manifest_path = control / "CONTROL_MANIFEST.json"
    if sha256_file(manifest_path) != DENSITY_MANIFEST_SHA256:
        raise ValueError("changed placement requires the exact density4 control manifest")
    manifest = read_json(manifest_path)
    files = manifest.get("files")
    if (
        manifest.get("schema") != "tabdata-r01-density4-control-bundle.v1"
        or not isinstance(files, dict)
        or not files
    ):
        raise ValueError("density4 control manifest is malformed")
    for name, digest in files.items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("unsafe density4 control member")
        if sha256_file(control / name) != digest:
            raise ValueError(f"changed density4 control member: {name}")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    if manifest.get("sha256") != hashlib.sha256(canonical).hexdigest() or {
        path.name for path in control.glob("*.py")
    } != set(files):
        raise ValueError("density4 control member set/identity differs")
    events_path = project_root / "state/r01-density4-v1/events.jsonl"
    if events_path.is_symlink() or not events_path.is_file():
        raise ValueError("density4 launch evidence is not a regular event log")
    matches = []
    with events_path.open(encoding="utf-8") as stream:
        for line in stream:
            # A concurrently appended final partial line is not durable evidence.
            if not line.endswith("\n"):
                continue
            event = json.loads(line)
            if event.get("event") == "fit_started" and event.get("job_id") == job.get("id"):
                matches.append(event)
    if len(matches) != 1:
        raise ValueError("changed placement needs exactly one durable density4 launch")
    event = matches[0]
    argv_sha = hashlib.sha256(json.dumps(job["argv"], separators=(",", ":")).encode()).hexdigest()
    cpus, pid = resource.get("cpu_affinity"), resource.get("pid")
    if (
        type(pid) is not int
        or pid <= 0
        or not isinstance(cpus, list)
        or len(cpus) != 2
        or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
        or len(set(cpus)) != 2
        or any(event.get(key) != job.get(key) for key in ("arm", "fold", "seed"))
        or event.get("pid") != pid
        or event.get("pgid") != pid
        or type(event.get("pid_start_ticks")) is not int
        or event["pid_start_ticks"] <= 0
        or event.get("cpu_ids") != cpus
        or event.get("gpu_uuid") != resource.get("visible_devices")
        or event.get("argv_sha256") != argv_sha
    ):
        raise ValueError("density4 launch does not bind the actual resource/PID/panel argv")
    encoded = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
    return {
        "authority": "sealed_density4_launch_event",
        "control_manifest_path": str(manifest_path),
        "control_manifest_sha256": DENSITY_MANIFEST_SHA256,
        "event_log_path": str(events_path),
        "launch_event": event,
        "launch_event_sha256": hashlib.sha256(encoded).hexdigest(),
        "panel_initial_cpu_ids": job["cpu_ids_initial"],
        "actual_cpu_ids": cpus,
    }
