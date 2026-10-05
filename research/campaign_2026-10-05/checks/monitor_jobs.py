from __future__ import annotations

import concurrent.futures
import fcntl
import json
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHECKS = ROOT / "research/campaign_2026-10-05/checks"
RUNTIME = CHECKS / "monitor_runtime"
STATE = RUNTIME / "STATUS.json"
KAGGLE = str(ROOT / ".venv/bin/kaggle")
TERMINAL = {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}


def command(argv: list[str], timeout: int = 15) -> dict[str, object]:
    """Observe a bounded command without changing a kernel or worker."""
    try:
        result = subprocess.run(
            argv,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {"exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        return {"exit": None, "stdout": "", "stderr": "observation timed out"}


def read_json(path: Path) -> dict[str, object]:
    """Read a snapshot, exposing malformed files as an observation error."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return {"observation_error": f"{path.name}: {error}"}


def save_json(path: Path, value: object) -> None:
    """Publish atomically so dashboards cannot see a partial snapshot."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def kernel_status(kernel: dict[str, object]) -> dict[str, object]:
    """Query one kernel and retain provider errors separately from run failures."""
    result = command([KAGGLE, "kernels", "status", str(kernel["ref"])])
    match = re.search(r"KernelWorkerStatus\.(\w+)", str(result["stdout"]))
    return {
        **kernel,
        "status": match.group(1) if result["exit"] == 0 and match else "OBSERVATION_ERROR",
        "observed_unix": time.time(),
        "observation_exit": result["exit"],
        "observation_error": (result["stderr"] or f"status exit {result['exit']}")
        if result["exit"] != 0 or not match
        else None,
    }


def provider(previous: dict[str, object], now: float) -> dict[str, object]:
    """Discover jobs and quota every 15 minutes; active statuses every five."""
    result = dict(previous.get("provider", {}))
    refresh = now - float(result.get("catalog_observed_unix", 0)) >= 900
    if refresh:
        catalog = command(
            [
                KAGGLE,
                "kernels",
                "list",
                "--mine",
                "--sort-by",
                "dateRun",
                "--page-size",
                "100",
                "--format",
                "json",
            ]
        )
        try:
            if catalog["exit"] != 0:
                raise ValueError(str(catalog["stderr"]))
            result["catalog"] = json.loads(str(catalog["stdout"]))
            result["catalog_observed_unix"] = now
            result.pop("catalog_error", None)
        except ValueError as error:
            result["catalog_error"] = str(error)
        quota = command([KAGGLE, "quota", "--format", "json"])
        try:
            if quota["exit"] != 0:
                raise ValueError(str(quota["stderr"]))
            result["quota"] = json.loads(str(quota["stdout"]))
            result["quota_observed_unix"] = now
            result.pop("quota_error", None)
        except ValueError as error:
            result["quota_error"] = str(error)
    catalog = result.get("catalog", [])
    pinned_file = CHECKS / "MONITORED_KERNELS.json"
    if pinned_file.exists():
        pinned = json.loads(pinned_file.read_text())["kernels"]
        seen = {value["ref"] for value in catalog}
        catalog = list(catalog) + [{"ref": key} for key in pinned if key not in seen]
    cached = result.get("kernels", {})
    pending = []
    for kernel in catalog:
        if "/wnn-" not in kernel["ref"]:
            continue
        old = cached.get(kernel["ref"], {})
        if old.get("status") not in TERMINAL or (
            old.get("lastRunTime") != kernel.get("lastRunTime")
        ):
            pending.append(kernel)
    # Jobs that fall off the recent catalog still need monitoring until terminal.
    refs = {kernel["ref"] for kernel in catalog}
    pending.extend(
        value
        for key, value in cached.items()
        if key not in refs and value.get("status") not in TERMINAL
    )
    pinned_refs = set(pinned) if pinned_file.exists() else set()
    pending.sort(
        key=lambda item: (
            0
            if item["ref"] in pinned_refs
            or cached.get(item["ref"], {}).get("status") not in TERMINAL | {None}
            else 1
        )
    )
    result["status_checks_deferred"] = max(0, len(pending) - 20)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for observed in pool.map(kernel_status, pending[:20]):
            key = str(observed["ref"])
            old = cached.get(key, {})
            if observed["status"] == "OBSERVATION_ERROR":
                observed["last_known_status"] = old.get("last_known_status", old.get("status"))
            cached[key] = {**old, **observed}
    missing_logs = [
        value
        for value in cached.values()
        if value.get("status") == "ERROR" and not value.get("error_log")
    ]
    for value in missing_logs[:2]:
        key = str(value["ref"])
        log = command([KAGGLE, "kernels", "logs", key])
        path = RUNTIME / (key.split("/")[-1] + ".error.log")
        path.write_text(str(log["stdout"])[-65536:])
        value["error_log"] = str(path.relative_to(ROOT))
        value["log_observation_exit"] = log["exit"]
    result["kernels"] = cached
    return result


def local(now: float) -> dict[str, object]:
    """Observe all guarded attempts, live services and aggregate resource use."""
    attempts = []
    for path in (ROOT / "research/campaign_2026-10-04/runner/attempts").glob("*.json"):
        value = read_json(path)
        state = value.get("status")
        terminal = state in {"finished", "failed", "lost", "cancelled"}
        if terminal and value.get("exit_code") == 0:
            if (
                value.get("finished_unix", 0) >= now - 86400
                and value.get("peak_observed_vram_bytes", 0) >= 4_500_000_000
            ):
                attempts.append(
                    {
                        "id": value.get("id"),
                        "status": state,
                        "resource_warning": "completed worker peak GPU near 5 GB cap",
                        "peak_observed_vram_bytes": value["peak_observed_vram_bytes"],
                    }
                )
            continue
        if terminal and value.get("finished_unix", 0) < now - 86400:
            continue
        entry = {
            key: value.get(key)
            for key in (
                "id",
                "status",
                "unit",
                "exit_code",
                "failure",
                "heartbeat_unix",
                "peak_ram_bytes",
                "peak_observed_vram_bytes",
                "teardown_verified",
            )
        }
        entry["path"] = str(path.relative_to(ROOT))
        if not terminal:
            entry["heartbeat_age_seconds"] = now - float(
                value.get("heartbeat_unix", value.get("reserved_unix", now))
            )
            if value.get("unit"):
                entry["service"] = command(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        str(value["unit"]),
                        "-p",
                        "ActiveState,Result,ExecMainStatus",
                    ],
                    5,
                )
        attempts.append(entry)
    slice_result = command(
        [
            "systemctl",
            "--user",
            "show",
            "wnn-campaign.slice",
            "-p",
            "MemoryCurrent,MemoryMax,MemoryHigh,MemorySwapMax,CPUQuotaPerSecUSec,ControlGroup",
        ],
        5,
    )
    properties = dict(
        line.split("=", 1) for line in str(slice_result["stdout"]).splitlines() if "=" in line
    )
    events = {}
    if properties.get("ControlGroup"):
        event_file = (
            Path("/sys/fs/cgroup") / properties["ControlGroup"].lstrip("/") / "memory.events"
        )
        try:
            events = dict(
                (key, int(value))
                for key, value in (line.split() for line in event_file.read_text().splitlines())
            )
        except OSError as error:
            events = {"observation_error": str(error)}
    return {
        "slice_properties": properties,
        "parent_memory_events": events,
        "attempts_requiring_attention": attempts,
        "slice": command(
            [
                "systemctl",
                "--user",
                "show",
                "wnn-campaign.slice",
                "-p",
                "MemoryCurrent,MemoryMax,MemoryHigh,MemorySwapMax,CPUQuotaPerSecUSec,ControlGroup",
            ],
            5,
        ),
        "services": command(
            [
                "systemctl",
                "--user",
                "list-units",
                "wnn-*",
                "--state=running,activating,failed",
                "--no-pager",
                "--plain",
                "--no-legend",
            ],
            5,
        ),
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            5,
        ),
    }


def alerts(snapshot: dict[str, object], now: float) -> list[str]:
    """Report run failures and stale observations without an automatic retry."""
    messages = []
    remote = snapshot["provider"]
    if remote.get("status_checks_deferred", 0):
        messages.append(
            f"Provider checks deferred: {remote['status_checks_deferred']}; observation incomplete"
        )
    for key in ("catalog_error", "quota_error"):
        if remote.get(key):
            messages.append(f"Provider {key}: {remote[key]}")
    for key, value in remote.get("kernels", {}).items():
        status = value.get("status")
        if status in {"ERROR", "OBSERVATION_ERROR", "CANCELLED", "CANCELED"}:
            messages.append(f"{key}: {status}")
        if status in {"QUEUED", "RUNNING"}:
            raw_time = value.get("lastRunTime")
            try:
                started = datetime.fromisoformat(str(raw_time))
                if started.tzinfo is None:
                    started = started.replace(tzinfo=UTC)
                age = now - started.timestamp()
                if age < -300:
                    messages.append(f"{key}: provider start timestamp is in the future")
                if status == "QUEUED" and age > 3600:
                    messages.append(f"{key}: queued >1 hour; no device result")
                if status == "RUNNING" and age > 8 * 3600:
                    messages.append(f"{key}: running >8 hours; inspect progress")
            except ValueError:
                messages.append(f"{key}: active run start time unavailable")
    for value in snapshot["local"]["attempts_requiring_attention"]:
        if value.get("resource_warning"):
            messages.append(f"Local {value['id']}: {value['resource_warning']}")
        if value.get("failure") or value.get("exit_code") not in (0, None):
            messages.append(f"Local {value['id']}: {value.get('failure')}")
        if value.get("heartbeat_age_seconds", 0) > 120:
            messages.append(f"Local {value['id']}: stale runner heartbeat")
        service = value.get("service", {})
        service_text = str(service.get("stdout", ""))
        if service and (
            service.get("exit") != 0
            or "ActiveState=inactive" in service_text
            or "ActiveState=failed" in service_text
        ):
            messages.append(f"Local {value['id']}: active attempt's service unavailable/ended")
    observed_local = snapshot["local"]
    properties = observed_local["slice_properties"]
    for field in ("MemoryCurrent", "MemoryHigh", "MemoryMax"):
        if not str(properties.get(field, "")).isdigit():
            messages.append(f"Local slice {field}: unavailable")
    if (
        str(properties.get("MemoryCurrent", "")).isdigit()
        and int(properties["MemoryCurrent"]) >= 27_000_000_000
    ):
        messages.append("Local aggregate RAM reached the 27 GB soft limit")
    if properties.get("MemoryHigh") != "27000000000":
        messages.append("Local RAM soft limit differs from 27 GB")
    if properties.get("MemoryMax") != "30000000000":
        messages.append("Local hard RAM cap differs from the user-authorized 30 GB")
    if properties.get("CPUQuotaPerSecUSec") != "12s":
        messages.append("Local CPU quota differs from the user-authorized 12 cores")
    if properties.get("MemorySwapMax") != "0":
        messages.append("Local worker swap is not disabled")
    for line in str(observed_local["services"]["stdout"]).splitlines():
        fields = line.split()
        if fields and fields[0].startswith("wnn-") and "failed" in fields[1:4]:
            messages.append(f"Local service {fields[0]}: failed")
    gpu = str(snapshot["local"]["gpu"]["stdout"]).strip()
    if gpu and snapshot["local"]["gpu"]["exit"] == 0:
        try:
            used = sum(int(line.split(",")[0]) * 1024**2 for line in gpu.splitlines())
            if used > 4_500_000_000:
                messages.append(f"Local GPU near 5 GB ceiling: {used} bytes")
        except ValueError:
            messages.append("Local GPU memory observation invalid")
    else:
        messages.append("Local GPU memory observation unavailable")
    return messages


def main() -> None:
    """Persist health and transitions under a nonoverlapping, read-only timer."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "checker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        previous = read_json(STATE) if STATE.exists() else {}
        now = time.time()
        observed_provider = provider(previous, now)
        now = time.time()
        snapshot = {
            "utc": datetime.now(UTC).isoformat(),
            "provider": observed_provider,
            "local": local(now),
            "checker_policy": "observe only; no launch, cancellation or retry",
        }
        snapshot["alerts"] = alerts(snapshot, now)
        current_events = snapshot["local"]["parent_memory_events"]
        previous_events = previous.get("local", {}).get("parent_memory_events", {})
        for key in ("oom", "oom_kill", "oom_group_kill"):
            if key in previous_events and current_events.get(key, 0) > previous_events[key]:
                snapshot["alerts"].append(f"Local parent new {key} event")
        save_json(STATE, snapshot)
        if snapshot["alerts"] != previous.get("alerts"):
            with (RUNTIME / "ALERT_HISTORY.jsonl").open("a") as history:
                history.write(
                    json.dumps({"utc": snapshot["utc"], "alerts": snapshot["alerts"]}) + "\n"
                )
        print(json.dumps({"utc": snapshot["utc"], "alerts": snapshot["alerts"]}))


if __name__ == "__main__":
    main()
