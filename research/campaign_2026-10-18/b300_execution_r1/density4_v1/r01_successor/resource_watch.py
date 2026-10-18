"""Publish fail-closed live cgroup, host-memory, swap, and worker-RSS samples."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

UNIT_CAPS = {
    "tabdata-r01-execution-v4": (192 * 1024**3, 0, 512),
    "tabdata-r01-density4-v1": (192 * 1024**3, 0, 512),
    "tabdata-pr15-execution-v2": (192 * 1024**3, 0, 512),
}
EVENT_KEYS = ("high", "max", "oom", "oom_group_kill", "oom_kill")
CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_ROOT = Path("/proc")
MEMINFO = Path("/proc/meminfo")
PR15_UNIT = "tabdata-pr15-execution-v2"
PR15_STATE = Path("/data/epanov_s/tabdata/state/pr15-execution-v2")
HISTORY_SCHEMA = "tabdata-b300-campaign-resource-history.v1"
PR15_TERMINAL_SCHEMA = "tabdata-pr15-terminal-success.v1"
PR15_TERMINAL_PROOF_SCHEMA = "tabdata-pr15-terminal-proof.v1"


class ObservedResourceViolation(ValueError):
    """A directly sampled hard-limit contradiction that invalidates admission now."""


def _properties(unit: str) -> dict[str, str]:
    command = [
        "systemctl",
        "--user",
        "show",
        unit,
        "--no-pager",
        "--property=LoadState",
        "--property=ControlGroup",
        "--property=MemoryMax",
        "--property=MemorySwapMax",
        "--property=TasksMax",
        "--property=ActiveState",
        "--property=Result",
        "--property=MainPID",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    if result.returncode:
        raise RuntimeError(f"systemctl show failed for {unit}: {result.stderr.strip()}")
    values = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def _integer(value: str, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not an integer byte/count value: {value!r}") from error
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _read_key_value_file(path: Path) -> dict[str, int]:
    values = {}
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) == 2:
            values[fields[0]] = _integer(fields[1], f"{path.name}:{fields[0]}")
    return values


def _unit_sample(
    unit: str, cgroup_root: Path
) -> tuple[dict[str, Any] | None, str | None, Path | None, dict[str, str]]:
    properties = _properties(unit)
    if properties.get("LoadState") != "loaded":
        return None, f"{unit} is not loaded", None, properties
    control_group = properties.get("ControlGroup", "")
    if not control_group.startswith("/"):
        return None, f"{unit} has no active cgroup path", None, properties
    root = cgroup_root.resolve(strict=True)
    cgroup = (root / control_group.lstrip("/")).resolve(strict=False)
    if not cgroup.is_relative_to(root):
        raise ValueError(f"{unit} cgroup escapes the unified cgroup root")
    if not cgroup.exists():
        return None, f"{unit} cgroup is absent", None, properties
    expected_memory, expected_swap, expected_pids = UNIT_CAPS[unit]
    record = {
        "control_group": control_group,
        "memory_current_bytes": _integer((cgroup / "memory.current").read_text().strip(), "memory.current"),
        "memory_max_bytes": _integer((cgroup / "memory.max").read_text().strip(), "memory.max"),
        "memory_swap_current_bytes": _integer(
            (cgroup / "memory.swap.current").read_text().strip(), "memory.swap.current"
        ),
        "memory_swap_max_bytes": _integer(
            (cgroup / "memory.swap.max").read_text().strip(), "memory.swap.max"
        ),
        "pids_current": _integer((cgroup / "pids.current").read_text().strip(), "pids.current"),
        "pids_max": _integer((cgroup / "pids.max").read_text().strip(), "pids.max"),
        "memory_events": _read_key_value_file(cgroup / "memory.events"),
    }
    if (
        record["memory_max_bytes"] != expected_memory
        or record["memory_swap_max_bytes"] != expected_swap
        or record["pids_max"] != expected_pids
    ):
        raise ObservedResourceViolation(
            f"{unit} live cgroup hard limits differ from the authorized envelope"
        )
    if any(key not in record["memory_events"] for key in EVENT_KEYS):
        raise ValueError(f"{unit} live cgroup event counters are incomplete")
    if (
        _integer(properties.get("MemoryMax", "-1"), "systemd MemoryMax") != expected_memory
        or _integer(properties.get("MemorySwapMax", "-1"), "systemd MemorySwapMax") != expected_swap
        or _integer(properties.get("TasksMax", "-1"), "systemd TasksMax") != expected_pids
    ):
        raise ObservedResourceViolation(f"{unit} systemd properties differ from its cgroup controls")
    sample = {
        "lifecycle": "live",
        "memory_current_bytes": record["memory_current_bytes"],
        "memory_max_bytes": record["memory_max_bytes"],
        "memory_swap_current_bytes": record["memory_swap_current_bytes"],
        "memory_swap_max_bytes": record["memory_swap_max_bytes"],
        "pids_current": record["pids_current"],
        "pids_max": record["pids_max"],
        "memory_events": record["memory_events"],
        "control_group": control_group,
        "systemd_load_state": properties["LoadState"],
        "systemd_memory_max_bytes": _integer(properties["MemoryMax"], "systemd MemoryMax"),
        "systemd_memory_swap_max_bytes": _integer(
            properties["MemorySwapMax"], "systemd MemorySwapMax"
        ),
        "systemd_pids_max": _integer(properties["TasksMax"], "systemd TasksMax"),
    }
    return sample, None, cgroup, properties


def _stat_identity(path: Path) -> tuple[int, str, int]:
    raw = path.read_text(encoding="ascii")
    close = raw.rfind(")")
    if close < 0:
        raise ValueError(f"malformed process stat: {path}")
    pid = int(raw[: raw.find(" ")])
    fields = raw[close + 2 :].split()
    if len(fields) < 20:
        raise ValueError(f"truncated process stat: {path}")
    return pid, fields[0], int(fields[2])


def _rss_bytes(pid: int, proc_root: Path) -> tuple[str, int, int]:
    process = proc_root / str(pid)
    _, state, pgid = _stat_identity(process / "stat")
    status = (process / "status").read_text(encoding="ascii")
    rss_kib = next(
        (int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:")),
        None,
    )
    if rss_kib is None:
        if state in {"Z", "X"}:
            rss_kib = 0
        else:
            raise ValueError(f"process {pid} has no readable VmRSS")
    return state, pgid, rss_kib * 1024


def _processes_in_unit(unit: str, proc_root: Path) -> list[int]:
    """Find all extant members, including zombies, of one exact service subtree."""
    component = unit if unit.endswith(".service") else unit + ".service"
    matches = []
    for item in proc_root.iterdir():
        if not item.name.isdigit():
            continue
        try:
            lines = (item / "cgroup").read_text(encoding="ascii").splitlines()
        except (FileNotFoundError, ProcessLookupError):
            continue
        unified = next((line.split(":", 2)[2] for line in lines if line.startswith("0::")), None)
        if unified is not None and component in unified.split("/"):
            matches.append(int(item.name))
    return matches


def _read_json_file(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"terminal receipt must be a regular non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"terminal receipt must be a JSON object: {path}")
    return value


def _terminal_pr15_sample(
    *,
    properties: dict[str, str],
    cgroup_root: Path,
    proc_root: Path,
    state_root: Path,
    previous: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Accept only a completed PR15 unit with durable final receipts and prior cgroup evidence."""
    load_state = properties.get("LoadState")
    if load_state not in {"loaded", "not-found"}:
        return None
    if (
        properties.get("ActiveState") != "inactive"
        or properties.get("MainPID") != "0"
    ):
        return None
    expected_memory, expected_swap, expected_pids = UNIT_CAPS[PR15_UNIT]
    if not isinstance(previous, dict) or previous.get("lifecycle") != "live":
        raise ValueError("terminal PR15 has no persisted last-observed live cgroup sample")
    last_group = previous.get("control_group")
    last_events = previous.get("memory_events")
    last_observed = previous.get("observed_unix")
    if (
        not isinstance(last_group, str)
        or not last_group.startswith("/")
        or not isinstance(last_events, dict)
        or any(type(last_events.get(key)) is not int or last_events[key] < 0 for key in EVENT_KEYS)
        or isinstance(last_observed, bool)
        or not isinstance(last_observed, (int, float))
        or not (0 < float(last_observed) <= time.time())
    ):
        raise ValueError("terminal PR15 last-observed cgroup history is malformed")
    history_caps = (
        previous.get("memory_max_bytes"),
        previous.get("memory_swap_max_bytes"),
        previous.get("pids_max"),
        previous.get("systemd_memory_max_bytes"),
        previous.get("systemd_memory_swap_max_bytes"),
        previous.get("systemd_pids_max"),
    )
    if history_caps != (
        expected_memory,
        expected_swap,
        expected_pids,
        expected_memory,
        expected_swap,
        expected_pids,
    ):
        raise ValueError("terminal PR15 lacks verified last-observed cgroup and systemd hard caps")
    proof = previous.get("terminal_proof")
    if proof is not None and not isinstance(proof, dict):
        raise ValueError("persisted PR15 terminal proof is malformed")
    if proof is None and time.time() - float(last_observed) > 20:
        raise ObservedResourceViolation(
            "first PR15 terminal observation is more than 20 seconds after its last live sample"
        )
    if load_state == "loaded":
        if properties.get("Result") != "success":
            raise ValueError("loaded terminal PR15 systemd unit did not finish successfully")
        if (
            _integer(properties.get("MemoryMax", "-1"), "systemd MemoryMax") != expected_memory
            or _integer(properties.get("MemorySwapMax", "-1"), "systemd MemorySwapMax")
            != expected_swap
            or _integer(properties.get("TasksMax", "-1"), "systemd TasksMax") != expected_pids
        ):
            raise ValueError("terminal PR15 systemd properties differ from its authorized hard caps")
        systemd_result: str | None = "success"
        cap_basis = "systemd_properties_and_last_observed_cgroup"
    else:
        # systemd garbage collection erases unit properties; never reinterpret its defaults.
        systemd_result = None
        cap_basis = "last_observed_live_cgroup_and_systemd_properties"
    active_property_group = properties.get("ControlGroup", "")
    if active_property_group.startswith("/") and active_property_group != last_group:
        raise ValueError("terminal PR15 systemd cgroup identity differs from last live observation")
    cgroup_base = cgroup_root.resolve(strict=True)
    cgroup = (cgroup_base / last_group.lstrip("/")).resolve(strict=False)
    if not cgroup.is_relative_to(cgroup_base) or cgroup.exists():
        raise ValueError("terminal PR15 cgroup still exists or escaped the unified hierarchy")
    processes = _processes_in_unit(PR15_UNIT, proc_root)
    if processes:
        raise ValueError(f"terminal PR15 still has process actors: {processes}")
    complete_path = state_root / "COMPLETE.json"
    results_path = state_root / "RESULTS.json"
    complete = _read_json_file(complete_path)
    results = _read_json_file(results_path)
    if complete != {"completed_successor_cells": 104, "preserved_original_fits": 16}:
        raise ValueError("PR15 COMPLETE receipt does not certify 104 delivered and 16 preserved fits")
    if len(results) != 104 or any(
        not isinstance(record, dict)
        or record.get("status") != "completed"
        or record.get("returncode") != 0
        for record in results.values()
    ):
        raise ValueError("PR15 RESULTS receipt does not contain 104 successful cells")
    complete_sha = hashlib.sha256(complete_path.read_bytes()).hexdigest()
    results_sha = hashlib.sha256(results_path.read_bytes()).hexdigest()
    if proof is None:
        proof = {
            "schema": PR15_TERMINAL_PROOF_SCHEMA,
            "first_terminal_observed_unix": time.time(),
            "last_live_observed_unix": float(last_observed),
            "last_live_control_group": last_group,
            "last_live_memory_events": dict(last_events),
            "last_live_memory_max_bytes": expected_memory,
            "last_live_memory_swap_max_bytes": expected_swap,
            "last_live_pids_max": expected_pids,
            "complete_sha256": complete_sha,
            "results_sha256": results_sha,
            "completed_successor_cells": 104,
            "preserved_original_fits": 16,
            "results_count": 104,
            "systemd_load_state": load_state,
            "systemd_result_observed": systemd_result,
            "hard_caps_basis": cap_basis,
        }
    else:
        accepted_at = proof.get("first_terminal_observed_unix")
        if (
            proof.get("schema") != PR15_TERMINAL_PROOF_SCHEMA
            or proof.get("complete_sha256") != complete_sha
            or proof.get("results_sha256") != results_sha
            or proof.get("completed_successor_cells") != 104
            or proof.get("preserved_original_fits") != 16
            or proof.get("results_count") != 104
            or proof.get("last_live_observed_unix") != float(last_observed)
            or proof.get("last_live_control_group") != last_group
            or proof.get("last_live_memory_events") != last_events
            or proof.get("last_live_memory_max_bytes") != expected_memory
            or proof.get("last_live_memory_swap_max_bytes") != expected_swap
            or proof.get("last_live_pids_max") != expected_pids
            or isinstance(accepted_at, bool)
            or not isinstance(accepted_at, (int, float))
            or not math.isfinite(accepted_at)
            or not 0 <= accepted_at - float(last_observed) <= 20
            or proof.get("systemd_load_state") not in {"loaded", "not-found"}
            or (
                proof.get("systemd_load_state") == "loaded"
                and proof.get("systemd_result_observed") != "success"
            )
            or (
                proof.get("systemd_load_state") == "not-found"
                and proof.get("systemd_result_observed") is not None
            )
        ):
            raise ValueError("persisted PR15 terminal proof does not match current receipts/history")
        if proof.get("systemd_load_state") == "not-found" and load_state == "loaded":
            raise ValueError("PR15 systemd unit reappeared after terminal garbage collection")
    return {
        "lifecycle": "terminal_success",
        "memory_current_bytes": 0,
        "memory_current_basis": "cgroup_absent_and_no_process_members",
        "memory_swap_current_bytes": 0,
        "pids_current": 0,
        "memory_max_bytes": expected_memory,
        "memory_swap_max_bytes": expected_swap,
        "pids_max": expected_pids,
        "memory_events_last_observed": dict(last_events),
        "memory_events_last_observed_unix": float(last_observed),
        "last_observed_control_group": last_group,
        "terminal_receipt": {
            "schema": PR15_TERMINAL_SCHEMA,
            "complete_sha256": complete_sha,
            "results_sha256": results_sha,
            "completed_successor_cells": 104,
            "preserved_original_fits": 16,
            "results_count": 104,
            "systemd_load_state": proof["systemd_load_state"],
            "systemd_active_state": properties["ActiveState"],
            "systemd_result_observed": proof["systemd_result_observed"],
            "systemd_main_pid": 0,
            "hard_caps_basis": proof["hard_caps_basis"],
            "current_systemd_load_state": load_state,
            "current_systemd_result_observed": systemd_result,
            "terminal_proof": proof,
            "cgroup_absent": True,
            "process_count": 0,
        },
    }


def _read_history(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"schema": HISTORY_SCHEMA, "units": {}}
    if path.is_symlink():
        raise ValueError("resource history may not be a symlink")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != HISTORY_SCHEMA
        or not isinstance(value.get("units"), dict)
    ):
        raise ValueError("resource history schema is malformed")
    return value


def _cgroup_pids(root: Path) -> set[int]:
    pids: set[int] = set()
    for directory, child_dirs, _ in os.walk(root, followlinks=False):
        child_dirs[:] = [name for name in child_dirs if not (Path(directory) / name).is_symlink()]
        path = Path(directory) / "cgroup.procs"
        try:
            pids.update(int(value) for value in path.read_text(encoding="ascii").split())
        except FileNotFoundError:
            continue
    return pids


def _process_cgroup(pid: int, proc_root: Path) -> str:
    lines = (proc_root / str(pid) / "cgroup").read_text(encoding="ascii").splitlines()
    unified = next((line.split(":", 2)[2] for line in lines if line.startswith("0::")), None)
    if unified is None:
        raise ValueError(f"process {pid} has no unified cgroup membership")
    return unified


def _worker_group_rss(units: dict[str, Path], proc_root: Path) -> dict[str, int]:
    groups: dict[str, int] = {}
    seen: set[int] = set()
    for unit, cgroup in units.items():
        expected = "/" + str(cgroup.relative_to(CGROUP_ROOT.resolve(strict=True))).replace("\\", "/")
        for pid in _cgroup_pids(cgroup):
            if pid in seen:
                raise ValueError(f"process {pid} appears in multiple campaign cgroups")
            seen.add(pid)
            try:
                observed = _process_cgroup(pid, proc_root)
                if observed != expected and not observed.startswith(expected + "/"):
                    continue
                _, pgid, rss = _rss_bytes(pid, proc_root)
            except FileNotFoundError:
                continue
            groups[f"{unit}:{pgid}"] = groups.get(f"{unit}:{pgid}", 0) + rss
    return groups


def _host_memory(path: Path) -> tuple[dict[str, int], int]:
    values = {}
    for line in path.read_text(encoding="ascii").splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[key] = _integer(value.strip().split()[0], f"/proc/meminfo:{key}") * 1024
    required = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    if not required <= set(values):
        raise ValueError("/proc/meminfo omits required host memory or swap counters")
    return {"MemTotal_bytes": values["MemTotal"], "MemAvailable_bytes": values["MemAvailable"]}, max(
        0, values["SwapTotal"] - values["SwapFree"]
    )


def sample_resources(
    *,
    cgroup_root: Path = CGROUP_ROOT,
    proc_root: Path = PROC_ROOT,
    meminfo: Path = MEMINFO,
    history_path: Path | None = None,
    pr15_state: Path = PR15_STATE,
) -> dict[str, Any]:
    """Collect one real sample; absent units are explicitly pending, never zero-filled."""
    units: dict[str, dict[str, Any]] = {}
    cgroups: dict[str, Path] = {}
    pending: list[str] = []
    errors: list[str] = []
    observed_violations: list[str] = []
    history = _read_history(history_path)
    history_units = history["units"]
    for unit in UNIT_CAPS:
        try:
            record, reason, cgroup, properties = _unit_sample(unit, cgroup_root)
            if reason is not None or record is None or cgroup is None:
                if unit == PR15_UNIT and properties.get("ActiveState") == "inactive":
                    previous = history_units.get(unit)
                    has_complete = (pr15_state / "COMPLETE.json").is_file()
                    has_results = (pr15_state / "RESULTS.json").is_file()
                    if previous is None and not has_complete and not has_results:
                        pending.append(unit)
                    else:
                        try:
                            terminal = _terminal_pr15_sample(
                                properties=properties,
                                cgroup_root=cgroup_root,
                                proc_root=proc_root,
                                state_root=pr15_state,
                                previous=previous,
                            )
                        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                            observed_violations.append(
                                f"{unit}: terminal evidence invalid: {type(error).__name__}: {error}"
                            )
                        else:
                            if terminal is None:
                                pending.append(unit)
                            else:
                                units[unit] = terminal
                                previous["terminal_proof"] = terminal["terminal_receipt"][
                                    "terminal_proof"
                                ]
                else:
                    pending.append(unit)
            else:
                prior = history_units.get(unit)
                if unit == PR15_UNIT and isinstance(prior, dict) and prior.get("terminal_proof"):
                    raise ObservedResourceViolation(
                        "PR15 v2 cgroup reappeared after its terminal completion proof"
                    )
                record["observed_unix"] = time.time()
                units[unit] = record
                cgroups[unit] = cgroup
                history_units[unit] = record
        except ObservedResourceViolation as error:
            observed_violations.append(f"{unit}: {error}")
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            errors.append(f"{unit}: {type(error).__name__}: {error}")
    try:
        host_memory, swap_current = _host_memory(meminfo)
        rss = _worker_group_rss(cgroups, proc_root)
    except (OSError, ValueError) as error:
        host_memory, swap_current, rss = {}, None, {}
        errors.append(f"host/process sample: {type(error).__name__}: {error}")
    verified = (
        not pending
        and not errors
        and not observed_violations
        and set(units) == set(UNIT_CAPS)
    )
    sample = {
        "schema": "tabdata-b300-campaign-live-resources.v1",
        "scope": "b300_campaign_aggregate",
        "verified": verified,
        "observed_unix": time.time(),
        "units": units,
        "pending_units": pending,
        "errors": errors,
        "observed_violations": observed_violations,
        "host_memory": host_memory,
        "memory_swap_current_bytes": swap_current,
        "worker_group_rss_bytes": rss,
        "rss_method": "sum VmRSS for /proc members of each exact unit process group",
    }
    if history_path is not None:
        write_atomic(history_path, history)
    return sample


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    """Publish each sample atomically with file and parent-directory fsync."""
    if path.is_symlink():
        raise ValueError("resource receipt destination may not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            parent_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def publish_sample(output: Path, sample: dict[str, Any]) -> Path:
    """Keep the last verified timestamp during transient partial observations."""
    if sample.get("verified") is True:
        write_atomic(output, sample)
        return output
    invalid_path = output.with_name(output.name + ".invalid.json")
    write_atomic(invalid_path, sample)
    if sample.get("observed_violations"):
        write_atomic(output, sample)
        return output
    return invalid_path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--pr15-state", type=Path, default=PR15_STATE)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Continuously publish fresh receipts; return only in explicit one-shot mode."""
    args = _args()
    if not (0.25 <= args.interval_seconds <= 20):
        raise ValueError("watch interval must be from 0.25 to 20 seconds")
    while True:
        sample = sample_resources(
            history_path=args.output.with_name(args.output.name + ".history.json"),
            pr15_state=args.pr15_state,
        )
        publish_sample(args.output, sample)
        if args.once:
            print(json.dumps(sample, sort_keys=True))
            return 0 if sample["verified"] else 2
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
