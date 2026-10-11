"""Lane watchdog: one pass over every compute lane, alerts only on state changes.

Lanes: the receipt-owned Vast instance and its job queue, the local GPU queue and its admission barrier, the Kaggle
capacity kernel, systemd timers and failed units, disk and memory.  Each pass compares the set of active problems with the
previous pass; new problems become ``ALERT`` rows and cleared ones ``RECOVERED`` rows in ``ALERTS.jsonl`` (and a desktop
notification with ``--notify``).  A broken check is itself an alert, so silence means every check ran and passed.

    scripts/watchdog.py --once --notify          # systemd timer, every few minutes
    tail -F research/.../watchdog/ALERTS.jsonl   # what the session monitor follows

Consecutive-failure counts keep one-off blips (a slow ssh, a gap between queued jobs) from alerting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BC = ROOT / "research/campaign_2026-10-06/bandit_continuation"
VAST = BC / "vast_teachers_r1"
WATCH = BC / "watchdog"
STATE_PATH = ROOT / ".runtime/watchdog_state.json"
HEARTBEAT = ROOT / ".runtime/watchdog_heartbeat"
KAGGLE_KERNELS = ["vladkochetov07/e529-gru512-768-muon-fit-v5r2-20261009", "vladkochetov07/wnn-native-device-tpu-20261009-v4"]
EXPECTED_TIMERS = ["wnn-local-queue-r1", "wnn-campaign-kaggle-poller"]
SPEND_WARN = 9.6  # fraction of the tick cap (12) where a warning starts


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def sh(args: list[str], timeout: int = 60) -> str:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False).stdout


class Problems:
    """Active problems of this pass with a persistence requirement (consecutive passes before alerting)."""

    def __init__(self, previous_counts: dict[str, int]) -> None:
        self.counts: dict[str, int] = {}
        self.messages: dict[str, tuple[str, str]] = {}
        self.previous = previous_counts

    def add(self, key: str, severity: str, message: str, needed: int = 1) -> None:
        count = self.previous.get(key, 0) + 1
        self.counts[key] = count
        self.messages[key] = (severity, message)
        self.needed = getattr(self, "needed", {})
        self.needed[key] = needed

    def active(self) -> dict[str, tuple[str, str]]:
        return {k: v for k, v in self.messages.items() if self.counts[k] >= self.needed[k]}


def check_vast(problems: Problems, state: dict) -> None:
    import importlib

    for lane in ("", "b"):
        os.environ["VT_LANE"] = lane
        sys.path.insert(0, str(VAST))
        import controller

        importlib.reload(controller)
        try:
            check_vast_lane(problems, state, controller, lane or "a")
        finally:
            os.environ.pop("VT_LANE", None)


def check_vast_lane(problems: Problems, state: dict, controller, lane: str) -> None:
    receipt_path = controller.RECEIPT
    if not receipt_path.exists():
        return
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("state") in ("destroyed_verified", "creating"):
        return
    if "instance_id" not in receipt:
        return
    row = json.loads(controller.vast("show", "instance", str(receipt["instance_id"])))
    if row.get("actual_status") != "running":
        age_min = (time.time() - float(row.get("start_date") or time.time())) / 60
        if age_min > 12:  # loading is normal for the first minutes
            problems.add(f"vast_{lane}_instance_not_running", "CRIT",
                         f"Vast lane {lane} instance {receipt['instance_id']} status {row.get('actual_status')} (intended {row.get('intended_status')}) after {age_min:.0f} min", 2)
        return
    try:
        info = controller.probe()
    except Exception as error:  # noqa: BLE001 - ssh failures are the signal
        problems.add(f"vast_{lane}_ssh_unreachable", "CRIT", f"lane {lane}: cannot read the Vast host over ssh: {type(error).__name__}", 2)
        return
    if not info.get("GPU"):
        problems.add(f"vast_{lane}_probe_empty", "CRIT", f"lane {lane}: ssh worked but the host probe returned nothing", 2)
        return
    pending = int(info.get("PENDING", "0 RUNNING 0").split()[0])
    in_flight = int(info.get("PENDING", "0 RUNNING 0").split("RUNNING")[1].split()[0])
    utils = [float(x.split(",")[0]) for x in info["GPU"].split(";") if x.strip()]
    if (pending or in_flight) and utils and max(utils) < 40:
        problems.add(f"vast_{lane}_gpu_idle_with_queue", "WARN",
                     f"lane {lane}: Vast GPUs idle (util {utils}) with {pending} pending and {in_flight} running jobs", 3)
    if pending <= 3:
        ready = len(controller.ready_items(lane)) if hasattr(controller, "ready_items") else 0
        problems.add(f"vast_{lane}_queue_low", "WARN",
                     f"lane {lane}: only {pending} jobs pending ({in_flight} running); {ready} backlog items ready -> run `VT_LANE={'' if lane == 'a' else lane} controller.py feed`", 2)
    daemon = controller.ssh("pgrep -fc '^python remote_refill.py' || true", timeout=40).stdout.strip()
    if (pending or in_flight) and daemon in ("", "0"):
        problems.add(f"vast_{lane}_refill_daemon_dead", "CRIT", f"lane {lane}: refill daemon not running with {pending} jobs pending", 2)
    stale = controller.ssh(
        "cd /root/run; for j in $(ls refill/running 2>/dev/null | sed 's/\\.json//'); do n=${j#*_}; "
        "f=$(ls -t logs/${n}.*.log 2>/dev/null | head -1); [ -n \"$f\" ] && echo \"$n $(( ($(date +%s)-$(stat -c %Y $f))/60 ))\"; done",
        timeout=40).stdout
    for line in stale.splitlines():
        name, _, age = line.partition(" ")
        if age.strip().isdigit() and int(age) > 15:
            problems.add(f"vast_{lane}_job_stalled:{name}", "WARN", f"lane {lane}: {name}: log not updated for {age} min", 2)
    failed = controller.ssh("ls /root/run/refill/done | grep -v '\\.rc0\\.json$' || true", timeout=40).stdout.split()
    seen = set(state.setdefault("vast_failed_seen", []))
    for name in failed:
        token = f"{lane}:{name}"
        if token not in seen:
            problems.add(f"vast_job_failed:{token}", "CRIT", f"lane {lane}: Vast job finished with a failure: {name}", 1)
            state["vast_failed_seen_new"] = state.get("vast_failed_seen_new", []) + [token]
    disk = int(info.get("DISKFREE", "999") or 999)
    if disk < 20:
        problems.add(f"vast_{lane}_disk_low", "CRIT" if disk < 8 else "WARN", f"lane {lane}: only {disk} GB disk free on the host (disk_free_gb)", 1)
    hours, cost = controller.spend()
    if cost >= SPEND_WARN:
        problems.add(f"vast_{lane}_spend_high", "WARN", f"lane {lane}: spend ${cost:.2f} on this instance ({hours:.1f} h); queue holds near the cap", 1)


def check_local(problems: Problems, state: dict) -> None:
    barrier = BC / "runtime/STOP_ADMISSION.json"
    if barrier.exists():
        reason = json.loads(barrier.read_text()).get("reason", "?")[:160]
        problems.add("local_admission_barrier_closed", "CRIT", f"local admission closed: {reason}", 1)
    queue = BC / "local_queue_r1"
    for folder in ("failed", "launch_failed", "blocked"):
        names = sorted(p.name for p in (queue / folder).glob("*.json"))
        seen = set(state.setdefault(f"local_{folder}_seen", names if not state.get("initialized") else []))
        for name in names:
            if name not in seen:
                problems.add(f"local_{folder}:{name}", "CRIT", f"local queue item moved to {folder}: {name}", 1)
                state.setdefault("local_new", []).append([folder, name])
    pending = list((queue / "pending").glob("*.json"))
    log = queue / "runner_log.jsonl"
    last_event_age = (time.time() - log.stat().st_mtime) / 60 if log.exists() else 9999
    gpu_apps = [line for line in sh(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).splitlines() if line.strip()]
    if pending and not gpu_apps and not barrier.exists() and last_event_age > 15:
        problems.add("local_gpu_idle_with_queue", "WARN",
                     f"local GPU idle {last_event_age:.0f} min with {len(pending)} queued items and no barrier", 2)
    if len(pending) < 3:
        problems.add("local_queue_low", "WARN", f"local GPU queue has only {len(pending)} pending items", 2)
    timers = sh(["systemctl", "--user", "list-timers", "--all", "--no-pager"])
    for name in EXPECTED_TIMERS:
        if name not in timers:
            problems.add(f"timer_missing:{name}", "CRIT", f"systemd timer {name} is not loaded", 1)


def check_systemd(problems: Problems, state: dict) -> None:
    failed = sorted(line.split()[0] for line in sh(["systemctl", "--user", "--failed", "--no-legend", "--plain"]).splitlines()
                    if line.strip() and "wnn" in line.split()[0])
    baseline = set(state.setdefault("failed_units_baseline", failed if not state.get("initialized") else []))
    for unit in failed:
        if unit not in baseline:
            problems.add(f"unit_failed:{unit}", "WARN", f"systemd unit failed: {unit}", 1)


def check_kaggle(problems: Problems, state: dict) -> None:
    last = state.get("kaggle_checked", 0)
    if time.time() - last < 900:
        for key, message in state.get("kaggle_problems", {}).items():
            problems.add(key, message[0], message[1], 1)
        return
    state["kaggle_checked"] = time.time()
    state["kaggle_problems"] = {}
    for kernel in KAGGLE_KERNELS:
        out = sh([str(ROOT / ".venv/bin/kaggle"), "kernels", "status", kernel], timeout=60)
        status = out.strip().splitlines()[-1] if out.strip() else "unknown"
        previous = state.setdefault("kaggle_status", {}).get(kernel)
        state["kaggle_status"][kernel] = status
        if "429" in out or "Too Many" in out or "unknown" in status:
            continue
        if any(word in status.upper() for word in ("ERROR", "CANCEL", "FAIL")):
            key = f"kaggle_kernel_bad:{kernel}"
            state["kaggle_problems"][key] = ["CRIT", f"Kaggle kernel {kernel}: {status[-120:]}"]
            problems.add(key, "CRIT", f"Kaggle kernel {kernel}: {status[-120:]}", 1)
        elif "COMPLETE" in status.upper() and previous and "COMPLETE" not in previous.upper():
            problems.add(f"kaggle_kernel_done:{kernel}", "INFO", f"Kaggle kernel finished: {kernel} (collect and grade)", 1)


def check_resources(problems: Problems, state: dict) -> None:
    home_free = shutil.disk_usage("/home").free / 1e9
    if home_free < 15:
        problems.add("disk_home_critical", "CRIT", f"/home has only {home_free:.0f} GB free", 1)
    elif home_free < 25:
        problems.add("disk_home_low", "WARN", f"/home has {home_free:.0f} GB free", 1)
    meminfo = dict(line.split(":") for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(meminfo["MemAvailable"].split()[0]) / 1e6
    if available < 3:
        problems.add("ram_low", "WARN", f"available RAM {available:.1f} GB", 2)


def check_oom(problems: Problems, state: dict) -> None:
    """Kernel OOM kills in the last 15 minutes (cgroup-limited campaign jobs, ad-hoc probes, anything else of this user)."""
    import re

    seen = set(state.get("oom_seen", []))
    out = sh(["journalctl", "-k", "--since", "-15min", "--no-pager"], timeout=30)
    for match in re.finditer(r"(Memory cgroup out of memory|Out of memory): Killed process (\d+) \((\S+)\) total-vm:\d+kB, anon-rss:(\d+)kB", out):
        pid = match.group(2)
        if pid in seen:
            continue
        seen.add(pid)
        scope = re.search(rf"oom-kill:.*?oom_memcg=(\S+?),.*?pid={pid}", out)
        where = scope.group(1).rsplit("/", 1)[-1] if scope else "?"
        problems.add(f"oom_kill:{pid}", "WARN", f"OOM-killed pid {pid} ({match.group(3)}), anon {int(match.group(4)) / 1e6:.1f} GB, cgroup {where}", 1)
    state["oom_seen"] = sorted(seen)[-200:]


def check_hpo(problems: Problems, state: dict) -> None:
    if (ROOT / ".runtime/hpo_disabled").exists():  # user stopped the HPO direction 2026-10-11: no alerts for a deliberately stopped study
        return
    """Optimizer health: ticks must keep landing and a running study must keep producing events."""
    for config in sorted((ROOT / "hpo/studies").glob("*/config.json")):
        study = config.parent
        events = study / "events.jsonl"
        if not events.exists() or (study / "STOP").exists() or (study / "DISABLED").exists():
            continue
        age_min = (time.time() - events.stat().st_mtime) / 60
        keepalive = ROOT / ".runtime/hpo_keepalive"
        tick_age = (time.time() - keepalive.stat().st_mtime) / 60 if keepalive.exists() else 9999
        if tick_age > 25:
            problems.add(f"hpo_{study.name}_ticks_stopped", "CRIT", f"hpo study {study.name}: no tick for {tick_age:.0f} min (timer or orchestrator down)", 1)
        if age_min > 150:
            problems.add(f"hpo_{study.name}_no_progress", "WARN", f"hpo study {study.name}: no event for {age_min:.0f} min", 1)
        # Training is deterministic given (config, seed): equal scores of different trials mean the sampler repeated a configuration.
        scores: dict[tuple, list[int]] = {}
        failed: set[int] = set()
        for line in events.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("ev") == "tell" and row.get("state") == "FAIL":
                failed.add(row.get("trial"))
            if row.get("ev") == "stage_result":
                scores.setdefault((row.get("fold"), row.get("wp")), []).append(row.get("trial"))
        scores = {key: [t for t in trials if t not in failed] for key, trials in scores.items()}
        repeated = [trials for trials in scores.values() if len(set(trials)) > 2]
        if repeated:
            problems.add(f"hpo_{study.name}_duplicate_scores", "WARN",
                         f"hpo study {study.name}: {len(repeated)} score value(s) shared by more than two trials (sampler repeating a configuration?)", 1)


CHECKS = [("hpo", check_hpo), ("vast", check_vast), ("local", check_local), ("systemd", check_systemd), ("kaggle", check_kaggle),
          ("resources", check_resources), ("oom", check_oom)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", required=True)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    WATCH.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    problems = Problems(state.get("counts", {}))
    for name, check in CHECKS:
        try:
            check(problems, state)
        except Exception as error:  # noqa: BLE001 - a broken check must be visible
            problems.add(f"watchdog_check_error:{name}", "WARN", f"watchdog check '{name}' failed: {type(error).__name__}: {str(error)[:160]}", 3)
    active = problems.active()
    previous_active = state.get("active", {})
    rows = []
    for key, (severity, message) in active.items():
        if key not in previous_active:
            rows.append({"utc": now(), "event": "ALERT", "severity": severity, "key": key, "message": message})
    for key, (_severity, message) in previous_active.items():
        if key not in active and not key.startswith(("vast_job_failed", "local_failed", "local_launch_failed", "local_blocked", "unit_failed", "kaggle_kernel_done", "oom_kill")):
            rows.append({"utc": now(), "event": "RECOVERED", "severity": "INFO", "key": key, "message": f"recovered: {message}"})
    # one-shot events become 'seen' once alerted so they do not repeat after the problem row disappears
    for name in state.pop("vast_failed_seen_new", []):
        state["vast_failed_seen"] = sorted(set(state.get("vast_failed_seen", [])) | {name})
    for folder, name in state.pop("local_new", []):
        state[f"local_{folder}_seen"] = sorted(set(state.get(f"local_{folder}_seen", [])) | {name})
    for key in list(active):
        if key.startswith(("unit_failed:", "kaggle_kernel_done")):
            state.setdefault("failed_units_baseline", []).append(key.split(":", 1)[-1])
    state["initialized"] = True
    state["counts"] = problems.counts
    state["active"] = {k: list(v) for k, v in active.items() if not k.startswith(("vast_job_failed", "local_failed", "local_launch_failed", "local_blocked", "oom_kill"))}
    state.pop("kaggle_status_prev", None)
    STATE_PATH.write_text(json.dumps(state, indent=1))
    HEARTBEAT.write_text(now())
    if rows:
        with (WATCH / "ALERTS.jsonl").open("a") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        for row in rows:
            print(f"{row['event']}|{row['severity']}|{row['key']}|{row['message']}", flush=True)
            if args.notify and row["event"] == "ALERT" and row["severity"] in ("CRIT", "WARN"):
                subprocess.run(["notify-send", "-u", "critical" if row["severity"] == "CRIT" else "normal",
                                f"WNN watchdog {row['severity']}", row["message"][:200]], check=False)
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.exit(main())
