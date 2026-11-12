from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

END_DRAIN_SECONDS = 30.0
TERM_GRACE_SECONDS = 5.0
KILL_BY_SECONDS = 10.0
WAVE_END_EPOCH = 1794471619.060462
POLL_SECONDS = 0.2
TERM_SIGNAL = getattr(signal, "SIGTERM", 15)
KILL_SIGNAL = getattr(signal, "SIGKILL", 9)
TRANSPORT_COMMS = {
    "ssh",
    "sshd",
    "ssh-agent",
    "rpcbind",
    "rpc.statd",
    "rpc.mountd",
    "rpc.gssd",
    "rpc.idmapd",
    "rpc.nfsd",
}

B300_ROOTS = {
    "/data/epanov_s/tabdata-all-20261112-r1/control/impact81-transformer-queue-r2",
    "/data/epanov_s/tabdata-all-20261112-r1/control/impact81-gru-queue-r1",
    "/data/epanov_s/tabdata-all-20261112-r1/control/impact81-check-r8",
    "/data/epanov_s/tabdata-all-20261112-r1/control/teacher-export-r2",
    "/data/epanov_s/tabdata-all-20261112-r1/control/partial-e3-readout-r1",
}
L4_ROOTS = {
    "/data/lc-epanov_s/tabdata-all-20261112-r1/control/root-fits-r1",
    "/data2/lc-epanov_s/tabdata-all-followups-20261112-r1/control/root-fits-r4",
    "/data2/lc-epanov_s/tabdata-ema-20261112-r1/control/root-fits-ema-r2",
    "/data/lc-epanov_s/tabdata-all-readouts-20261112-r3/control/canonical-readouts-r3",
    "/data2/lc-epanov_s/tabdata-followup-readouts-20261112-r3/control/canonical-readouts-r3",
    "/data/lc-epanov_s/tabdata-ema-readouts-20261112-r3/control/canonical-readouts-r3",
    "/data/lc-epanov_s/tabdata-all-readouts-20261112-r4/control/canonical-readouts-r3",
    "/data2/lc-epanov_s/tabdata-followup-readouts-20261112-r4/control/canonical-readouts-r3",
    "/data/lc-epanov_s/tabdata-ema-readouts-20261112-r4/control/canonical-readouts-r3",
}
ROOTS_BY_ROLE = {"b300": B300_ROOTS, "l4": L4_ROOTS}
OUTPUT_BY_ROLE = {
    "b300": "/data/epanov_s/tabdata-all-20261112-r1/control/wave-drain-r1",
    "l4": "/data/lc-epanov_s/tabdata-all-20261112-r1/control/wave-drain-r1",
}
PID_KEYS = {"pid", "process_id"}
BIRTH_KEYS = {
    "birth",
    "birth_ticks",
    "start_ticks",
    "starttime_ticks",
    "start_time_ticks",
    "proc_start_ticks",
}
EXPLICIT_PAIRS = {
    ("owner_pid", "owner_birth"),
    ("owner_pid", "owner_birth_ticks"),
    ("child_pid", "child_birth"),
    ("child_pid", "child_birth_ticks"),
    ("controller_pid", "controller_birth_ticks"),
}


@dataclass(frozen=True)
class ReceiptIdentity:
    pid: int
    birth_ticks: int
    source: str


@dataclass(frozen=True)
class ProcessRow:
    pid: int
    birth_ticks: int
    ppid: int
    uid: int
    comm: str
    exe_name: str
    state: str = "S"


@dataclass(frozen=True)
class OwnedProcess:
    pid: int
    birth_ticks: int
    ppid: int
    uid: int
    comm_sha256: str
    depth: int
    source: str


class Procfs:
    """Read Linux process identity fields without using command names as ownership."""

    def __init__(self, proc_root: Path = Path("/proc")) -> None:
        self.proc_root = proc_root

    def read(self, pid: int) -> ProcessRow | None:
        try:
            stat_text = (self.proc_root / str(pid) / "stat").read_text(encoding="utf-8")
            fields = stat_text.rsplit(")", 1)[1].split()
            status = (self.proc_root / str(pid) / "status").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, IndexError, OSError):
            return None
        uid_line = next(
            (line for line in status.splitlines() if line.startswith("Uid:")), None
        )
        if uid_line is None:
            return None
        try:
            exe_path = os.readlink(self.proc_root / str(pid) / "exe")
            exe_name = Path(exe_path).name
            comm = stat_text.split("(", 1)[1].rsplit(")", 1)[0]
            return ProcessRow(
                pid,
                int(fields[19]),
                int(fields[1]),
                int(uid_line.split()[2]),
                comm,
                exe_name,
                fields[0],
            )
        except (IndexError, ValueError):
            return None
        except OSError:
            return None

    def all(self) -> list[ProcessRow]:
        rows: list[ProcessRow] = []
        try:
            entries = list(self.proc_root.iterdir())
        except OSError:
            return rows
        for entry in entries:
            if entry.name.isdigit():
                row = self.read(int(entry.name))
                if row is not None:
                    rows.append(row)
        return rows


def allowed_root(path: str, role: str) -> bool:
    """Require an exact absolute native control-root path from the role allowlist."""
    candidate = PurePosixPath(path)
    if not candidate.is_absolute() or ".." in candidate.parts or str(candidate) != path:
        return False
    return path in ROOTS_BY_ROLE[role]


def path_is_within(path: str, root: str) -> bool:
    """Check lexical POSIX containment, including paths whose root is not staged yet."""
    candidate, base = PurePosixPath(path), PurePosixPath(root)
    return (
        candidate.is_absolute()
        and ".." not in candidate.parts
        and candidate.parts[: len(base.parts)] == base.parts
    )


def receipt_root_for(path: str, roots: Iterable[str]) -> str | None:
    """Return the most specific explicitly supplied root containing a receipt path."""
    matches = [root for root in roots if path_is_within(path, root)]
    return max(matches, key=lambda root: len(PurePosixPath(root).parts), default=None)


def _as_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def extract_identities(document: object, source: str) -> list[ReceiptIdentity]:
    """Extract only PID/start-birth pairs from an explicitly supplied owner receipt."""
    found: set[tuple[int, int]] = set()

    def add(pid_value: object, birth_value: object) -> None:
        pid, birth = _as_positive_int(pid_value), _as_positive_int(birth_value)
        if pid is not None and birth is not None:
            found.add((pid, birth))

    def visit(value: object) -> None:
        if isinstance(value, dict):
            lowered = {str(key).casefold(): item for key, item in value.items()}
            for pid_key in PID_KEYS:
                for birth_key in BIRTH_KEYS:
                    if pid_key in lowered and birth_key in lowered:
                        add(lowered[pid_key], lowered[birth_key])
            for pid_key, birth_key in EXPLICIT_PAIRS:
                if pid_key in lowered and birth_key in lowered:
                    add(lowered[pid_key], lowered[birth_key])
            for key, child in lowered.items():
                if key.endswith("_pid_births"):
                    if isinstance(child, dict):
                        for pid_key, birth in child.items():
                            if isinstance(birth, dict):
                                inner = {str(k).casefold(): v for k, v in birth.items()}
                                for birth_key in BIRTH_KEYS:
                                    if birth_key in inner:
                                        add(pid_key, inner[birth_key])
                            else:
                                add(pid_key, birth)
                    elif isinstance(child, list):
                        for item in child:
                            visit(item)
                else:
                    visit(child)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(document)
    return [ReceiptIdentity(pid, birth, source) for pid, birth in sorted(found)]


def read_receipt(
    path: str, kind: str, roots: Iterable[str], role: str
) -> dict[str, object]:
    """Read one designated control receipt; absent roots/files are recorded, not fatal."""
    if (
        Path(path).suffix.casefold() != ".json"
        or "config" in Path(path).name.casefold()
    ):
        raise ValueError(f"configuration files are not valid {kind} receipts: {path}")
    if not any(
        path_is_within(path, root) and allowed_root(root, role) for root in roots
    ):
        raise ValueError(
            f"{kind} receipt is outside explicit whitelisted roots: {path}"
        )
    receipt: dict[str, object] = {"kind": kind, "path": path, "status": "missing"}
    if not Path(path).is_file():
        return receipt
    resolved = Path(path).resolve(strict=True)
    parent_roots = []
    for root in roots:
        root_path = Path(root)
        if root_path.exists():
            resolved_root = root_path.resolve(strict=True)
            if Path(resolved).is_relative_to(resolved_root):
                parent_roots.append(resolved_root)
    if not parent_roots:
        raise ValueError(
            f"{kind} receipt resolves outside its whitelisted root: {path}"
        )
    payload = resolved.read_bytes()
    document = json.loads(payload)
    identities = extract_identities(document, path)
    if not identities:
        raise ValueError(
            f"{kind} receipt contains no recognized PID/birth identity: {path}"
        )
    receipt.update(
        status="read",
        sha256=hashlib.sha256(payload).hexdigest(),
        evidence_source_id=hashlib.sha256(payload).hexdigest(),
        identities=[asdict(item) for item in identities],
    )
    return receipt


def _is_transport(row: ProcessRow) -> bool:
    comm = row.comm.casefold()
    executable = row.exe_name.casefold()
    return (
        comm in TRANSPORT_COMMS
        or comm.startswith(("rpc.", "ssh"))
        or executable in TRANSPORT_COMMS
        or executable.startswith(("rpc.", "ssh"))
    )


def _is_live(row: ProcessRow | None) -> bool:
    """Treat zombie and exiting procfs entries as non-running processes."""
    return row is not None and row.state not in {"Z", "X", "x"}


def snapshot_owned_tree(
    owners: Iterable[ReceiptIdentity],
    explicit_owned: Iterable[ReceiptIdentity],
    procfs: Procfs,
    auditor_pid: int,
    expected_uid: int | None = None,
) -> tuple[list[OwnedProcess], list[dict[str, object]]]:
    """Snapshot receipt-proven roots plus same-UID descendants.

    A dead controller does not invalidate a separate live PID/birth receipt. In
    that case expected_uid (the native auditor's UID) still binds the child.
    """
    owner_ids = list(owners)
    explicit_ids = list(explicit_owned)
    rows = {row.pid: row for row in procfs.all()}
    excluded: list[dict[str, object]] = []
    seed_rows: dict[int, tuple[ProcessRow, str]] = {}
    live_owners: list[ProcessRow] = []
    for identity in owner_ids:
        row = rows.get(identity.pid)
        if row is None or row.birth_ticks != identity.birth_ticks:
            excluded.append(
                {
                    "pid": identity.pid,
                    "birth_ticks": identity.birth_ticks,
                    "reason": "owner_missing_or_birth_mismatch",
                    "source": identity.source,
                }
            )
        elif identity.pid == auditor_pid:
            excluded.append(
                {
                    "pid": identity.pid,
                    "birth_ticks": identity.birth_ticks,
                    "reason": "auditor_excluded",
                    "source": identity.source,
                }
            )
        elif not _is_live(row):
            excluded.append(
                {
                    "pid": identity.pid,
                    "birth_ticks": identity.birth_ticks,
                    "reason": "zombie_or_dead_process",
                    "source": identity.source,
                }
            )
        elif _is_transport(row):
            excluded.append(
                {
                    "pid": identity.pid,
                    "birth_ticks": identity.birth_ticks,
                    "reason": "transport_process_excluded",
                    "source": identity.source,
                }
            )
        elif expected_uid is not None and row.uid != expected_uid:
            excluded.append(
                {
                    "pid": identity.pid,
                    "birth_ticks": identity.birth_ticks,
                    "reason": "owner_uid_mismatch",
                    "source": identity.source,
                }
            )
        else:
            live_owners.append(row)
            seed_rows[identity.pid] = (row, identity.source)
    owner_uids = {row.uid for row in live_owners}
    if expected_uid is None and len(owner_uids) > 1:
        raise ValueError("verified owner receipts span multiple UIDs")
    owner_uid = (
        expected_uid if expected_uid is not None else next(iter(owner_uids), None)
    )
    for identity in explicit_ids:
        row = rows.get(identity.pid)
        if row is None or row.birth_ticks != identity.birth_ticks:
            excluded.append(
                {
                    "pid": identity.pid,
                    "birth_ticks": identity.birth_ticks,
                    "reason": "owned_missing_or_birth_mismatch",
                    "source": identity.source,
                }
            )
        elif identity.pid == auditor_pid:
            excluded.append(
                {
                    "pid": identity.pid,
                    "birth_ticks": identity.birth_ticks,
                    "reason": "auditor_excluded",
                    "source": identity.source,
                }
            )
        elif not _is_live(row):
            excluded.append(
                {
                    "pid": identity.pid,
                    "birth_ticks": identity.birth_ticks,
                    "reason": "zombie_or_dead_process",
                    "source": identity.source,
                }
            )
        elif owner_uid is None or row.uid != owner_uid:
            excluded.append(
                {
                    "pid": identity.pid,
                    "birth_ticks": identity.birth_ticks,
                    "reason": "owned_uid_mismatch",
                    "source": identity.source,
                }
            )
        elif _is_transport(row):
            excluded.append(
                {
                    "pid": identity.pid,
                    "birth_ticks": identity.birth_ticks,
                    "reason": "transport_process_excluded",
                    "source": identity.source,
                }
            )
        else:
            seed_rows[identity.pid] = (row, identity.source)
    children: dict[int, list[ProcessRow]] = {}
    for row in rows.values():
        children.setdefault(row.ppid, []).append(row)
    pending = [(row, 0, source) for row, source in seed_rows.values()]
    seen: set[int] = set()
    owned: list[OwnedProcess] = []
    while pending:
        row, depth, source = pending.pop(0)
        if row.pid in seen:
            continue
        seen.add(row.pid)
        if row.pid == auditor_pid:
            excluded.append(
                {
                    "pid": row.pid,
                    "birth_ticks": row.birth_ticks,
                    "reason": "auditor_excluded",
                    "source": source,
                }
            )
            continue
        if not _is_live(row):
            excluded.append(
                {
                    "pid": row.pid,
                    "birth_ticks": row.birth_ticks,
                    "reason": "zombie_or_dead_process",
                    "source": source,
                }
            )
            continue
        if owner_uid is None or row.uid != owner_uid:
            excluded.append(
                {
                    "pid": row.pid,
                    "birth_ticks": row.birth_ticks,
                    "reason": "descendant_uid_mismatch",
                    "source": source,
                }
            )
            continue
        if _is_transport(row):
            excluded.append(
                {
                    "pid": row.pid,
                    "birth_ticks": row.birth_ticks,
                    "reason": "transport_process_excluded",
                    "source": source,
                }
            )
            continue
        comm_hash = hashlib.sha256(row.comm.encode("utf-8")).hexdigest()
        owned.append(
            OwnedProcess(
                row.pid, row.birth_ticks, row.ppid, row.uid, comm_hash, depth, source
            )
        )
        for child in children.get(row.pid, []):
            pending.append((child, depth + 1, "same_uid_descendant"))
    return sorted(owned, key=lambda item: (item.depth, item.pid)), excluded


def _same_process(ref: OwnedProcess, procfs: Procfs) -> bool:
    row = procfs.read(ref.pid)
    return _is_live(row) and row.birth_ticks == ref.birth_ticks and row.uid == ref.uid


def signal_exact(ref: OwnedProcess, sig: int, procfs: Procfs) -> str:
    """Signal one proven process, preferring pidfd and rechecking birth and UID."""
    row = procfs.read(ref.pid)
    if not _is_live(row) or row.birth_ticks != ref.birth_ticks or row.uid != ref.uid:
        return "identity_changed_or_gone"
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send = getattr(signal, "pidfd_send_signal", None)
    if pidfd_open is not None and pidfd_send is not None:
        try:
            fd = pidfd_open(ref.pid, 0)
        except ProcessLookupError:
            return "already_gone"
        except PermissionError:
            return "permission_denied"
        except OSError as exc:
            if exc.errno not in {errno.ENOSYS, errno.EINVAL}:
                return f"pidfd_open_error_{exc.errno}"
            fd = -1
        if fd < 0:
            return _signal_pid_fallback(ref, sig, procfs)
        try:
            row = procfs.read(ref.pid)
            if (
                not _is_live(row)
                or row.birth_ticks != ref.birth_ticks
                or row.uid != ref.uid
            ):
                return "identity_changed_after_pidfd"
            try:
                pidfd_send(fd, sig)
            except ProcessLookupError:
                return "already_gone"
            except PermissionError:
                return "permission_denied"
            except OSError as exc:
                return f"pidfd_signal_error_{exc.errno}"
        finally:
            os.close(fd)
        return "pidfd_signal_sent"
    return _signal_pid_fallback(ref, sig, procfs)


def _signal_pid_fallback(ref: OwnedProcess, sig: int, procfs: Procfs) -> str:
    """Use exact PID only when birth and UID still match immediately before os.kill."""
    row = procfs.read(ref.pid)
    if not _is_live(row) or row.birth_ticks != ref.birth_ticks or row.uid != ref.uid:
        return "identity_changed_before_kill_fallback"
    try:
        os.kill(ref.pid, sig)
    except ProcessLookupError:
        return "already_gone"
    except PermissionError:
        return "permission_denied"
    except OSError as exc:
        return f"kill_fallback_error_{exc.errno}"
    return "birth_rechecked_kill_fallback"


def drain_snapshot(
    owned: list[OwnedProcess],
    procfs: Procfs,
    end_epoch: float,
    signal_fn: Callable[[OwnedProcess, int, Procfs], str],
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> list[dict[str, object]]:
    """TERM the verified snapshot, wait five seconds, then KILL survivors by T-10."""
    actions: list[dict[str, object]] = []
    for ref in sorted(owned, key=lambda item: (-item.depth, item.pid)):
        signal_epoch = now()
        result = signal_fn(ref, TERM_SIGNAL, procfs)
        actions.append(
            {
                "pid": ref.pid,
                "birth_ticks": ref.birth_ticks,
                "signal": "TERM",
                "result": result,
                "signal_epoch": signal_epoch,
                "release_after_deadline": signal_epoch >= end_epoch,
            }
        )
    kill_by = end_epoch - KILL_BY_SECONDS
    grace_deadline = min(now() + TERM_GRACE_SECONDS, kill_by)
    while now() < grace_deadline and any(_same_process(ref, procfs) for ref in owned):
        sleep(min(POLL_SECONDS, max(0.0, grace_deadline - now())))
    for ref in sorted(owned, key=lambda item: (-item.depth, item.pid)):
        if _same_process(ref, procfs):
            signal_epoch = now()
            result = signal_fn(ref, KILL_SIGNAL, procfs)
            actions.append(
                {
                    "pid": ref.pid,
                    "birth_ticks": ref.birth_ticks,
                    "signal": "KILL",
                    "result": result,
                    "signal_epoch": signal_epoch,
                    "missed_t10_deadline": signal_epoch > kill_by,
                    "release_after_deadline": signal_epoch >= end_epoch,
                }
            )
    return actions


def gpu_process_records(timeout: float = 5.0) -> tuple[list[dict[str, object]], str]:
    """Query only PID, GPU UUID, and memory fields from NVIDIA compute processes."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"unavailable:{type(exc).__name__}"
    records: list[dict[str, object]] = []
    for row in csv.reader(result.stdout.splitlines()):
        if not row:
            continue
        try:
            pid = int(row[0].strip())
            memory_text = row[2].strip().split()[0]
            memory_mib: int | str = (
                int(memory_text) if memory_text.isdigit() else memory_text
            )
            records.append(
                {"pid": pid, "gpu_uuid": row[1].strip(), "used_memory_mib": memory_mib}
            )
        except (IndexError, ValueError):
            if any(cell.strip() for cell in row):
                return [], "unparseable_nvidia_smi_output"
    return records, "ok"


def gpu_ownership_report(
    records: Iterable[dict[str, object]], owned: Iterable[OwnedProcess], procfs: Procfs
) -> dict[str, object]:
    """Classify only PID/birth matches as ours; foreign GPU PIDs are never signalled."""
    by_pid = {ref.pid: ref for ref in owned}
    ours: list[dict[str, object]] = []
    foreign: list[dict[str, object]] = []
    for record in records:
        pid = int(record["pid"])
        ref = by_pid.get(pid)
        if ref is not None and _same_process(ref, procfs):
            ours.append({**record, "birth_ticks": ref.birth_ticks})
        else:
            foreign.append(record)
    return {"owned_gpu_processes": ours, "foreign_gpu_processes_untouched": foreign}


def write_immutable_json(path: str, document: dict[str, object]) -> None:
    """Create a new final JSON exactly once and remove write permission after fsync."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        fchmod = getattr(os, "fchmod", None)
        if fchmod is not None:
            fchmod(fd, 0o444)
        else:
            os.chmod(target, 0o444)
    finally:
        os.close(fd)


def pin_to_one_cpu() -> int | None:
    """Restrict this auditor to one available CPU where affinity APIs are present."""
    get_affinity, set_affinity = (
        getattr(os, "sched_getaffinity", None),
        getattr(os, "sched_setaffinity", None),
    )
    if get_affinity is None or set_affinity is None:
        raise RuntimeError("one-CPU affinity controls are unavailable")
    available = get_affinity(0)
    if not available:
        raise RuntimeError("no CPU affinity available")
    cpu = min(available)
    set_affinity(0, {cpu})
    return cpu


def wait_for_epoch(
    target_epoch: float, now: Callable[[], float], sleep: Callable[[float], None]
) -> None:
    """Wait in short bounded intervals until the supplied UTC epoch."""
    while now() < target_epoch:
        sleep(min(1.0, max(0.0, target_epoch - now())))


def _read_inputs(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, object]], list[ReceiptIdentity], list[ReceiptIdentity], list[str]
]:
    receipts: list[dict[str, object]] = []
    owners: list[ReceiptIdentity] = []
    owned: list[ReceiptIdentity] = []
    errors: list[str] = []
    for kind, paths in (
        ("owner", args.owner_receipt),
        ("start", args.start_receipt),
        ("owned", args.owned_receipt),
    ):
        for path in paths:
            try:
                item = read_receipt(path, kind, args.root, args.role)
                receipts.append(item)
                identities = [
                    ReceiptIdentity(**entry) for entry in item.get("identities", [])
                ]
                if item["status"] == "read":
                    if kind == "owner":
                        owners.extend(identities)
                    else:
                        owned.extend(identities)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{kind}:{path}:{type(exc).__name__}:{exc}")
    return receipts, owners, owned, errors


def designated_receipt_paths(
    role: str, roots: Iterable[str]
) -> tuple[list[str], list[str], list[str]]:
    """Select only owner.json, controller-owner.json, and owned/*.json."""
    owners: list[str] = []
    starts: list[str] = []
    owned: list[str] = []
    for root in roots:
        base = Path(root)
        owner_json = base / "owner.json"
        controller_owner = base / "controller-owner.json"
        if role == "l4" and controller_owner.is_file():
            owner_path = controller_owner
        elif owner_json.is_file():
            owner_path = owner_json
        else:
            owner_path = controller_owner if role == "l4" else owner_json
        start_path = owner_json if owner_json.is_file() else owner_path
        owners.append(str(owner_path))
        starts.append(str(start_path))
        owned_dir = base / "owned"
        if owned_dir.is_dir():
            owned.extend(
                str(path)
                for path in sorted(owned_dir.glob("*.json"))
                if "config" not in path.name.casefold()
            )
    return owners, starts, owned


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Drain only receipt-owned native processes at the fixed wave cutoff."
    )
    parser.add_argument("--role", choices=sorted(ROOTS_BY_ROLE), required=True)
    parser.add_argument("--end-epoch", type=float, required=True)
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="explicit whitelisted control root; repeat",
    )
    parser.add_argument("--owner-receipt", action="append", default=[])
    parser.add_argument("--start-receipt", action="append", default=[])
    parser.add_argument("--owned-receipt", action="append", default=[])
    parser.add_argument(
        "--discover-control-receipts",
        action="store_true",
        help="read only owner.json/controller-owner.json and owned/*.json under exact roots",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="final proof is created once as OUTPUT_ROOT/final.json",
    )
    parser.add_argument(
        "--execute", action="store_true", help="send signals after the T-30 drain start"
    )
    parser.add_argument(
        "--gpu-check",
        action="store_true",
        help="query nvidia-smi PIDs at final verification",
    )
    args = parser.parse_args(argv)

    if not sys.platform.startswith("linux"):
        raise SystemExit("native process cleanup requires Linux /proc")
    if abs(args.end_epoch - WAVE_END_EPOCH) > 1e-6:
        raise SystemExit("--end-epoch must match the fixed wave cutoff")
    if not args.execute:
        raise SystemExit("--execute is required to create the end-of-wave proof")
    if set(args.root) != ROOTS_BY_ROLE[args.role] or len(args.root) != len(
        ROOTS_BY_ROLE[args.role]
    ):
        raise SystemExit("all exact role roots must be supplied once")
    for root in args.root:
        if not allowed_root(root, args.role):
            raise SystemExit(f"root is not whitelisted for {args.role}: {root}")
    if args.output_root != OUTPUT_BY_ROLE[args.role]:
        raise SystemExit(f"output root is not the fixed {args.role} proof path")
    if args.discover_control_receipts:
        if args.owner_receipt or args.start_receipt or args.owned_receipt:
            raise SystemExit("do not mix receipt discovery with manual receipt paths")
        (
            args.owner_receipt,
            args.start_receipt,
            args.owned_receipt,
        ) = designated_receipt_paths(args.role, args.root)
    elif not args.owner_receipt or not args.start_receipt:
        raise SystemExit("owner and start receipt paths are required")
    for path in [*args.owner_receipt, *args.start_receipt, *args.owned_receipt]:
        if (
            Path(path).suffix.casefold() != ".json"
            or "config" in Path(path).name.casefold()
            or not any(
                path_is_within(path, root) and allowed_root(root, args.role)
                for root in args.root
            )
        ):
            raise SystemExit(
                f"receipt path must be a designated non-config JSON under a whitelisted root: {path}"
            )
    cpu = pin_to_one_cpu()
    drain_epoch = args.end_epoch - END_DRAIN_SECONDS
    if args.execute:
        wait_for_epoch(drain_epoch, time.time, time.sleep)
    drain_observed_epoch = time.time()
    procfs = Procfs()
    root_status = [
        {"path": root, "status": "present" if Path(root).is_dir() else "missing"}
        for root in args.root
    ]
    receipts, owners, explicit_owned, errors = _read_inputs(args)
    owned, excluded = snapshot_owned_tree(
        owners, explicit_owned, procfs, os.getpid(), expected_uid=os.geteuid()
    )
    actions: list[dict[str, object]] = []
    if args.execute:
        actions = drain_snapshot(
            owned, procfs, args.end_epoch, signal_exact, time.time, time.sleep
        )
        # The immutable proof is a cutoff observation, not an early post-drain
        # snapshot. Wait through the remaining wave window before final checks.
        wait_for_epoch(args.end_epoch, time.time, time.sleep)
    remaining = [asdict(ref) for ref in owned if _same_process(ref, procfs)]
    gpu_report: dict[str, object] = {"status": "skipped"}
    if args.gpu_check:
        records, status = gpu_process_records()
        gpu_report = {"status": status, **gpu_ownership_report(records, owned, procfs)}
    missing_receipts = [
        item["path"] for item in receipts if item["status"] == "missing"
    ]
    missing_inputs = [
        item["path"] for item in root_status if item["status"] == "missing"
    ] + missing_receipts
    live_unresolved = {
        "owner_uid_mismatch",
        "owned_uid_mismatch",
        "descendant_uid_mismatch",
        "transport_process_excluded",
    }
    ownership_proof_complete = (
        bool(owners)
        and not errors
        and not any(item.get("reason") in live_unresolved for item in excluded)
    )
    owned_processes_clear = not remaining
    owned_gpu_processes_clear = gpu_report.get("status") == "ok" and not gpu_report.get(
        "owned_gpu_processes"
    )
    if ownership_proof_complete and owned_processes_clear and owned_gpu_processes_clear:
        final_status = "clear" if not missing_inputs else "clear_with_missing_inputs"
    elif (
        missing_inputs
        and not remaining
        and not errors
        and gpu_report.get("status") == "ok"
    ):
        final_status = "incomplete_missing_inputs"
    else:
        final_status = "incomplete_or_owned_process_remains"
    doc: dict[str, object] = {
        "schema": "wnn.native-end-of-wave-audit.v1",
        "role": args.role,
        "end_epoch": args.end_epoch,
        "drain_start_epoch": drain_epoch,
        "observed_epoch": time.time(),
        "drain_observed_epoch": drain_observed_epoch,
        "drain_started_late": drain_observed_epoch > drain_epoch + 1.0,
        "cleanup_after_deadline": drain_observed_epoch >= args.end_epoch,
        "execute": args.execute,
        "cpu_affinity": cpu,
        "control_roots": root_status,
        "owner_receipts": [item for item in receipts if item["kind"] == "owner"],
        "start_receipts": [item for item in receipts if item["kind"] == "start"],
        "owned_receipts": [item for item in receipts if item["kind"] == "owned"],
        "receipt_errors": errors,
        "owned_snapshot": [asdict(ref) for ref in owned],
        "excluded_identities": excluded,
        "signal_actions": actions,
        "remaining_owned_pid_births": remaining,
        "gpu": gpu_report,
        "foreign_gpu_action": "none",
        "missing_inputs": missing_inputs,
        "ownership_proof_complete": ownership_proof_complete,
        "owned_processes_clear": owned_processes_clear,
        "owned_gpu_processes_clear": owned_gpu_processes_clear,
        "scope_inputs_complete": not missing_inputs,
        "artifacts_preserved": True,
        "status": final_status,
    }
    final_path = str(Path(args.output_root) / "final.json")
    write_immutable_json(final_path, doc)
    return (
        0
        if final_status
        in {"clear", "clear_with_missing_inputs", "incomplete_missing_inputs"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
