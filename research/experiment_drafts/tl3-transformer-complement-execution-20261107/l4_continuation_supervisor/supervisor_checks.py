from __future__ import annotations

import hashlib
import json
from pathlib import Path

FIRST_PROFILE = 1794071266.717658
DEADLINE = FIRST_PROFILE + 82800


def digest(path: Path) -> str:
    """Stream actual checkpoint or bounded control bytes into SHA256."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path: Path) -> dict:
    """Read a regular control artifact without following an unexpected symbolic link."""
    if path.is_symlink():
        raise ValueError("control artifact cannot be a link")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("control artifact must contain an object")
    return value


def pinned(spec: dict) -> dict:
    """Bind ROOT-supplied metadata/config/profile/source/clock artifacts to native hashes."""
    path = Path(spec["path"])
    if not path.is_absolute() or digest(path) != spec["sha256"]:
        raise ValueError("ROOT-pinned artifact changed")
    return read(path)


def marker(path: Path) -> tuple[int, int, int, int]:
    """Identify one complete atomic safe-state publication."""
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def archive_complete(output: Path, proof_sha: str) -> bool:
    """Only the final immutable12E archive proof permits a clean archiver exit."""
    path = output / "archive_complete.json"
    if not path.exists():
        return False
    value = read(path)
    return (
        value.get("schema") == "wnn.gru-safe-state-archive-complete.v1"
        and value.get("completed_epochs") == 12
        and value.get("transition_proof_sha256") == proof_sha
        and value.get("archive_epochs") == list(range(value.get("initial_epoch", -100), 13))
    )


def archiver_guard(
    output: Path, proof_sha: str, receipt: dict | None, result: tuple | None, same: object
) -> None:
    """Fail immediately on archiver death except its verified final12E success."""
    if result is not None:
        if result[0] == 0 and result[1] is False and archive_complete(output, proof_sha):
            return
        raise RuntimeError("external retention archiver exited without complete12E proof")
    if receipt is None or not same(receipt["child"]):
        raise RuntimeError("external retention archiver lost its exact live identity")


def owned_trainer(
    receipt: dict, old: Path, source: Path, seed: int, gpu: int, identity: object
) -> dict:
    """Prove the exact trainer is a same-group PPID descendant of the native owned leader."""
    if (
        receipt.get("schema") != "wnn.multistage-owned-process.v1"
        or receipt.get("mode") != "fit"
        or receipt.get("owner", {}).get("cell_id") != f"gru_s{seed}"
        or Path(receipt["output_dir"]).resolve() != old.resolve()
        or receipt.get("gpu_key") != str(gpu)
    ):
        raise ValueError("old receipt is not the intended single-GPU GRU fit")
    leader = receipt["child"]
    if identity(int(leader["pid"])) != leader:
        raise ValueError("original owned leader PID/birth/PGID differs")
    records = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = path.read_text()
            fields = raw[raw.rfind(")") + 2 :].split()
            records[int(path.parent.name)] = (int(fields[1]), int(fields[2]), fields[19])
        except (OSError, ValueError, IndexError):
            continue
    family = {int(leader["pid"])}
    while True:
        found = {pid for pid, (parent, _, _) in records.items() if parent in family} - family
        if not found:
            break
        family.update(found)
    candidates = []
    for pid in family:
        try:
            args = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
            if (
                len(args) > 1
                and args[1] == str(source / "scripts/train_sequence.py")
                and "--output" in args
                and args[args.index("--output") + 1] == str(old)
            ):
                actual = identity(pid)
                if (
                    actual is not None
                    and actual["pgid"] == leader["pgid"]
                    and pid in records
                    and actual["birth"] == records[pid][2]
                ):
                    candidates.append(actual)
        except (OSError, UnicodeError, IndexError):
            continue
    if len(candidates) != 1:
        raise ValueError("expected exactly one PPID-proven same-group original trainer")
    return candidates[0]
