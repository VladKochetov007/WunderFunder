from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

COMMIT = "6ab2f679e70928f913b97d5a29f0d1e09ceacaa3"
SOURCE_MANIFEST_SHA = "9a3b147c20e902786f026a29741cd9f5135de91183e305105b2039f735905d7a"
SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
DEFAULT_RUNTIME = Path("/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1")
SHORT_RUN_ROOT = Path("/data/epanov_s/tdn7run")
DEFAULT_SUPPORT = Path(
    "/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1/operator-r26/src"
)
SPLIT = Path(
    "/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1/data/chronological_primary.json"
)
PACKET = "research/experiment_drafts/tl3-transformer-complement-execution-20261107"
CELLS = (
    ("labels_s101001", "0,1"),
    ("complement_s101001", "2,3"),
    ("labels_s101002", "4,5"),
    ("complement_s101002", "6,7"),
)


def digest(path: Path) -> str:
    """Hash bounded source/receipt files without reading weights or private rows."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_object(path: Path) -> dict[str, object]:
    """Read a bounded regular JSON receipt, rejecting links and incomplete records."""
    if path.is_symlink() or not 0 < path.stat().st_size <= 32_000_000:
        raise ValueError(f"invalid receipt file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def verify_acceptance(run_root: Path) -> bool:
    """Verify a completed source-bound check receipt without rerunning existing checks."""
    path = run_root / "acceptance/COMPLETE.json"
    if not path.exists():
        return False
    value = read_object(path)
    if (
        value.get("commit") != COMMIT
        or value.get("split_sha256") != SPLIT_SHA
        or value.get("status") != "checks_completed"
        or set(value.get("evidence_sha256", {}))
        != {
            "pytest.xml",
            "bf16.xml",
            "bf16x9.xml",
            "ddp.json",
            "complement-ddp.json",
            "muon-ddp.json",
        }
    ):
        raise ValueError("existing acceptance uses another source or split")
    for name, expected in value["evidence_sha256"].items():
        if Path(name).name != name or digest(path.parent / name) != expected:
            raise ValueError("acceptance evidence changed")
    return True


def reference_ready(runtime: Path) -> bool:
    """Require ROOT completion plus the independently reproducible full-valid baseline."""
    path = runtime / "receipts/reference-preparation-complete-r1.json"
    if not path.exists():
        return False
    value = read_object(path)
    baseline_path = runtime / "b300/baseline-score.json"
    if (
        value.get("status") not in ("completed", "complete", "passed")
        or value.get("split_sha256") != SPLIT_SHA
        or value.get("baseline_score_sha256") != digest(baseline_path)
    ):
        raise ValueError("reference preparation receipt is not bound to this baseline/split")
    baseline = read_object(baseline_path)
    score = baseline["score"]
    wp = score["wp"]
    if (
        baseline.get("schema") != "wnn.tl3-reference-score.v1"
        or baseline.get("split_sha256") != SPLIT_SHA
        or baseline.get("sequences") != 1873
        or score.get("metric_id") != "global_wp_clip2_v1"
        or not isinstance(wp, (int, float))
        or not math.isfinite(wp)
        or abs(wp - 0.6923774245652725) > 2e-6
    ):
        raise ValueError("baseline replay does not reproduce the qualified full-valid endpoint")
    for membership in ("fit", "evaluation"):
        manifest = runtime / "reference" / membership / "manifest.json"
        cache = read_object(manifest)
        if (
            cache.get("complete") is not True
            or cache.get("membership") != membership
            or cache.get("split_sha256") != SPLIT_SHA
            or cache.get("reference_sha256") != baseline["producer"]["reference_sha256"]
        ):
            raise ValueError("reference cache differs from qualified baseline")
        if (
            membership == "evaluation"
            and digest(manifest) != baseline["predictions_manifest_sha256"]
        ):
            raise ValueError("baseline reference manifest changed")
    return True


def approval_ready(run_root: Path, cell: str) -> bool:
    """Require ROOT's unchanged approved production copy; never set approval in this helper."""
    folder = run_root / cell
    path = folder / "approved_profile.json"
    if not path.exists():
        return False
    approved = read_object(path)
    if approved.get("approved") is not True:
        return False
    native = read_object(folder / "profile-two/profile_receipt.json")
    if (
        approved.get("reviewer") != "ROOT"
        or native.get("status") != "profile_complete"
        or native.get("updates") != 12
        or native.get("errors") != []
        or native.get("scientific_checkpoint_written") is not False
        or native["identity"]["world_size"] != 2
        or any(approved.get(key) != item for key, item in native.items() if key != "approved")
    ):
        raise ValueError("production profile approval changed native evidence or lacks ROOT review")
    return True


def directory_bytes(root: Path) -> int:
    """Count new physical outputs without following restored data/cache links."""
    total = 0
    for folder, directories, files in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if not (Path(folder) / name).is_symlink()]
        for name in files:
            path = Path(folder) / name
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                continue
    return total


def process_family_records(identities: dict[int, int]) -> dict[int, dict[str, object]]:
    """Track exact birth-bound descendants across session changes and orphan reparenting."""
    processes: dict[int, dict[str, object]] = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = path.read_text()
            fields = raw[raw.rfind(")") + 2 :].split()
            pid = int(path.parent.name)
            processes[pid] = {
                "pid": pid,
                "parent_pid": int(fields[1]),
                "pgid": int(fields[2]),
                "birth": fields[19],
                "rss_pages": int(fields[21]),
                "state": fields[0],
            }
        except (OSError, ValueError, IndexError):
            continue
    tracked = {
        pid
        for pid, birth in identities.items()
        if pid in processes and int(processes[pid]["birth"]) == birth
    }
    while True:
        found = {
            pid
            for pid, record in processes.items()
            if record["parent_pid"] in tracked and pid not in identities
        }
        if not found:
            break
        for pid in found:
            identities[pid] = int(processes[pid]["birth"])
        tracked.update(found)
    return {pid: processes[pid] for pid in tracked}


def process_family_rss(identities: dict[int, int]) -> int:
    """Sum the complete tracked family, including torchrun's separate rank sessions."""
    return sum(
        record["rss_pages"]
        for record in process_family_records(identities).values()
        if record["state"] != "Z"
    ) * os.sysconf("SC_PAGE_SIZE")
