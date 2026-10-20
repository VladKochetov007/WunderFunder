from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import subprocess
import time
from pathlib import Path

PAYLOAD = {}
P = PAYLOAD
B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
D = B / "research_tmp/fg19_parent_fit_r2_verify_v2"
RUN = B / "data/artifacts/runs/fg19_parent_h256x2_f0_s1001_fit_r2_v1"
SOURCE = B / "source-fg19-cache-68ff6829"
SPLIT = B / "splits/fold0.json"
CACHE = B / "research_tmp/fg19_t3_fit_homogeneous_cache_v1"
UNIT = "computer-vision-homework-fg19-parent-fit-r2-verify-v2.service"
NATIVE_UNIT = "computer-vision-homework-fg19-parent-fit-r2-v1.service"
INVOCATION = "5186b3016ff54b1fa754c10a1f87b3e4"
COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
MANIFEST_SHA = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
TERMINAL_RAW_SHA = "a65af766fafaaa811d0991072035b51ec04dc01353da3059d7c594dffef239cb"
CACHE_METADATA_SHA = "03d9cf1fbd20b8bb3ad7b7d0c4513293fd1d5128e816256e4bf3cdfbfa4b43e3"
RNG_DIAGNOSTIC_SHA = "0bb2a87b9796872d5e26666ea7398bca69b7f0a0ce27bb0895714b5ed654598b"
RNG_DIAGNOSTIC_ACCEPTANCE_SHA = "655c5f12db5e166cfabf68c343aeb10d709f95db4abffeaf2925b413226837ec"
RNG_DIAGNOSTIC_STOP_SHA = "8e935356da08ac249e47389bb65bc5597134e6442ab9d914da1e7ec256b73c67"
RNG_DIAGNOSTIC_RAW_SHA = "2f6ced10d3812a3b3ae8354268a193323df6a22ea8523fdd2458060f4b8fcf69"
RNG_DIAGNOSTIC_SOURCE_SHA = "75f3b53762f5faf60f785126a8dec4b24e2bf52f620fce1397d7958eb25906b2"
RNG_DIAGNOSTIC_UNIT = "computer-vision-homework-fg19-parent-rng-diag-v2.service"
RNG_DIAGNOSTIC_INVOCATION = "b9dcbfc0b8054e51a49fc3792fd037a0"
MAX_SOURCE_BYTES = 128_000_000
MAX_JSON_BYTES = 16_000_000
MAX_MODEL_BYTES = 32_000_000
MAX_STATE_BYTES = 64_000_000
MAX_ARTIFACT_BYTES = 128_000_000
SUPPORT_MAX_BYTES = 1_500_000_000
PARENT_TARGET_BYTES = 17_000_000_000
PARENT_RESERVE_BYTES = 1_000_000_000


def require(condition: bool, message: str) -> None:
    """Stop before side effects when a pinned or fresh gate differs."""
    if not condition:
        raise RuntimeError(message)


def digest(raw: bytes) -> str:
    """Return the SHA-256 digest of a bounded byte string."""
    return hashlib.sha256(raw).hexdigest()


def utc_now() -> str:
    """Return an ISO UTC timestamp without requiring a specific Python minor."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def payload_bytes() -> dict[str, bytes]:
    """Decode the fixed package, checking its embedded byte counts and hashes."""
    result = {}
    for name, item in P.items():
        raw = base64.b64decode(item["base64"], validate=True)
        require(type(item.get("bytes")) is int, f"invalid payload byte count: {name}")
        require(len(raw) == item["bytes"], f"payload byte count differs: {name}")
        require(digest(raw) == item.get("sha256"), f"payload digest differs: {name}")
        result[name] = raw
    return result


def json_bytes(raw: bytes, name: str) -> dict[str, object]:
    """Decode one bounded UTF-8 JSON payload as an object."""
    value = json.loads(raw)
    require(isinstance(value, dict), f"JSON object expected: {name}")
    return value


def stable_file(path: Path, cap: int) -> tuple[bytes, dict[str, int]]:
    """Read and hash one small regular metadata file without following links."""
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= cap, f"unsafe/large file: {path}")
    require(not path.is_symlink(), f"symlink file: {path}")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(len(raw) == before.st_size, f"file changed or exceeded cap: {path}")
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while read: {path}",
    )
    return raw, {
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def file_hash(path: Path, cap: int) -> tuple[str, dict[str, int]]:
    """Hash a regular file in chunks while confirming its stat identity."""
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(), f"unsafe file: {path}")
    require(before.st_size <= cap, f"file exceeds cap: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            hasher.update(block)
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while hashing: {path}",
    )
    return hasher.hexdigest(), {
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def show(unit: str, fields: tuple[str, ...], timeout: int = 15) -> dict[str, str]:
    """Read explicit systemd properties for one exact unit or slice."""
    command = ["systemctl", "--user", "show", "--all", unit]
    for field in fields:
        command.extend(("-p", field))
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout, stderr=result.stderr
        )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def physical(path: Path) -> None:
    """Reject symlink redirection anywhere from the fixed project root downward."""
    require(path.exists() and not path.is_symlink(), f"path missing or linked: {path}")
    require(path.resolve(strict=True) == path, f"path is not physical: {path}")


def checked_source(manifest: dict[str, object], admission: dict[str, object]) -> dict[str, object]:
    """Verify the exact 167-file immutable source tree before staging support."""
    require(manifest.get("source_commit") == COMMIT, "source commit differs")
    rows = manifest.get("files")
    require(isinstance(rows, list) and len(rows) == 167, "source manifest count differs")
    source_map = {}
    total = 0
    for row in rows:
        require(isinstance(row, dict), "malformed source manifest row")
        rel, size, sha = row.get("path"), row.get("bytes"), row.get("sha256")
        require(isinstance(rel, str) and type(size) is int and size >= 0, "malformed source pin")
        require(isinstance(sha, str) and len(sha) == 64, "malformed source digest")
        relative = Path(rel)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source path")
        require(rel not in source_map, "duplicate source path")
        source_map[rel] = sha
        total += size
        require(total <= MAX_SOURCE_BYTES, "source byte budget exceeded")
        path = SOURCE / relative
        current = path
        while current != SOURCE:
            require(not current.is_symlink(), f"source path contains symlink: {current}")
            current = current.parent
        actual, stat_info = file_hash(path, size)
        require((actual, stat_info["bytes"]) == (sha, size), f"source pin differs: {rel}")
    require(len(source_map) == 167, "source manifest contains duplicates")
    inventory = set()
    for root, dirs, files in os.walk(SOURCE, followlinks=False):
        for name in dirs:
            require(not (Path(root) / name).is_symlink(), "source contains a linked directory")
        for name in files:
            path = Path(root) / name
            require(not path.is_symlink(), "source contains a linked file")
            inventory.add(str(path.relative_to(SOURCE)))
    require(inventory == set(source_map), "source inventory differs from manifest")
    require(
        (admission.get("source_files_verified"), admission.get("source_manifest_sha256"))
        == (167, MANIFEST_SHA),
        "source identity differs from parent admission",
    )
    return {
        "commit": COMMIT,
        "files": len(source_map),
        "bytes": total,
        "manifest_sha256": MANIFEST_SHA,
    }


def checked_data(admission: dict[str, object]) -> dict[str, object]:
    """Recheck the six registered data/cache paths by stat only."""
    data = admission.get("data_identity_stats_only")
    require(isinstance(data, dict) and len(data) == 6, "registered data identity is incomplete")
    observed = {}
    for rel, expected in data.items():
        require(isinstance(expected, dict), f"malformed registered data row: {rel}")
        path = B / "data" / rel
        require(str(path) == expected.get("path"), f"registered path differs: {rel}")
        lst = path.lstat()
        current = path.stat()
        actual = {
            "resolved": str(path.resolve(strict=True)),
            "device": current.st_dev,
            "bytes": current.st_size,
            "inode": current.st_ino,
            "mtime_ns": current.st_mtime_ns,
            "islink": stat.S_ISLNK(lst.st_mode),
        }
        require(
            actual == {key: expected[key] for key in actual}, f"registered data stat differs: {rel}"
        )
        observed[rel] = actual
    return observed


def checked_split(admission: dict[str, object]) -> dict[str, object]:
    """Hash the small campaign split manifest and bind it to the fit admission."""
    raw, info = stable_file(SPLIT, 2_000_000)
    sha = digest(raw)
    require(sha == SPLIT_SHA == admission.get("split_sha256"), "split manifest digest differs")
    json_bytes(raw, "split manifest")
    return {**info, "sha256": sha}


def checked_run_artifacts() -> dict[str, object]:
    """Record bounded terminal artifact stats without opening any checkpoint."""
    physical(RUN)
    expected = {
        "model.pt": MAX_MODEL_BYTES,
        "model_ep0.pt": MAX_MODEL_BYTES,
        "model_ep1.pt": MAX_MODEL_BYTES,
        "model_ep2.pt": MAX_MODEL_BYTES,
        "model_ep3.pt": MAX_MODEL_BYTES,
        "training_state.pt": MAX_STATE_BYTES,
        "provenance.json": MAX_JSON_BYTES,
        "history.json": MAX_JSON_BYTES,
    }
    rows = {}
    total = 0
    for name, cap in expected.items():
        path = RUN / name
        info = path.lstat()
        require(
            stat.S_ISREG(info.st_mode) and not path.is_symlink(), f"unsafe terminal file: {name}"
        )
        require(info.st_size <= cap, f"terminal file exceeds cap: {name}")
        rows[name] = {
            "bytes": info.st_size,
            "device": info.st_dev,
            "inode": info.st_ino,
            "mtime_ns": info.st_mtime_ns,
        }
        total += info.st_size
    require(total <= MAX_ARTIFACT_BYTES, "terminal artifact aggregate exceeds cap")
    return {"files": rows, "aggregate_bytes": total}


def checked_cache(
    admission: dict[str, object], split: dict[str, object], payload: dict[str, bytes]
) -> dict[str, object]:
    """Bind cache metadata and accepted outputs by JSON and stat only."""
    cache = admission.get("cache")
    require(isinstance(cache, dict), "admitted cache identity is missing")
    acceptance = json_bytes(payload["CACHE_ACCEPTANCE.json"], "cache acceptance")
    exact_stop = json_bytes(payload["CACHE_EXACT_STOP.json"], "cache exact stop")
    require(
        digest(payload["CACHE_ACCEPTANCE.json"]) == cache.get("acceptance_sha256"),
        "cache acceptance pin differs",
    )
    require(
        digest(payload["CACHE_EXACT_STOP.json"]) == cache.get("exact_stop_sha256"),
        "cache stop pin differs",
    )
    require(
        cache.get("metadata_sha256") == CACHE_METADATA_SHA, "admitted cache metadata pin differs"
    )
    require(
        (
            acceptance.get("status"),
            acceptance.get("metadata_sha256"),
            acceptance.get("source_commit"),
            acceptance.get("source_files_verified"),
        )
        == ("ACCEPTED", CACHE_METADATA_SHA, COMMIT, 167),
        "cache acceptance identity differs",
    )
    require(
        acceptance.get("unit") == cache.get("unit")
        and acceptance.get("invocation_id") == cache.get("invocation_id"),
        "cache invocation differs",
    )
    unit = cache.get("unit")
    before = exact_stop.get("before", {}).get(unit, {})
    after = exact_stop.get("after", {}).get(unit, {})
    stopped = exact_stop.get("stop", {}).get(unit, {})
    require(
        (
            before.get("InvocationID"),
            before.get("ExecMainStatus"),
            stopped.get("returncode"),
            after.get("LoadState"),
            after.get("ActiveState"),
            after.get("SubState"),
            after.get("InvocationID"),
            exact_stop.get("populated"),
        )
        == (cache.get("invocation_id"), "0", 0, "not-found", "inactive", "dead", "", {}),
        "cache exact-stop receipt differs",
    )
    physical(CACHE)
    metadata_raw, metadata_info = stable_file(CACHE / "metadata.json", MAX_JSON_BYTES)
    require(digest(metadata_raw) == CACHE_METADATA_SHA, "current cache metadata digest differs")
    metadata = json_bytes(metadata_raw, "cache metadata")
    require(metadata.get("split_manifest_sha256") == split.get("sha256"), "cache split differs")
    output = acceptance.get("output")
    stats = cache.get("array_stats_only")
    require(isinstance(output, dict) and isinstance(stats, dict), "cache output stats missing")
    arrays = {}
    for name, count, size, sha in (
        (
            "train",
            10535,
            1_685_600_128,
            "4abbb9be31ec1135a5e0ff5bb87c93a2db6e44943e95d23ff70835b8044c5194",
        ),
        (
            "valid",
            1405,
            224_800_128,
            "6963d3c565caf469b023a61ba6d2dd376d929b2bcf724af88492dc8adb76f591",
        ),
    ):
        filename = f"main_homogeneous_{name}.npy"
        path = CACHE / filename
        st = path.lstat()
        require(
            stat.S_ISREG(st.st_mode) and not path.is_symlink(), f"unsafe cache output: {filename}"
        )
        actual = {
            "bytes": st.st_size,
            "device": st.st_dev,
            "inode": st.st_ino,
            "mtime_ns": st.st_mtime_ns,
        }
        require(actual == stats.get(name), f"cache output stat differs: {name}")
        accepted = output.get(name)
        require(isinstance(accepted, dict), f"cache acceptance output missing: {name}")
        require(
            (
                st.st_size,
                accepted.get("sha256"),
                accepted.get("bytes"),
                accepted.get("shape"),
                accepted.get("dtype"),
            )
            == (size, sha, size, [count, 20000, 2], "float32"),
            f"cache accepted output differs: {name}",
        )
        arrays[name] = actual
    return {
        "metadata_sha256": CACHE_METADATA_SHA,
        "metadata_bytes": metadata_info["bytes"],
        "acceptance_sha256": cache["acceptance_sha256"],
        "exact_stop_sha256": cache["exact_stop_sha256"],
        "arrays_stat_only": arrays,
    }


def checked_rng_diagnostic(payload: dict[str, bytes]) -> dict[str, object]:
    """Bind the small diagnostic report, root acceptance, and exact-stop bytes."""
    report_raw = payload["RNG_DIAGNOSTIC.json"]
    acceptance_raw = payload["RNG_DIAGNOSTIC_ACCEPTANCE.json"]
    stop_raw = payload["RNG_DIAGNOSTIC_EXACT_STOP.json"]
    require(digest(report_raw) == RNG_DIAGNOSTIC_SHA, "RNG diagnostic report digest differs")
    require(
        digest(acceptance_raw) == RNG_DIAGNOSTIC_ACCEPTANCE_SHA,
        "RNG diagnostic acceptance digest differs",
    )
    require(digest(stop_raw) == RNG_DIAGNOSTIC_STOP_SHA, "RNG diagnostic stop digest differs")
    report = json_bytes(report_raw, "RNG diagnostic")
    acceptance = json_bytes(acceptance_raw, "RNG diagnostic acceptance")
    stop = json_bytes(stop_raw, "RNG diagnostic exact stop")
    require(
        (report.get("schema"), report.get("status"), report.get("torch_version"))
        == (
            "fg19.parent-fit-r2-rng-diagnostic.v2",
            "DIAGNOSTIC_ONLY_NO_ACCEPTANCE",
            "2.14.0+cu130",
        ),
        "RNG diagnostic schema/runtime differs",
    )
    require(
        (
            acceptance.get("unit"),
            acceptance.get("invocation_id"),
            acceptance.get("report_sha256"),
            acceptance.get("raw_sha256"),
            acceptance.get("source_sha256"),
            acceptance.get("quality_score"),
            acceptance.get("native_untouched"),
        )
        == (
            RNG_DIAGNOSTIC_UNIT,
            RNG_DIAGNOSTIC_INVOCATION,
            RNG_DIAGNOSTIC_SHA,
            RNG_DIAGNOSTIC_RAW_SHA,
            RNG_DIAGNOSTIC_SOURCE_SHA,
            None,
            True,
        ),
        "RNG diagnostic acceptance identity differs",
    )
    before = stop.get("before")
    after = stop.get("after")
    action = stop.get("stop")
    require(
        isinstance(before, dict) and isinstance(after, dict) and isinstance(action, dict),
        "RNG diagnostic stop receipt is incomplete",
    )
    require(
        (
            stop.get("unit"),
            stop.get("invocation_id"),
            stop.get("capture_acceptance_sha256"),
            stop.get("diagnostic_report_sha256"),
            stop.get("raw_capture_sha256"),
            before.get("InvocationID"),
            before.get("ActiveState"),
            before.get("SubState"),
            before.get("MainPID"),
            before.get("ExecMainCode"),
            before.get("ExecMainStatus"),
            action.get("returncode"),
            after.get("LoadState"),
            after.get("ActiveState"),
            after.get("SubState"),
            after.get("InvocationID"),
            stop.get("populated_before"),
            stop.get("populated_after"),
        )
        == (
            RNG_DIAGNOSTIC_UNIT,
            RNG_DIAGNOSTIC_INVOCATION,
            RNG_DIAGNOSTIC_ACCEPTANCE_SHA,
            RNG_DIAGNOSTIC_SHA,
            RNG_DIAGNOSTIC_RAW_SHA,
            RNG_DIAGNOSTIC_INVOCATION,
            "active",
            "exited",
            "0",
            "1",
            "0",
            0,
            "not-found",
            "inactive",
            "dead",
            "",
            {},
            {},
        ),
        "RNG diagnostic exact stop differs",
    )
    return {
        "report_sha256": RNG_DIAGNOSTIC_SHA,
        "capture_acceptance_sha256": RNG_DIAGNOSTIC_ACCEPTANCE_SHA,
        "exact_stop_sha256": RNG_DIAGNOSTIC_STOP_SHA,
        "torch_version": report.get("torch_version"),
        "cpu_rng": report.get("rng", {}).get("entries", {}).get("torch"),
        "cuda_rng": report.get("rng", {}).get("entries", {}).get("cuda"),
    }


def checked_terminal(
    payload: dict[str, bytes], manifest: dict[str, object], admission: dict[str, object]
) -> tuple[dict[str, str], dict[str, object]]:
    """Join the captured raw native bytes to typed evidence and current unit state."""
    evidence = json_bytes(payload["TERMINAL_EVIDENCE.json"], "terminal evidence")
    raw = payload["NATIVE_STDOUT.raw.json"]
    require(
        digest(raw) == TERMINAL_RAW_SHA == evidence.get("raw_sha256"),
        "native raw capture digest differs",
    )
    require(len(raw) == evidence.get("raw_bytes"), "native raw capture byte count differs")
    captured = json_bytes(raw, "native raw capture")
    require(
        captured.get("terminal_success") is True, "raw native capture did not report exit success"
    )
    raw_unit = captured.get("unit")
    require(isinstance(raw_unit, dict), "raw native unit record is missing")
    require(
        (
            raw_unit.get("InvocationID"),
            raw_unit.get("ActiveState"),
            raw_unit.get("SubState"),
            raw_unit.get("MainPID"),
            raw_unit.get("ExecMainCode"),
            raw_unit.get("ExecMainStatus"),
        )
        == (INVOCATION, "active", "exited", "0", "1", "0"),
        "raw native exit differs",
    )
    require(
        (
            evidence.get("schema"),
            evidence.get("unit"),
            evidence.get("invocation_id"),
            evidence.get("run_name"),
            evidence.get("active_state"),
            evidence.get("sub_state"),
        )
        == (
            "fg19.parent-fit-r2-native-terminal-evidence.v1",
            NATIVE_UNIT,
            INVOCATION,
            "fg19_parent_h256x2_f0_s1001_fit_r2_v1",
            "active",
            "exited",
        ),
        "typed terminal evidence identity differs",
    )
    for key, expected in (
        ("main_pid", 0),
        ("exec_main_code", 1),
        ("exec_main_status", 0),
        ("raw_bytes", len(raw)),
    ):
        require(
            type(evidence.get(key)) is int and evidence[key] == expected,
            f"typed terminal field differs: {key}",
        )
    raw_files = captured.get("files")
    require(
        isinstance(raw_files, dict)
        and set(raw_files) == {"ADMISSION.json", "LAUNCH.json", "PROVENANCE.json", "unit.log"},
        "raw capture metadata file set differs",
    )
    for name, record in raw_files.items():
        require(isinstance(record, dict), f"raw captured file identity malformed: {name}")
        content = base64.b64decode(record["base64"], validate=True)
        require(
            len(content) == record.get("bytes") and digest(content) == record.get("sha256"),
            f"raw captured file digest differs: {name}",
        )
        if name in {"ADMISSION.json", "LAUNCH.json"}:
            require(content == payload[name], f"captured {name} differs from pinned metadata")
    captured_sources = captured.get("source_files")
    expected_sources = {row["path"]: row["sha256"] for row in manifest["files"]}
    require(captured_sources == expected_sources, "native capture source map differs")
    keys = (
        "InvocationID",
        "ActiveState",
        "SubState",
        "MainPID",
        "ExecMainCode",
        "ExecMainStatus",
        "Result",
        "CPUUsageNSec",
        "ExecMainStartTimestamp",
        "ExecMainExitTimestamp",
        "ExecMainStartTimestampMonotonic",
        "ExecMainExitTimestampMonotonic",
        "MemoryPeak",
    )
    current = show(NATIVE_UNIT, keys)
    require(
        (
            current.get("InvocationID"),
            current.get("ActiveState"),
            current.get("SubState"),
            current.get("MainPID"),
            current.get("ExecMainCode"),
            current.get("ExecMainStatus"),
            current.get("Result"),
        )
        == (INVOCATION, "active", "exited", "0", "1", "0", "success"),
        "fresh native unit state differs",
    )
    require(
        current.get("CPUUsageNSec", "").isdigit() and int(current["CPUUsageNSec"]) > 0,
        "fresh native CPU evidence missing",
    )
    for key in keys[8:]:
        require(current.get(key), f"fresh native terminal property missing: {key}")
    return current, evidence


def checked_parent() -> dict[str, object]:
    """Check the two-core parent, all descendants, headroom and physical disk."""
    fields = (
        "ControlGroup",
        "MemoryCurrent",
        "CPUQuotaPerSecUSec",
        "MemoryHigh",
        "MemoryMax",
        "MemorySwapMax",
    )
    parent = show("wnn-campaign.slice", fields)
    control_group = parent.get("ControlGroup", "")
    require(
        control_group not in ("", "/") and "/wnn-campaign.slice" in control_group,
        "parent cgroup identity is unscoped",
    )
    require(
        (
            parent.get("CPUQuotaPerSecUSec"),
            parent.get("MemoryHigh"),
            parent.get("MemoryMax"),
            parent.get("MemorySwapMax"),
        )
        == ("2s", "18000000000", "20000000000", "0"),
        "parent resource envelope differs",
    )
    cgroup = Path("/sys/fs/cgroup") / control_group.lstrip("/")
    require(
        cgroup.is_dir() and cgroup.resolve(strict=True) == cgroup,
        "parent cgroup path is unavailable or redirected",
    )
    proc_files = [cgroup / "cgroup.procs", *cgroup.rglob("cgroup.procs")]
    require(len(proc_files) < 128, "parent cgroup inventory exceeds bound")
    populated = {}
    for path in proc_files:
        value = path.read_text(encoding="ascii")
        if value.strip():
            populated[str(path)] = value
    require(not populated, "parent campaign slice has populated descendants")
    current = int((cgroup / "memory.current").read_text(encoding="ascii").strip())
    require(
        current + SUPPORT_MAX_BYTES + PARENT_RESERVE_BYTES < PARENT_TARGET_BYTES,
        "parent memory headroom is insufficient",
    )
    events = {
        name: (cgroup / name).read_text(encoding="ascii")
        for name in ("memory.events", "memory.events.local")
    }
    free = os.statvfs(B).f_bavail * os.statvfs(B).f_frsize
    require(free >= 25 * 1024**3, "project filesystem free space is below the 25 GiB floor")
    result = subprocess.run(
        ["findmnt", "-T", str(D.parent), "-n", "-o", "FSTYPE"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    require(
        result.returncode == 0 and result.stdout.strip() not in ("tmpfs", "ramfs"),
        "research_tmp is not on physical storage",
    )
    return {
        "systemd": parent,
        "cgroup": str(cgroup),
        "populated": populated,
        "memory_current_bytes": current,
        "memory_events": events,
        "free_bytes": free,
        "research_tmp_fstype": result.stdout.strip(),
    }


def write_exclusive(path: Path, raw: bytes) -> None:
    """Create one file once, flush its bytes and sync the parent directory."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_json_exclusive(path: Path, value: dict[str, object]) -> None:
    """Persist a small JSON record without replacement."""
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    write_exclusive(path, raw)


def text_value(value: object) -> str:
    """Make subprocess timeout bytes safe for the durable JSON record."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def main() -> None:
    """Run fresh gates, stage only pinned inputs, then submit one support unit."""
    output = D
    require(
        B.is_dir() and not B.is_symlink() and B.resolve(strict=True) == B,
        "project root is unavailable",
    )
    physical(D.parent)
    require(
        D.parent == B / "research_tmp" and not os.path.lexists(D),
        "verifier output already exists or is redirected",
    )
    require(
        not os.path.lexists(B / "runtime/STOP_ADMISSION.json"),
        "global stop-admission marker exists",
    )
    require(
        show(UNIT, ("LoadState",)).get("LoadState") == "not-found", "verifier unit already exists"
    )
    physical(SOURCE)
    payload = payload_bytes()
    expected_names = {
        "parent_fit_terminal_check_v2.py",
        "teacher_s5_terminal_check_v1.py",
        "propensity_terminal_support.py",
        "SOURCE_MANIFEST.json",
        "ADMISSION.json",
        "LAUNCH.json",
        "LAUNCH_ACCEPTANCE.json",
        "TERMINAL_EVIDENCE.json",
        "NATIVE_STDOUT.raw.json",
        "CACHE_ACCEPTANCE.json",
        "CACHE_EXACT_STOP.json",
        "RNG_DIAGNOSTIC.json",
        "RNG_DIAGNOSTIC_ACCEPTANCE.json",
        "RNG_DIAGNOSTIC_EXACT_STOP.json",
    }
    require(set(payload) == expected_names, "fixed verifier payload file set differs")
    manifest = json_bytes(payload["SOURCE_MANIFEST.json"], "source manifest")
    admission = json_bytes(payload["ADMISSION.json"], "parent admission")
    launch = json_bytes(payload["LAUNCH.json"], "parent launch")
    acceptance = json_bytes(payload["LAUNCH_ACCEPTANCE.json"], "parent launch acceptance")
    require(
        (
            admission.get("run_name"),
            admission.get("unit"),
            admission.get("source_commit"),
            admission.get("source_files_verified"),
            admission.get("source_manifest_sha256"),
        )
        == ("fg19_parent_h256x2_f0_s1001_fit_r2_v1", NATIVE_UNIT, COMMIT, 167, MANIFEST_SHA),
        "parent admission identity differs",
    )
    require(
        launch.get("unit") == NATIVE_UNIT and acceptance.get("invocation_id") == INVOCATION,
        "parent launch identity differs",
    )
    require(
        acceptance.get("status") == "ACTUAL_LAUNCHED_FULL_FIT_PENDING",
        "parent fit is outside its pre-acceptance verification stage",
    )
    require(
        acceptance.get("admission_sha256") == digest(payload["ADMISSION.json"])
        and acceptance.get("launch_sha256") == digest(payload["LAUNCH.json"]),
        "parent launch acceptance links differ",
    )
    require(
        (manifest.get("source_commit"), len(manifest.get("files", []))) == (COMMIT, 167),
        "source manifest identity differs",
    )
    source = checked_source(manifest, admission)
    split = checked_split(admission)
    data = checked_data(admission)
    outputs = checked_run_artifacts()
    cache = checked_cache(admission, split, payload)
    rng_diagnostic = checked_rng_diagnostic(payload)
    native, terminal = checked_terminal(payload, manifest, admission)
    parent = checked_parent()

    output.mkdir(mode=0o700, exist_ok=False)
    staged = {}
    for name, raw in payload.items():
        write_exclusive(output / name, raw)
        staged[name] = {"bytes": len(raw), "sha256": digest(raw)}
    verifier_admission = {
        "schema": "fg19.parent-fit-r2-terminal-verifier-admission.v1",
        "created_utc": utc_now(),
        "run_name": "fg19_parent_h256x2_f0_s1001_fit_r2_v1",
        "unit": UNIT,
        "native_unit": NATIVE_UNIT,
        "invocation_id": INVOCATION,
        "terminal_evidence_sha256": digest(payload["TERMINAL_EVIDENCE.json"]),
        "terminal_raw_sha256": TERMINAL_RAW_SHA,
        "rng_diagnostic": rng_diagnostic,
        "source": source,
        "split": split,
        "registered_data_stat_only": data,
        "terminal_files_stat_only": outputs,
        "cache_stat_only": cache,
        "native_unit_snapshot": native,
        "typed_terminal_evidence": terminal,
        "parent": parent,
        "staged_files": staged,
        "resource_limits": {
            "cpu_quota": "50%",
            "memory_high_bytes": 1400000000,
            "memory_max_bytes": SUPPORT_MAX_BYTES,
            "swap_max_bytes": 0,
            "runtime_seconds": 600,
            "cuda_visible_devices": "",
            "blas_threads": 1,
        },
        "scope": "Exact checker, two pinned helpers, and bounded JSON metadata only; no prediction-array bytes copied or read by admission.",
    }
    write_json_exclusive(output / "VERIFIER_ADMISSION.json", verifier_admission)

    python = "/home/grinchenkoeg/WunderFunder/.venv/bin/python"
    command = [
        "systemd-run",
        "--user",
        "--unit=" + UNIT,
        "--property=Type=exec",
        "--property=RemainAfterExit=yes",
        "--property=Slice=wnn-campaign.slice",
        "--property=CPUQuota=50%",
        "--property=MemoryHigh=1400000000",
        "--property=MemoryMax=1500000000",
        "--property=MemorySwapMax=0",
        "--property=RuntimeMaxSec=600",
        "--property=TimeoutStopSec=45",
        "--property=StandardOutput=append:" + str(output / "unit.log"),
        "--property=StandardError=append:" + str(output / "unit.log"),
        "--working-directory=" + str(output),
    ]
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TMPDIR": str(B / "research_tmp"),
        "PYTHONPATH": str(output) + ":" + str(SOURCE / "src"),
        "WNN_PROJECT_ROOT": str(B / "data"),
    }
    command.extend("--setenv=" + key + "=" + value for key, value in environment.items())
    command.extend(
        [
            python,
            "-u",
            str(output / "parent_fit_terminal_check_v2.py"),
            "--run-dir",
            str(RUN),
            "--source-root",
            str(SOURCE),
            "--source-manifest",
            str(output / "SOURCE_MANIFEST.json"),
            "--split",
            str(SPLIT),
            "--admission",
            str(output / "ADMISSION.json"),
            "--launch",
            str(output / "LAUNCH.json"),
            "--launch-acceptance",
            str(output / "LAUNCH_ACCEPTANCE.json"),
            "--terminal-evidence",
            str(output / "TERMINAL_EVIDENCE.json"),
            "--cache-root",
            str(CACHE),
            "--cache-acceptance",
            str(output / "CACHE_ACCEPTANCE.json"),
            "--cache-exact-stop",
            str(output / "CACHE_EXACT_STOP.json"),
            "--rng-diagnostic",
            str(output / "RNG_DIAGNOSTIC.json"),
            "--rng-diagnostic-acceptance",
            str(output / "RNG_DIAGNOSTIC_ACCEPTANCE.json"),
            "--rng-diagnostic-exact-stop",
            str(output / "RNG_DIAGNOSTIC_EXACT_STOP.json"),
            "--output",
            str(output / "VERIFICATION.json"),
        ]
    )
    submission = {"outcome": "unknown", "command": command}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=25, check=False)
        submission.update(
            {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        )
        if result.returncode == 0:
            try:
                submission["post"] = show(
                    UNIT,
                    (
                        "InvocationID",
                        "ActiveState",
                        "SubState",
                        "MainPID",
                        "ControlGroup",
                        "ExecMainCode",
                        "ExecMainStatus",
                    ),
                    timeout=10,
                )
                submission["outcome"] = "submitted"
            except subprocess.TimeoutExpired as error:
                submission["post_show_timeout"] = {
                    "stdout": text_value(error.stdout),
                    "stderr": text_value(error.stderr),
                }
            except subprocess.CalledProcessError as error:
                submission["post_show_error"] = {
                    "returncode": error.returncode,
                    "stdout": text_value(error.stdout),
                    "stderr": text_value(error.stderr),
                    "command": error.cmd,
                }
            except (OSError, RuntimeError) as error:
                submission["post_show_error"] = {
                    "returncode": None,
                    "stdout": None,
                    "stderr": None,
                    "error": repr(error),
                }
    except subprocess.TimeoutExpired as error:
        submission.update(
            {
                "stdout": text_value(error.stdout),
                "stderr": text_value(error.stderr),
                "submit_timeout": True,
            }
        )
    except OSError as error:
        submission.update({"os_error": repr(error), "submit_failed_before_confirmation": True})
    launch_record = {
        "schema": "fg19.parent-fit-r2-terminal-verifier-launch.v1",
        "created_utc": utc_now(),
        "unit": UNIT,
        "invocation_id": submission.get("post", {}).get("InvocationID")
        if isinstance(submission.get("post"), dict)
        else None,
        "submission": submission,
        "unknown_outcome_policy": "Never retry. Inspect this exact unit and durable files before any further action.",
    }
    write_json_exclusive(output / "VERIFIER_LAUNCH.json", launch_record)
    print(
        json.dumps(
            {
                "unit": UNIT,
                "launch_outcome": submission["outcome"],
                "launch_record": str(output / "VERIFIER_LAUNCH.json"),
            },
            sort_keys=True,
        )
    )


main()
