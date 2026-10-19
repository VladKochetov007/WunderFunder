"""Admit one clean F0 strict-FP32 propensity fit on Beleriand."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

PAYLOAD_B64 = "__PAYLOAD_B64__"
PAYLOAD = json.loads(base64.b64decode(PAYLOAD_B64, validate=True))
B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
SOURCE, DATA = B / "source-fg19-propensity-235db75", B / "data"
SPLIT, REGISTRY = B / "splits/fold0.json", B / "support-source/terminal-v1/PORTABLE_SOURCES.json"
PACKET = B / "research/full_gru_tweaks_r1/profile_v1"
PROFILE = B / "research_tmp/fg19_propensity_profile_v1/PROFILE_RESULT.json"
PROFILE_REVIEW = PACKET / "PROFILE_REVIEW.json"
OUTPUT = B / "research_tmp/fg19_propensity_clean_f0_admission_v1"
RUN_NAME = "fg19_propensity_clean_f0_seed0_e4"
RUN_DIR = DATA / "artifacts/runs" / RUN_NAME
UNIT, GPU = (
    "computer-vision-homework-fg19-propensity-f0-e4-v1.service",
    "GPU-a50d0c3b-d28f-a2a7-8b43-3e02b543fc0c",
)
PYTHON = Path("/home/grinchenkoeg/WunderFunder/.venv/bin/python")
COMMIT = "235db75ab91e24c69f21e637a83a48512dc674fe"
REGISTRY_SHA = "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
FIT_ARGS = f"""--name {RUN_NAME} --hidden 64 --epochs 4 --lr 0.002 --seed 0
--checkpoint-policy fixed --checkpoint-epoch 3 --no-eval-during-fit
--split-manifest {SPLIT} --precision-profile strict_fp32_v1 --loader-workers 0""".split()
MIN_GPU_FREE, FIT_MEMORY = 4_500 * 1024 * 1024, 8_000_000_000


def _run(args: list[str], timeout: int = 20) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{args!r}: {result.returncode}: {result.stderr}")
    return result.stdout


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: dict[str, object]) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _text(value: str | bytes | None) -> str | None:
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _pins() -> dict[str, str]:
    expected = PAYLOAD["file_hashes"]
    if len(expected) != 157 or len(PAYLOAD["source_files"]) != 153:
        raise ValueError("source manifest must bind 153 canonical and 4 profile files")
    observed, total = {}, 0
    for relative, digest in expected.items():
        if relative.startswith("source/"):
            root, name = SOURCE, relative.removeprefix("source/")
        elif relative.startswith("packet/"):
            root, name = PACKET, relative.removeprefix("packet/")
        else:
            raise ValueError(f"pinned path outside source/packet roots: {relative}")
        path = root / name
        if ".." in Path(relative).parts or len(digest) != 64 or path.is_symlink():
            raise ValueError(f"unsafe or malformed pinned file: {relative}")
        if any(parent.is_symlink() for parent in path.parents if parent != root.parent):
            raise ValueError(f"symlink in pinned path: {relative}")
        if not path.resolve(strict=True).is_relative_to(root) or not stat.S_ISREG(
            path.stat().st_mode
        ):
            raise ValueError(f"pinned file is not a regular in-root file: {relative}")
        if path.stat().st_size > 8_000_000:
            raise ValueError(f"pinned code/config file exceeds bound: {relative}")
        total += path.stat().st_size
        observed[relative] = _sha(path)
        if observed[relative] != digest:
            raise ValueError(f"pinned source changed: {relative}")
    if total > 200_000_000:
        raise ValueError("pinned source tree exceeds bounded rehash allowance")
    return observed


def _profile(observed: dict[str, str]) -> dict[str, object]:
    if any(path.is_symlink() for path in (SPLIT, REGISTRY, PROFILE)):
        raise ValueError("split, registry, or profile receipt is a symlink")
    if (
        _sha(SPLIT) != SPLIT_SHA
        or _sha(REGISTRY) != REGISTRY_SHA
        or _sha(PROFILE) != PAYLOAD["profile_result_sha256"]
    ):
        raise ValueError("accepted split, registry, or actual-shape profile receipt changed")
    result = json.loads(PROFILE.read_bytes())
    if (
        result.get("status"),
        result.get("source_commit"),
        result.get("split_manifest_sha256"),
        result.get("configuration_sha256"),
    ) != ("completed_engineering_profile_only", COMMIT, SPLIT_SHA, PAYLOAD["config_sha256"]):
        raise ValueError("profile receipt does not match the declared F0 fit")
    run = result.get("profile", {})
    progress, runtime, records = (
        run.get("optimizer_progress", {}),
        result.get("runtime", {}),
        run.get("repeat_records", []),
    )
    checks = (
        run.get("precision_profile"),
        run.get("checkpoint_written"),
        run.get("quality_metric_computed"),
        progress.get("optimizer_completed_updates"),
        progress.get("nonfinite_updates"),
        runtime.get("cuda_visible_devices"),
        runtime.get("allocator_cap_bytes"),
        runtime.get("tf32"),
        runtime.get("autocast"),
    )
    wanted = ("strict_fp32_v1", False, False, 20, 0, GPU, 3_000_000_000, False, False)
    if (
        checks != wanted
        or len(records) != 2
        or any(row.get("full_gpu_input_shape") != [16, 20_000, 211] for row in records)
    ):
        raise ValueError("profile lacks required finite strict-FP32 evidence")
    hashes = result.get("propensity_fit_source_sha256", {})
    if len(hashes) != 25 or hashes != PAYLOAD["profile_source_hashes"]:
        raise ValueError("profile must bind all 25 expected source pins")
    if any(observed.get(f"source/{name}") != digest for name, digest in hashes.items()):
        raise ValueError("profile source pins differ from current source")
    review = json.loads(PROFILE_REVIEW.read_bytes())
    if (
        review.get("status"),
        review.get("profile_sha256"),
        review.get("profile_core_sha256"),
        review.get("prerequisite_sha256"),
    ) != (
        "PASS",
        observed["packet/propensity_profile.py"],
        observed["packet/propensity_profile_core.py"],
        PAYLOAD["config_sha256"],
    ):
        raise ValueError("profile review no longer binds the completed profile")
    return result


def _parent() -> tuple[dict[str, str], dict[str, str]]:
    keys = [
        "ControlGroup",
        "MemoryCurrent",
        "MemoryHigh",
        "MemoryMax",
        "MemorySwapMax",
        "CPUQuotaPerSecUSec",
    ]
    args = ["systemctl", "--user", "show", "--all", "wnn-campaign.slice"]
    args.extend(part for key in keys for part in ("-p", key))
    props = dict(line.split("=", 1) for line in _run(args).splitlines())
    limits = (
        props.get("CPUQuotaPerSecUSec"),
        props.get("MemoryHigh"),
        props.get("MemoryMax"),
        props.get("MemorySwapMax"),
    )
    if (
        not props.get("ControlGroup")
        or limits != ("2s", "18000000000", "20000000000", "0")
        or int(props["MemoryCurrent"]) + FIT_MEMORY > 17_000_000_000
    ):
        raise RuntimeError(f"campaign parent lacks CPU/memory headroom: {props}")
    group = Path("/sys/fs/cgroup") / props["ControlGroup"].lstrip("/")
    files = [group / "cgroup.procs", *group.rglob("cgroup.procs")]
    if len(files) > 128:
        raise RuntimeError("campaign cgroup descendant scan exceeded bound")
    populated = {str(path): value for path in files if (value := path.read_text().strip())}
    if populated:
        raise RuntimeError(f"campaign cgroup has active descendants: {populated}")
    return props, populated


def _gpu() -> tuple[str, list[dict[str, object]]]:
    base = ["nvidia-smi", "-i", GPU]
    line = _run(
        base + ["--query-gpu=uuid,name,memory.free,memory.total", "--format=csv,noheader"]
    ).strip()
    fields = [value.strip() for value in line.split(",")]
    if (
        len(fields) != 4
        or fields[0] != GPU
        or int(fields[2].split()[0]) * 1024 * 1024 < MIN_GPU_FREE
    ):
        raise RuntimeError(f"selected GPU identity/free-memory gate failed: {line}")
    output = _run(base + ["--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"])
    processes = []
    for row in output.splitlines():
        cols = [value.strip() for value in row.split(",")]
        if len(cols) != 3 or cols[1] != GPU:
            if row.strip().startswith("No running processes found"):
                continue
            raise RuntimeError("GPU process query returned an unexpected row")
        try:
            owner = (Path("/proc") / cols[0] / "cgroup").read_text().strip()
        except FileNotFoundError:
            processes.append(
                {"pid": cols[0], "observation": "exited_before_cgroup_read", "preserved": True}
            )
            continue
        if "wnn-campaign.slice" in owner:
            raise RuntimeError("selected GPU has a process in campaign cgroup")
        processes.append(
            {"pid": cols[0], "cgroup": owner, "used_memory": cols[2], "preserved": True}
        )
    return line, processes


def _data() -> dict[str, dict[str, object]]:
    registered, expected = (
        json.loads(REGISTRY.read_bytes()).get("sources", {}),
        PAYLOAD["profile_data_stats"],
    )
    if set(registered) != set(expected):
        raise ValueError("portable source registry membership changed")
    result = {}
    for relative, previous in expected.items():
        path = DATA / relative
        if (
            ".." in Path(relative).parts
            or path.is_symlink()
            or not path.resolve(strict=True).is_relative_to(DATA)
        ):
            raise ValueError(f"unsafe registered data path: {relative}")
        current, entry = path.stat(), registered[relative]
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_size != entry["original_identity"]["bytes"]
        ):
            raise ValueError(f"registered data file size/type changed: {relative}")
        identity = {
            "bytes": current.st_size,
            "mtime_ns": current.st_mtime_ns,
            "inode": current.st_ino,
        }
        if identity != {key: previous[key] for key in identity}:
            raise ValueError(f"registered data identity changed since profile: {relative}")
        result[relative] = {
            **identity,
            "registered_sha256": entry["sha256"],
            "payload_hashed": False,
        }
    return result
