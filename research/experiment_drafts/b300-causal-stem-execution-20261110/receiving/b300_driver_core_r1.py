"""Pure gates and Linux process identity helpers for the PR74 B300 receiver."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any  # JSON documents are boundary data with recipe-specific fields.

COMMIT = "8a8afec964985a6ba8b13d1c9af1ab00e14c6504"
SOURCE_SHA256 = "959fc8a335f97a83de31c12dfa6a24f5db4776b78b70bf45f7890cf09b96b5d6"
MANIFEST_SHA256 = "1cb6bf60c668f34ea3c49e6f4c3b0c774ea6cdc7e8f8c5f29e5d9f04322f1936"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
CELLS = (
    "plain_s203201",
    "k1_s203201",
    "k16_s203201",
    "plain_s203202",
    "k1_s203202",
    "k16_s203202",
)
PROFILE_WAVES = (
    ("plain_s203201", "k1_s203201", "k16_s203201", "plain_s203202"),
    ("k1_s203202", "k16_s203202"),
)
FIT_STREAMS = {
    "0,1": ("plain_s203201", "k1_s203202"),
    "2,3": ("k1_s203201", "k16_s203202"),
    "4,5": ("k16_s203201",),
    "6,7": ("plain_s203202",),
}
FIT_WAVES = tuple(
    tuple(stream[index] for stream in FIT_STREAMS.values() if len(stream) > index)
    for index in range(max(map(len, FIT_STREAMS.values())))
)
AUTHORITY_SCHEMA = "wnn.b300-pr74-fourpairs-authority.v1"
FIT_ADMISSION_SCHEMA = "wnn.b300-pr74-fourpairs-fit-admission.v1"
PROFILE_RECEIPT_SCHEMA = "wnn.sequence-profile.v1"
DRIVER_SCHEMA = "wnn.b300-pr74-fourpairs-driver-r1"
FIT_UPDATES = 3_280
UPDATES_PER_EPOCH = 820
TRAIN_MEMBERS, VALID_MEMBERS, SCORED_ROWS = 10_325, 1_873, 3_528_273
MAX_MANIFEST_BYTES = 8_000_000
MAX_JSON_BYTES = 32_000_000


@dataclass(frozen=True)
class ProcIdentity:
    """Minimal `/proc` identity needed to distinguish reused PIDs and owned groups."""

    pid: int
    state: str
    ppid: int
    pgid: int
    session: int
    start_ticks: int
    rss_bytes: int


def sha256_file(path: Path) -> str:
    """Hash a file in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Match the trainer's canonical JSON identity digest."""
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path: Path, limit: int = MAX_JSON_BYTES) -> dict[str, Any]:
    """Load a bounded regular JSON object without following a symbolic link."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"missing, linked or oversized JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON value is not an object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any], *, replace: bool = False) -> None:
    """Durably publish one JSON receipt, preserving no-retry files by default."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and (path.exists() or path.is_symlink()):
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def verify_source_capsule(
    root: Path, manifest_path: Path, expected_sha: str, commit: str
) -> tuple[str, dict[str, str]]:
    """Verify all archived files, reject extra importable Python, and derive trainer identity."""
    raw = manifest_path.read_bytes()
    if len(raw) > MAX_MANIFEST_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("source capsule manifest SHA or size is invalid")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "commit", "files"}:
        raise ValueError("source capsule manifest schema differs")
    files = manifest["files"]
    root = root.resolve(strict=True)
    if (
        manifest.get("schema") != "wnn.git-source-capsule.v1"
        or manifest.get("commit") != commit
    ):
        raise ValueError("source capsule commit/schema differs from the explicit pins")
    if not isinstance(files, dict) or not files:
        raise ValueError("source capsule has no file inventory")
    for relative, digest in files.items():
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError("source capsule entry is malformed")
        posix = PurePosixPath(relative)
        path = root / relative
        if (
            posix.is_absolute()
            or ".." in posix.parts
            or "\\" in relative
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ValueError(f"source capsule path is unsafe or missing: {relative}")
        if not path.resolve().is_relative_to(root) or sha256_file(path) != digest:
            raise ValueError(f"source capsule content differs: {relative}")
    python_files = {
        path.relative_to(root).as_posix()
        for folder in (root / "src", root / "scripts")
        for path in folder.rglob("*.py")
    }
    expected_python = {
        name
        for name in files
        if name.endswith(".py") and name.startswith(("src/", "scripts/"))
    }
    if python_files != expected_python or "scripts/predict_run.py" not in files:
        raise ValueError("source capsule importable Python inventory differs")
    trainer_files = {
        name: digest
        for name, digest in files.items()
        if name.startswith("src/") and name.endswith(".py")
    }
    if "scripts/train_sequence.py" not in files:
        raise ValueError("source capsule omits canonical sequence trainer")
    trainer_files["scripts/train_sequence.py"] = files["scripts/train_sequence.py"]
    return canonical_sha256(trainer_files), files


def validate_authority(
    value: dict[str, Any], *, root: Path, manifest_sha: str, commit: str
) -> dict[str, dict[str, Any]]:
    """Validate the frozen authority's exact source, six cells, four pair streams, and resource contract."""
    source = value.get("source")
    configs = value.get("configs")
    budget = value.get("budget")
    profile_waves = value.get("profile_waves")
    streams = value.get("fit_pair_streams")
    if (
        not isinstance(source, dict)
        or not isinstance(configs, dict)
        or not isinstance(budget, dict)
    ):
        raise TypeError("authority lacks source/config/resource objects")
    if (
        value.get("schema") != AUTHORITY_SCHEMA
        or value.get("approved") is not True
        or Path(str(value.get("native_root"))).as_posix() != root.as_posix()
        or source.get("commit") != commit
        or source.get("manifest_sha256") != manifest_sha
        or source.get("source_sha256") != SOURCE_SHA256
        or value.get("split_sha256") != SPLIT_SHA256
        or value.get("world_size") != 2
        or value.get("new_fits") != 6
        or value.get("epochs") != 4
        or value.get("updates_per_epoch") != UPDATES_PER_EPOCH
        or value.get("updates_per_fit") != FIT_UPDATES
        or value.get("fit_members") != TRAIN_MEMBERS
        or value.get("evaluation_members") != VALID_MEMBERS
        or value.get("scored_rows") != SCORED_ROWS
        or value.get("precision") != "bf16_bf16x9"
        or tuple(tuple(wave) for wave in profile_waves) != PROFILE_WAVES
        or {key: tuple(cells) for key, cells in streams.items()} != FIT_STREAMS
    ):
        raise ValueError(
            "authority bytes do not match the approved PR74 four-pair allocation"
        )
    expected_budget = {
        "wall_seconds": 21_600,
        "fit_seconds": 18_000,
        "collection_seconds": 3_600,
        "profile_stage_timeout_seconds": 1_800,
        "output_bytes": 100_000_000_000,
        "minimum_free_bytes": 100_000_000_000,
        "owned_rss_bytes": 274_877_906_944,
        "threads_per_rank": 8,
    }
    if any(budget.get(key) != expected for key, expected in expected_budget.items()):
        raise ValueError(
            "authority resource/deadline caps differ from the approved contract"
        )
    profile_contract = value.get("profiles")
    if not isinstance(profile_contract, dict) or (
        profile_contract.get("count") != 12
        or profile_contract.get("updates_each") != 50
        or profile_contract.get("kinds") != ["production", "zero_warmup_peak"]
        or profile_contract.get("require_both_rank_full_state_reload") is not True
        or profile_contract.get("scientific_fits_admitted") != 0
    ):
        raise ValueError(
            "authority profile gates differ from the approved twelve-profile screen"
        )
    if (
        Path(str(value.get("clock_path"))).as_posix()
        != (root / "receipts" / "first-profile-r1.json").as_posix()
        or Path(str(value.get("fit_admission_path"))).as_posix()
        != (root / "control" / "ROOT-fit-admission-r1.json").as_posix()
        or value.get("fit_admission_schema") != FIT_ADMISSION_SCHEMA
    ):
        raise ValueError("authority clock or manual fit-admission path differs")
    if set(configs) != set(CELLS):
        raise ValueError("authority config roster differs from the six approved cells")
    uuids: list[str] = []
    validated: dict[str, dict[str, Any]] = {}
    for cell in CELLS:
        item = configs[cell]
        if not isinstance(item, dict):
            raise TypeError(f"authority config is malformed for {cell}")
        expected_seed = int(cell.rsplit("s", 1)[1])
        expected_kernel = (
            0 if cell.startswith("plain_") else 1 if cell.startswith("k1_") else 16
        )
        pair = item.get("pair")
        gpu_uuids = item.get("gpu_uuids")
        if (
            item.get("seed") != expected_seed
            or item.get("kernel") != expected_kernel
            or not isinstance(pair, list)
            or len(pair) != 2
            or not all(type(index) is int for index in pair)
            or not isinstance(gpu_uuids, list)
            or len(gpu_uuids) != 2
            or not all(isinstance(uuid, str) and uuid for uuid in gpu_uuids)
            or len(set(gpu_uuids)) != 2
        ):
            raise ValueError(f"cell identity/pair is malformed for {cell}")
        uuids.extend(gpu_uuids)
        validated[cell] = item
    if len(set(uuids)) != 8:
        raise ValueError("authority reuses a physical B300 UUID")
    return validated


def validate_config(
    path: Path,
    authority_config: dict[str, Any],
    cell: str,
    normalized_config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Verify raw config bytes, then bind the canonical source loader's normalized recipe."""
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != authority_config.get("raw_sha256")
    ):
        raise ValueError(f"production config bytes differ from authority for {cell}")
    raw_config = read_json(path, 64_000)
    config = normalized_config
    seed, kernel = authority_config["seed"], authority_config["kernel"]
    model = config.get("model")
    expected = {
        "epochs": 4,
        "batch_size": 128,
        "accumulate_chunks": 4,
        "chunk_rows": 256,
        "gradient_horizon_rows": 256,
        "optimizer": "muon",
        "lr": 0.0003,
        "weight_decay": 0.01,
        "warmup_fraction": 0.05,
        "schedule": "warmup_cosine_v2",
        "min_lr_ratio": 0.1,
        "precision_mode": "bf16_bf16x9",
        "required_world_size": 2,
        "required_device": "B300",
        "checkpoint_retention": "all_epochs",
        "checkpoint_endpoint_policy": "configured_final_epoch",
        "objective_mode": "standalone_labels_wmse_v2",
    }
    if (
        any(config.get(key) != value for key, value in expected.items())
        or config.get("seed") != seed
    ):
        raise ValueError(f"optimizer/schedule/config contract differs for {cell}")
    raw_model = raw_config.get("model")
    if (
        not isinstance(raw_model, dict)
        or raw_model.get("input_stem_kernel", 0) != kernel
        or raw_config.get("seed") != seed
    ):
        raise ValueError(
            f"raw model kernel or seed differs from the frozen cell for {cell}"
        )
    if (
        not isinstance(model, dict)
        or any(
            model.get(key) != value
            for key, value in {
                "architecture": "transformer",
                "hidden": 768,
                "layers": 24,
                "heads": 12,
                "context_rows": 1024,
                "hint_dim": 0,
            }.items()
        )
        or model.get("input_stem_kernel", 0) != kernel
    ):
        raise ValueError(f"model/context/input-stem differs for {cell}")
    if (
        config.get("training_mask") != "all_after_warmup"
        or config.get("normalization_scope") != "all_fit"
    ):
        raise ValueError(f"train-only fitting ancestry differs for {cell}")
    return config, canonical_sha256(config)


def source_identity_files(files: dict[str, str]) -> dict[str, str]:
    """Build the same trainer source-files map from the already verified capsule manifest."""
    selected = {
        name: digest
        for name, digest in files.items()
        if name.startswith("src/") and name.endswith(".py")
    }
    if "scripts/train_sequence.py" not in files:
        raise ValueError("source manifest lacks the exact public training entrypoint")
    selected["scripts/train_sequence.py"] = files["scripts/train_sequence.py"]
    return selected


def validate_profile_receipt(
    path: Path,
    *,
    config: dict[str, Any],
    config_sha: str,
    source_sha: str,
    source_files: dict[str, str],
    expected_split: str,
    expected_uuids: list[str],
) -> dict[str, Any]:
    """Require a successful 50-update profile and exact two-rank full-state reload evidence."""
    receipt = read_json(path)
    identity = receipt.get("identity")
    if not isinstance(identity, dict):
        raise TypeError("profile receipt lacks native identity")
    data = identity.get("data")
    if not isinstance(data, dict):
        raise TypeError("profile receipt lacks split/data provenance")
    fit, evaluation = data.get("fit_members"), data.get("evaluation_members")
    fit_ok = (
        isinstance(fit, list)
        and len(fit) == TRAIN_MEMBERS
        and all(
            isinstance(item, list) and len(item) == 2 and item[0] == "train"
            for item in fit
        )
        and len(
            {
                (item[0], item[1])
                for item in fit
                if isinstance(item, list) and len(item) == 2
            }
        )
        == TRAIN_MEMBERS
    )
    eval_ok = (
        isinstance(evaluation, list)
        and len(evaluation) == VALID_MEMBERS
        and all(
            isinstance(item, list) and len(item) == 2 and item[0] == "valid"
            for item in evaluation
        )
        and len(
            {
                (item[0], item[1])
                for item in evaluation
                if isinstance(item, list) and len(item) == 2
            }
        )
        == VALID_MEMBERS
    )
    reload = receipt.get("engineering_reload")
    ranks = reload.get("by_rank") if isinstance(reload, dict) else None
    runtime = identity.get("runtime")
    devices = runtime.get("devices") if isinstance(runtime, dict) else None
    precision = runtime.get("precision") if isinstance(runtime, dict) else None
    memory = receipt.get("memory_by_rank")
    if isinstance(memory, dict):
        memory = [memory]
    if (
        receipt.get("receipt_schema") != PROFILE_RECEIPT_SCHEMA
        or receipt.get("status") != "profile_complete"
        or receipt.get("approved") is not False
        or type(receipt.get("updates")) is not int
        or receipt.get("updates") != 50
        or receipt.get("errors") != []
        or receipt.get("error_count") != 0
        or receipt.get("scientific_checkpoint_written") is not False
        or identity.get("source_sha256") != source_sha
        or identity.get("source_files") != source_files
        or identity.get("config_sha256") != config_sha
        or identity.get("world_size") != 2
        or data.get("split_sha256") != expected_split
        or not fit_ok
        or not eval_ok
        or data.get("normalization", {}).get("scope") != "all_fit"
        or data.get("normalization", {}).get("sequence_count") != TRAIN_MEMBERS
        or not isinstance(devices, list)
        or len(devices) != 2
        or [str(device.get("uuid", "")).removeprefix("GPU-") for device in devices]
        != expected_uuids
        or not isinstance(precision, dict)
        or precision.get("mode") != "bf16_bf16x9"
        or not isinstance(memory, list)
        or len(memory) != 2
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("peak_reserved_bytes"), int)
            or not isinstance(row.get("device_total_memory_bytes"), int)
            or row["peak_reserved_bytes"] >= 0.9 * row["device_total_memory_bytes"]
            for row in memory
        )
        or not isinstance(reload, dict)
        or reload.get("status") != "passed"
        or not isinstance(ranks, list)
        or [rank.get("rank") for rank in ranks] != [0, 1]
        or any(
            rank.get("model_optimizer_scheduler_rng_equal") is not True
            for rank in ranks
        )
        or not isinstance(config, dict)
    ):
        raise ValueError(
            "profile status/identity/full-state reload/resource receipt is incomplete"
        )
    return receipt


def parse_proc_stat(pid: int, raw: str, page_size: int) -> ProcIdentity:
    """Parse Linux proc stat safely when comm contains spaces or closing parentheses."""
    left = raw.find("(")
    right = raw.rfind(")")
    if left < 0 or right <= left or int(raw[:left].strip()) != pid:
        raise ValueError("malformed /proc/PID/stat identity")
    fields = raw[right + 1 :].split()
    if len(fields) < 22:
        raise ValueError("short /proc/PID/stat")
    return ProcIdentity(
        pid=pid,
        state=fields[0],
        ppid=int(fields[1]),
        pgid=int(fields[2]),
        session=int(fields[3]),
        start_ticks=int(fields[19]),
        rss_bytes=max(0, int(fields[21])) * page_size,
    )


def proc_entry(pid: int, proc_root: Path = Path("/proc")) -> ProcIdentity | None:
    """Read one current process identity; disappearing processes are normal during polling."""
    try:
        return parse_proc_stat(
            pid, (proc_root / str(pid) / "stat").read_text(), os.sysconf("SC_PAGE_SIZE")
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return None


def process_table(proc_root: Path = Path("/proc")) -> dict[int, ProcIdentity]:
    """Take a bounded identity/RSS snapshot for existing local processes."""
    result = {}
    for child in proc_root.iterdir():
        if (
            child.name.isdigit()
            and (entry := proc_entry(int(child.name), proc_root)) is not None
        ):
            result[entry.pid] = entry
    return result


def owned_group_members(
    table: dict[int, ProcIdentity], pgid: int, birth_ticks: int
) -> list[ProcIdentity]:
    """Return only processes in this receiver-created session after checking its leader birth."""
    leader = table.get(pgid)
    if leader is not None and leader.start_ticks != birth_ticks:
        return []
    return [
        entry
        for entry in table.values()
        if entry.pgid == pgid and entry.session == pgid
    ]


def directory_bytes(root: Path) -> int:
    """Count regular output files without following symlinked data/cache directories."""
    total = 0
    for folder, directories, files in os.walk(root, followlinks=False):
        directories[:] = [
            name for name in directories if not (Path(folder) / name).is_symlink()
        ]
        for name in files:
            path = Path(folder) / name
            if path.is_symlink():
                continue
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue
    return total


def validate_clock(value: dict[str, Any]) -> tuple[float, float, float]:
    """Accept only ROOT's original first-profile+5h/+6h deadline receipt."""
    first, fit, wave = (
        value.get("first_profile_epoch"),
        value.get("fit_deadline_epoch"),
        value.get("wave_deadline_epoch"),
    )
    if not all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        for item in (first, fit, wave)
    ):
        raise ValueError("first-profile clock fields are malformed")
    first, fit, wave = float(first), float(fit), float(wave)
    if abs(fit - first - 18_000) > 1 or abs(wave - first - 21_600) > 1:
        raise ValueError(
            "first-profile clock does not retain 5h fit plus 1h collection"
        )
    return first, fit, wave


def validate_fit_admission(
    admission: dict[str, Any],
    *,
    expected_sha: str,
    actual_sha: str,
    authority_sha: str,
    source_sha: str,
    source_root: Path,
    configs: dict[str, dict[str, Any]],
    root: Path,
) -> dict[str, dict[str, Any]]:
    """Check ROOT's post-profile six-fit release and every referenced receipt/config byte."""
    profiles = admission.get("profiles")
    if (
        actual_sha != expected_sha
        or admission.get("schema") != FIT_ADMISSION_SCHEMA
        or admission.get("approved") is not True
        or admission.get("max_fresh_fits") != 6
        or admission.get("source_sha256") != source_sha
        or admission.get("source_root") != str(source_root)
        or admission.get("authority_sha256") != authority_sha
        or admission.get("split_sha256") != SPLIT_SHA256
        or not isinstance(profiles, dict)
        or set(profiles) != set(CELLS)
    ):
        raise ValueError(
            "ROOT fit admission is missing, late, unapproved or differs from source/authority"
        )
    approved: dict[str, dict[str, Any]] = {}
    for cell in CELLS:
        item = profiles[cell]
        authority = configs[cell]
        if not isinstance(item, dict):
            raise TypeError(f"fit admission entry malformed for {cell}")
        paths = {
            "production_receipt_path": root
            / "runs"
            / cell
            / "profile"
            / "profile_receipt.json",
            "peak_receipt_path": root
            / "runs"
            / cell
            / "profile-peak"
            / "profile_receipt.json",
            "approved_receipt_path": root / "runs" / cell / "approved_profile.json",
        }
        for key, path in paths.items():
            hash_key = key.removesuffix("_path") + "_sha256"
            if (
                item.get(key) != str(path)
                or path.is_symlink()
                or not path.is_file()
                or sha256_file(path) != item.get(hash_key)
            ):
                raise ValueError(
                    f"ROOT admission does not bind the {key} bytes for {cell}"
                )
        if (
            item.get("config_sha256") != authority["config_sha256"]
            or item.get("raw_config_sha256") != authority["raw_sha256"]
            or item.get("pair") != authority["pair"]
            or item.get("gpu_uuids") != authority["gpu_uuids"]
        ):
            raise ValueError(
                f"ROOT fit admission changes config or device binding for {cell}"
            )
        approved[cell] = item
    return approved
