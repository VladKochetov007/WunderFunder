"""Mirror admitted native fits into the dashboard tree and export real evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import stat
import sys
import tempfile
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.multistage_plot import export_stage_trial_plots  # noqa: E402
from wnn.multistage_search import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
    file_lock,
    file_sha256,
    read_json,
)

LOG = logging.getLogger("wnn.multistage_monitor")
LOG_FILES = ("monitor_events.jsonl", "monitor_evaluations.jsonl")
METADATA_FILE = "run_metadata.json"
STATE_FILE = ".multistage_monitor_state.json"
MAX_METADATA_BYTES = 16 * 1_048_576
MAX_TRIAL_REQUEST_BYTES = 1_048_576
MAX_LOG_LINE_BYTES = 131_072
COPY_BLOCK_BYTES = 1 << 20


@dataclass(frozen=True)
class FitRun:
    """A registered fitting run whose metadata binds an inner control cohort."""

    source_dir: Path
    run_id: str
    stage: int
    family: str
    config_sha256: str
    split_sha256: str
    primary_split_sha256: str
    source_sha256: str
    metadata_sha256: str
    stage_evidence: str
    metadata: dict[str, object]
    metadata_bytes: bytes


def _is_sha256(value: object) -> bool:
    """Return whether a value is a lowercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _as_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _lstat_regular(path: Path) -> os.stat_result:
    """Return metadata for a regular file without following a final symlink."""
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"refusing non-regular source path: {path}")
    return info


def _open_regular(path: Path) -> BinaryIO:
    """Open a regular file without following its final path component."""
    before = _lstat_regular(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ValueError(f"source file changed while opening: {path}")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _read_trial_request(run_dir: Path) -> tuple[dict[str, object], Path] | None:
    """Read a colocated or direct-parent request without following request symlinks."""
    matches: list[tuple[dict[str, object], Path]] = []
    for request_dir in (run_dir, run_dir.parent):
        request_path = request_dir / "trial_request.json"
        try:
            raw = _read_bounded_regular(request_path, MAX_TRIAL_REQUEST_BYTES)
        except FileNotFoundError:
            continue
        request = _as_mapping(json.loads(raw), "trial request")
        if request.get("schema") != "wnn.multistage-trial-plan.v1":
            raise ValueError("run has an unsupported adjacent trial request")
        matches.append((request, request_dir))
    if not matches:
        return None
    if len(matches) == 2 and matches[0][0] != matches[1][0]:
        raise ValueError("run has conflicting colocated and parent trial requests")
    return matches[0]


def _bound_request_stage(
    run_dir: Path,
    request_dir: Path,
    request: dict[str, object],
    config: dict[str, object],
    family: str,
    data: dict[str, object],
    source_info: dict[str, object],
) -> int:
    """Validate the adjacent owned-campaign request against immutable run identity."""
    stage = request.get("stage")
    if not isinstance(stage, int) or isinstance(stage, bool) or stage not in {1, 2, 3}:
        raise ValueError("trial request stage must be one of 1, 2, or 3")
    stage_directory = f"stage{stage}"
    if stage_directory not in {parent.name for parent in request_dir.parents}:
        raise ValueError("trial request is outside its declared owned-campaign stage")
    if request.get("family") != family:
        raise ValueError("trial request family differs from run metadata")
    if (
        request.get("primary_split_sha256") != data.get("primary_split_sha256")
        or request.get("auxiliary_split_sha256") != data.get("split_sha256")
        or request.get("source_sha256") != source_info.get("sha256")
    ):
        raise ValueError("trial request data identity differs from run metadata")
    if request.get("objective_scope") != "inner_train_proxy":
        raise ValueError("trial request does not bind the inner train proxy scope")
    if request.get("primary_valid_used") is not False:
        raise ValueError("trial request does not explicitly exclude primary validation")
    config_path_value = request.get("config_path")
    config_digest = request.get("config_sha256")
    if not isinstance(config_path_value, str) or not _is_sha256(config_digest):
        raise ValueError("trial request config identity is incomplete")
    config_path = Path(config_path_value)
    try:
        config_path = config_path.resolve(strict=True)
        request_parent = request_dir.resolve(strict=True)
        if config_path.parent != request_parent:
            raise ValueError("trial request config is outside its owned trial directory")
        config_bytes = _read_bounded_regular(config_path, MAX_TRIAL_REQUEST_BYTES)
    except OSError as error:
        raise ValueError("trial request config is unavailable") from error
    if hashlib.sha256(config_bytes).hexdigest() != config_digest:
        raise ValueError("trial request config bytes differ from their bound digest")
    request_config = _as_mapping(json.loads(config_bytes), "trial request config")
    if canonical_json_sha256(request_config) != canonical_json_sha256(config):
        raise ValueError("trial request config differs from run metadata")
    return stage


def _read_bounded_regular(path: Path, maximum_bytes: int) -> bytes:
    """Read a regular file with a strict size bound and no final symlink following."""
    with _open_regular(path) as source:
        if os.fstat(source.fileno()).st_size > maximum_bytes:
            raise ValueError(f"file exceeds its {maximum_bytes}-byte bound: {path}")
        payload = source.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError(f"file exceeds its {maximum_bytes}-byte bound: {path}")
    return payload


def _read_metadata(
    run_dir: Path, registered_stage: int | None = None
) -> tuple[dict[str, object], bytes]:
    """Read and validate native run metadata needed to label one mirror."""
    with _open_regular(run_dir / METADATA_FILE) as source:
        if os.fstat(source.fileno()).st_size > MAX_METADATA_BYTES:
            raise ValueError(f"run metadata exceeds {MAX_METADATA_BYTES} bytes")
        raw = source.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError(f"run metadata exceeds {MAX_METADATA_BYTES} bytes")
    value = json.loads(raw)
    metadata = _as_mapping(value, "run metadata")
    config = _as_mapping(metadata.get("config"), "run config")
    provenance = _as_mapping(metadata.get("provenance"), "run provenance")
    data = _as_mapping(provenance.get("data"), "run data provenance")
    source_info = _as_mapping(metadata.get("source"), "run source")
    if data.get("evaluation_role") != "control":
        raise ValueError("run is not an inner-control fit")
    if not _is_sha256(data.get("split_sha256")):
        raise ValueError("run lacks its source-bound auxiliary split digest")
    if not _is_sha256(data.get("primary_split_sha256")):
        raise ValueError("run lacks its source-bound primary split digest")
    if not _is_sha256(source_info.get("sha256")):
        raise ValueError("run lacks its source digest")
    model = _as_mapping(config.get("model"), "run model config")
    family = model.get("architecture")
    if not isinstance(family, str) or not family.isascii() or not family.isalnum():
        raise ValueError("run model architecture is not a safe family label")
    kd_weight = config.get("kd_weight", 0.0)
    has_kd = (
        isinstance(kd_weight, (int, float)) and not isinstance(kd_weight, bool) and kd_weight > 0
    )
    inferred_stage = (
        3
        if data.get("teacher") is not None or has_kd
        else 2
        if (data.get("initializer") is not None)
        else None
    )
    trial_request_binding = _read_trial_request(run_dir)
    if trial_request_binding is not None:
        trial_request, request_dir = trial_request_binding
        request_stage = _bound_request_stage(
            run_dir, request_dir, trial_request, config, family, data, source_info
        )
        if inferred_stage is not None and request_stage < inferred_stage:
            raise ValueError("trial request stage conflicts with checkpoint ancestry")
        if registered_stage is not None and registered_stage != request_stage:
            raise ValueError("registered stage conflicts with the bound trial request")
        stage = request_stage
        stage_evidence = "trial_request"
    elif registered_stage is not None:
        if registered_stage not in {1, 2, 3}:
            raise ValueError("registered stage must be one of 1, 2, or 3")
        if inferred_stage is not None and registered_stage < inferred_stage:
            raise ValueError("registered stage conflicts with checkpoint ancestry")
        stage = registered_stage
        stage_evidence = "registered_role"
    elif inferred_stage is not None:
        stage = inferred_stage
        stage_evidence = "checkpoint_ancestry"
    else:
        raise ValueError(
            "label-only fit needs an owned trial_request.stage or an explicit registered stage"
        )
    config_sha256 = canonical_json_sha256(config)
    metadata_sha256 = canonical_json_sha256(metadata)
    run_id = f"stage{stage}-{family}-{config_sha256[:10]}-{metadata_sha256[:12]}"
    if len(run_id) > 80:
        raise ValueError("generated dashboard run ID exceeds the reader limit")
    details = {
        "run_id": run_id,
        "stage": stage,
        "family": family,
        "config_sha256": config_sha256,
        "split_sha256": data["split_sha256"],
        "primary_split_sha256": data["primary_split_sha256"],
        "source_sha256": source_info["sha256"],
        "metadata_sha256": metadata_sha256,
        "stage_evidence": stage_evidence,
    }
    return details, raw


def _source_directory(path: Path) -> Path:
    """Resolve one source directory after rejecting a symlink directory."""
    requested = path.expanduser().absolute()
    info = os.lstat(requested)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"source runs root is not a real directory: {requested}")
    return requested.resolve(strict=True)


def _candidate_directories(roots: list[Path], registered: list[Path]) -> list[Path]:
    """Find direct run directories below roots and include explicitly registered paths."""
    candidates: set[Path] = set()
    for raw_root in roots:
        try:
            root = _source_directory(raw_root)
        except FileNotFoundError:
            continue
        with os.scandir(root) as children:
            for child in children:
                try:
                    info = child.stat(follow_symlinks=False)
                except OSError:
                    continue
                if child.is_symlink() or not stat.S_ISDIR(info.st_mode):
                    continue
                run_dir = root / child.name
                try:
                    _lstat_regular(run_dir / METADATA_FILE)
                except (OSError, ValueError):
                    continue
                candidates.add(run_dir.resolve(strict=True))
    for raw_path in registered:
        path = raw_path.expanduser().absolute()
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"registered fit is not a real directory: {path}")
        _lstat_regular(path / METADATA_FILE)
        candidates.add(path.resolve(strict=True))
    return sorted(candidates)


def _parse_registered_stage(value: str) -> tuple[Path, int]:
    """Parse one explicit ``PATH=STAGE`` role binding from the CLI."""
    path_text, separator, stage_text = value.rpartition("=")
    if not separator or not path_text or stage_text not in {"1", "2", "3"}:
        raise ValueError("registered stage must have the form PATH=1|2|3")
    return Path(path_text), int(stage_text)


def _load_fit_run(source_dir: Path, registered_stage: int | None = None) -> FitRun:
    """Bind run labels and split/config/source identities from native metadata."""
    metadata, raw = _read_metadata(source_dir, registered_stage)
    native_metadata = _as_mapping(json.loads(raw), "run metadata")
    stage = metadata.get("stage")
    if not isinstance(stage, int) or isinstance(stage, bool):
        raise ValueError("derived stage label is invalid")
    run_id = metadata.get("run_id")
    family = metadata.get("family")
    config_sha256 = metadata.get("config_sha256")
    split_sha256 = metadata.get("split_sha256")
    primary_split_sha256 = metadata.get("primary_split_sha256")
    source_sha256 = metadata.get("source_sha256")
    metadata_sha256 = metadata.get("metadata_sha256")
    stage_evidence = metadata.get("stage_evidence")
    if (
        not isinstance(run_id, str)
        or not isinstance(family, str)
        or not isinstance(config_sha256, str)
        or not isinstance(split_sha256, str)
        or not isinstance(primary_split_sha256, str)
        or not isinstance(source_sha256, str)
        or not isinstance(metadata_sha256, str)
        or not isinstance(stage_evidence, str)
    ):
        raise ValueError("derived fit metadata is incomplete")
    return FitRun(
        source_dir=source_dir,
        run_id=run_id,
        stage=stage,
        family=family,
        config_sha256=config_sha256,
        split_sha256=split_sha256,
        primary_split_sha256=primary_split_sha256,
        source_sha256=source_sha256,
        metadata_sha256=metadata_sha256,
        stage_evidence=stage_evidence,
        metadata=native_metadata,
        metadata_bytes=raw,
    )


def _stage_preference(run: FitRun) -> tuple[int, int]:
    """Prefer bound campaign roles, then explicit registrations, then ancestry."""
    evidence_rank = {
        "checkpoint_ancestry": 0,
        "registered_role": 1,
        "trial_request": 2,
    }.get(run.stage_evidence, -1)
    return evidence_rank, run.stage


def _ensure_directory(path: Path) -> None:
    """Create one mirror directory and reject symlink or non-directory collisions."""
    if path.exists() or path.is_symlink():
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"dashboard mirror path is unsafe: {path}")
        return
    path.mkdir(parents=True, exist_ok=False)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_copy_bytes(path: Path, payload: bytes) -> None:
    """Replace one mirror file durably using a same-directory temporary file."""
    if path.is_symlink():
        raise ValueError(f"refusing to replace a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _marker(file: BinaryIO, offset: int) -> bytes:
    """Read the short prefix marker that proves an append-only file still matches."""
    if offset <= 0:
        return b""
    file.seek(max(0, offset - 64))
    return file.read(min(64, offset))


def _open_mirror_log(path: Path) -> BinaryIO:
    """Open or create one mirror journal without following a final symlink."""
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"dashboard journal is not a regular file: {path}")
        return os.fdopen(descriptor, "r+b")
    except BaseException:
        os.close(descriptor)
        raise


def _last_complete_offset(source: BinaryIO, size: int) -> int:
    """Return the byte after the latest newline in a potentially growing JSONL file."""
    cursor = size
    while cursor:
        start = max(0, cursor - COPY_BLOCK_BYTES)
        source.seek(start)
        block = source.read(cursor - start)
        newline = block.rfind(b"\n")
        if newline >= 0:
            return start + newline + 1
        cursor = start
    return 0


def _copy_range(source: BinaryIO, destination: BinaryIO, start: int, end: int) -> None:
    """Copy a fixed byte range while the caller holds both files open."""
    source.seek(start)
    destination.seek(start)
    remaining = end - start
    while remaining:
        chunk = source.read(min(COPY_BLOCK_BYTES, remaining))
        if not chunk:
            raise OSError("source JSONL changed during a bounded copy")
        destination.write(chunk)
        remaining -= len(chunk)


def _reset_log(path: Path, source: BinaryIO, complete_end: int) -> bytes:
    """Atomically replace a mirror journal with the complete source prefix."""
    if path.is_symlink():
        raise ValueError(f"refusing to replace a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    marker = b""
    try:
        with os.fdopen(descriptor, "w+b") as output:
            _copy_range(source, output, 0, complete_end)
            output.flush()
            os.fsync(output.fileno())
            marker = _marker(output, complete_end)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return marker


def _sync_log(source_path: Path, mirror_path: Path, previous: object) -> dict[str, object] | None:
    """Append complete source lines or reset the mirror after a source rotation."""
    try:
        source = _open_regular(source_path)
    except FileNotFoundError:
        return None
    with source:
        source_stat = os.fstat(source.fileno())
        source_size = source_stat.st_size
        complete_end = _last_complete_offset(source, source_size)
        old = previous if isinstance(previous, dict) else {}
        offset = old.get("offset")
        same_source = (
            old.get("device") == source_stat.st_dev and old.get("inode") == source_stat.st_ino
        )
        with _open_mirror_log(mirror_path) as mirror:
            mirror_size = os.fstat(mirror.fileno()).st_size
            expected_marker = old.get("marker")
            state_valid = (
                same_source
                and isinstance(offset, int)
                and not isinstance(offset, bool)
                and offset >= 0
                and mirror_size == offset
                and complete_end >= offset
                and isinstance(expected_marker, str)
                and _marker(source, offset).hex() == expected_marker
                and _marker(mirror, offset).hex() == expected_marker
            )
            if not state_valid:
                marker = _reset_log(mirror_path, source, complete_end)
                return {
                    "device": source_stat.st_dev,
                    "inode": source_stat.st_ino,
                    "offset": complete_end,
                    "marker": marker.hex(),
                }
            if isinstance(offset, int) and not isinstance(offset, bool) and complete_end > offset:
                _copy_range(source, mirror, offset, complete_end)
                mirror.flush()
                os.fsync(mirror.fileno())
            marker = _marker(source, complete_end)
        return {
            "device": source_stat.st_dev,
            "inode": source_stat.st_ino,
            "offset": complete_end,
            "marker": marker.hex(),
        }


def _evaluation_identity(evaluation_path: Path, run: FitRun) -> tuple[str, int] | None:
    """Validate real control evaluations and return their bytes digest and count."""
    if not evaluation_path.is_file():
        return None
    digest = file_sha256(evaluation_path)
    count = 0
    identity_sha: str | None = None
    with _open_regular(evaluation_path) as stream:
        for line_number, line in enumerate(stream, start=1):
            if len(line) > MAX_LOG_LINE_BYTES:
                raise ValueError(f"evaluation journal line {line_number} exceeds its bound")
            if not line.endswith(b"\n"):
                raise ValueError("mirrored evaluation journal has an incomplete line")
            event = json.loads(line)
            if not isinstance(event, dict) or event.get("kind") != "evaluation":
                raise ValueError("evaluation journal contains a non-evaluation record")
            context = _as_mapping(event.get("context"), "evaluation context")
            control = _as_mapping(context.get("control"), "inner control context")
            if control.get("role") != "control" or control.get("scope") != "inner_train_proxy":
                raise ValueError("evaluation is not the source-bound inner train proxy")
            if control.get("split_sha256") != run.split_sha256:
                raise ValueError("evaluation split digest differs from native run metadata")
            value = context.get("run_identity_sha256")
            if not isinstance(value, str) or not _is_sha256(value):
                raise ValueError("evaluation lacks its source-qualified run identity")
            if identity_sha is not None and identity_sha != value:
                raise ValueError("one fit emitted conflicting source-qualified identities")
            identity_sha = value
            count += 1
    return (digest, count) if count else None


def _plot_request_path(mirror_dir: Path) -> Path:
    return mirror_dir / "multistage_plot_request.json"


def _plot_request(run: FitRun, profile_identity: str) -> dict[str, object]:
    """Create a plot request from immutable run metadata and emitted evaluations."""
    return {
        "schema": "wnn.multistage-monitor-plot-request.v1",
        "trial_number": None,
        "stage": run.stage,
        "family": run.family,
        "run_id": run.run_id,
        "config_sha256": run.config_sha256,
        "auxiliary_split_sha256": run.split_sha256,
        "primary_split_sha256": run.primary_split_sha256,
        "profile_identity_sha256": profile_identity,
        "source_sha256": run.source_sha256,
        "native_metadata_sha256": run.metadata_sha256,
        "stage_evidence": run.stage_evidence,
    }


def _write_plot_request(mirror_dir: Path, run: FitRun, profile_identity: str) -> Path:
    """Bind plot labels to the actual run metadata and evaluation identity."""
    request = _plot_request(run, profile_identity)
    path = _plot_request_path(mirror_dir)
    if path.exists():
        existing = read_json(path)
        if existing != request:
            raise ValueError("plot request identity changed for an immutable fit")
    else:
        atomic_write_json(path, request, replace=False)
    return path


def _promote_run_view(old: FitRun, new: FitRun, dashboard_root: Path) -> None:
    """Move one generated mirror when a stronger campaign role appears later."""
    old_dir = dashboard_root / old.run_id
    new_dir = dashboard_root / new.run_id
    if old_dir == new_dir:
        return
    if new_dir.exists() or new_dir.is_symlink():
        raise ValueError(f"cannot promote run view into an occupied path: {new_dir}")
    old_info = os.lstat(old_dir)
    if stat.S_ISLNK(old_info.st_mode) or not stat.S_ISDIR(old_info.st_mode):
        raise ValueError(f"existing run mirror is unsafe: {old_dir}")
    old_metadata = _as_mapping(
        json.loads(_read_bounded_regular(old_dir / METADATA_FILE, MAX_METADATA_BYTES)),
        "mirrored metadata",
    )
    if canonical_json_sha256(old_metadata) != old.metadata_sha256:
        raise ValueError("existing mirror metadata does not match its canonical run identity")
    state = _load_state(old_dir)
    os.replace(old_dir, new_dir)
    _fsync_directory(dashboard_root)
    request_path = _plot_request_path(new_dir)
    evaluation_path = new_dir / "monitor_evaluations.jsonl"
    if evaluation_path.is_file():
        identity = _evaluation_profile_identity(evaluation_path)
        atomic_write_json(request_path, _plot_request(new, identity))
        state.pop("last_plot_evaluation_sha256", None)
        state.pop("last_plot_evaluation_count", None)
        atomic_write_json(new_dir / STATE_FILE, state)


def _load_state(mirror_dir: Path) -> dict[str, object]:
    """Read the monitor's durable append offsets, failing closed on unsafe state."""
    path = mirror_dir / STATE_FILE
    if not path.exists():
        return {"schema": "wnn.multistage-monitor-state.v1", "files": {}}
    if path.is_symlink() or not stat.S_ISREG(os.lstat(path).st_mode):
        raise ValueError(f"mirror state is unsafe: {path}")
    state = read_json(path)
    if state.get("schema") != "wnn.multistage-monitor-state.v1":
        raise ValueError("unsupported monitor mirror state")
    return state


def _sync_run(run: FitRun, dashboard_root: Path) -> None:
    """Mirror one native fit, then export plots for each new complete evaluation."""
    mirror_dir = dashboard_root / run.run_id
    _ensure_directory(mirror_dir)
    metadata_path = mirror_dir / METADATA_FILE
    if metadata_path.is_symlink():
        raise ValueError(f"mirror metadata path is a symlink: {metadata_path}")
    same_metadata = (
        metadata_path.is_file()
        and file_sha256(metadata_path) == hashlib.sha256(run.metadata_bytes).hexdigest()
    )
    if not same_metadata:
        _atomic_copy_bytes(metadata_path, run.metadata_bytes)
    state = _load_state(mirror_dir)
    old_files = state.get("files", {})
    if not isinstance(old_files, dict):
        raise ValueError("monitor file-offset state is malformed")
    new_files: dict[str, object] = {}
    changed = False
    for name in LOG_FILES:
        source_path = run.source_dir / name
        mirror_path = mirror_dir / name
        old = old_files.get(name)
        current = _sync_log(source_path, mirror_path, old)
        if current is not None:
            new_files[name] = current
            if current != old:
                changed = True
    state["files"] = new_files
    evaluation = _evaluation_identity(mirror_dir / "monitor_evaluations.jsonl", run)
    old_eval = state.get("last_plot_evaluation_sha256")
    if evaluation is not None and evaluation[0] != old_eval:
        profile_identity = _evaluation_profile_identity(mirror_dir / "monitor_evaluations.jsonl")
        request_path = _write_plot_request(mirror_dir, run, profile_identity)
        result = export_stage_trial_plots(request_path, mirror_dir, run.stage)
        if result.get("status") not in {"exported", "unchanged"}:
            raise RuntimeError(f"plot export did not complete: {result}")
        source_path = mirror_dir / "plots" / "plot_source.json"
        plot_source = read_json(source_path)
        plot_source["run_id"] = run.run_id
        plot_source["native_source_sha256"] = run.source_sha256
        plot_source["native_config_sha256"] = run.config_sha256
        plot_source["native_metadata_sha256"] = run.metadata_sha256
        plot_source["stage_evidence"] = run.stage_evidence
        plot_source["auxiliary_split_sha256"] = run.split_sha256
        plot_source["primary_split_sha256"] = run.primary_split_sha256
        plot_source["run_identity_sha256"] = profile_identity
        atomic_write_json(source_path, plot_source)
        state["last_plot_evaluation_sha256"] = evaluation[0]
        state["last_plot_evaluation_count"] = evaluation[1]
        LOG.info(
            "exported actual evaluation plots run_id=%s stage=%s family=%s evaluation_count=%d status=%s",
            run.run_id,
            run.stage,
            run.family,
            evaluation[1],
            result["status"],
        )
        changed = True
    if changed or not (mirror_dir / STATE_FILE).exists():
        atomic_write_json(mirror_dir / STATE_FILE, state)


def _evaluation_profile_identity(evaluation_path: Path) -> str:
    """Read the identity digest recorded by the actual native evaluation producer."""
    identity: str | None = None
    with _open_regular(evaluation_path) as stream:
        for line in stream:
            event = json.loads(line)
            if not isinstance(event, dict):
                continue
            context = event.get("context")
            if not isinstance(context, dict):
                continue
            value = context.get("run_identity_sha256")
            if isinstance(value, str) and _is_sha256(value):
                if identity is not None and identity != value:
                    raise ValueError("evaluation journal changed run identity")
                identity = value
    if identity is None:
        raise ValueError("native evaluation journal has no run identity")
    return identity


def watch(args: argparse.Namespace) -> None:
    """Watch the bounded admitted-run set until its wall-clock deadline."""
    dashboard_root = args.dashboard_runs_root.expanduser().absolute()
    dashboard_root.mkdir(parents=True, exist_ok=True)
    if dashboard_root.is_symlink() or not dashboard_root.is_dir():
        raise ValueError("dashboard runs root must be a real directory")
    deadline = (
        args.deadline_epoch
        if args.deadline_epoch is not None
        else time.time() + args.duration_seconds
    )
    if not math.isfinite(deadline) or deadline <= time.time() or deadline > time.time() + 21_600:
        raise ValueError("monitor deadline must be in the next six hours")
    roots = list(args.runs_root)
    registered = list(args.registered_run)
    registered_stage_by_path: dict[Path, int] = {}
    for value in args.registered_stage:
        stage_path, stage = _parse_registered_stage(value)
        resolved_path = _source_directory(stage_path)
        previous = registered_stage_by_path.get(resolved_path)
        if previous is not None and previous != stage:
            raise ValueError(f"conflicting registered stage roles for {resolved_path}")
        registered_stage_by_path[resolved_path] = stage
        registered.append(resolved_path)
    if not roots and not registered:
        raise ValueError("provide at least one --runs-root or --registered-run")
    lock_path = dashboard_root / ".multistage_monitor.lock"
    lock_context: AbstractContextManager[None] = file_lock(lock_path, blocking=False)
    try:
        with lock_context:
            seen: dict[str, FitRun] = {}
            metadata_by_run_id: dict[str, str] = {}
            while time.time() < deadline:
                try:
                    directories = _candidate_directories(roots, registered)
                    current_runs: dict[str, FitRun] = {}
                    for path in directories:
                        try:
                            run = _load_fit_run(path, registered_stage_by_path.get(path))
                        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                            LOG.warning("skipping unqualified fit path=%s reason=%s", path, error)
                            continue
                        previous_candidate = current_runs.get(run.metadata_sha256)
                        if previous_candidate is not None:
                            if previous_candidate.stage != run.stage:
                                LOG.warning(
                                    "identical native metadata has conflicting stage labels metadata_sha256=%s stages=%d,%d",
                                    run.metadata_sha256,
                                    previous_candidate.stage,
                                    run.stage,
                                )
                            if _stage_preference(run) > _stage_preference(previous_candidate):
                                current_runs[run.metadata_sha256] = run
                            continue
                        current_runs[run.metadata_sha256] = run

                    for metadata_sha256, run in current_runs.items():
                        previous = seen.get(metadata_sha256)
                        if previous is not None:
                            if _stage_preference(run) > _stage_preference(previous):
                                if run.run_id != previous.run_id:
                                    try:
                                        _promote_run_view(previous, run, dashboard_root)
                                    except (OSError, ValueError) as error:
                                        LOG.exception(
                                            "could not promote stage-bound mirror metadata_sha256=%s reason=%s",
                                            metadata_sha256,
                                            error,
                                        )
                                        run = previous
                                if run is not previous:
                                    seen[metadata_sha256] = run
                                    metadata_by_run_id.pop(previous.run_id, None)
                                    metadata_by_run_id[run.run_id] = metadata_sha256
                            else:
                                run = previous
                        elif len(seen) >= args.max_runs:
                            LOG.warning("monitor cap reached; skipping run_id=%s", run.run_id)
                            continue
                        else:
                            seen[metadata_sha256] = run
                            prior_digest = metadata_by_run_id.get(run.run_id)
                            if prior_digest is not None and prior_digest != metadata_sha256:
                                LOG.error("dashboard run ID collision run_id=%s", run.run_id)
                                seen.pop(metadata_sha256, None)
                                continue
                            metadata_by_run_id[run.run_id] = metadata_sha256
                        try:
                            _sync_run(run, dashboard_root)
                        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
                            LOG.exception(
                                "mirror/export failed run_id=%s reason=%s", run.run_id, error
                            )
                    LOG.info(
                        "watch cycle complete tracked_runs=%d deadline_epoch=%.3f",
                        len(seen),
                        deadline,
                    )
                except (OSError, ValueError) as error:
                    LOG.exception("watch cycle failed reason=%s", error)
                time.sleep(min(args.poll_seconds, max(0.0, deadline - time.time())))
    except OSError as error:
        raise RuntimeError(f"another monitor may own {lock_path}") from error


def build_parser() -> argparse.ArgumentParser:
    """Build the no-GPU bounded monitor CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", action="append", type=Path, default=[])
    parser.add_argument("--registered-run", action="append", type=Path, default=[])
    parser.add_argument("--registered-stage", action="append", default=[])
    parser.add_argument("--dashboard-runs-root", required=True, type=Path)
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--duration-seconds", type=float, default=21_600)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-runs", type=int, default=48)
    return parser


def main() -> None:
    """Parse arguments and run the bounded mirror watcher."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args()
    if (
        not math.isfinite(args.duration_seconds)
        or args.duration_seconds <= 0
        or args.duration_seconds > 21_600
    ):
        raise SystemExit("--duration-seconds must be in (0, 21600]")
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0 or args.poll_seconds > 60:
        raise SystemExit("--poll-seconds must be in (0, 60]")
    if args.max_runs < 1 or args.max_runs > 64:
        raise SystemExit("--max-runs must be in [1, 64]")
    watch(args)


if __name__ == "__main__":
    main()
