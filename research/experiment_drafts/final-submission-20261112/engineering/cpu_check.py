"""Bounded, CPU-only integrity and engineering timing check for the final TL3 ZIP."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parent
PAYLOAD = ROOT / "payload"
OUTPUT = ROOT / "output"
MAX_ZIP_BYTES = 20_000_000
MAX_ARCHIVE_BYTES = 300_000_000
MEMORY_LIMIT_BYTES = 16 * (1 << 30)
WALL_LIMIT_SECONDS = 900
ROW_COUNT = 20_000
PREFIX_ROWS = 1_000
OFFICIAL_ROWS = 1_970 * 20_000
HISTORICAL_W18_SECONDS = 872.0
FINAL17_SHA256 = "207b9251b89ad27c814fe26110ac30fd2d1a209cbcabd1498ecb315392d783b8"
W18_EVIDENCE_SHA256 = "2dc00d8a16f316775ddb979382312ee8a1d60b3c3555128726b8c4a44786c29e"
ROLES = ("w18", "tiedlead_s5", "tiedlead_s6", "source4_lstm")
CHECKPOINT_MEMBERS = {
    "w18": "models/tl3final_w18final_distilled_s5/model_ep1.pt",
    "tiedlead_s5": "models/fttall_tlead_s5/model.pt",
    "tiedlead_s6": "models/fttall_tlead_s6/model.pt",
    "source4_lstm": "models/source4_lstm192x1/model.pt",
}
ISA_REQUIREMENTS = {
    "v2": {"cx16", "lahf_lm", "popcnt", "ssse3", "sse4_1", "sse4_2"},
    "avx2": {"avx", "avx2", "fma"},
    "v3": {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave"},
}


def _sha256_bytes(value: bytes) -> str:
    """Return SHA-256 for a bounded byte string."""
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash a regular file without loading it all into memory."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member(name: str) -> PurePosixPath:
    """Reject absolute, traversal, and platform-dependent archive paths."""
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe archive member: {name!r}")
    return path


def _set_worker_limits() -> None:
    """Pin this process to one CPU and bound virtual memory and CPU time."""
    if sys.version_info[:2] != (3, 11) or not sys.platform.startswith("linux"):
        raise RuntimeError("the isolated checker requires Linux Python 3.11")
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (WALL_LIMIT_SECONDS, WALL_LIMIT_SECONDS))
    if hasattr(os, "sched_getaffinity"):
        available = os.sched_getaffinity(0)
        if not available:
            raise RuntimeError("the container has no available CPU for the checker")
        os.sched_setaffinity(0, {min(available)})
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _safe_child_environment(isa: str | None = None) -> dict[str, str]:
    """Pass only runtime essentials to checker subprocesses, never inherited tokens."""
    allowed = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL")
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    if isa is not None:
        environment["WNN_NATIVE_LEVEL"] = isa
    return environment


def _regular_zip_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Validate ZIP paths, duplicate names, member types, and expanded size."""
    result: dict[str, zipfile.ZipInfo] = {}
    expanded = 0
    for info in archive.infolist():
        _safe_member(info.filename.rstrip("/"))
        if info.filename in result:
            raise ValueError(f"ZIP repeats member {info.filename!r}")
        mode = (info.external_attr >> 16) & 0o170000
        if mode not in (0, 0o100000, 0o040000):
            raise ValueError(f"ZIP contains a link or special file: {info.filename}")
        expanded += info.file_size
        if expanded > MAX_ARCHIVE_BYTES:
            raise ValueError("ZIP expanded size exceeds the 300 MB checker limit")
        result[info.filename] = info
    return result


def _extract_zip(archive_path: Path, destination: Path) -> dict[str, str]:
    """Extract a checked ZIP and return the SHA-256 of every regular member."""
    destination.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = _regular_zip_members(archive)
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC failed for {bad}")
        for name, info in members.items():
            relative = _safe_member(name.rstrip("/"))
            target = destination.joinpath(*relative.parts)
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.read(info)
            target.write_bytes(payload)
            hashes[name] = _sha256_bytes(payload)
    return hashes


def _extract_project_tar(archive_path: Path, destination: Path) -> Path:
    """Extract only regular source files from the small pinned project tarball."""
    destination.mkdir(parents=True, exist_ok=False)
    expanded = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            path = _safe_member(member.name.rstrip("/"))
            if member.isdir():
                destination.joinpath(*path.parts).mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(
                    f"project tar contains a link or special entry: {member.name}"
                )
            expanded += member.size
            if expanded > MAX_ARCHIVE_BYTES:
                raise ValueError(
                    "project tar expanded size exceeds the 300 MB checker limit"
                )
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"project tar member cannot be read: {member.name}")
            with stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output, length=1 << 20)
    candidates = [
        path.parent.parent
        for path in destination.rglob("check_final_tl3_synthetic.py")
        if path.parent.name == "scripts"
    ]
    matching = [path for path in candidates if (path / "src" / "wnn").is_dir()]
    if len(matching) != 1:
        raise ValueError(
            "project tar must contain exactly one pinned src/wnn and synthetic checker"
        )
    return matching[0]


def _read_json(path: Path) -> dict[str, Any]:
    """Read a bounded JSON object."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
        raise ValueError(f"expected bounded JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _verify_payload(payload: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    """Bind the exact candidate package and reference archives to frozen hashes."""
    candidate = payload / "candidate.zip"
    build_receipt_path = payload / "build_receipt.json"
    project_tar = payload / "project.tar.gz"
    final17 = payload / "final17.zip"
    w18_evidence = payload / "w18_evidence.zip"
    for required_path in (
        candidate,
        build_receipt_path,
        project_tar,
        final17,
        w18_evidence,
    ):
        if required_path.is_symlink() or not required_path.is_file():
            raise FileNotFoundError(
                f"required payload file is missing or unsafe: {required_path.name}"
            )
    expected_recipe_members = recipe.get("members")
    if not isinstance(expected_recipe_members, list) or [
        item.get("role") for item in expected_recipe_members
    ] != list(ROLES):
        raise ValueError("frozen recipe does not bind the four expected roles in order")
    expected_checkpoints = {
        item["role"]: item["checkpoint_sha256"] for item in expected_recipe_members
    }
    if any(len(value) != 64 for value in expected_checkpoints.values()):
        raise ValueError("frozen recipe checkpoint hash is malformed")
    if _sha256_file(final17) != FINAL17_SHA256:
        raise ValueError(
            "17-checkpoint reference archive differs from the frozen local archive"
        )
    if _sha256_file(w18_evidence) != W18_EVIDENCE_SHA256:
        raise ValueError(
            "W18 evidence archive differs from the authenticated local archive"
        )
    build_receipt = _read_json(build_receipt_path)
    artifact = build_receipt.get("artifact")
    if not isinstance(artifact, dict):
        raise TypeError("build receipt lacks an artifact record")
    zip_bytes = candidate.stat().st_size
    zip_sha = _sha256_file(candidate)
    if (
        zip_bytes > MAX_ZIP_BYTES
        or artifact.get("bytes") != zip_bytes
        or artifact.get("sha256") != zip_sha
    ):
        raise ValueError(
            "candidate ZIP bytes or hash differ from the exact build receipt"
        )
    input_artifacts = build_receipt.get("input_artifacts")
    if not isinstance(input_artifacts, dict) or (
        input_artifacts.get("checkpoint_archive_sha256") != FINAL17_SHA256
        or input_artifacts.get("w18_evidence_archive_sha256") != W18_EVIDENCE_SHA256
    ):
        raise ValueError(
            "build receipt does not bind both exact FP32 reference archives"
        )
    with zipfile.ZipFile(candidate) as archive:
        members = _regular_zip_members(archive)
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"candidate ZIP CRC failed for {bad}")
        expected_member_hashes = artifact.get("members")
        if not isinstance(expected_member_hashes, dict) or set(
            expected_member_hashes
        ) != set(members):
            raise ValueError("candidate ZIP member list differs from its build receipt")
        member_hashes = {
            name: _sha256_bytes(archive.read(info)) for name, info in members.items()
        }
        if member_hashes != expected_member_hashes:
            raise ValueError(
                "one or more exact candidate ZIP members differ from the build receipt"
            )
        manifest = json.loads(archive.read("members.json"))
    if (
        manifest.get("serving_policy") != "full_mix"
        or manifest.get("precision") != "q8vnni"
        or manifest.get("active_member_indices") != [0, 1, 2, 3]
        or manifest.get("independent_evaluation_sequences") != 0
        or manifest.get("independent_score_claim") != "forbidden"
        or manifest.get("candidate_scope") != "approved_full_corpus_final_model_set"
        or manifest.get("final_fit", {}).get("manifest_sha256")
        != recipe.get("fit_membership_sha256")
        or manifest.get("final_fit", {}).get("fit_row_groups", {}).get("train")
        != list(range(10_607))
        or manifest.get("final_fit", {}).get("fit_row_groups", {}).get("valid")
        != list(range(1_873))
    ):
        raise ValueError(
            "candidate manifest is not the declared all-labeled full-mix package"
        )
    expected_package_members = {
        "solution.py",
        "members.json",
        "precision.txt",
        *(f"weights_{role}.npz" for role in ROLES),
        *(f"{level}/wnn_native.abi3.so" for level in ("v2", "avx2", "v3")),
    }
    if set(members) != expected_package_members:
        raise ValueError(
            "candidate ZIP file inventory differs from the fixed full-mix layout"
        )
    manifest_members = manifest.get("members")
    if not isinstance(manifest_members, list) or [
        item.get("role") for item in manifest_members
    ] != list(ROLES):
        raise ValueError("candidate ZIP has unexpected model member order")
    for item in manifest_members:
        role = item["role"]
        if item.get("checkpoint_sha256") != expected_checkpoints[role]:
            raise ValueError(
                f"candidate ZIP {role} checkpoint does not match the frozen recipe"
            )
    source = build_receipt.get("source")
    source_files = source.get("files") if isinstance(source, dict) else None
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("build receipt lacks its pinned source-file digest map")
    return {
        "candidate_path": str(candidate),
        "candidate_sha256": zip_sha,
        "candidate_bytes": zip_bytes,
        "build_receipt_sha256": _sha256_file(build_receipt_path),
        "final17_archive_sha256": FINAL17_SHA256,
        "w18_evidence_archive_sha256": W18_EVIDENCE_SHA256,
        "expected_checkpoint_sha256": expected_checkpoints,
        "source_files": source_files,
        "candidate_member_sha256": member_hashes,
        "manifest": manifest,
        "build_receipt": build_receipt,
    }


def _copy_reference_checkpoints(
    payload: Path, workspace: Path, expected_hashes: dict[str, str]
) -> Path:
    """Extract only the four frozen FP32 checkpoints into the helper's expected layout."""
    import zipfile

    reference_root = workspace / "reference-checkpoints"
    reference_root.mkdir()
    archive_sources = {"w18": payload / "w18_evidence.zip"}
    archive_sources.update({role: payload / "final17.zip" for role in ROLES[1:]})
    destinations = {
        "w18": reference_root
        / "tl3c_w18final_distilled_s5"
        / "checkpoints"
        / "model.pt",
        "tiedlead_s5": reference_root
        / "tl3c_fttall_tlead_s5"
        / "checkpoints"
        / "model.pt",
        "tiedlead_s6": reference_root
        / "tl3c_fttall_tlead_s6"
        / "checkpoints"
        / "model.pt",
        "source4_lstm": reference_root
        / "models"
        / "primary-refit-03-s1t000000-lstm"
        / "model.pt",
    }
    for role in ROLES:
        archive_path = archive_sources[role]
        member_name = CHECKPOINT_MEMBERS[role]
        with zipfile.ZipFile(archive_path) as archive:
            members = _regular_zip_members(archive)
            if member_name not in members:
                raise ValueError(
                    f"reference archive lacks {role} checkpoint {member_name}"
                )
            payload_bytes = archive.read(member_name)
        if _sha256_bytes(payload_bytes) != expected_hashes[role]:
            raise ValueError(
                f"{role} FP32 checkpoint hash differs from the frozen recipe"
            )
        target = destinations[role]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload_bytes)
    return reference_root


def _load_synthetic_helper(project_root: Path, expected_hashes: dict[str, str]) -> Any:
    """Load the existing synthetic checker and replace only its old fixed hash tuple."""
    sys.path[:0] = [str(project_root), str(project_root / "src")]
    import importlib

    checker = importlib.import_module("scripts.check_final_tl3_synthetic")
    checker.EXPECTED_CHECKPOINT_SHA256 = tuple(expected_hashes[role] for role in ROLES)
    return checker


def _make_synthetic_sequence(models: list[Any]) -> Any:
    """Generate a seeded, correlated 20k-row raw112 stream from fit-only normalization."""
    import numpy as np

    mean = np.asarray(models[3].mean.detach().cpu().numpy(), dtype=np.float32)
    std = np.asarray(models[3].std.detach().cpu().numpy(), dtype=np.float32)
    scale = np.maximum(np.abs(std), np.float32(1e-4))
    rng = np.random.default_rng(20261112)
    rows = (
        mean[None, :]
        + rng.normal(0.0, 0.012, (ROW_COUNT, 112)).astype(np.float32) * scale
    )
    for start, stop in ((22, 52), (74, 104)):
        increments = rng.normal(0.0, 0.001, (ROW_COUNT, stop - start)).astype(
            np.float32
        )
        rows[:, start:stop] += np.cumsum(increments, axis=0) * scale[start:stop]
    for step in range(1, ROW_COUNT):
        if step % 97 == 11:
            rows[step, 52] += np.float32(0.08) * scale[52]
        if step % 251 == 19:
            rows[step, 0] += np.float32(0.05) * scale[0]
            rows[step, 11] -= np.float32(0.04) * scale[11]
    if not np.isfinite(rows).all():
        raise ValueError("generated 20k synthetic raw112 stream is not finite")
    return np.ascontiguousarray(rows, dtype=np.float32)


def _prepare_reference(
    payload: Path, verified: dict[str, Any], workspace: Path, data_path: Path | None
) -> dict[str, Any]:
    """Verify source and checkpoints, then compute FP32 reference streams without labels."""
    import numpy as np
    import torch

    project_root = _extract_project_tar(
        payload / "project.tar.gz", workspace / "project-unpacked"
    )
    source_root = verified["build_receipt"]["source"]["files"]
    for relative, expected in source_root.items():
        path = project_root.joinpath(*_safe_member(relative).parts)
        if _sha256_file(path) != expected:
            raise ValueError(f"pinned source member digest mismatch: {relative}")
    reference_root = _copy_reference_checkpoints(
        payload, workspace, verified["expected_checkpoint_sha256"]
    )
    helper = _load_synthetic_helper(
        project_root, verified["expected_checkpoint_sha256"]
    )
    checkpoint_paths = helper._checkpoint_paths(reference_root)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    models, _ = helper._load_reference_models(checkpoint_paths)
    sequence_ids: list[int] = []
    raw_sequences: list[Any] = []
    if data_path is None:
        raw_sequences.append(_make_synthetic_sequence(models))
        sequence_ids.append(91001)
        basis = "synthetic raw112 generated from final-fit LSTM normalization; engineering-only"
        data_sha256 = None
        labels_read = False
    else:
        resolved_data_path = data_path.resolve(strict=True)
        if not resolved_data_path.is_file():
            raise ValueError("timing data must resolve to a regular local file")
        data_sha256 = _sha256_file(resolved_data_path)
        final_fit = verified["manifest"].get("final_fit")
        source_hashes = (
            final_fit.get("source_sha256") if isinstance(final_fit, dict) else None
        )
        if not isinstance(source_hashes, dict) or data_sha256 != source_hashes.get(
            "valid"
        ):
            raise ValueError(
                "real timing data is not the exact receipt-bound official valid file"
            )
        from wnn.data import SequenceFile

        sequences = SequenceFile(resolved_data_path)
        if len(sequences) < 2:
            raise ValueError("real-data timing needs at least two canonical sequences")
        for index in (0, 1):
            sequence = sequences.read(index, targets=False, mask=False)
            if (
                sequence.features.shape != (ROW_COUNT, 112)
                or not np.isfinite(sequence.features).all()
            ):
                raise ValueError(
                    f"data row group {index} is not finite canonical raw112"
                )
            sequence_ids.append(int(sequence.seq_ix))
            raw_sequences.append(
                np.ascontiguousarray(sequence.features, dtype=np.float32)
            )
        basis = "official validation feature streams only; targets and masks were not read; engineering-only"
        labels_read = False
    references = [helper._reference_predictions(models, raw) for raw in raw_sequences]
    if any(
        reference.shape != (ROW_COUNT, 4, 2) or not np.isfinite(reference).all()
        for reference in references
    ):
        raise ValueError(
            "FP32 reference outputs have an unexpected shape or nonfinite values"
        )
    for index, (raw, reference) in enumerate(
        zip(raw_sequences, references, strict=True)
    ):
        np.save(workspace / f"raw112_{index}.npy", raw, allow_pickle=False)
        np.save(
            workspace / f"fp32_reference_{index}.npy", reference, allow_pickle=False
        )
    (workspace / "sequence_ids.json").write_text(
        json.dumps(sequence_ids, separators=(",", ":")), encoding="utf-8"
    )
    (workspace / "reference_basis.json").write_text(
        json.dumps({"basis": basis, "labels_read": labels_read}, sort_keys=True),
        encoding="utf-8",
    )
    checkpoint_hashes = {
        role: _sha256_file(path)
        for role, path in zip(ROLES, checkpoint_paths, strict=True)
    }
    del models
    return {
        "project_root": str(project_root),
        "reference_root": str(reference_root),
        "source_file_count": len(source_root),
        "checkpoint_sha256": checkpoint_hashes,
        "sequence_ids": sequence_ids,
        "sequences": len(raw_sequences),
        "rows_per_sequence": ROW_COUNT,
        "fp32_reference_shape_per_sequence": [ROW_COUNT, 4, 2],
        "basis": basis,
        "input_data_sha256": data_sha256,
        "input_data_resolved_path": str(resolved_data_path)
        if data_sha256 is not None
        else None,
        "labels_read": labels_read,
    }


def _cpu_flags() -> set[str]:
    """Read Linux CPU feature flags for safe ISA selection."""
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    for line in lines:
        if line.startswith(("flags", "Features")):
            return set(line.split(":", 1)[1].lower().split())
    return set()


def _available_isas() -> list[str]:
    """Return packaged ISA variants supported by the current CPU."""
    flags = _cpu_flags()
    return [name for name, required in ISA_REQUIREMENTS.items() if flags >= required]


def _run_one_isa(
    level: str, workspace: Path, verified: dict[str, Any], remaining_seconds: float
) -> dict[str, Any]:
    """Run exact-ZIP callback checks in a fresh process forced to one bundled ISA."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--one-isa",
        level,
        "--workspace",
        str(workspace),
        "--candidate-sha256",
        verified["candidate_sha256"],
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1.0, remaining_seconds),
            env=_safe_child_environment(level),
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return {"isa": level, "status": "timed_out", "passed": False}
    if result.returncode != 0:
        return {
            "isa": level,
            "status": "failed",
            "passed": False,
            "returncode": result.returncode,
            "stderr_tail": result.stderr[-3000:],
            "stdout_tail": result.stdout[-1000:],
        }
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "isa": level,
            "status": "invalid_worker_receipt",
            "passed": False,
            "stdout_tail": result.stdout[-3000:],
            "stderr_tail": result.stderr[-1000:],
        }
    return document


def _numeric_summary(actual: Any, reference: Any) -> dict[str, Any]:
    """Summarize served-vs-FP32 drift without turning it into an unapproved score gate."""
    import numpy as np

    difference = actual.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(difference)
    return {
        "rms": float(np.sqrt(np.mean(difference * difference))),
        "p99_absolute": float(np.quantile(absolute, 0.99)),
        "maximum_absolute": float(np.max(absolute)),
        "allclose_atol_2e_4_rtol_2e_4": bool(
            np.allclose(actual, reference, atol=2e-4, rtol=2e-4)
        ),
    }


def _run_isa_child(
    level: str, workspace: Path, expected_candidate_sha: str
) -> dict[str, Any]:
    """Import the exact packaged callback and test full-mix parity and sequence behavior."""
    import numpy as np

    _set_worker_limits()
    package_root = workspace / "candidate-package"
    if _sha256_file(workspace / "candidate.zip") != expected_candidate_sha:
        raise ValueError("candidate ZIP changed between integrity and runtime checks")
    spec = importlib.util.spec_from_file_location(
        "candidate_fullmix_solution", package_root / "solution.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot import solution.py extracted from the candidate ZIP")
    solution = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = solution
    spec.loader.exec_module(solution)
    callback = solution.PredictionModel()
    native_path = Path(sys.modules["wnn_native"].__file__).resolve()
    expected_native = (package_root / level / "wnn_native.abi3.so").resolve()
    if native_path != expected_native:
        raise RuntimeError(
            "callback imported a native library outside the exact ZIP ISA member"
        )
    if callback.model.serving_policy != "full_mix" or tuple(
        callback.model.active_member_indices
    ) != (0, 1, 2, 3):
        raise RuntimeError(
            "exact ZIP callback did not activate the declared full-mix members"
        )
    if not all(
        name.startswith("q8vnni") for name in callback.model.effective_precisions
    ):
        raise RuntimeError(
            "exact ZIP callback did not use the declared Q8 serving precision"
        )

    sequence_ids = json.loads(
        (workspace / "sequence_ids.json").read_text(encoding="utf-8")
    )
    raw_sequences = [
        np.load(workspace / f"raw112_{index}.npy", allow_pickle=False)
        for index in range(len(sequence_ids))
    ]
    references = [
        np.load(workspace / f"fp32_reference_{index}.npy", allow_pickle=False)
        for index in range(len(sequence_ids))
    ]
    reference_basis = _read_json(workspace / "reference_basis.json")
    if any(raw.shape != (ROW_COUNT, 112) for raw in raw_sequences) or any(
        reference.shape != (ROW_COUNT, 4, 2) for reference in references
    ):
        raise ValueError("prepared input or FP32 reference has the wrong shape")
    members_by_sequence = []
    member_pass_seconds = []
    callback_outputs = []
    callback_seconds = []
    for sequence_id, raw in zip(sequence_ids, raw_sequences, strict=True):
        members = np.empty((ROW_COUNT, 4, 2), dtype=np.float32)
        callback.model.reset()
        start = time.perf_counter()
        for step, row in enumerate(raw):
            point = _DataPoint(sequence_id, step, True, row)
            members[step] = np.asarray(
                callback.model.step_members(point), dtype=np.float32
            )
        member_pass_seconds.append(time.perf_counter() - start)
        if not np.isfinite(members).all():
            raise ValueError("full-mix member output contains NaN or infinity")
        members_by_sequence.append(members)

        blended = np.empty((ROW_COUNT, 2), dtype=np.float32)
        callback.model.reset()
        start = time.perf_counter()
        for step, row in enumerate(raw):
            prediction = callback.predict(_DataPoint(sequence_id + 1, step, True, row))
            if prediction is None:
                raise ValueError(
                    f"callback returned no prediction on synthetic row {step}"
                )
            blended[step] = np.asarray(prediction, dtype=np.float32)
        callback_seconds.append(time.perf_counter() - start)
        if not np.isfinite(blended).all():
            raise ValueError("full-mix callback output contains NaN or infinity")
        member_blend = np.stack(
            [
                np.asarray(solution.wnn_native.FinalTl3Predictor.blend_members(row))
                for row in members
            ]
        ).astype(np.float32, copy=False)
        if not np.array_equal(blended, member_blend):
            raise AssertionError(
                "exact callback output differs from the native nested FP64/FP32 blend"
            )
        callback_outputs.append(blended)

    first_raw = raw_sequences[0]
    first_members = members_by_sequence[0]
    first_sequence_id = sequence_ids[0]
    callback.model.reset()
    replay = np.empty_like(first_members)
    for step, row in enumerate(first_raw):
        replay[step] = np.asarray(
            callback.model.step_members(_DataPoint(first_sequence_id, step, True, row)),
            dtype=np.float32,
        )
    reset_replay_identical = bool(np.array_equal(first_members, replay))

    changed = first_raw.copy()
    changed[PREFIX_ROWS:, 0] += np.float32(0.25)
    changed[PREFIX_ROWS:, 11] -= np.float32(0.5)
    changed[PREFIX_ROWS:, 52] += np.float32(0.3)
    callback.model.reset()
    future = np.empty((ROW_COUNT, 4, 2), dtype=np.float32)
    for step, row in enumerate(changed):
        future[step] = np.asarray(
            callback.model.step_members(
                _DataPoint(first_sequence_id + 2, step, True, row)
            ),
            dtype=np.float32,
        )
    prefix_invariant = bool(
        np.array_equal(first_members[:PREFIX_ROWS], future[:PREFIX_ROWS])
    )
    if not reset_replay_identical or not prefix_invariant:
        raise AssertionError("callback reset or causal future-prefix check failed")

    all_members = np.concatenate(members_by_sequence, axis=0)
    all_references = np.concatenate(references, axis=0)
    all_callbacks = np.concatenate(callback_outputs, axis=0)
    member_drift = {
        role: _numeric_summary(all_members[:, index, :], all_references[:, index, :])
        for index, role in enumerate(ROLES)
    }
    reference_blend = np.stack([_blend_fp64_fp32(row) for row in all_references])
    blend_drift = _numeric_summary(all_callbacks, reference_blend)
    median_callback_seconds = float(np.median(callback_seconds))
    projected = median_callback_seconds * OFFICIAL_ROWS / ROW_COUNT
    return {
        "isa": level,
        "status": "passed_integrity_checks",
        "passed": True,
        "native_library_sha256": _sha256_file(native_path),
        "native_library_import_path": str(
            native_path.relative_to(package_root).as_posix()
        ),
        "sequence_ids": sequence_ids,
        "timing_sample_basis": reference_basis.get("basis"),
        "member_pass_seconds_20000_rows_by_sequence": member_pass_seconds,
        "callback_seconds_20000_rows_by_sequence": callback_seconds,
        "median_callback_seconds_20000_rows": median_callback_seconds,
        "callback_microseconds_per_row": median_callback_seconds * 1e6 / ROW_COUNT,
        "linear_projection_seconds_1970x20000_rows_at_measured_median": projected,
        "projection_comparison_w18_historical_seconds": HISTORICAL_W18_SECONDS,
        "projection_ratio_to_historical_w18": projected / HISTORICAL_W18_SECONDS,
        "projection_is_official_runtime_gate": False,
        "served_q8_vs_fp32_member_drift": member_drift,
        "served_q8_blend_vs_fp32_blend_drift": blend_drift,
        "callback_matches_served_member_blend_bitwise": True,
        "reset_replay_identical": reset_replay_identical,
        "future_prefix_invariant_first_1000_rows": prefix_invariant,
        "outputs_finite": True,
    }


class _DataPoint:
    """Minimal competition callback object required by the native predictor."""

    def __init__(
        self, seq_ix: int, step_in_seq: int, need_prediction: bool, state: Any
    ) -> None:
        self.seq_ix = seq_ix
        self.step_in_seq = step_in_seq
        self.need_prediction = need_prediction
        self.state = state


def _blend_fp64_fp32(members: Any) -> Any:
    """Reproduce the fixed nested FP64 accumulation and FP32 casts."""
    import numpy as np

    result = np.empty(2, dtype=np.float32)
    for target in range(2):
        inner = 0.0
        inner += float(members[0, target]) * 0.34
        inner += float(members[1, target]) * 0.33
        inner += float(members[2, target]) * 0.33
        middle = np.float32(inner)
        outer = 0.0
        outer += float(middle) * 0.9
        outer += float(members[3, target]) * 0.1
        result[target] = np.float32(outer)
    return result


def _run_main(
    payload: Path, recipe_path: Path, report_path: Path, data_path: Path | None
) -> dict[str, Any]:
    """Run immutable artifact checks, FP32 references, and each supported packaged ISA."""
    _set_worker_limits()
    if not payload.is_dir() or payload.is_symlink():
        raise FileNotFoundError(f"fixed payload directory is missing: {payload}")
    recipe = _read_json(recipe_path)
    verified = _verify_payload(payload, recipe)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with tempfile.TemporaryDirectory(prefix="final-cpu-check-") as temp:
        workspace = Path(temp)
        package_path = workspace / "candidate-package"
        shutil.copyfile(payload / "candidate.zip", workspace / "candidate.zip")
        with zipfile.ZipFile(payload / "candidate.zip") as archive:
            _regular_zip_members(archive)
            archive.extractall(package_path)
        reference_info = _prepare_reference(payload, verified, workspace, data_path)
        available = _available_isas()
        isa_reports: list[dict[str, Any]] = []
        for level in available:
            remaining = WALL_LIMIT_SECONDS - (time.monotonic() - started) - 5.0
            if remaining <= 0:
                isa_reports.append(
                    {"isa": level, "status": "not_run_deadline", "passed": False}
                )
                continue
            report = _run_one_isa(level, workspace, verified, remaining)
            isa_reports.append(report)
        supported_runs_pass = bool(available) and all(
            item.get("passed") is True for item in isa_reports
        )
        result: dict[str, Any] = {
            "schema": "wnn.final-tl3-cpu-integrity-check.v1",
            "status": "complete" if supported_runs_pass else "failed_or_incomplete",
            "started_utc": started_utc,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "limits": {
                "cpu_cores": 1,
                "address_space_bytes": MEMORY_LIMIT_BYTES,
                "wall_seconds": WALL_LIMIT_SECONDS,
            },
            "candidate": {
                "zip_sha256": verified["candidate_sha256"],
                "zip_bytes": verified["candidate_bytes"],
                "build_receipt_sha256": verified["build_receipt_sha256"],
                "serving_policy": "full_mix",
                "precision": "q8vnni",
                "active_members": list(ROLES),
            },
            "reference": reference_info,
            "supported_isa_variants": available,
            "isa_results": isa_reports,
            "integrity_and_causality_checks_pass": supported_runs_pass,
            "serving_precision_numeric_drift": "reported per member and blend; no new pass threshold imposed",
            "runtime_interpretation": "feature-only engineering timing; no timing projection establishes official scorer runtime or candidate quality",
            "independent_evaluation_sequences": 0,
            "score_computed": False,
            "independent_score_claim": "forbidden",
            "elapsed_wall_seconds": time.monotonic() - started,
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, report_path)
    return result


def main() -> None:
    """Run the fixed payload checker or its one-ISA subprocess worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=PAYLOAD)
    parser.add_argument("--recipe", type=Path, default=ROOT / "FROZEN_RECIPE_R1.json")
    parser.add_argument(
        "--report",
        "--out",
        dest="report",
        type=Path,
        default=OUTPUT / "final-cpu-report.json",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="optional official feature-only parquet for two 20k-row checks",
    )
    parser.add_argument("--one-isa", choices=tuple(ISA_REQUIREMENTS))
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--candidate-sha256")
    args = parser.parse_args()
    try:
        if args.one_isa is not None:
            if args.workspace is None or args.candidate_sha256 is None:
                parser.error("--one-isa requires --workspace and --candidate-sha256")
            document = _run_isa_child(
                args.one_isa, args.workspace, args.candidate_sha256
            )
        else:
            document = _run_main(args.payload, args.recipe, args.report, args.data)
    except Exception as error:
        document = {
            "schema": "wnn.final-tl3-cpu-integrity-check.v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "independent_evaluation_sequences": 0,
            "score_computed": False,
            "independent_score_claim": "forbidden",
        }
        if args.one_isa is not None:
            document["isa"] = args.one_isa
        else:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.report.with_suffix(args.report.suffix + ".tmp")
            temporary.write_text(
                json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, args.report)
        print(json.dumps(document, sort_keys=True, indent=2, allow_nan=False))
        raise SystemExit(2) from error
    print(json.dumps(document, sort_keys=True, indent=2, allow_nan=False))
    if (
        document.get("passed") is False
        or document.get("integrity_and_causality_checks_pass") is False
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
