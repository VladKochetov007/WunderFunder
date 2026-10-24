"""Source-bound receiver checks and cached-teacher prerequisites for R4."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from campaign_io import canonical_sha256
from catalog import normalized_runtime_config
from parent_cache_config import validate_parent_config_binding

# Manifests and receipts contain heterogeneous JSON values at this I/O boundary.
PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]
EMULATION_ENV = "CUBLAS_EMULATE_SINGLE_PRECISION"


def read(path: Path) -> dict[str, Any]:
    """Read a bounded JSON object rather than an unbounded artifact payload."""
    if path.stat().st_size > 32_000_000:
        raise ValueError(f"oversized receipt: {path}")
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return result


def file_hash(path: Path) -> str:
    """Stream an artifact hash without materializing model weights."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def receiver_check_environment() -> dict[str, str]:
    """Copy the receiver environment with process-wide emulation disabled for checks."""
    environment = os.environ.copy()
    environment[EMULATION_ENV] = "0"
    return environment


def execute(
    command: list[str], log: Path | None = None, env: Mapping[str, str] | None = None
) -> None:
    """Execute one explicit receiver command, preserving failed command output."""
    child_environment = None if env is None else dict(env)
    if log is None:
        subprocess.run(command, cwd=ROOT, check=True, env=child_environment)
        return
    with (
        log.open("x") as stream,
        subprocess.Popen(
            command,
            cwd=ROOT,
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process,
    ):
        if process.stdout is None:
            raise RuntimeError("failed to capture receiver output")
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


def torchrun(ranks: int) -> list[str]:
    """Use the same locked Python environment for every distributed worker."""
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={ranks}",
    ]


def run_checks(campaign_root: Path, check_id: str) -> None:
    """Run complete R3/R4 checks once and publish a source-bound receipt pointer."""
    if not check_id or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for ch in check_id):
        raise ValueError("check-id must contain lowercase ASCII letters, digits, '-' or '_'")
    pointer = campaign_root / "acceptance.json"
    if pointer.exists():
        raise FileExistsError("this campaign already has an immutable acceptance receipt")
    directory = campaign_root / "checks" / check_id
    directory.mkdir(parents=True, exist_ok=False)
    for args, name in ((["-q"], "nvidia-smi.txt"), (["topo", "-m"], "topology.txt")):
        execute(["nvidia-smi", *args], directory / name)
    tests = [
        "test_b300_sequence_models",
        "test_b300_sequence_contracts",
        "test_large_teacher_runtime_guards",
        "test_sequence_full_fit_data",
        "test_sequence_recovery_cuda",
        "test_sequence_muon",
        "test_sequence_inner_splits",
        "test_sequence_cache_ensemble",
        "test_high_compute_campaign",
        "test_high_compute_runner",
        "test_sequence_precision",
        "test_sequence_attention_backends",
        "test_sequence_compile",
        "test_sequence_precision_cuda",
    ]
    check_environment = receiver_check_environment()
    execute(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *[f"tests/{name}.py" for name in tests],
            f"--junitxml={directory / 'pytest.xml'}",
        ],
        directory / "pytest.log",
        check_environment,
    )
    for script, name in (
        ("check_sequence_ddp.py", "ddp"),
        ("check_sequence_muon_ddp.py", "muon-ddp"),
    ):
        execute(
            torchrun(2)
            + [f"scripts/{script}", "--output-receipt", str(directory / f"{name}.json")],
            directory / f"{name}.log",
            check_environment,
        )
    execute(
        [sys.executable, str(PACKET / "admit.py"), "checks", "--directory", str(directory)],
        env=check_environment,
    )
    receipt = directory / "r4_acceptance.json"
    with pointer.open("x") as stream:
        json.dump({"receipt": str(receipt), "sha256": file_hash(receipt)}, stream, indent=2)
        stream.write("\n")


def accepted_checks(campaign_root: Path, source_hash: str) -> Path:
    """Require unchanged successful checks for this exact runtime and packet."""
    pointer = read(campaign_root / "acceptance.json")
    path = Path(pointer["receipt"])
    if file_hash(path) != pointer["sha256"]:
        raise ValueError("receiver acceptance receipt changed")
    receipt = read(path)
    if (
        receipt.get("status") != "passed"
        or receipt.get("extension") != "wnn.high-compute-checks.v1"
        or receipt.get("source_sha256") != source_hash
    ):
        raise ValueError("receiver checks do not match current R4 source")
    for name, expected in receipt["files"].items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f"receiver acceptance evidence changed: {name}")
    return path


def verify_admission_binding(run_directory: Path, raw_config_path: Path) -> None:
    """Require intact raw, normalized, and approved profile evidence before fitting."""
    run_root = run_directory.resolve()
    binding_path = run_root / "r4_admission_binding.json"
    normalized_path = run_root / "admission-normalized-config.json"
    approved_path = run_root / "approved_profile.json"
    raw_path = raw_config_path.resolve(strict=True)
    for path in (binding_path, normalized_path, approved_path):
        if not path.is_file():
            raise ValueError(f"R4 profile admission evidence is incomplete: {path}")
    binding = read(binding_path)
    expected_paths = {
        "raw_config_path": str(raw_path),
        "normalized_config_path": str(normalized_path.resolve()),
        "approved_profile_path": str(approved_path.resolve()),
    }
    if binding.get("schema") != "wnn.high-compute-r4-admission-binding.v1" or any(
        binding.get(name) != value for name, value in expected_paths.items()
    ):
        raise ValueError("R4 profile admission paths differ from the frozen cell")
    expected_hashes = {
        "raw_config_sha256": file_hash(raw_path),
        "normalized_config_sha256": file_hash(normalized_path),
        "approved_profile_sha256": file_hash(approved_path),
    }
    if any(binding.get(name) != value for name, value in expected_hashes.items()):
        raise ValueError("R4 profile admission evidence changed")
    raw_config = read(raw_path)
    normalized_config = read(normalized_path)
    if normalized_config != normalized_runtime_config(raw_config):
        raise ValueError("R4 normalized admission config differs from its raw cell config")
    approved = read(approved_path)
    identity = approved.get("identity")
    if (
        approved.get("approved") is not True
        or not isinstance(identity, dict)
        or identity.get("config_sha256") != canonical_sha256(normalized_config)
    ):
        raise ValueError("R4 approved profile does not bind its normalized cell config")


def parent_cache_guards(campaign_root: Path, cell: dict[str, Any], source_hash: str) -> None:
    """Bind KD prerequisites to completed G teachers and actual export parity."""
    inputs = cell.get("cache_inputs", [])
    parents = cell.get("parent_run_dirs", [])
    declarations = cell.get("cache_producers", [])
    if (
        not isinstance(inputs, list)
        or not inputs
        or not isinstance(parents, list)
        or len(inputs) != len(parents)
        or not isinstance(declarations, list)
        or len(inputs) != len(declarations)
        or not all(isinstance(item, str) for item in inputs + parents)
        or not all(isinstance(item, dict) for item in declarations)
    ):
        raise ValueError("KD cell lacks exact teacher-cache parent bindings")
    for cache_name, parent_name, declaration in zip(inputs, parents, declarations, strict=True):
        cache, parent = campaign_root / cache_name, campaign_root / parent_name
        if cache.resolve() != (parent / "teacher-cache").resolve():
            raise ValueError("KD cache is outside its declared teacher run")
        status = read(parent / "fit/terminal_status.json")
        manifest = read(cache / "manifest.json")
        parity = read(parent / "cache-parity.json")
        validate_parent_config_binding(
            campaign_root, cache_name, parent_name, declaration, manifest, status
        )
        if (
            status.get("status") != "completed"
            or status["identity"]["source_sha256"] != source_hash
            or manifest.get("complete") is not True
            or manifest["split_sha256"] != cell["split_sha256"]
            or manifest["producer"]["checkpoint_sha256"] != status["model_sha256"]
            or manifest["producer"]["provenance"]["identity"] != status["identity"]
            or parity.get("schema") != "wnn.sequence-cache-parity.v1"
            or parity.get("passed") is not True
            or parity["checkpoint_sha256"] != status["model_sha256"]
            or parity["split_sha256"] != cell["split_sha256"]
            or parity["manifest_sha256"]["eight"] != file_hash(cache / "manifest.json")
            or parity["manifest_sha256"]["one"]
            != file_hash(parent / "teacher-cache-one/manifest.json")
        ):
            raise ValueError("KD parent endpoint/export/parity identities do not agree")


def merged_cache_guard(campaign_root: Path, cell: dict[str, Any]) -> None:
    """Require the mean cache to use exactly the frozen ordered G teacher parents."""
    inputs = cell.get("cache_inputs", [])
    target = campaign_root / cell["teacher_cache"]
    if len(inputs) == 1:
        if target.resolve() != (campaign_root / inputs[0]).resolve():
            raise ValueError("single KD cache differs from its frozen teacher")
        return
    if len(inputs) != 3:
        raise ValueError("KD requires one or three frozen teacher caches")
    manifest = read(target / "manifest.json")
    producer = manifest.get("producer", {})
    expected = [file_hash(campaign_root / name / "manifest.json") for name in inputs]
    actual = [parent.get("manifest_sha256") for parent in producer.get("parents", [])]
    if (
        manifest.get("complete") is not True
        or manifest.get("split_sha256") != cell["split_sha256"]
        or producer.get("kind") != "mean_of_three_fp32"
        or actual != expected
    ):
        raise ValueError("merged KD cache differs from its frozen three-teacher composition")
