from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/e540_v10_uploaded_packet"
V10_BUNDLE_SHA256 = "5be47eedbfddd2893ca4ae3cf6dcdb7cf762c95225e7694c10ef35b2a8cfd195"
V10_WORKER_SHA256 = "e96640d60a3874ecc4f530384d73cf46fec8857e5bb06da76de54f54a4844b26"


def _extract_packet(bundle: Path, destination: Path) -> None:
    """Extract only regular source files from the frozen packet bundle."""
    with tarfile.open(bundle, "r:gz") as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe packet member path: {member.name}")
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unexpected non-file packet member: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"packet member has no bytes: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _load_extracted_worker(root: Path, module_name: str) -> ModuleType:
    """Load the worker with its imports resolved from the extracted packet."""
    worker_path = root / "scripts/e540_kaggle_v10_export_worker.py"
    scripts_path = str(root / "scripts")
    source_path = str(root / "src")
    sys.path.insert(0, source_path)
    sys.path.insert(0, scripts_path)
    for name in ("e540_cli_preflight", "e540_export_validation"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(module_name, worker_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load extracted E540 worker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_runtime_environment(spec_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    for key, value in spec["environment"].items():
        monkeypatch.setenv(key, str(value))


def _approved_authorization(worker: ModuleType, packet_root: Path) -> dict[str, object]:
    binding_path = packet_root / worker.BINDING_PATH
    recipe_path = packet_root / worker.RECIPE_PATH
    binding_sha = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    recipe_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    subset_sha = hashlib.sha256((packet_root / worker.SUBSET_PATH).read_bytes()).hexdigest()
    expected = worker._expected_authorization(binding_sha, recipe_sha, subset_sha)
    return worker.validate_authorization(packet_root / worker.AUTHORIZATION_PATH, expected)


def test_actual_uploaded_v10_packet_reproduces_and_narrowly_repairs_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = FIXTURES / "bundle.wnn"
    runtime_spec = FIXTURES / "runtime_spec.json"
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == V10_BUNDLE_SHA256
    _load_runtime_environment(runtime_spec, monkeypatch)

    packet_root = tmp_path / "packet"
    packet_root.mkdir()
    _extract_packet(bundle, packet_root)
    old_worker_path = packet_root / "scripts/e540_kaggle_v10_export_worker.py"
    assert hashlib.sha256(old_worker_path.read_bytes()).hexdigest() == V10_WORKER_SHA256
    binding = json.loads(
        (
            packet_root
            / "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/v10_export_only/E540_V10_ARTIFACT_BINDING.json"
        ).read_text(encoding="utf-8")
    )
    recipe = json.loads(
        (
            packet_root
            / "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/v10_export_only/RECIPE_V10_EXPORT_ONLY.json"
        ).read_text(encoding="utf-8")
    )
    assert "recipe_sha256" not in binding

    old_worker = _load_extracted_worker(packet_root, "e540_v10_uploaded_worker")
    with pytest.raises(
        ValueError, match="V10 artifact binding, recipe, subset or split identity changed"
    ):
        old_worker._validate_upstream(binding, recipe)
    old_auth = _approved_authorization(old_worker, packet_root)
    assert old_auth["status"] == "approved"

    patched_worker = ROOT / "scripts/e540_kaggle_v10_export_worker.py"
    shutil.copyfile(patched_worker, old_worker_path)
    fixed_worker = _load_extracted_worker(packet_root, "e540_v11_patched_worker")
    fixed_worker._validate_upstream(binding, recipe)
    fixed_auth = _approved_authorization(fixed_worker, packet_root)
    cli_checks = fixed_worker.validate_prediction_clis()

    assert fixed_auth["authorization_id"] == old_auth["authorization_id"]
    assert [check["mode"] for check in cli_checks] == ["linear", "ple"]
    assert all(check["status"] == "parsed" for check in cli_checks)
    assert all(check["reader_requests_no_targets_or_mask"] for check in cli_checks)
