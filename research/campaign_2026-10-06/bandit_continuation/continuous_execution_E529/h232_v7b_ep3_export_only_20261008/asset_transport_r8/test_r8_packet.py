from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

R8_ROOT = Path(__file__).resolve().parent


def load_module(name: str, source: Path) -> ModuleType:
    """Load one packet helper by its stable path without changing sys.path."""
    specification = importlib.util.spec_from_file_location(name, source)
    if specification is None or specification.loader is None:
        raise ImportError(f"could not load packet helper: {source}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


packet = load_module("h232_r8_packet_builder", R8_ROOT / "build_r8_packet.py")
smoke = load_module("h232_r8_predictor_import_smoke", R8_ROOT / "smoke_predictor_import.py")


def test_title_and_provider_slug_are_canonical() -> None:
    assert packet.kernel_slug(packet.R8_TITLE) == packet.R8_ID.split("/", maxsplit=1)[1]


def test_rendered_kernel_only_adds_child_import_path_and_new_identity() -> None:
    r7_source = (packet.R7_ROOT / "kernel_packet_r7_v1/kernel.py").read_text()
    rendered = packet.render_kernel(r7_source)
    source_root = f'/ "{packet.SOURCE_ROOT_NAME}"'
    source_package = 'source_package = source_root / "src"'
    worker_import = "from export_worker import main as run_export"
    assert "KAGGLE_WORKING" in rendered
    assert source_root in rendered
    assert source_package in rendered
    assert rendered.index('os.environ["PYTHONPATH"]') < rendered.index(worker_import)
    assert f'KERNEL_ID = "{packet.R8_ID}"' in rendered
    assert "h232_export_asset_transport_receipt_r8.json" in rendered
    assert "h232_v7b_ep3_verified_payload_r8" in rendered
    original = rendered
    source_block_start = original.index(
        '    working_root = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))'
    )
    source_block_end = original.index(
        "    sys.path.insert(0, str(destination))", source_block_start
    )
    original = original[:source_block_start] + original[source_block_end:]
    for current, previous in (
        (f'KERNEL_ID = "{packet.R8_ID}"', f'KERNEL_ID = "{packet.R7_ID}"'),
        (
            f'KERNEL_TITLE = "{packet.R8_TITLE}"',
            'KERNEL_TITLE = "H232 GRU384 V7B EP3 Export 20261008 R7"',
        ),
        (
            "h232_export_asset_transport_receipt_r8.json",
            "h232_export_asset_transport_receipt_r7.json",
        ),
        ("h232_v7b_ep3_verified_payload_r8", "h232_v7b_ep3_verified_payload_r7"),
    ):
        assert original.count(current) == 1
        original = original.replace(current, previous)
    assert original == r7_source


def test_rendered_bootstrap_sets_the_workers_future_source_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r7_source = (packet.R7_ROOT / "kernel_packet_r7_v1/kernel.py").read_text()
    rendered = packet.render_kernel(r7_source)
    begin = '    working_root = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))'
    end = "    sys.path.insert(0, str(destination))"
    block = rendered[rendered.index(begin) : rendered.index(end)]
    namespace: dict[str, object] = {"Path": Path, "os": os}
    exec(compile(f"def configure():\n{block}", "<rendered-bootstrap>", "exec"), namespace)
    monkeypatch.setenv("KAGGLE_WORKING", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", "/existing/python/path")
    configure = namespace["configure"]
    assert callable(configure)
    configure()
    source_package = tmp_path / packet.SOURCE_ROOT_NAME / "src"
    assert os.environ["PYTHONPATH"] == f"{source_package}:/existing/python/path"
    assert not source_package.exists()


def test_child_environment_prepends_verified_src_and_preserves_prior_path(
    tmp_path: Path,
) -> None:
    source_package = tmp_path / "src"
    (source_package / "wnn").mkdir(parents=True)
    base_environment = {"PYTHONPATH": "/existing/site:/existing/project"}
    environment = smoke.predictor_environment(tmp_path, base_environment)
    assert environment["PYTHONPATH"] == f"{source_package}:/existing/site:/existing/project"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["WNN_THREADS"] == "4"
    assert environment["OMP_NUM_THREADS"] == "4"
    assert environment["MKL_NUM_THREADS"] == "4"
    assert environment["OPENBLAS_NUM_THREADS"] == "4"
    assert environment["WNN_LOADER_WORKERS"] == "2"
    assert environment["WNN_LOADER_RAM_GB"] == "0.30"
    assert environment["WNN_GPU_FRACTION"] == "0.75"
    assert base_environment["PYTHONPATH"] == "/existing/site:/existing/project"


def test_recorded_real_predictor_import_smoke_used_frozen_source() -> None:
    smoke = packet.read_json(packet.SMOKE_RECEIPT)
    assert smoke["status"] == "passed"
    assert smoke["return_code"] == 0
    assert smoke["source_archive_sha256"] == packet.SOURCE_ARCHIVE_SHA256
    assert "usage:" in smoke["stdout"].lower()
    assert smoke["child_environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert smoke["child_environment"]["PYTHONPATH"].endswith("/src")


def test_verified_remote_r7_payload_and_failure_are_bound_before_successor() -> None:
    receipt, terminal = packet.validate_r7_basis()
    asset = packet.read_json(packet.R7_ASSET_VERIFICATION)
    assert receipt["asset_bytes"] == 8_325_215
    assert asset["version_pin"] == f"{packet.ASSET_ID}/1"
    assert asset["expected_remote_identity_verified"] is True
    assert terminal["terminal_collection"]["file_count"] == 98
    assert terminal["diagnosis"]["exception"] == "ModuleNotFoundError: No module named 'wnn'"
    assert packet.SOURCE_ROOT_NAME == "h232_v7b_ep3_export_source_r2"


def test_r8_metadata_keeps_the_exact_r7_private_inputs() -> None:
    metadata = json.loads((packet.R7_ROOT / "kernel_packet_r7_v1/kernel-metadata.json").read_text())
    assert metadata["dataset_sources"][-1] == f"{packet.ASSET_ID}/{packet.ASSET_VERSION}"
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"


def test_built_r8_receipt_is_source_only_and_not_push_authority() -> None:
    receipt = packet.read_json(packet.BUILD_RECEIPT)
    kernel_path = packet.KERNEL_PACKET / "kernel.py"
    metadata_path = packet.KERNEL_PACKET / "kernel-metadata.json"
    metadata = packet.read_json(metadata_path)
    assert receipt["status"] == "prepared_for_independent_root_review"
    assert receipt["kernel_id"] == packet.R8_ID
    assert receipt["kernel_slug_matches_title"] is True
    assert receipt["reused_private_asset_dataset_version"] == f"{packet.ASSET_ID}/1"
    assert receipt["checkpoint_sha256"] == packet.CHECKPOINT_SHA256
    assert receipt["export_worker_sha256"] == packet.WORKER_SHA256
    assert receipt["science_or_model_changes"] is False
    assert receipt["training_profile_scoring_or_grading_enabled"] is False
    assert receipt["provider_actions"] == 0
    assert receipt["root_push_authorized"] is False
    assert packet.sha256_file(kernel_path) == receipt["kernel_source_sha256"]
    assert packet.sha256_file(metadata_path) == receipt["kernel_metadata_sha256"]
    assert metadata["id"] == packet.R8_ID
    assert metadata["title"] == packet.R8_TITLE
