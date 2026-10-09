from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import seal_sampled_cohort as sealer  # noqa: E402


def _copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)


def test_sealed_admitted_bootstrap_reaches_fold0_boundary_before_worker(tmp_path: Path, monkeypatch) -> None:
    pending = tmp_path / "pending"
    seal_receipt = sealer.seal_candidate(pending)
    pending_packet = json.loads((pending / "sampled_cohort.packet.json").read_text())
    admission = sealer._admission_payload(
        pending_packet["source"]["cohort_spec_sha256"],
        pending_packet["source"]["runner_overlay_manifest_sha256"],
        pending_packet["source"]["runner_worker_sha256"],
        pending_packet["source"]["code_manifest_sha256"],
        push_allowed=True,
    )
    assert seal_receipt["state"] == "sealed_pending_admission"
    admission_path = tmp_path / "fixture_root_admission.json"
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    admitted = tmp_path / "admitted"
    sealer.seal_candidate(admitted, admission_path)

    input_root = tmp_path / "input"
    source_mount = input_root / "datasets/vladkochetov07/wnn-d02-muon-loss-code-20261009-r5"
    overlay_mount = input_root / "datasets/vladkochetov07/wnn-sampled-recipe-runner-code-20261009-r1"
    _copy_tree(sealer.SOURCE_ASSET, source_mount)
    _copy_tree(admitted / "runner_overlay_asset", overlay_mount)
    monkeypatch.setenv("KAGGLE_INPUT", str(input_root))
    monkeypatch.setenv("KAGGLE_WORKING", str(tmp_path / "working"))
    monkeypatch.delenv("WNN_SAMPLED_COHORT_OFFLINE", raising=False)

    module_path = admitted / "kernel" / sealer.KERNEL_FILENAME
    spec = importlib.util.spec_from_file_location("sealed_sampled_bootstrap_fixture", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def mocked_fold0_boundary():
        raise RuntimeError("test reached fold0 mount boundary after admission validation")

    module.discover_fold0_asset_manifest = mocked_fold0_boundary
    assert module.main() == 1
    work_root = tmp_path / "working"
    failure = json.loads((work_root / "failure.json").read_text())
    assert failure["error"] == "test reached fold0 mount boundary after admission validation"
    assert failure["error_type"] == "RuntimeError"
    assert (work_root / "boot_marker.json").is_file()
    assert not (work_root / "out").exists()

    extracted_repo = work_root / "wnn"
    sys.path.insert(0, str(extracted_repo / "scripts"))
    sys.path.insert(0, str(extracted_repo / "src"))
    sys.modules.pop("loss_pair_gpu_worker", None)
    worker_path = extracted_repo / "scripts/sampled_recipe_worker.py"
    worker_spec = importlib.util.spec_from_file_location("extracted_sampled_worker", worker_path)
    assert worker_spec is not None and worker_spec.loader is not None
    worker_module = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(worker_module)
    loaded_cohort = worker_module.load_spec()
    worker_bootstrap = {
        "runner_overlay_manifest_sha256": pending_packet["source"]["runner_overlay_manifest_sha256"],
        "code_manifest_sha256": pending_packet["source"]["code_manifest_sha256"],
    }
    monkeypatch.setenv("WNN_SAMPLED_ROOT_ADMISSION_JSON", json.dumps(admission))
    validated_admission, worker_seconds = worker_module._validate_admission(
        loaded_cohort, worker_bootstrap
    )
    assert validated_admission == admission
    assert worker_seconds == 3300
